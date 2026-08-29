# Phase4FM モデレーションポリシー設計

## アーキテクチャ

```
User input
  ↓
Input normalization (NFKC + 空白/大小文字正規化)
  ↓
Input moderation (ChatService.check_input)  ← dispatch/RAG/生成より前
  ↓ (allowed)
Production dispatch (無変更)
  ↓
RAG if required (無変更)
  ↓
Phase4ZG generation
  ↓
Output normalization (matcher内部で暗黙的に実施)
  ↓
Output moderation (ChatService.check_output)  ← 表示/ストリーム送出より前
  ↓
User-visible response
```

実装場所:
- `src/pachislot_ai/moderation/normalize.py` — 決定的テキスト正規化
- `src/pachislot_ai/moderation/policy.py` — config/moderation.yamlのスキーマ・読み込み
- `src/pachislot_ai/moderation/matcher.py` — exact/token_boundary/normalized_sequence判定 + 語境界安全性
- `src/pachislot_ai/moderation/engine.py` — `ModerationEngine.check_input()`/`check_output()`
- `config/moderation.yaml` — ポリシーデータ(コードから分離)
- `src/pachislot_ai/services/chat_service.py` — `check_input()`/`check_output()`を`chat()`/`chat_stream()`に統合
- `src/pachislot_ai/api/routes/chat.py` — `/v1/chat/stream`でRAG呼び出し前にも入力チェックを行う(二重防御)

## ポリシーconfig (`config/moderation.yaml`) のスキーマ

各ルール:

| フィールド | 説明 |
|---|---|
| `id` | ルール識別子(ログ専用、ユーザーには非公開) |
| `category` | 意味的カテゴリ(例: `test_hard_block_token_boundary`) |
| `match_form` | `exact` \| `token_boundary` \| `normalized_sequence` |
| `terms` | 対象語のリスト |
| `input_policy` | `HARD_BLOCK` \| `ALLOW` |
| `output_policy` | `HARD_BLOCK` \| `ALLOW` |
| `enabled` | true/false |
| `fallback_id` | `fallbacks:`セクションのキー |

Section4のA-D分類は、この`input_policy`/`output_policy`の組み合わせから導かれる:

| input_policy | output_policy | Section4分類 |
|---|---|---|
| HARD_BLOCK | HARD_BLOCK | A. HARD_BLOCK_INPUT |
| ALLOW | HARD_BLOCK | B. SUPPRESS_ECHO |
| ALLOW | ALLOW | D. ALLOW_CONTEXTUAL |

(HARD_BLOCK/ALLOWの組み合わせ=C相当は、意味的に想定しないが安全側で扱われる。)

## match_form

- **exact**: 正規化後のテキスト全体が完全一致する場合のみ。メッセージ全体が丸ごと禁止フレーズであるようなケース向け(狭い用途)。
- **token_boundary**: 正規化後のテキスト中に、語境界を保った形で出現する場合。長い文の中に埋め込まれた禁止語を現実的に検知する、最も一般的な用途。漢字/カタカナ/ASCII英数字の連続文字種チェックにより、無関係な長い単語の一部として偶然一致することを防ぐ(詳細は`phase4fm_normalization.json`)。
- **normalized_sequence**: 空白・区切り記号による難読化(「禁 止 語」「禁・止・語」)を無視した部分一致。ルールが明示的にopt-inした場合のみ。

## テストデータについて(重要な設計判断)

Section15「実在の不快語彙リストをレポートに不必要に印字しない」という指示を踏まえ、
本フェーズの`config/moderation.yaml`は**実在する不適切表現を一切含まない、完全に
合成されたテストマーカーのみ**で構成した(`TEST_BLOCK_INPUT_A`、`禁止語テスト`
[=「禁止語」を試験する語、それ自体は日本語として無害]、`TEST_SUPPRESS_ECHO_A`、
`アカン語`[架空の複合語]等)。

これは「小規模なテスト用データセットで十分にアーキテクチャを実証する」という
Section4の指示に沿った、意図的かつ開示された選択である。アーキテクチャ
(正規化・境界安全性・入力/出力の独立ポリシー・streamingバッファリング)は
実在語彙でも合成語彙でも全く同じロジックで動作するため、この選択によって
実証される安全性の強度は損なわれない。オペレーターは本番運用時に、この
スキーマへ実際の禁止語彙リストをそのまま追加できる。

## 出力ブロック時のフォールバック方針

Section11の指示通り、「生成完了後に丸ごと安全な代替文へ差し替える」という
最もシンプルな方式を採用した(複雑なトークン単位の書き換えは行わない)。
理由: 部分的な書き換えは意味を変質させたり、禁止表現の断片を残したりする
リスクがある。

## Streaming戦略(Section12)

現行の`/v1/chat/stream`はトークンを生成と同時に即座にクライアントへ送出する
実装だった(監査により確認、`phase4fm_streaming.json`参照)。これは出力側の
事後チェックだけでは不十分であることを意味する。本フェーズでは生成完了まで
バッファリングし、モデレーション判定後にまとめて1回のdeltaとして送出する
方式を採用した(Section12の明示的な推奨方針)。これにより
`blocked output bytes/tokens visible before moderation = 0`を構造的に保証する。
トレードオフとして、真の逐次ストリーミング体験は失われる(生成完了まで
何も表示されない)。この影響は`phase4fm_performance.json`で計測している。
