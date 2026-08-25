# 使用モデルのライセンス情報

このプロジェクトがローカルで使用するLLM / Embeddingモデルのライセンス概要と、
将来どちらのLLMを正式採用したかを追跡するためのメモ。**モデルの重み自体は
Gitにコミットしない**（`D:\AI\models` はGit管理外）。本ファイルは参照情報のみを記録する。

最終更新: 2026-08-24（Phase 3.5: Qwen/Swallow A/B比較時点）

---

## 1. Qwen2.5-14B-Instruct（現行モデル・稼働中）

| 項目 | 内容 |
|---|---|
| 取得元リポジトリ | [`Qwen/Qwen2.5-14B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF) |
| ファイル | `qwen2.5-14b-instruct-q4_k_m-0000{1,2,3}-of-00003.gguf`（3分割） |
| 合計サイズ | 約 8.9 GB |
| 保存先 | `D:\AI\models\llm\` |
| **ライセンス** | **Apache License 2.0** |
| 商用利用 | 可（Apache 2.0は制限の少ない寛容なライセンス。著作権表示・変更点の明示等の一般的条件のみ） |
| 参照 | https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF （モデルカード `license: apache-2.0`） |
| 取得日 | Phase 0（2026-08-24） |

---

## 2. Llama-3.1-Swallow-8B-Instruct-v0.5（比較対象・Phase 3.5で追加）

| 項目 | 内容 |
|---|---|
| 開発元 | 東京科学大学（Tokyo Institute of Technology / 現・東京科学大学） [`tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5`](https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5) |
| GGUF取得元 | [`okamototk/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf`](https://huggingface.co/okamototk/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf)（第三者による量子化・再配布。imatrix量子化、日本語データセット [TFMC/imatrix-dataset-for-japanese-llm](https://huggingface.co/datasets/TFMC/imatrix-dataset-for-japanese-llm) を使用） |
| ファイル | `Llama-3.1-Swallow-8B-Instruct-v0.5_Q4_K_M.gguf` |
| ファイルサイズ | 4.92 GB（4,920,736,128 バイト、ダウンロード後に実測確認済み） |
| 保存先 | `D:\AI\models\llm\swallow\` |
| **ライセンス** | **デュアルライセンス**（両方に同時に従う必要がある） |
| ライセンス1 | **META LLAMA 3.1 COMMUNITY LICENSE AGREEMENT**（Release Date: 2024-07-23）。ベースモデル (Llama 3.1-8B-Instruct) 由来。原文を `D:\AI\models\llm\swallow\LICENSE` に保存済み |
| ライセンス2 | **Gemma Terms of Use**（Last modified: 2024-04-01）。v0.5のinstruction tuningでgemma-3-27b-itの挙動を模倣する形で学習されているため付随。原文を `D:\AI\models\llm\swallow\GEMMA_TERMS_OF_USE.md` に保存済み |
| 商用利用 | **条件付きで可** |

### Swallow (Llama 3.1 Community License) の主な条件

- 月間アクティブユーザーが7億人を超えるサービスで利用する場合、Metaから別途明示的なライセンスを取得する必要がある（本プロジェクトの社内評価用途では非該当）
- 再配布時は「Built with Llama」の表示、AIモデル名の先頭に "Llama" を含める、"Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved." の表示が必要
- Metaに対する知的財産訴訟を起こすとライセンスが終了する条項あり
- Acceptable Use Policy の遵守が必須

### Swallow (Gemma Terms of Use) の主な条件

- 商用利用の明示的な禁止はないが、[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) の遵守が必須
- 再配布する場合、本規約の制限事項を再配布先にも引き継ぐ契約とすること、規約全文の提供、改変箇所の明示、「Gemmaは ai.google.dev/gemma/terms のGemma利用規約に基づいて提供されている」旨の通知が必要

### 参照

- https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5
- https://huggingface.co/okamototk/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf
- https://ai.meta.com/llama/license/（Llama 3.1 Community License 参考）
- https://ai.google.dev/gemma/terms（Gemma Terms of Use 参考）
- https://ai.google.dev/gemma/prohibited_use_policy（Gemma Prohibited Use Policy）

取得日: Phase 3.5（2026-08-24）

---

## 2.5 LLM-jp-3-13B-Instruct（比較対象・Phase 3.6で追加検討）

| 項目 | 内容 |
|---|---|
| 開発元 | 大学共同利用機関法人 情報・システム研究機構 国立情報学研究所（NII） LLM勉強会（LLM-jp）大規模言語モデル研究開発センター [`llm-jp/llm-jp-3-13b-instruct`](https://huggingface.co/llm-jp/llm-jp-3-13b-instruct) |
| モデル構成 | Transformerベース、130億パラメータ、40層、hidden size 5120、attention head 40、context長4096 |
| **ライセンス（公式モデル）** | **Apache License 2.0** |
| モデルカード上の注記 | 「研究開発の初期段階であり、人間の意図や安全性への配慮に沿うようチューニングされていない」旨の記載あり（Apache 2.0上の追加利用制限ではないが、実運用前の留意事項として記録） |
| 商用利用 | 可（Apache 2.0、追加のAcceptable Use Policy等の明記なし） |
| 公式参照 | https://huggingface.co/llm-jp/llm-jp-3-13b-instruct （モデルカード `license: apache-2.0`） |

### GGUF版（第三者量子化）

| 項目 | 内容 |
|---|---|
| 取得元候補 | [`alfredplpl/llm-jp-3-13b-instruct-gguf`](https://huggingface.co/alfredplpl/llm-jp-3-13b-instruct-gguf) |
| 変換元 | 公式 `llm-jp/llm-jp-3-13b-instruct` と一致（README記載を確認済み） |
| 変換内容 | **量子化のみ（ファインチューニング・再学習なし）**。README記載の手順（npaka氏によるLLM-jp-3のGGUF変換手順を参照して変換）に基づく |
| 量子化バリエーション | IQ4_XS (7.43GB) / **Q4_K_M (8.35GB)** / Q8_0 (14.6GB) |
| ライセンス表示 | Apache 2.0（公式モデルと整合） |
| 採用予定ファイル | `llm-jp-3-13b-instruct-Q4_K_M.gguf`（8.35GB） |
| 保存予定先 | `D:\AI\models\llm\llm-jp-3-13b\` |
| 備考 | 別候補 `aipib/llm-jp-3-13b-instruct-gguf` はアクセス制限（HTTP 401）により内容確認不可のため不採用。`mmnga/llm-jp-3-13b-instruct3-gguf` および `mmnga/llm-jp-3.1-13b-instruct4-gguf` は元モデルが指定の `llm-jp-3-13b-instruct`（v1）とは異なるチェックポイント（instruct3 / 3.1系）のため対象外 |

取得日: Phase 3.6（2026-08-24、公式モデルカード・GGUF配布元の確認まで完了。ダウンロードはユーザー許可後に実施）

---

## 2.6 Llama-3-ELYZA-JP-8B（比較対象・Phase 3.7で追加検討）

| 項目 | 内容 |
|---|---|
| 開発元 | 株式会社ELYZA（日本） [`elyza/Llama-3-ELYZA-JP-8B`](https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B) |
| ベースモデル | `meta-llama/Meta-Llama-3-8B-Instruct` に対する追加事前学習＋instruction tuningで日本語性能を強化したモデル |
| パラメータ数 | 8B |
| **ライセンス** | **Meta Llama 3 Community License Agreement**（ https://llama.meta.com/llama3/license/ ） |
| 商用利用 | **可**。ただし月間アクティブユーザー数が7億人を超えるサービスで利用する場合は別途Metaへのライセンス申請が必要（本プロジェクトの社内評価用途では非該当） |
| 帰属表示義務 | 再配布・製品組込み時に (1) ライセンス全文の同梱、(2) 関連Webサイト/製品ドキュメントへの「Built with Meta Llama 3」の明示、(3) 派生モデル名の先頭に "Llama 3" を含めること、(4) 著作権表示 "Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright © Meta Platforms, Inc." の保持、が必要 |
| 禁止事項 | Meta Llama 3 Acceptable Use Policy 遵守が必須（違法行為・児童搾取・暴力助長・欺瞞行為等の禁止）。また、Llama Materialsの出力を「他の（Llama系以外の）大規模言語モデルの改善」に使用することを禁止する条項あり |
| ELYZA独自の追加制限 | **確認した限り、Meta標準のライセンス・AUP以外にELYZA独自の追加制限は見当たらない**（README・LICENSE・USE_POLICY.md・Noticeファイルをすべて確認済み） |
| LoRA / fine-tuning | 明示的な禁止条項なし。Hugging Face上に既に267件のadapter・13件のfinetune・66件のmergeが公開されており、コミュニティでの追加学習が実践されている実績あり |
| 商用利用条件の不明点 | **なし**（Swallowと同じMeta Llama系ライセンスの枠組みで、Gemmaのような追加のデュアルライセンス層がなく、むしろSwallowより単純明快） |
| 公式参照 | https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B （モデルカード `license: llama3`） |

### GGUF版（ELYZA公式配布）

| 項目 | 内容 |
|---|---|
| 取得元 | [`elyza/Llama-3-ELYZA-JP-8B-GGUF`](https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B-GGUF)（**ELYZA公式による直接配布**。第三者による量子化ではない） |
| 変換内容 | llama.cppによる**量子化のみ**（README記載）。GGUF版はGPT-4評価スコアが3.57（オリジナルの3.655からわずかに低下）と量子化による軽微な性能劣化が明記されている |
| 量子化バリエーション | `Llama-3-ELYZA-JP-8B-q4_k_m.gguf`（**Q4_K_M**, 4.92GB）が唯一のファイル |
| ライセンス表示 | Meta Llama 3 Community License（公式モデルと整合、LICENSE/USE_POLICY.md/Noticeを同梱） |
| 採用予定ファイル | `Llama-3-ELYZA-JP-8B-q4_k_m.gguf`（4.92GB） |
| 保存予定先 | `D:\AI\models\llm\elyza\` |
| 推奨system prompt（ELYZA公式README記載） | 「You are a sincere and excellent Japanese assistant. Unless otherwise instructed, always respond in Japanese.」（参考情報として記録。本プロジェクトでは既存の`system.jinja2`をそのまま使い、既存3モデルとの比較条件を統一する） |

取得日: Phase 3.7（2026-08-24、公式モデルカード・GGUF配布元の確認まで完了。ダウンロードはユーザー許可後に実施）

---

## 3. multilingual-e5-base（Embeddingモデル、稼働中）

| 項目 | 内容 |
|---|---|
| 取得元 | [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) |
| 保存先 | `D:\AI\models\embedding\multilingual-e5-base\` |
| ライセンス | MIT License |
| 商用利用 | 可 |
| 取得日 | Phase 0（2026-08-24） |

---

## 4. 正式採用モデルの追跡

| 日付 | 状態 | 採用モデル | 備考 |
|---|---|---|---|
| Phase 0〜3 | 稼働中 | Qwen2.5-14B-Instruct (Q4_K_M) | `.env` の `LLM_MODEL_PATH` で指定 |
| Phase 3.5 | A/B比較実施（Qwen vs Swallow） | （比較結果は `docs/llm_comparison.md` §1-12参照） | Swallowはプロンプト改善再テスト後も「採用保留」 |
| Phase 3.6 | 3モデル比較実施（+LLM-jp-3-13B） | **引き続きQwen2.5-14B-Instructを稼働継続** | LLM-jpは4基準すべて未達で「採用不可」。詳細は `docs/llm_comparison.md` §13-14参照 |
| Phase 3.7 | 4モデル比較実施（+Llama-3-ELYZA-JP-8B） | **引き続きQwen2.5-14B-Instructを稼働継続** | ELYZAはQ1-Q4・Q9・Q12-Q14の8問は完全正確、速度・VRAM効率も4モデル中最良だったが、Q6でSGGの数値をGGに誤転用する問題が確認され、判定ルール上「採用保留」。Swallow「採用保留」・LLM-jp「採用不可」の既存評価は変更なし。詳細は `docs/llm_comparison.md` §15-16参照 |
| Phase 3.8 | ELYZA専用改善プロンプト再テスト実施 | **引き続きQwen2.5-14B-Instructを稼働継続** | 専用system prompt (`system_elyza.jinja2`) でQ6の誤転用は解消したが、Q7型の「偽の出典ラベル付き捏造」が未解消。新規追加した5問のうち1問(N1)で、渡されていないPGGの数値を捏造する重大なハルシネーションを新規検出。判定は「採用保留」のまま。速度・VRAMへの悪影響はなし。詳細は `docs/llm_comparison.md` §17-18参照 |

**次のアクション**: `docs/llm_comparison.md` の推奨結果を確認し、正式採用が決まった時点でこの表の「稼働中」欄を更新すること。各モデルとも商用利用可能なライセンスだが、条件（帰属表示・禁止事項）が異なるため、実際にサービス化する際は採用モデルのライセンス条件を必ず遵守すること。ELYZAは「RAG検索の再現率改善」「より強制力の強い出力フォーマット制約」を試した上での再々テストが次の一手として考えられる（`docs/llm_comparison.md` §17.7参照）。
