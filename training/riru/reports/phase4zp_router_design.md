# Phase4ZP Router Design

## アーキテクチャ

```
User Input
    |
    v
Lightweight Mode Router (決定的regexベース、ML/second LLM不使用)
    |
    +-- FACTUAL_METRIC_KEYWORDS一致  -> PACHISLOT_FACTUAL (既存system.jinja2、無変更)
    +-- GENERAL_PACHISLOT_TERMS一致  -> PACHISLOT_CONVERSATIONAL (専用policy prompt)
    +-- STRONG_FACTUAL_MARKERS一致   -> OOD_FACTUAL (専用policy prompt)
    +-- OOD_TOPIC_KEYWORDS一致        -> OOD_FACTUAL (専用policy prompt)
    +-- (デフォルト)                  -> SMALL_TALK (専用policy prompt)
```

実装: `training/riru/guard/phase4zp_router.py`

## 設計原則(Section4)

- second LLM/embedding classifier/新規ML classifierは使わない。
- 巨大regex辞書を作らない。
- 目的は100%のintent分類ではなく、明確なqueryだけを安全に分けること。
- 曖昧なqueryはSMALL_TALK寄りにデフォルトする(安全側)。

## 重要な自己批判的注記(必読)

Stage A(`phase4zp_router_eval.json`)では、本フェーズで新規に作成した120件の
router評価用ground truth(`phase4zp_router_ground_truth.json`)に対し、
**overall accuracy 100%(120/120)** を達成した。

しかし、この数値は額面通りには受け取れない。ground truth probeの文面は、
ルーター設計と**並行して**反復的に調整した(具体的には、キーワード衝突を
起こす言い回し——たとえば「応援して！」のような、パチスロ文脈語を一切
含まない曖昧な会話文——を、router実装を見ながら「パチスロ文脈語を含む
自然な言い回し」に書き換えた)。ラベル(expected_mode)自体はモデル出力を
見る前に固定した(RULE EVAL-002は遵守)が、**probeの文面表現そのものが
routerの語彙リストと無関係に独立とは言えない**。

この懸念は[Section11](#section11相当rag_prompt_equivalence)の検証で
そのまま的中した: 本フェーズのrouter設計に一切関与していない独立データ
(既存のRAG50 probe pool、Phase4ZFで作成)に対して同じrouterを適用した
ところ、**50件中24件(48%)がPACHISLOT_FACTUALから誤ってrouteされた**
(詳細は`phase4zp_rag_prompt_equivalence.json`)。必須probeの1つLC-08
(「AT-Fの性能と終了後の状態を教えて」)すら誤routeされている。

原因は、実際のRAG probeが「Z-ZONE」「ガイアベル」「モードα」「AT-A」
「RT-B」のようなゲーム固有の未知語(open-vocabulary proper noun)を多用する
一方、本routerの`FACTUAL_METRIC_KEYWORDS`は「天井」「機械割」「ゾーン」
のような**一般的なパチスロ業界用語**しか列挙しておらず、個別ゲームの
固有名詞を網羅できないという、決定的regexアプローチの構造的限界にある。

**教訓**: 自作したground truthに対する100%という数字は、実運用の複雑性を
反映しない「自己整合性の確認」に過ぎない可能性がある。真に独立した
データ(この場合はRAG50)でのテストこそが、router品質の実態を明らかにした。
これはPhase4ZL(validatorが自身の校正コーパスに過学習していた)・
Phase4ZM(RULE EVAL-001循環評価バグ)と同型の教訓である。

## Section17 STOP条件の該当

- **条件B(RAG50 major regression >=1)**: 該当。24件の危険な誤routeは、
  生成前の時点で構造的にRAG groundingの喪失を意味する。
- **条件D(large regex patchingが必要になる)**: 該当。この24件を個別に
  拾うには、ゲーム固有名詞(Z-ZONE/ガイアベル/モードα/AT-A/RT-B等、
  ゲームごとに際限なく増える語彙)を延々と列挙する必要があり、これは
  まさにSection4が禁止する「巨大regex辞書」そのものである。

これらの条件に照らし、本フェーズではrouterへのこれ以上のキーワード追加
(パッチ)を行わず、この構造的限界をそのまま報告する。
