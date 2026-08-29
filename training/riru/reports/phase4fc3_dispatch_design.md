# Phase4FC3 dispatch/evidence-arbitration設計

## 1. 実装ファイル

- `src/pachislot_ai/dispatch/conservative_dispatch.py`(新規): production dispatch。Phase4ZR/ZPの`training/riru/guard/phase4zr_conservative_dispatch.py`・`phase4zp_router.py`の**意味論のみ**を移植(コードをそのままコピーしていない)。FACTUAL_METRIC_KEYWORDS/GENERAL_PACHISLOT_TERMS/OOD_TOPIC_KEYWORDS等の語彙は値も含め完全に同一のまま再利用し、新規に`IDENTITY_PERSONA`カテゴリのみ追加した。
- `src/pachislot_ai/rag/evidence_arbitration.py`(新規): chunk側no-evidenceマーカーとstructured facts側の実データの矛盾を解消する統合層。entity_attribution.py・structured_lookup.pyのどちらの内部ロジックも変更していない。
- `src/pachislot_ai/rag/pipeline.py`(変更): `select_grounded_chunks()`の直後に`arbitrate()`呼び出しを1行追加。
- `src/pachislot_ai/services/chat_service.py`(変更): `build_rag_context()`冒頭でdispatch()を呼び、SMALL_TALK/IDENTITY_PERSONA/OOD_FACTUALと確信を持って判定された場合はRagPipelineを一切呼ばずNoneを返す。それ以外(PACHISLOT_FACTUAL/CONVERSATIONAL/UNKNOWN)は既存のRagPipeline呼び出しへそのまま委譲する。

## 2. 生産モード

1. SMALL_TALK — RAG context注入なし
2. IDENTITY_PERSONA — RAG context注入なし(SMALL_TALKの内部的特殊化として実装、テスト/報告では区別)
3. PACHISLOT_FACTUAL — 既存RAG pipelineフル使用
4. PACHISLOT_CONVERSATIONAL — 既存RAG pipelineフル使用(Section10の設計方針通り、単純化のため区別せず一律RAGへ委譲。詳細はSection10の項参照)
5. OOD_FACTUAL — RAG context注入なし
6. UNKNOWN — 既存RAG pipelineフル使用(常に、is_emptyでも省略しない。理由はSection3参照)

## 3. 重大な設計判断の記録(2件の安全性修正)

### 3.1 UNKNOWN + is_empty の扱い(当初案を撤回)

**当初実装**: UNKNOWNと判定され、かつRAG pipelineがis_empty=Trueを返した場合、RAG contextをNoneとして返す(小談義同様に扱う)案を実装した。

**発覚した問題**: 「GGプラスとは何か説明して」のような、実在entity+phantom接尾辞のクエリ(GENERAL_PACHISLOT_TERMSに一致しないため必ずUNKNOWNになる)で、ablation testを実施したところ、RAG context(空contextのfallback文言含む)を完全に省略すると、モデルが「GGプラスは、GGシリーズの最新作で...」という具体的で自信満々な架空説明を創作することを確認した(with fallback: 正しくdecline / without any context: 詳細な作り話)。これはこの投稿全体が解決してきたQ6/AT-F/RT-A系の fabrication パターンの再来であり、看過できない安全性regressionと判断した。

**最終実装**: UNKNOWNは常にRAG pipelineの結果をそのまま返す(is_emptyでも省略しない)。既存の空context fallback文言による安全網を維持する。この判断はSection11の「obvious non-RAG conversation」という条件を、SMALL_TALK/IDENTITY_PERSONA/OOD_FACTUALのような**確信を持って**分類できたケースのみを指すと解釈した結果である。

### 3.2 preference-question正規表現の「の？」終端パターン(発見・即時修正)

**当初実装**: hedge削減のため、`(ある|してる|した|の)[？?]$`という文末パターンでSMALL_TALKを検出しようとした。

**発覚した問題**: 手動dispatchテストで「GG中はどんな状態なの？」(機種固有の実在entityについての重要な事実質問)が、この「の？」終端に一致してSMALL_TALKへ**危険に誤ルーティング**されることを発見した(GG/GG中は機種固有語彙のためFACTUAL_METRIC_KEYWORDS/GENERAL_PACHISLOT_TERMSに一致せず、優先度の高い判定を素通りしてしまう)。

**最終実装**: 「の」単独の終端パターンを削除し、`(ある|してる|した)[？?]$`のみに限定した。この修正後、GT260全260件で再検証し、dangerous misroute=0を確認した。回帰テスト`test_machine_specific_factual_question_with_no_da_ending_not_misrouted_to_small_talk`を追加。

## 4. 評価中に発見した3つ目のバグ(evidence arbitration境界チェック)

`structured_lookup._value_matches_query_with_boundary()`の境界チェック用正規表現`[ァ-ー]`(カタカナ範囲U+30A1-U+30FC)が、区切り記号「・」(U+30FB、中点)を数値的に含んでしまうため、structured factsのラベル(例:「[低確A・低確B・天国準備滞在時]」)に対しentity名を照合する際、直前の「・」を誤って「単語継続」と判定し、境界安全な一致を妨げることが判明した。この関数はstructured_lookup.py内で無変更のまま維持し(Section26の指示)、evidence_arbitration.py内でstructured factのdetail文字列の「・」をスペースへ正規化してから照合する対処とした。

**既知の残存制約**: DBのラベルが「天国準備滞在時」のように、entity名に別の語が**直接融合**した複合語である場合、境界安全チェックは(意図的に)これを単体一致とみなさない。これは複合語誤衝突を防ぐという設計原則を優先した結果であり、バグではない。安全側(decline)に倒れるのみで、fabricationにはならない。

## 5. 意図的に対応しなかったこと

- PACHISLOT_CONVERSATIONALの一部クエリ(例:「パチスロ打ちに行こうと思うんだけど、おすすめある？」)で、既存RAG pipelineを通した結果hedgeが発生する例が複数観測された。Section10は「purely conversational: natural Riru response is allowed. If response requires factual claims: use grounded RAG」と、両者を区別する設計を示唆しているが、本フェーズでは安全性を優先し、CONVERSATIONALは一律で既存RAG pipelineへ委譲する単純な実装に留めた(SMALL_TALK同様にRAGを省略する設計へ変更するリスク・ベネフィットは未検証であり、Section21の評価予算内で新たなablationを行う余裕がなかったため)。
- small-talk hedgeの残存17/65(26.2%)は、全件`rag_context_injected=False`(RAG context完全に注入なし)であるにもかかわらず発生しており、Phase4ZG自身の学習済み挙動(個人的な好み・性格質問に「登録データにない」という定型文で応答する傾向)に起因することをablation testで確認済み。これは本フェーズのスコープ外(Section26で明示的に禁止されているretrain/Phase4ZG変更が必要)であり、意図的に未対応のまま報告する。
