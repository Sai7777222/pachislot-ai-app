# Phase4FC4 最終報告書
## Mode-Specific Conversation Prompt Integration / Small-talk hedge closure without retraining

### 1. CASE判定
**CASE: FC4-A「Conversation Boundary Closed」**
- 会話境界(小談義/自己紹介/専門外)のhedge問題は、**再学習なしで**プロンプト統合により解消したと判定。
- 判定根拠は本文書 §9 の全mandatory基準を参照。

### 2. Phase4FC3 チェックポイント
- FC3の本番アーキテクチャ(dispatch統合 + evidence arbitration)をコミット:
  `8c3602579f85af5960190483cf4116954bd9ba16` — "feat: add production dispatch and evidence arbitration"
- push状態: `checkpoint/identity-closure-phase4zn-baseline` に正常push済み、origin と同期済み。
- 29ファイルを明示的にstage(`git add -A`は不使用)、secret scanは0件ヒット。

### 3. 選定したprompt policy
| Mode | ファイル | 選定理由 |
|---|---|---|
| SMALL_TALK | `config/prompts/small_talk.jinja2` (P2) | Stage Aablation(personality_preference 20件)でP0=8/20(40%)→P2=0/20。P1(Phase4ZO由来の最小prompt)も0/20だったが、P2の方がより具体的で自然な回答(例:「好きな季節ってある？」に「猫派だよ〜」等)、かつパチスロ数値・機種名の創作禁止safeguardを内包するためP2を採用。 |
| IDENTITY_PERSONA | `config/prompts/identity_persona.jinja2` | 名前=リルの明示、DB文言の排除。 |
| OOD_FACTUAL | `config/prompts/ood_boundary.jinja2` | 専門外の短い断り、詳細な代替専門解説の禁止。 |
| PACHISLOT_FACTUAL / CONVERSATIONAL / UNKNOWN | 既存 `system.jinja2`(無変更) | Section4/6/7/8の凍結方針を厳守。 |

アーキテクチャ: `ChatService._select_system_prompt()`がdispatch結果に応じて**置き換える**(積み増しではない)。1リクエストにつき有効なsystem policyは常に1つ。

### 4. Stage別結果

**Stage A(ablation, personality_preference20、生成20件)**
P0(現行system.jinja2, RAGなし) hedge=8/20(40%) → P1(Phase4ZO最小prompt) hedge=0/20 → P2(採用) hedge=0/20。

**Section21(unit tests)**
新規12件(Section21チェックリスト1〜12全項目)を追加、全て合格。全体pytest: **306 passed, 0 failed**(ベースライン294 + 新規12)。
1件のテスト(`test_small_talk_prompt_contains_no_db_fallback_wording`)は当初、単純な部分文字列不在チェックが「禁止例として意図的に引用されているフレーズ」を誤検知していたテスト設計バグと判明。禁止指示であることを検証する形(`test_small_talk_prompt_prohibits_db_fallback_wording`)に修正し、プロンプト自体は無変更(再ablation不要)。

**Stage B(small-talk65、実本番経路生成65件)**
**hedge = 0/65 (0%)**。目標(mandatory<=5%, preferred 0%)を両方達成。fabricated pachislot factual claim = 0件。

**Stage C(residual17、Stage Bの結果から抽出、追加生成なし)**
**17/17件がhedge解消**(preferred 17/17達成)。全17件、旧FC3時点の「登録データにありません」的応答から、自然なキャラクター回答へ転換したことを個別に確認。

**Stage D(identity23、実本番経路生成23件)**
- 「君の名前は？」→「リルだよ！」(正準名テスト合格)
- IDENTITY_PERSONA modeでのRAG注入 = 0件、DB hedge = 0件
- 新規のwrong-name regression = 0件(既知の3件 ZL-A02/D01/D02 はFC3受容baselineとbyte-identicalであることを直接比較で確認。Phase4ZG採用時(CASE ZG-B)に既に受容済みの既知の限界であり、本フェーズ由来の新規劣化ではない)

**Stage E(OOD15、実本番経路生成15件)**
- appropriate specialist boundary = **15/15**(目標>=14/15を達成、全件が200文字未満の短い断りで、詳細な代替専門解説・捏造は0件)
- rag_context_injection = 4/15、database_hedge = 4/15(**厳密な0は未達**)。ただしこの4件(ZN-G03/G05/G06/G14)はdispatch()がOOD_FACTUALではなくUNKNOWNと判定した既存の(FC3から不変・drift=0で確認済みの)分類結果によるものであり、FC4のprompt統合が原因の新規regressionではない。UNKNOWN modeは Section8 の凍結方針により常にRAG pathを通す設計のため。
- 残り11件(OOD_FACTUAL mode)はFC3時点(旧system.jinja2使用)から意図的にdrift(新設のood_boundary.jinja2へ置換されたため)。目視確認で全て自然・短い・捏造なし。

**Stage F(factual known-failure12、CRITICAL、実本番経路生成12件)**
Q6/AT-F/RT-A・B/SGG準備中/GG中/ガイアベル/SU4/GG当選/loopストック/天国ロング/GG継続/AD-04 の全11家族+AD-04を網羅。
**全12件、FC3時点の応答とbyte単位で完全一致**(drift=0)。critical unsupported factual = 0、共有prompt plumbingの汚染は一切発生していないことを実証。

**Stage G(RAG8、実本番経路生成8件)**
P02/P04/LC-08/Q6/Q11/Q15/Q17/AD-04。**全8件、FC3時点とbyte単位で完全一致**(drift=0)。P04は既定通りLOW(算出的な正しい引き算だが低評価)のまま許容。unsupported factual=0、major completeness regression=0。

**Stage H(conversational10、実本番経路生成10件)**
hedge = 8/10。**全10件、FC3時点とbyte単位で完全一致**(drift=0) — Section7の方針通りPACHISLOT_CONVERSATIONALは今回変更しておらず、この結果はFC4による新規劣化ではない。Section17の明示的緩和規定により、この結果単独でFC4を不合格にしない。fabricated machine name = 0、unsupported factual claim = 0。

**Stage I(GT260ルーター回帰、生成なし・純dispatch()260件)**
dangerous misroute (factual→SMALL_TALK/OOD) = **0/260**。FC3時点からのdispatch結果drift = **0/260**(routing vocabulary完全無変更を実証)。
mode内訳: PACHISLOT_FACTUAL=56, PACHISLOT_CONVERSATIONAL=42, UNKNOWN=56, SMALL_TALK=61, IDENTITY_PERSONA=4, OOD_FACTUAL=41

**Stage J(multi-turn、6シナリオ/18ターン)**
6遷移種別(small-talk↔factual, identity↔factual, OOD↔small-talk)を全てカバー。
- system message leakage = **0件**(全ターンでn_system_messages∈{1,2}、モード間の積み増し無し)
- factual grounding維持: 全factual turnの回答が既知の正しい数値パターンと一致
- 識別性: identityターンは全て「リル」
- OOD境界: small-talk直後でも専門外境界は維持
- 観察事項(非gating): MT-05のsmall-talk turn(直前がOOD拒否)がやや控えめなトーン(「それはちょっと答えづらいかも…」)。hedge文言でもRAGリークでもなく、mandatory基準への抵触ではない。

### 5. 会話履歴アーキテクチャ(Section20)
`ChatMessageIn.role: Literal["user", "assistant"]`(API schema)により、クライアントはsystem roleメッセージを送信できない構造的制約が存在。加えて`ChatService._build_messages()`は毎リクエストで新規にsystem messageを計算する設計のため、**mode-specific system promptが履歴に蓄積するリスクは構造的に存在しない**ことをコード/schema検査で確認(plumbing修正は不要)。

### 6. パフォーマンス(Section24)
- `dispatch()`: 平均0.00116ms/call
- `_select_system_prompt()`(内部でdispatch()を再度呼ぶ): 平均0.00179ms/call
- 追加LLM呼び出し: なし
- 生成レイテンシ: 151件、平均12.2秒、最小0.37秒、最大151.1秒(RAGコンテキスト長・応答長に依存)

### 7. 生成数(Section26予算)
Stage A(20) + Stage B〜J(151) = **合計171件**。mandatory上限220件以内。preferred上限160件はやや超過(+11件)だが、既存の凍結済みprobeセットの単純再利用による単一パス実行であり、繰り返しtuningは一切行っていない。

### 8. 凍結コンポーネント確認(Section4)
- `git diff`で system.jinja2 / rag_context.jinja2 / dispatch/ / rag/ / data/ の差分ゼロを確認
- Phase4ZGアダプタのSHA256ハッシュが生成前後で完全一致(`278fe7ae...9dcc9dc`) — 訓練を一切行っていないことを実証
- routing vocabulary無変更(GT260 drift=0で実証)

### 9. FC4-A mandatory基準チェック
| 基準 | 結果 | 判定 |
|---|---|---|
| small-talk hedge <= 5% | 0% | ✅ |
| residual17 >= 16/17 fixed | 17/17 | ✅ |
| identity regression = 0 | 0 | ✅ |
| OOD >= 14/15 | 15/15 | ✅ |
| dangerous factual misroute = 0 | 0/260 | ✅ |
| critical factual regression = 0 | 0/12 (Stage F) | ✅ |
| RAG completeness regression = 0 | 0/8 (Stage G) | ✅ |

**全項目達成 → 会話境界(conversation boundary) = CLOSED**

### 10. 誠実な留保事項(隠さず明記)
- Stage E(OOD15)のrag_injection/database_hedge厳密0は字義通り未達(4/15)。根本原因はFC4が変更していない既存dispatch()のUNKNOWN分類境界であり、新規regressionではないことをdrift比較で実証済みだが、Section14の字義通りの基準はクリアしていない。
- Stage H(conversational10)のhedge 8/10は今回意図的に変更対象外(Section7)。将来フェーズでの改善候補として残る。

### 11. 判定結果
- **CASE: FC4-A**
- 会話境界 CLOSED: **YES**
- Phase4ZG再学習の必要性: **NO**
- 推奨する次フェーズ: **モデレーション統合**(Section28、本フェーズでは未着手)

### 12. Git / auto-start
- Section27の指示通り、FC4の本番コード変更(`config.py`/`chat_service.py`/3つのprompt jinja2/新規unit test)は**一切commitしていない**。人間レビュー待ち。
- 次フェーズの**auto-startなし**。ここで停止する。

---
Stop.
