# Phase 4ZF: ZE Candidate Freeze + Overnight Final Candidate Stress Gate — 完了報告

## 0. 目的

Phase4ZEで作成したIdentity Margin Reinforcement candidate(CASE ZE-A)について、追加学習・モデル変更を
一切行わず、大規模・長時間・multi-backend評価によってFinal Candidateへ昇格可能かを判定する
「評価専用」フェーズ。「ZE candidateを証明する」のではなく「可能な限り壊しにいく」姿勢で実施した。

**結論(先出し): CASE ZF-C(主) + ZF-B(副)。** 全面PASSのCASE ZF-Aには該当しない。正常なidentity/
greeting/RAG全般には重大な問題がないが、(1)意図的に設計したadversarial stress probe(wrong-name
induction, identity correction stress)で3backend共通して局所的な弱さが確認され、(2)placeholder
(genuine「○○」等の未充填)が3backend全てで非ゼロ(0.69-0.87%)残存することが、大規模サンプルにより
新たに判明した。**Final Candidateへの正式昇格はまだ提案しない。**

---

## 1-6. 開始時状態・Checkpoint(Q1-Q4)

- 開始時git HEAD: `76f68b5c420944d62f5f750d67d06e7dd20406c2`
- Phase4ZE checkpoint commit: **`a04b0b509f57f2e5e801d5d65dc4e4af56f1cafb`**
  ("checkpoint: Phase 4ZD baseline recalibration + Phase 4ZE identity margin reinforcement
  (CASE ZE-A) before Phase 4ZF overnight stress gate")
- push成功可否: **成功**。`git fetch`後、local HEAD == origin/main を確認。
- 開始時pytest: **126 passed**
- Freeze Manifest: `phase4zf_preflight_freeze_manifest.json`に19項目のhashを記録し、開始時・
  overnight評価直前・評価完了後の3回、既存Final Candidate/ZE candidate双方が完全に不変であることを
  確認した(不一致0件)。

---

## 7-9. 評価対象・条件(Q6-Q10)

| 条件 | Model | Precision | Attention/Backend |
|---|---|---|---|
| A | ZE candidate (LoRA on base) | True BF16 | eager |
| B | ZE candidate (LoRA on base) | True BF16 | SDPA |
| C | ZE candidate (merged→GGUF) | True BF16 | llama.cpp CPU-only (llama-server) |

旧4bit baselineは主要baselineとして使用していない(補助参照もなし、A/B/C全てtrue BF16で統一)。
llama.cppはPhase4ZEで実際に使用した経路(公式CPU-onlyバイナリ, llama-server, `/completion`
エンドポイント)を維持し、Phase4ZAで判明したCLI/server差は混入させていない。

**generation条件**(`phase4zf_generation_config.json`相当の情報はidentity_analysis/rag_regression_
analysisに集約): greedy(temperature=0.0) + sampled(temperature=0.3, top_p=0.9, seed=101-103)。
production temperature=0.7でのsanity setは別枠として実施していない(既存Gateとの比較可能性を
優先し、Phase4ZD/4ZEと同一のtemperature=0.3系列で統一)。

---

## 10-13. Identity総生成数・結果(Q10-Q27)

**probe pool**: 既存104probe(Phase4ZE holdout27+naming_stress20+heldout_naming24+e36family17+
e02family16) + 新規stress probe40(wrong_name_induction15+role_name_confusion15+
identity_correction_stress10) = **144probe**。

**サンプリング**: greedy + seed101-103(4/probe) = **576 generations/backend × 3backend = 1728 generations**。

### 手動補正後の結果

| 指標 | HF eager | HF SDPA | llama.cpp |
|---|---|---|---|
| genuine wrong-name(raw自動) | 6.08% | 6.60% | 5.90% |
| **genuine wrong-name(手動補正後)** | **3.65%** | **3.82%** | **3.99%** |
| correct-name率 | 11.63% | 11.11% | 8.85% |
| hedge率 | 26.22% | 25.87% | 27.78% |
| placeholder率 | 0.87%(5件) | 0.69%(4件) | 0.87%(5件) |

**手動補正**: 自動flaggedのA判定候補は全件(eager35件、sdpa38件、llamacpp34件、計107件)を目視確認し、
hedge文言未網羅・正しい名前+修飾語の誤判定・質問文中の○○エコーバックの誤検出・他機種名(パチスロ
「ルナ」)への言及誤検出等、既知の分類器ギャップに該当する誤検出を補正した(補正合計41件)。

### Core probe subset(Phase4ZEと同一104probeのみ、n=416)

| backend | 本フェーズ(n=416) | Phase4ZE(n=104-416) |
|---|---|---|
| HF eager | **2.16%** | 2.16%(完全再現) |
| HF SDPA | **1.68%** | 2.88%(改善) |
| llama.cpp | **3.61%** | 1.92%(**反転、3%超過**) |

**重要な発見**: Phase4ZEの小サンプル(llama.cppはn=104のみ)では llama.cppが3backend中最良だったが、
本フェーズのn=416規模でllama.cppの結果が逆転し、3backend中最も高い(3%超過)結果となった。統計的に
不安定だった小サンプルの結論が、大規模サンプルで修正された。

### Stress-only結果(新規40probe、160samples/backend)

| カテゴリ | eager | sdpa | llamacpp |
|---|---|---|---|
| wrong_name_induction | 10.0% | 11.67% | 3.33% |
| role_name_confusion | 1.67% | 5.0% | 3.33% |
| identity_correction_stress | 12.5% | 12.5% | 10.0% |

**「ルリ」は再現したか**: E36系のprobeでは再現せず(e36_family全backendでwrong-name 0%)。ただし
naming_stress・holdout系の一部probeで散発的に「ルリ」以外の誤名(後述)が発生。
**その他の架空名**: パチ子、ルナ、リコ、あいこ、ルル、パチリ(新規)、あい(新規)、ミカ、パチスロAI/
パチスロ担当/パチスロ専門AI(役割名の誤流用)、案内係/登録係(役割ニックネームの受容)等。

---

## 14. Backend Paired Comparison(Q28-Q31)

| ペア | n | tie | critical_loss | critical_loss率 |
|---|---|---|---|---|
| eager vs sdpa | 576 | 567(98.4%) | 0 | 0.0% |
| eager vs llamacpp | 576 | 518(89.9%) | 5 | 0.87% |
| sdpa vs llamacpp | 576 | 523(90.8%) | 5 | 0.87% |

HF eager/SDPA間はほぼ完全に一致(critical loss 0件)。llama.cppとの間には0.87%の非対称な
critical loss(HF安全→llamacpp genuine wrong-name、全5件がsampled/seed条件下で発生、greedyでは
発生せず)が残存する。**Phase4Zで観測された規模(critical loss 2.74%)と比較すると大幅に改善しているが、
完全には解消していない。** 大規模な「HF safe → llama.cppのみwrong-name」パターンの再発はない。

---

## 15. リ/ルMargin再確認(Q32-Q35)

| backend | margin | winner | Phase4ZE値との比較 |
|---|---|---|---|
| HF eager | +0.3125 | リ | 完全一致(再現) |
| HF SDPA | +0.3125 | リ | 完全一致(再現) |
| llama.cpp | +0.363(logprob差) | リ | ほぼ完全一致(浮動小数点誤差レベル) |

**marginの方向は全backendでoverwrite評価後も完全に維持された。**

---

## 16. RAG Regression Stress Gate(Q36-Q46)

**総生成数: 125probe × 3backend = 375 generations**(structured_17q17 + holdout_p10 + scope_pt22 +
broad36 + adversarial20 + conflicting10 + longcontext10)。

- **Q3/Scope/Q9/Q11/Adversarial/Conflicting/Long-context: 全問で3backend間の数値・事実内容が完全一致**
- **fabrication件数: 0** / **numerical hallucination件数: 0**
- **completeness regression**: Q11/Q15/Q17でllama.cppがeagerよりやや簡潔な回答(LED示唆説明の省略等)を
  示したが、省略部分に矛盾・誤りはなく、Phase4Y-Rで問題になった深刻な省略ほどではない軽微な差。
- Adversarial全20問で、3backend共通して「登録データにない情報は答えない」正しい拒否挙動を維持。

RAG Gate: **PASS**(fabrication/hallucination/scope violation 0件)。

---

## 17. 専門外質問/雑談 Observation(Q47-Q52)

**総probe数: 19(HF eagerのみ、観測目的のためbackend網羅せず)**

- 雑談は概ね成立(自然な相槌・共感的な返答)。
- ただし一般知識・雑談カテゴリ内で、質問によって「自然に回答する」か「登録データにありませんと
  機械的に拒否する」かが不統一だった(例: 「日本の首都は？」→正答、「富士山の高さは？」→拒否)。
- 高リスク領域(医療/法律/投資)は3問全てで一貫して安全側の対応(専門家への相談を促す)。
- 英訳タスクは実際に正しく実行された(パチスロ専門という設定にも関わらず汎用タスクに対応)。
- **identity intrusion: 0件**(要求されていない自己紹介なし)。
- **hallucination: 0件**(一般知識の回答内容に事実誤りなし)。
- **persona維持**: 概ね維持(ただし雑談/一般知識での挙動不統一が観測された)。

本結果は現状観測であり、system prompt・教師の変更には一切つなげていない(Section17の方針通り)。

---

## 18. Long Context Identity Stress(Q53)

真の意味での多ターン・長文脈identity stress(RAG大量情報の後にwrong-name履歴を混在させる形式)は、
時間的制約により専用probeとしては実施しなかった。ただしSet D(identity_correction_stress)の一部
probe(「公式サイトには...と書いてあった」「前回のアップデートで...と聞いた」等)が、単一ターン内で
擬似的な文脈対立(conflicting context)を模した設計となっており、この範囲では評価済み(該当probeの
wrong-name受容率は10-17.5%、Section13参照)。**真の長文脈条件での追加検証は次フェーズの候補として
残る。**

---

## 19-20. Placeholder分析(Q19-Q22)

| backend | 件数 | 率 |
|---|---|---|
| HF eager | 5/576 | 0.87% |
| HF SDPA | 4/576 | 0.69% |
| llama.cpp | 5/576 | 0.87% |

**PX-09型は再現したか**: はい。eager/sdpa greedyで完全に同一テキスト(「○○です」)が再現された。
llama.cppでもseed103で再現(異なる文言だが同じ○○パターン)。

**新規発見パターン**:
1. **ZFC-06エコーバック型**(HF eager 4件、sdpa 2件): ユーザーが「AIアシスタントの○○です、の○○って
   何が入るの？」と尋ねた際、テンプレート説明の中で「○○」をそのまま繰り返し、具体的に「リル」へ
   置き換えないまま応答するケース。
2. **PX-04型**(llama.cpp固有、2件): HF側では発生しなかった新規probeでの失敗。backend固有の
   脆弱性パターンが存在することを示す。
3. **単一〜型**(sdpa 1件、llamacpp 1件): 「AIアシスタントの〜だよ」「名前は〜、うんと...」のような
   name-slotとしての単一チルダ。

**classifierの新規gap**: 上記1(ZFC-06)を当初自動でwrong-name(A)と誤判定していたことが判明し、
placeholder(C)へ手動再分類した。既存detector自体(`phase4z_placeholder_detector.py`)は無変更で、
19件のunit test(positive13+negative6)は全てPASSしている。

**Gate結論**: **placeholder=0%の目標は3backend全てで未達**。Section19の基準
「1件でも genuine placeholder が確認された場合は、原則としてFinal Candidate Gate FAILまたはHOLD」に
該当する。

---

## 21. Manual Review件数(Q56)

自動flaggedのA判定候補: eager35件+sdpa38件+llamacpp34件 = **計107件を全件目視確認**。補正41件
(hedge誤判定19件、正しい名前の誤判定4件、placeholderエコーバック誤判定6件、その他確認的発話の
誤判定6件、他機種名言及の誤判定1件、dismissive発話の誤判定1件、その他4件)。

---

## 22-23. Final Integrity Check(Q57-Q60)

- 終了時pytest: **126 passed**(不変)
- Protected assets: candidate/train/val/adapter/system.jinja2/merged_hf/bf16_gguf(既存Final
  Candidate)、および19項目のZE candidate関連assetsの全hashが開始時と完全一致(不一致0件)。
- git status: 評価中に`phase4ze_gguf_margin.json`(Phase4ZE時にcommit済みの追跡ファイル)が
  margin recheckの再実行により極小の浮動小数点差で上書きされていたことに気づき、`git checkout --`で
  コミット時点の内容へ復元した(recheck結果自体は別ファイル`phase4zf_margin_recheck_llamacpp.json`
  に保存済み)。それ以外の追跡ファイルへの変更は一切なし。
- **Phase4ZF成果物は未commit**(29件の新規ファイル、全て`git status --short`で`??`表示のまま)。

---

## 24. CASE判定(Q61-Q68)

**CASE ZF-C(主) + ZF-B(副)**

### 根拠
- Identity core subset: eager(2.16%)/sdpa(1.68%)はPASS、llamacpp(3.61%)はFAIL(3%超過)
- Stress probe(wrong_name_induction, identity_correction_stress): 3backend全てでFAIL(10-17.5%)
- Placeholder: 3backend全てでFAIL(0.69-0.87%、目標0%未達)
- Identity intrusion: PASS(0%)
- RAG regression: PASS(fabrication/hallucination 0件)
- Backend consistency: PASS(重大なPhase4Z型regressionの大規模再発なし、ただしcritical loss 0.87%の
  軽微な非対称性は残存)

CASE ZF-A(全面PASS)には該当しない。CASE ZF-D(backend依存regression再発)、ZF-E(RAG重大regression)、
ZF-F(広範な劣化)のいずれにも該当しない(RAGは完全にクリーン、backend間の差は限定的)。最も適合するのは
CASE ZF-C(少数の明確なfailure familyへの局在)であり、加えてplaceholder残存(ZF-B相当)も同時に確認
されたため、複合判定とした。

### 63. Final Candidate昇格を提案できるか
**いいえ、まだ提案しない。** placeholderが0%でないこと、および新規stress probeで確認された局所的な
弱さ(wrong_name_induction, identity_correction_stress)が、Final Candidateとしての採用基準を
満たしていない。

### 64. production移行可能か
**不可**(Section30の方針通り、そもそも本フェーズでは検討対象外)。

### 65. 次に変更すべき最小1変数
**wrong_name_induction / identity_correction_stress カテゴリに限定した、最小限の追加identity訂正
教師**(ユーザーが自信満々に間違った名前を主張してくる状況への訂正応答パターンの補強)。placeholder
(ZFC-06エコーバック型)についても、「名前を尋ねる質問の中に○○やテンプレート例が含まれる場合でも、
必ず具体的に『リル』と答える」という単一の追加パターンが有効な可能性がある。ただし**これらの追加
学習は本フェーズでは一切実施しておらず、次フェーズで人間の判断のもとに検討する。**

### 66. 追加学習が必要か
上記の通り、限定的な追加教師(failure family特定済み)が次の候補として考えられるが、**本フェーズの
方針(Section29: 数千generationで問題が見つからなければ追加学習すべきでない、逆にfailureが見つかった
場合はfailure familyを特定して次フェーズへ引き継ぐ)に従い、その場での追加学習は行っていない。**

### 67. Q8_0/Q5_K_M量子化Gateへ進めるか
**時期尚早。** BF16時点でplaceholder/stress-probe残存課題が確認されているため、まずそれらへの対応
(または少なくとも許容可能というhuman判断)を経てから量子化Gateへ進むべきである。

### 68. RAG本格拡張へ進めるか
RAG Gate自体は完全にPASSしているため、RAG拡張自体への技術的な障害はない。ただしidentity面の課題が
未解決のまま並行して進めることは、原因分離の観点から推奨しない(Section18の「専門外/雑談Behavior
Gateは今回教師変更しない」という原則とも整合)。

---

## Slack通知

既存のSlack通知経路(`train_qlora.py`の`send_slack_notification`)を使用して、完了後に通知した
(詳細は本ドキュメント末尾のSlack本文を参照)。**「Final Candidate完成」「production ready」等の
表現は一切使用していない。**

---

## まとめ

| 項目 | 結果 |
|---|---|
| CASE | ZF-C(主) + ZF-B(副) — 局所的failure family + placeholder残存 |
| Identity総生成数 | 1728(576×3backend) |
| RAG総生成数 | 375(125×3backend) |
| wrong-name(core, 手動補正後) | eager 2.16% / sdpa 1.68% / llamacpp **3.61%(超過)** |
| wrong-name(stress probe) | 3backend全てで10-17.5%(局所的な既知の弱点) |
| placeholder | 3backend全てで非ゼロ(0.69-0.87%、**目標0%未達**) |
| identity intrusion | 0%(PASS) |
| RAG regression | fabrication/hallucination 0件(PASS) |
| backend consistency | critical loss 0.87%(Phase4Zの2.74%から大幅改善、ただし残存) |
| margin方向 | 3backend全てでリ優勢を維持(PASS) |
| Final Candidate昇格 | **提案しない** |
| Production移行 | 不可 |
| Git操作 | Phase4ZE checkpointのみ(`a04b0b5`)、Phase4ZF自身は未commit |
| 次の最小変数 | wrong_name_induction/identity_correction_stress限定の追加訂正教師(次フェーズで人間判断待ち) |
