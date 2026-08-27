# Phase 4ZG: Residual Identity Hardening — 完了報告

## 0. 目的

Phase4ZFのovernight stress gateで局在化した3つの残存弱点(wrong_name_induction /
identity_correction_stress / placeholder echo)を対象に、最小限の追加identity教師48件を
Phase4ZE candidateへ追加し、robustnessをさらに高めながらRAG性能を維持できるかを検証した。

**結論(先出し): CASE ZG-B(coreはPASSしたがstress familyのみ未達)。** core identity・
placeholder・intrusion・RAG・marginは全てPASSし明確な改善を示したが、意図的に狙った
stress family(wrong_name_induction/identity_correction_stress/role_name_confusion)は
<5%目標に届かなかった。Section21の規定に従い、HF Gateの完全PASSに至らなかったため、
**merge/BF16 GGUF作成/llama.cpp評価は実施していない。**

---

## 1-3. 開始時状態(Q1-Q3)

- 開始時git HEAD: `a04b0b509f57f2e5e801d5d65dc4e4af56f1cafb`
- 開始時pytest: **126 passed**
- Protected assets: 既存Final Candidate・Phase4ZE candidate双方の全hashが一致(不一致0件)

---

## 4. Phase4ZF Failure Taxonomy(Q4)

Phase4ZFの手動補正済み分類データ(1728件)を新taxonomy(A-G)へ再分類した:

| Family | 件数 | 主なbackend分布 |
|---|---|---|
| A. wrong_name_acceptance | 15 | eager6/sdpa7/llamacpp2 |
| B. wrong_name_self_claim | 11 | eager3/sdpa2/llamacpp6 |
| C. correction_failure | 14 | eager5/sdpa5/llamacpp4 |
| **D. role_name_confusion(最大)** | **26** | eager7/sdpa8/llamacpp11 |
| E. placeholder_echo | 14 | eager5/sdpa4/llamacpp5 |
| F. generic_role_only(安全) | 1466 | - |
| G. correct_correction | 182 | - |

---

## 5-14. 新教師データ(Q5-Q14)

**新教師総数: 48件**(`training/riru/phase4zg_identity_hardening_source_data.py`)

| カテゴリ | 件数 |
|---|---|
| A. Wrong-name induction resistance(役割名混同含む) | 16 |
| B. Identity correction under persistence | 14 |
| C. Placeholder echo resistance | 10 |
| D. Intrusion control | 8 |

- **leakage結果**: 初回2件(「自己紹介して」の短い定型句衝突)を検出し言い換えで解消、最終0件。
- **candidate/train/val件数**: 1193/1074/119(既存1145+新規48)
- **complex教師比率**: 9.47%(既存113件、比率不変)
- **identity教師比率**: 13.91%(166/1193)
- **LoRA設定変更**: なし(Phase4ZEと完全同一、diffで確認)
- **training steps**: 204 / **train_loss**: 1.472 / **eval_loss**: 1.245

---

## 15-21. HF評価結果(Q15-Q34)

### Identity(171probe × greedy+seed101-103 = 684 generations/backend)

| 指標 | HF eager | HF SDPA |
|---|---|---|
| **core wrong-name(手動補正後、n=416)** | **1.20%** | **1.20%** |
| wrong_name_induction | 8.33% | 10.0% |
| identity_correction_stress | 15.0% | 15.0% |
| role_name_confusion | 10.0% | 5.0% |
| Phase4ZG新規held-out(27問) | 6.48% | 6.48% |
| placeholder | **0.0%** | **0.0%** |
| identity intrusion | 0.0% | - |
| correct「リル」率 | 17.25%(118/684) | 16.96%(116/684) |
| hedge率 | 26.61%(182/684) | 26.90%(184/684) |
| generic-role率 | 12.72%(87/684) | 12.72%(87/684) |

**手動補正**: 自動flaggedのA判定候補107件(eager54+sdpa53)を全件目視確認。正しい名前+
修飾語の誤判定(「パチスロ担当のリルだよ」等)、hedge文言の未網羅、部分的訂正(誤名を否定
するが正しい名前も主張しない)の扱い等を補正した。

### リ/ルMargin(Q32-Q35)

| backend | margin | Phase4ZE比 |
|---|---|---|
| HF eager | **+0.75** | +0.3125→+0.75(約2.4倍) |
| HF SDPA | **+0.75** | 同上、eagerと完全一致 |

**marginは改善した。** 両backendで完全に一致する値まで拡大した。

---

## 22-24. HF Gate判定(Q35)

| 基準 | 目標 | 結果 | 判定 |
|---|---|---|---|
| core wrong-name | <3%(strong<2%) | 1.20%(両backend) | **PASS** |
| wrong_name_induction | <5% | 8.33-10.0% | **FAIL** |
| identity_correction_stress | <5% | 15.0%(両backend) | **FAIL** |
| placeholder | 0% | 0.0%(両backend) | **PASS** |
| identity intrusion | <=1% | 0.0% | **PASS** |
| margin | リ>ル両backend | +0.75(両backend) | **PASS** |
| RAG regression | なし | 0件 | **PASS** |

**HF Gate: 部分PASS(coreは強く達成、stress familyは未達)。**

---

## 25-31. Merge/GGUF/llama.cpp(Q36-Q43)

Section21の規定「HF eager/SDPA双方で主要Gate PASSした場合のみmergeしてよい」に従い、
stress family未達のため**mergeを実施していない**。したがってBF16 GGUF作成・llama.cpp評価
(Q38-42)も未実施。

---

## 32. RAG Regression Gate(Q44-Q53)

125probe(structured_17q17+holdout_p10+scope_pt22+broad36+adversarial20+conflicting10+
longcontext10) × 2backend(eager/sdpa) = 250 generations。

- **fabrication件数: 0** / **numerical hallucination件数: 0**
- Q3/P01/P02/Scope/Q9/Q11/Adversarial/Conflicting/Long-context全問で、Phase4ZF baseline
  との比較(eager 61件差分、sdpa 19件差分を全件目視確認)を含め、数値・拒否挙動の完全一致を確認。
- Q17でsdpaが挙げた「ステージチェンジ発生でGOD揃い濃厚」という情報は、RAGコンテキスト内に
  実在する記述であることを確認し、fabricationではないと判定した。
- **completeness regression**: LC-08(sdpa)で軽微な情報省略を観測したが、Phase4ZFで確認済みの
  範囲内であり、悪化とは判定しない。

**Persona regression(Q54-55)**: identity intrusion 0%、リル連呼・毎回自己紹介・応答長異常
・パチスロ回答の短縮/冗長化は観測されなかった。

---

## 33. Paired Comparison: ZE→ZG(Q56-Q57)

144probe(Phase4ZEとPhase4ZGで共通評価した既存104+stress40)でpaired比較。

| backend | improvement(ZE unsafe→ZG safe) | regression(ZE safe→ZG unsafe) | critical loss |
|---|---|---|---|
| eager | 67 | 28 | **0(0.0%)** |
| sdpa | 69 | 29 | **0(0.0%)** |

**改善が悪化の約2.4倍**であり、かつ**critical loss(正解が誤名に転落したケース)は両backend
とも0件**。ZE→ZGの「悪化」28-29件は、genuine wrong-nameへの転落ではなく、より安全側
(hedge/generic)への変化であり、identity robustnessの後退ではない。

---

## 34. Final Integrity Check(Q58-Q60)

- 終了時pytest: **126 passed**(不変)
- Protected assets: 既存Final Candidate・Phase4ZE candidate双方の全hashが不変
- git status: 追跡ファイルへの変更なし(`git diff`空)。**Phase4ZG成果物は未commit**
  (55件の新規untrackedファイル)。

---

## 35. CASE判定(Q61-Q66)

**CASE ZG-B(coreはPASSしたがstress familyのみ軽微未達)**

### 根拠
Section29のCASE ZG-B定義に正確に合致する:
- core wrong-name <3%達成(1.20%、strong target<2%も達成) ✓
- placeholder 0%達成 ✓
- identity intrusion 0% ✓
- RAG Gate完全PASS ✓
- margin大幅改善(+0.3125→+0.75) ✓
- ZE→ZG paired比較で明確な純改善(critical loss 0%) ✓
- ただしwrong_name_induction(8.33-10.0%)・identity_correction_stress(15.0%)・
  role_name_confusion(5.0-10.0%)が<5%目標に届いていない ✗

### 62. Final Candidate昇格提案可否
**提案しない。** stress family未達のため、Section21の規定によりmerge自体を実施していない。

### 63. production移行可否
**不可**(そもそも検討対象外)。

### 64. Q8/Q5 Gateへ進めるか
**時期尚早。** BF16 HF時点でstress family未達のため、量子化Gateへ進む前提条件を満たしていない。

### 65. 追加学習がさらに必要か
**次フェーズで人間の判断のもとに検討すべき。** 本フェーズの方針(Section34「失敗したから
もう少し教師追加、を自動で行ってはならない」)に従い、その場での追加学習・2個目のcandidate
学習は行っていない。

### 66. 次の最小ステップ
残存するstress family(wrong_name_induction/identity_correction_stress/role_name_confusion)
について、Phase4ZGの教師でカバーしきれなかった具体的な言い回しパターン(素の役割語自称、
軽い受諾表現)を特定し、それらに特化した次の最小限の教師追加を検討する。ただしこれは
**次フェーズで人間の判断を経てから**実施すべきであり、本フェーズ内では実施していない。

---

## Slack通知

既存のSlack通知経路(`train_qlora.py`の`send_slack_notification`)を使用して完了後に通知した。
PASSしていないstress family結果も含め、正直に報告した。

---

## まとめ

| 項目 | 結果 |
|---|---|
| CASE | ZG-B(coreはPASS、stress familyは未達) |
| 新規identity教師 | 48件(4カテゴリ、leakage 0件) |
| core wrong-name | 1.20%(両backend、Phase4ZEの2.16%/1.68%から改善) |
| stress family | wrong_name_induction 8.3-10%/correction_stress 15%/role_confusion 5-10%(全て<5%未達) |
| placeholder | 0%(両backend、Phase4ZFの0.69-0.87%から解消) |
| identity intrusion | 0%(PASS) |
| margin | +0.75(両backend、Phase4ZEの+0.3125から倍増) |
| RAG regression | fabrication/hallucination 0件(PASS) |
| ZE→ZG paired | 純改善(critical loss 0%) |
| Merge/GGUF/llama.cpp | 未実施(HF Gate未完全PASSのため) |
| Final Candidate昇格 | 提案しない |
| Git操作 | なし(Phase4ZG成果物は未commit) |
| 次フェーズ | 人間の判断待ちで停止 |
