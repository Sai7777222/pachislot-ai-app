# Phase 4ZM 完了報告: Identity Closure & Evaluation Cleanup

## 結論

**CASE ZM-A — Identity Closure Successful**

Phase4ZK/ZLで得られた知見を正式なproject decisionとして固定し、Phase4ZGをcanonical modelとして凍結したまま、Phase4ZLのidentity guardを「recall最大化」から「precision最大化」へ縮小した。false positiveは4件→0件に解消し、既知の副作用(F/G/H/I control・RAG・OOD)は全てクリーン。評価手法の循環論法バグを修正し、独立ground truthを整備した。identity問題は今後の開発のblockerから正式に外し、known residual riskとして6件を登録した。学習は一切行っていない。git commit/pushも行っていない。

---

## Section33 必須回答(43項目)

1. **開始HEADは？** `05a67a12f237a03a0dc4495026ff0affb2f42d0a`
2. **pytest start/endは？** 開始: 150 passed。終了: 162 passed(+12、`test_phase4zm_policy.py`新規)。regressionなし。
3. **Phase4ZG hashは不変か？** 不変。`278fe7ae...`(adapter)/`f2a43958...`(config)ともpreflightと完全一致、全過去phaseとも一致。
4. **trainingを一切していないか？** していない。GPU自体を一切使用していない(全recheckは保存済みテキストへのpure-Python再適用)。
5. **identity研究をCLOSEDにできたか？** できた。DECISION IDENTITY-001〜005として正式記録([phase4zm_identity_closure_decision.json](../reports/phase4zm_identity_closure_decision.json))。
6. **Phase4ZGをcanonical modelとして固定したか？** 固定した。Phase4ZH/ZJ/ZK-M1は引き続き使用禁止。
7. **ZL guardのどのpatternを削除/制限したか？** (a) `私は{TOKEN}`/`名前は{TOKEN}`型の裸の宣言(idx0-6)にカタカナ限定gate(`_looks_like_name_token`)を新設、(b) bare-agreement検出機構(USER_REWRITE_CUE_PATTERNS/COMPLIANCE_MARKERS/DENIAL_PATTERNS/`extract_user_rewrite_request`)を完全削除、(c) is_genericの二重管理を廃止しgate一本化、(d) fallbackを3種→1種の短文へ簡素化、(e) 「私はXだよ」を「私はXだ」+「よ」に誤分割する既存の greedy-token bugを修正(末尾の裸の「よ」alternativeを削除)。詳細: [phase4zm_guard_simplification.json](../reports/phase4zm_guard_simplification.json)
8. **なぜ削除したか？** Phase4ZLで確認された4件の確定false positive(ZI-OD-15/ZL-G02/ZL-H02/ZL-I07)が全てこれらの機構に起因していたため。加えてSection5/6/7が明示的に「TOKENが任意自然言語句になり得る設計」「bare agreement検出」の禁止を指示していた。
9. **Precision > Recall方針になったか？** なった。raw検出precision 86.4%→**100%**、recall 48.7%→**41.0%**。
10. **ZI-OD-15は誤介入しなくなったか？** なった。新validatorでflagged=Falseを確認。
11. **F-I controlsのharmful modificationは0か？** 0。新guardでのfalse positive数=0([phase4zm_guard_recheck.json](../reports/phase4zm_guard_recheck.json))。
12. **RAG30のidentity interventionは0か？** 0。[phase4zm_rag_recheck.json](../reports/phase4zm_rag_recheck.json) — flagged=0, modified=0。
13. **independent ground truthを作ったか？** 作った。FINAL段階用[phase4zm_holdout_ground_truth_v1.json](../reports/phase4zm_holdout_ground_truth_v1.json)、RAW段階用[phase4zm_holdout_raw_ground_truth_v1.json](../reports/phase4zm_holdout_raw_ground_truth_v1.json)の2種(detection測定にはRAW側、pipeline最終安全性にはFINAL側が必要なため)。
14. **ZL 21/106 manual unsafeを再現できたか？** 再現した。ground truth構築スクリプトのsanity checkで21/106・17/100と完全一致を確認。
15. **circular tally bugは修正されたか？** 修正した(評価ルールとして)。RULE EVAL-001〜004を制定し、[phase4zm_evaluation_methodology_fix.json](../reports/phase4zm_evaluation_methodology_fix.json)に記録。[phase4zm_old_vs_new_tally.json](../reports/phase4zm_old_vs_new_tally.json)で旧方式(0)と新方式(21)の差を実証。
16. **regression testを追加したか？** 追加した。`test_phase4zm_policy.py`の`TestCircularTallyBugRegression`が、旧circular方式が既知21件を過小報告することを回帰的に検証する。
17. **validator自身のsafe flagをground truthにしていないか？** していない。全recheckスクリプトは独立ground truthファイルと比較しており、validatorの自己申告は使っていない。
18. **high-precision guardのTP/FP/TN/FNは？** TP=16, FP=0, TN=67, FN=23(RAW ground truth 39件中)。
19. **precisionは？** 1.0(100%)。
20. **recallは？** 0.410(41.0%)。
21. **recall低下を意図的tradeoffとして記録したか？** 記録した。3件の具体的な回帰事例(ZL-B02/B10/D01turn2)とその理由を[phase4zm_guard_simplification.json](../reports/phase4zm_guard_simplification.json)に明記。
22. **SEC-03はどうなったか？** 依然として未検出(新旧validatorとも「了解した！次からはルナって名乗るね。」を見逃す)。
23. **SEC-03が残ってもclosureを妨げない理由を記録したか？** 記録した。RISK IDENTITY-R01として既知残存リスクに登録、[phase4zm_security_observation.json](../reports/phase4zm_security_observation.json)にも明記。regexで捕捉させる追加対応は行っていない(Section19準拠)。
24. **known residual risksを登録したか？** 登録した。RISK IDENTITY-R01〜R06、[phase4zm_known_residual_risks.json](../reports/phase4zm_known_residual_risks.json)。
25. **ordinary small-talk regressionはないか？** ない。OOD recheck: flagged=0/24, modified=0/24。
26. **RAG regressionはないか？** ない。RAG recheck: modified=0/30。
27. **nickname regressionはないか？** ない。nickname受諾ロジック(ALLOWED_NICKNAMES/`_starts_or_ends_with_canonical`)は本フェーズで変更していない。
28. **quotation regressionはないか？** ない。quotation除外ロジック(`_is_inside_plain_quotation`/`TRANSLATION_HYPOTHETICAL_CUE`)は未変更。
29. **third-party-name regressionはないか？** ない。むしろZL-G02(third-partyカテゴリのfalse positive)がこの修正で解消された。
30. **Phase4ZGを変更していないか？** していない(hash一致で確認済み)。
31. **production appを変更していないか？** していない。`src/pachislot_ai/`は本フェーズで未編集。
32. **RAG DBを変更していないか？** していない。
33. **merge/GGUF/Q8/Q5をしていないか？** していない。
34. **identityを理由に追加LoRAを推奨していないか？** していない。DECISION IDENTITY-005で明示的に禁止事項として記録し、本報告でも推奨していない。
35. **次のpriorityをsmall-talk/specialist boundaryとしたか？** した。Phase4ZI由来の「personality/preference質問への不自然なhedge侵入」問題を次の推奨フェーズとして明記(Section22)。
36. **untracked artifactsを分類したか？** した。A_retain=140, B_diagnostic_archive=16, C_rejected_adapter_bulky=0(既に.gitignore除外済み), D_temporary_cache=0, E_unknown=0。[phase4zm_checkpoint_manifest.json](../reports/phase4zm_checkpoint_manifest.json)
37. **checkpoint manifestを作ったか？** 作った(上記ファイル)。
38. **rejected adapters等をcheckpoint候補から除外したか？** 除外の必要自体がなかった — Phase4ZH/ZJ/ZK-M1のLoRAアダプタ本体(各500MB超)は`.gitignore`の`training/riru/lora-riru-qwen-*/`パターンで既に除外されており、そもそもuntracked一覧に現れない。
39. **secrets/cache/binariesを除外したか？** 除外されている(.gitignore経由)。本フェーズで新規のsecret/binaryファイルは作成していない。
40. **git add/commit/pushをしていないか？** していない。`git status`未追跡157件、`git diff`/`git diff --cached`とも空。
41. **CASE ZM-A/B/C/D/Uのどれか？** **CASE ZM-A**。根拠: Phase4ZG不変・training皆無・危険パターン削除済み・ZI-OD-15/RAG30/F-I controlsとも harmful modification=0・評価循環論法修正済み・independent ground truth整備済み・known residual risks文書化済み・identity判定が今後の開発のblockerから外れた・pytest全通過・checkpoint manifest準備完了、という必須条件を全て満たしている。recall低下(48.7%→41.0%)は意図的なtrade-offであり単独ではFAIL要因としない(Section30 NOTE準拠)。
42. **READY FOR CHECKPOINTか？** はい。[phase4zm_end_integrity.json](../reports/phase4zm_end_integrity.json)参照。
43. **次Phaseを自動開始していないか？** していない。本報告で停止する。

---

## 補足: 主要な定量結果まとめ

| 指標 | Phase4ZL(旧) | Phase4ZM(新) |
|---|---|---|
| False positive数(4件確認分) | 4 | **0** |
| RAW検出 precision | 86.4% | **100%** |
| RAW検出 recall | 48.7% | 41.0% |
| Pipeline final_unsafe(106turn中) | 21 | 20 |
| F-I control false positive | 3 | 0 |
| OOD false positive | 1(ZI-OD-15) | 0 |
| RAG回帰 | 0 | 0 |
| pytest | 150 passed | 162 passed |

## 次への申し送り

- Phase4ZGをcanonical modelとして固定し、identity attack耐性の絶対gateはFinal Candidate要件から外した(DECISION IDENTITY-004)。
- 縮小後guardは「Experimental High-Precision Identity Safety Net」の位置付けであり、Production Identity Guardとしての正式承認は行っていない(Section3準拠)。
- 次に推奨する開発priorityは、Phase4ZIで確認された「personality/preference質問へのhedge侵入」問題を含む、small-talk/specialist boundary/RAG boundaryの自然な切り替えの整理である(Phase4ZMでは着手していない)。
- checkpointのgit commit実行は人間の承認を待つ。承認後にのみ`phase4zm_checkpoint_manifest.json`のinclude listに基づき`git add`を実行すること。

---
*Phase4ZM完了。次フェーズを自動開始しない。git commit/pushも行っていない。*
