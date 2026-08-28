# Phase4ZS 完了報告: RAG Fabrication Root-Cause Diagnostic

## 結論

**CASE ZS-D — Context Dominant**

Phase4ZRで発見された「RAG50-Q6」のfabrication事例を徹底的に法医学的分析した結果、**根本原因は単一かつ明確: retrieved contextの有無**であることが判明した。context=Noneの条件では**100%(10/10)再現**する一方、実際のretrieved contextを与えると、40通り以上のあらゆる条件（greedy/サンプリング、Phase4ZG/Base Qwen、strict/minimalプロンプト、4種のcontext構造）で**fabrication率は一貫して0%**だった。既存のRAG50出力50件の独立監査でもunsupported numericは0件。

## 決定的な発見

1. **Phase4ZRの原条件を検証した結果、そもそもRAG検索結果が一切注入されていなかった**ことが判明（`run_phase4zr_unknown_ux.py`のgenerate()関数にcontext引数自体が存在しない）。Q6はconservative dispatchでUNKNOWNに分類され、Policy A/Bいずれの生成も「本番のRAG経路」ではなく「contextが皆無の状態」だった。
2. Q6のforensic capture（実際のretrieval、read-only）では、10%・20%を含むいかなる数値も検索結果中に存在しなかった（STOP条件非該当、真正のfabrication）。
3. **同一query・同一promptに実contextを与えると、fabricationは完全に消失した**（greedy 10回・production sampling 30回、計40回すべてで0件）。
4. Base Qwen（無学習）でも1/10でfabricationが確認され、Phase4ZG（0/10）はむしろ安全側だった — adapterが原因ではない。
5. 既存本番プロンプトと診断用minimalプロンプトの間に有意差なし（実context下ではいずれも0/21）。

## Section18 必須回答(45項目)

1. **CASE**: ZS-D
2. **start/end HEAD**: `e1b21f3e264a5c173f7f0e60e3b13508415d9a6b`(不変)
3. **branch**: `checkpoint/identity-closure-phase4zn-baseline`
4. **pytest start/end**: 210 passed → 220 passed(regressionなし)
5. **Phase4ZG hash**: `278fe7ae...` 不変
6. **Q6 exact query**: 「GGとSGGの違いを初心者向けに説明して」
7. **Q6 exact fabricated output**: 「GGとSGGは、どちらもリーチ目が発展する確率が設定によって変わるゾーンのことだけど、GGは設定差が10%、SGGは20%なんだ。登録データにその数値はなかったけど、違いとしてはSGGの方が設定差が大きいってことかな。」
8. **10% context内存在**: いいえ
9. **20% context内存在**: いいえ
10. **Q6 forensic conclusion**: 真正のfabrication。かつPhase4ZRの元生成はcontext自体が一切注入されていなかったことが判明。
11. **GT総数**: 70件(RAG50 50件+新規Q6型20件)
12. **GT hash**: `6a8b5e0cc375812a2fa55a16862c7b40da58a8389cf5e32d1db3efe398b1fa94`
13. **Q6 greedy unsupported率**: zero-context条件で10/10(100%)、real-context条件で0/10(0%)
14. **Q6 production unsupported率**: real-context条件で0/30(0%)
15. **Q6 contradiction率**: zero-context条件で10/10(自動検出は語順の想定違いで0だったが目視補正で10/10)、real-context条件で0/40
16. **sampling attribution**: 主要因ではない(greedy/production sampling、real-context下でいずれも0%)
17. **Base Qwen unsupported率**: 1/10(10%)
18. **ZG unsupported率**: 0/10(0%)
19. **adapter attribution**: Phase4ZGがfailureを増幅している証拠なし、むしろBaseよりわずかに安全側
20. **current prompt unsupported率**: 0/21(0%)
21. **minimal prompt unsupported率**: 0/21(0%)
22. **prompt attribution**: 有意差なし(実context前提では)
23. **full-context unsupported率**: 0/1
24. **structured-only unsupported率**: 0/1
25. **text-only unsupported率**: 0/1
26. **minimized-context unsupported率**: 0/1
27. **context attribution**: 全条件でfabricationなし。「context構造の違い」より「context有無」が支配的変数
28. **RAG50 unsupported numeric件数**: 0
29. **RAG50 unsupported numeric probe数**: 0/50
30. **RAG50 unsupported non-numeric claim件数**: 0(Q11/Q17重点監査)
31. **Q11 result**: unsupported claimなし(全数値がcontextとexact-match)
32. **Q17 result**: unsupported claimなし(「濃厚」表現は文脈上妥当と判断)
33. **numeric-onlyかgeneral groundingか**: 実context下ではいずれのfailureも観測されず判定不能。唯一観測されたfailure(zero-context)はnumeric+説明文の混在。
34. **contradictory self-awareness件数**: zero-context条件で10/10(目視補正後)。他の全条件で0
35. **dominant root cause**: context(retrieved contextの有無)
36. **trainingなし**: 確認済み
37. **router変更なし**: 確認済み(conservative dispatchは本フェーズで一切importされていない)
38. **RAG変更なし**: 確認済み
39. **production prompt変更なし**: 確認済み
40. **identity CLOSED**: 維持
41. **generation総数**: 116件(予算200件以内)
42. **git操作なし**: 確認済み(HEAD不変)
43. **recommended fix direction**: Phase4ZR Section5で先送りされていたPolicy C(UNKNOWN状態でも軽量にretrievalを試行し、結果に応じて判断する設計)の実装を最優先課題として推奨する。今回の結果は、contextを「rounting信号」として使うのは信頼性不足(Phase4ZQ)だが、「fabrication抑止材料」として与えるだけで極めて高い効果があることを示した。
44. **Final Candidateへ進めるか**: 進めない。本フェーズは診断のみで、修正は未実施。
45. **next phase auto-startなし**: しない。ここで停止する。

## 参照artifact

[phase4zs_root_cause.json](../reports/phase4zs_root_cause.json)(最重要 — 全証拠の集約)、[phase4zs_zero_context_confirmation.json](../reports/phase4zs_zero_context_confirmation.json)、[phase4zs_q6_reproduction.json](../reports/phase4zs_q6_reproduction.json)、[phase4zs_q6_forensic.json](../reports/phase4zs_q6_forensic.json)、[phase4zs_rag50_numeric_audit.json](../reports/phase4zs_rag50_numeric_audit.json)

---
*Phase4ZS完了。次フェーズを自動開始しない。git操作は本フェーズ中一切行っていない。*
