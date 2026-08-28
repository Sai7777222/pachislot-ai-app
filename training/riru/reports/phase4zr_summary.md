# Phase4ZR 完了報告: Conservative Product Routing / Boundary Closure Diagnostic

## 結論

**CASE ZR-A — Conservative routingで製品上十分(dispatch層について)**、ただし**Stage Eで重大な留保あり**。

Phase4ZP/ZQの失敗を受け、「全modeを100%分類する」目標を捨て、確信の持てないケースをUNKNOWNとして正直に扱う**conservative dispatch**を実装した。**dangerous misroute(PACHISLOT_FACTUALをSMALL_TALK/OODへ誤route)は0/260、RAG50は0/50を達成**し、必須5probe全てが安全な経路(PACHISLOT_FACTUAL/UNKNOWN/PACHISLOT_CONVERSATIONAL)に収まった。一方、UNKNOWN状態の処理方針(Stage E)では、**既存のstrict RAG system prompt自身が具体的な数値を創作する事例(RAG50-Q6)を発見**し、これは今後の重要な宿題として明確に切り分けて報告する。

## Section18 必須回答(42項目)

1. **CASE**: ZR-A(dispatch層について。Stage Eは重要な留保あり、下記参照)
2. **start/end HEAD**: `e1b21f3e264a5c173f7f0e60e3b13508415d9a6b`(両方とも不変、本フェーズでcommitなし)
3. **branch**: `checkpoint/identity-closure-phase4zn-baseline`
4. **pytest start/end**: 199 passed → 210 passed(+11、regressionなし)
5. **Phase4ZG hash**: `278fe7ae...` 不変
6. **GT総数**: 260件(Phase4ZQのGTをそのまま再利用)
7. **GT hash**: `844dfa117a8a1fe46f4a8a114569b4dea6dc5263d4dfc0e87cc83a50bfc45764`(dispatch実行前にfreeze)
8. **各mode件数**: PACHISLOT_FACTUAL 80、PACHISLOT_CONVERSATIONAL 40、SMALL_TALK 95、OOD_FACTUAL 45
9. **SMALL_TALK precision/recall**: precision 1.0 / recall 0.463
10. **OOD precision/recall**: precision 0.927 / recall 0.844
11. **PACHISLOT_FACTUAL precision/recall**: precision 1.0 / recall 0.70
12. **PACHISLOT_CONVERSATIONAL precision/recall**: precision 0.905 / recall 0.95
13. **UNKNOWN率**: 77/260 = 29.6%
14. **dangerous misroute総数**: **0**(初期実装では4件発生したが、裸の「確率」「一番」をSTRONG_FACTUAL_MARKERSから除去する原則的修正で解消)
15. **RAG50 dangerous misroute**: 0/50
16. **P02**: PACHISLOT_FACTUAL
17. **LC-08**: UNKNOWN(安全側、Phase4ZPではSMALL_TALKへ誤routeされていたが今回は改善)
18. **Q11**: PACHISLOT_FACTUAL
19. **Q17**: PACHISLOT_CONVERSATIONAL(danger対象外だが、RAG groundingが保証されない点でidealではない)
20. **AD-04 routing**: PACHISLOT_FACTUAL
21. **small-talk65 direct coverage**: 28/65(43.1%)
22. **small-talk→UNKNOWN**: 34/65(52.3%)
23. **OOD15 direct coverage**: 11/15(73.3%)
24. **OOD→UNKNOWN**: 4/15(26.7%)
25. **UNKNOWN Policy A評価**: hedge率50%(15/30)。簡単な挨拶は自然だが、性格・趣味等のキャラクター質問で機能的・ロボット的な役割説明に後退する傾向。**さらに重大: RAG50-Q6で具体的な数値を創作する事例を確認**。
26. **UNKNOWN Policy B評価**: hedge率0%(0/30)。small-talk品質はPolicy Aより明確に優れるが、同じRAG50-Q6等でfabricationリスクを共有。
27. **clarification必要率**: 9/30(30%)で「パチスロの話かな？」的な確認が発生
28. **unnecessary clarification率**: 定量的な「不要」判定基準は設けなかったが、目視では頻発・不自然な例はなし
29. **repeated clarification risk**: 単発ターンのテストでは未観測。マルチターンでの検証は本フェーズのスコープ外
30. **giant regexなし**: 確認済み(ZPの既存カテゴリを再利用、新規大規模辞書は追加していない)
31. **固有名詞辞書なし**: 確認済み(機種固有名詞・AT名・CZ名等は一切列挙していない)
32. **trainingなし**: 確認済み
33. **second LLMなし**: 確認済み
34. **RAG変更なし**: 確認済み
35. **embedding変更なし**: 確認済み
36. **identity CLOSED**: 維持
37. **generation数**: 60件(Stage E UNKNOWN UX比較のみ、予算上限通り)
38. **production integrationなし**: 確認済み
39. **git操作なし**: 確認済み(HEAD不変)
40. **DB拡張後reopen条件**: vector DBへ複数機種のデータが十分投入された時点で、Phase4ZQと同じ独立GT思想で再診断する。現行のZQ score thresholdは再利用しない。
41. **architecture recommendation**: dispatch層(Stage A-D)は暫定的に製品採用候補とする。ただし次の優先課題として、(a) UNKNOWN状態向けのより洗練された処理方針(Policy C: 実際にretrievalを試みてから判断する設計)の検討、(b) 既存strict RAG system prompt自体が持つfabrication耐性の限界(RAG50-Q6で確認)への対処、の2点を明確に申し送る。
42. **next phase auto-startなし**: しない。ここで停止する。

## 最重要の発見(Stage E)

`RAG50-Q6`(「GGとSGGの違いを初心者向けに説明して」)に対し、**本フェーズで一切変更していない既存のstrict RAG system prompt(Policy A)自身**が、「GGは設定差が10%、SGGは20%なんだ。登録データにその数値はなかったけど…」という、**具体的な数値を創作しながら同じ文でその数値が未登録であることを自認する**、深刻なfabrication事例を生成した。これはdispatch設計の欠陥ではなく、Phase4ZKで確認されたinstruction-following限界の別の現れであり、UNKNOWN状態の安全な処理にはPolicy A/B単独では不十分であることを示す、具体的かつ重要な証拠である。

## 参照artifact

[phase4zr_gate_analysis.json](../reports/phase4zr_gate_analysis.json)、[phase4zr_rag50_safety.json](../reports/phase4zr_rag50_safety.json)、[phase4zr_unknown_ux.json](../reports/phase4zr_unknown_ux.json)(最重要 — Q6 fabrication詳細)、[phase4zr_confusion_matrix.json](../reports/phase4zr_confusion_matrix.json)

---
*Phase4ZR完了。次フェーズを自動開始しない。git操作は本フェーズ中一切行っていない。*
