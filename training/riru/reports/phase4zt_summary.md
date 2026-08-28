# Phase4ZT 完了報告: UNKNOWN Retrieval Policy C Integration & Safety Validation

## 結論

**CASE ZT-A — Policy C Successful**、**採用variant: C3(Conservative Mixed Policy、字句重なりベース)**

Phase4ZSで特定された「UNKNOWN + context皆無 + strict RAG generation」という危険な経路を、Policy C(3案比較)で完全に閉じた。C1(常にRAG化)はsmall-talk hedgeを27.1%まで再発させ、C2(常にclarification)は安全だがfactual回答能力を完全に放棄する。**C3(字句重なりに基づく振り分け)が両者の長所を両立**し、全gateを達成した。

## Section18 必須回答(44項目)

1. **CASE**: ZT-A
2. **selected Policy C variant**: C3(Conservative Mixed Policy)
3. **start/end HEAD**: `e1b21f3e264a5c173f7f0e60e3b13508415d9a6b`(不変)
4. **branch**: `checkpoint/identity-closure-phase4zn-baseline`
5. **pytest start/end**: 220 passed → 233 passed(regressionなし)
6. **Phase4ZG hash**: `278fe7ae...` 不変
7. **conservative dispatch unchanged**: 変更なし(無編集、テストで判定結果の同一性も確認)
8. **GT総数/hash**: 260件 / `6c5a2357794784324f86601e90e3e196a2081beec166df709cb2577928a4b48a`(Phase4ZRのGTを再利用)
9. **UNKNOWN総数**: 77
10. **contextless strict-RAG generation数**: **0**(全3variant、mandatory invariant達成)
11. **Q6 unsupported numeric**: 0(全variant、greedy10+production20)
12. **Q6 contradiction**: 0
13. **UNKNOWN unsupported numeric**: 0/77(C3)
14. **UNKNOWN unsupported non-numeric**: 目視サンプリング範囲で顕著な事例なし(Phase4ZSのRAG50監査結果と整合)
15. **UNKNOWN clarification率**(C3): 60/77(77.9%)
16. **UNKNOWN irrelevant RAG answer数**: OOD 2件が字句重なりでRAG経路に入ったが、いずれも安全な専門外宣言に着地(実害0)
17. **small-talk hedge**: 0/62(評価対象62件、目標<=5%を大幅に下回る)
18. **small-talk irrelevant RAG混入**: 0/62
19. **OOD boundary**: 15/15(目視補正後、目標>=14/15を上回る)
20. **OOD irrelevant RAG混入**: 2/15(C3の字句重なり誤爆、ただし実害なし)
21. **RAG50 fabrication**: 0/50
22. **RAG50 unsupported numeric**: 0/50
23. **RAG50 numerical hallucination**: 0/50
24. **RAG50 major completeness regression**: 0(49件は既存Phase4ZN raw output無変更、1件[LC-08]はPolicy C3経由でむしろ改善)
25. **P02**: PACHISLOT_FACTUAL(direct、無変更)
26. **LC-08**: UNKNOWN → Policy C3 → 「登録データにAT-Fの性能と終了後の状態についての情報は見つかりませんでした。」(誠実な不明回答に改善。Phase4ZPではSMALL_TALKへ誤routeされ黙殺されていた)
27. **Q6**: unsupported numeric 0(全条件で再現、原条件[context=None]では10/10だったのがPolicy C導入で解消)
28. **Q11**: PACHISLOT_FACTUAL(direct、無変更)
29. **Q17**: PACHISLOT_CONVERSATIONAL(direct、無変更)
30. **AD-04**: PACHISLOT_FACTUAL(direct、無変更)
31. **pachislot conversational fabricated names**: 0/10
32. **F01**: direct pathで無変更(Phase4ZP時点の応答を維持)
33. **retrieval hitをclassifierにしていない**: 確認済み。C3は字句重なり(lexical overlap)を判断材料とし、retrieval scoreは一切使用しない(コード内に"score"という語が存在しないことをテストで確認)。
34. **RAG DB unchanged**: 確認済み
35. **retriever unchanged**: 確認済み
36. **embedding unchanged**: 確認済み
37. **production RAG prompt unchanged**: 確認済み
38. **trainingなし**: 確認済み
39. **identity CLOSED**: 維持
40. **generation総数**: 321件(Stage B: 90件、Stage C: 231件)
41. **git操作なし**: 確認済み(HEAD不変)
42. **Final Candidate architecture候補にできるか**: **できる**。conservative dispatch(Phase4ZR)+ Policy C3(本フェーズ)を提案する。
43. **recommended next phase**: Final Candidate総合評価(comprehensive evaluation)、または実際のproduction統合に向けた実装計画の策定。
44. **next phase auto-startなし**: しない。ここで停止する。

## Policy C3の設計要点

字句重なり(lexical overlap)——クエリから一般的な助詞・機能語を除いた残りの語が、検索結果のタイトル・本文に文字列として出現するか——のみを判断材料とする。retrieval scoreのしきい値でも、機種固有名詞辞書でもない、汎用的かつ検証可能な信号。SMALL_TALK48件では一度もRAG経路に入らず(hedge=0)、PACHISLOT_FACTUAL20件中15件(75%)を正しくRAG経路へ導いた。

## 参照artifact

[phase4zt_gate_analysis.json](../reports/phase4zt_gate_analysis.json)(最重要)、[phase4zt_policy_comparison.json](../reports/phase4zt_policy_comparison.json)(C1/C2/C3比較の全データ)、[phase4zt_path_trace.json](../reports/phase4zt_path_trace.json)(mandatory invariant検証)、[phase4zt_rag50_recheck.json](../reports/phase4zt_rag50_recheck.json)

---
*Phase4ZT完了。次フェーズを自動開始しない。git操作は本フェーズ中一切行っていない。*
