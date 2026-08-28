# Phase4ZQ 完了報告: Retrieval-Signal Routing Feasibility Diagnostic

## 結論

**CASE ZQ-C — Retrieval Signal Not Reliable**

既存RAG retrieverの検索結果(hit有無・score)をPACHISLOT_FACTUAL判定の信号として使う案を診断した結果、**現行のvector DB(単一機種のみで構成)では、無関係な雑談・OOD質問に対しても意味的に無関係だが高スコア(0.77〜0.85)の「それらしい」検索結果が返ってしまい、スコア単独ではPACHISLOT_FACTUALと非PACHISLOT_FACTUALを実用的に分離できない**ことが判明した。retrieval-based routingは中止する。

## Section17 必須回答(48項目)

1. **CASE**: ZQ-C
2. **start HEAD**: `e1b21f3e264a5c173f7f0e60e3b13508415d9a6b`
3. **end HEAD**: `e1b21f3e264a5c173f7f0e60e3b13508415d9a6b`(不変)
4. **branch**: `checkpoint/identity-closure-phase4zn-baseline`
5. **pytest start/end**: 191 passed → 199 passed(+8、regressionなし)
6. **Phase4ZG hash**: `278fe7ae...` 不変
7. **training有無**: なし
8. **RAG DB変更有無**: なし(読み取り専用の`Retriever.search()`のみ使用)
9. **retriever変更有無**: なし(`src/pachislot_ai/rag/`配下は無編集)
10. **embedding変更有無**: なし
11. **GT総数**: 260件
12. **各mode件数**: PACHISLOT_FACTUAL 80(RAG50の50件含む)、PACHISLOT_CONVERSATIONAL 40、SMALL_TALK 95、OOD_FACTUAL 45
13. **GT freeze/hash**: retrieval実行前にfreeze、`phase4zq_ground_truth_hash.txt`にsha256記録済み(`62c1ccf6...`)
14. **RAG50 retrieval non-empty率**: 50/50 (100%)
15. **RAG50 relevant-hit率**: 内容レベルでは全件topically妥当(単一機種DBのため機種違いの心配なし)。ただしscoreレベルでの識別力はない。
16. **RAG50 factual detection recall**: non-empty基準では100%だが信号として無意味。閾値ベースでは最良でも70%前後(全PACHISLOT_FACTUAL80件に対する数値、目標98%未達)。
17. **P02 detection**: non-empty、top1=0.8492
18. **LC-08 detection**: non-empty、top1=0.8291
19. **Q11 detection**: non-empty、top1=0.836
20. **Q17 detection**: non-empty、top1=0.8555
21. **AD-04 detection**: non-empty、top1=0.8055
22. **factual全体recall**: 閾値ベースで最良70%(探索的分析、本番閾値ではない)
23. **factual false negatives**: 最良閾値(0.828、探索的)でも24/80が閾値未満
24. **small-talk retrieval non-empty率**: 95/95 (100%)
25. **small-talk relevant false-positive率**: 目視確認した5件全てで意味的に無関係な結果が高scoreで返った。実質ほぼ100%相当。
26. **OOD retrieval non-empty率**: 45/45 (100%)
27. **OOD relevant false-positive率**: 同上、目視5件全てで無関係。
28. **pachislot conversational retrieval率**: 40/40 (100%) non-empty、top1 score中央値0.8264(PACHISLOT_FACTUALと近い範囲で重複)
29. **score separation**: なし。PACHISLOT_FACTUALの71%(57/80)が、非PACHISLOT_FACTUALの99%(138/140)と同じscore帯域(0.786〜0.846)に存在。
30. **overlapの程度**: 深刻。`phase4zq_score_distribution.json`参照。
31. **retrieval-only feasibility**: 不可(recall最大70%、目標98%に届かず)
32. **ZP-router + retrieval-rescue feasibility**: 部分的にのみ機能(下記参照)。ただし新規false positive発生と閾値の事後最適化という2つの問題がある。
33. **rescueできたZP RAG50 24件中の件数**: 20/24(探索的閾値0.828時点)
34. **rescueで新たに壊す件数**: 14件(SMALL_TALK/OOD計140件中、新たにPACHISLOT_FACTUALへ誤って救済されてしまう)
35. **giant regexなし？**: その通り、追加していない
36. **固有名詞辞書追加なし？**: その通り、追加していない
37. **second LLMなし？**: その通り、使用していない
38. **classifier trainingなし？**: その通り、行っていない
39. **generation実施数**: 0件(Phase4ZG呼び出しなし。retrieval診断のみでgenerationは不要と判断)
40. **RAG50生成を行ったか**: 行っていない(retrieval診断のみで十分と判断)
41. **static analysisで十分だった箇所**: 全stage。retrievalのstatic capture(260件、8秒で完了)だけで、H1〜H3の全仮説を検証するのに十分なデータが得られた。
42. **evaluation leakage有無**: なし。GTはretrieval実行前にfreeze・hash化した。
43. **circular GT有無**: なし。GTは既存の人間定義category(Phase4ZN/ZP由来)を再利用しており、retrieval予測を一切参照していない。
44. **identity CLOSED維持**: 維持(本フェーズで一切触れていない)
45. **production integrationなし**: その通り、行っていない
46. **git操作なし**: その通り、`git add/commit/push`は一切実行していない(HEADは開始時のまま不変)
47. **次Phase推奨**: retrieval-based routingは中止。次の現実的な選択肢は(a) vector DBに複数機種データが投入された時点での再診断、(b) keyword router(ZP)とretrieval signalを組み合わせた高度なhybrid設計の検討、(c) 軽量ML classifierの検討 — いずれも本フェーズのスコープ外で未着手。
48. **next phase auto-startなし？**: しない。ここで停止する。

## 重要な注記

本フェーズの結論は、**現行のRAG DBが単一機種(ミリオンゴッド神々の軌跡)のみで構成されているという特殊な状態に強く依存している可能性がある**。DBに複数機種・複数トピックのデータが投入され、embeddingが真に多様な負例を区別できるようになれば、結果が変わる可能性がある。「retrieval signalは原理的にrouting信号として使えない」という普遍的な結論として一般化すべきではなく、**現時点のDB構成では使えない**という限定的な結論として扱うべきである。

## 参照artifact

[phase4zq_score_distribution.json](../reports/phase4zq_score_distribution.json)(最重要 — score重複の詳細)、[phase4zq_false_positive_analysis.json](../reports/phase4zq_false_positive_analysis.json)、[phase4zq_rag50_coverage.json](../reports/phase4zq_rag50_coverage.json)、[phase4zq_hybrid_simulation.json](../reports/phase4zq_hybrid_simulation.json)、[phase4zq_gate_analysis.json](../reports/phase4zq_gate_analysis.json)

---
*Phase4ZQ完了。次フェーズを自動開始しない。git操作は本フェーズ中一切行っていない。*
