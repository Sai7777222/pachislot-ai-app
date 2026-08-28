# Phase4ZP 完了報告: Lightweight Mode Router Prototype

## 結論

**CASE ZP-C — Router Not Reliable Enough**

3モード(SMALL_TALK/OOD_FACTUAL/PACHISLOT_CONVERSATIONAL)それぞれに専用の軽量policy promptを与えるアーキテクチャ自体は**極めて有効**で、hedge intrusion 0/65、OOD境界14/15、機種名創作0/10と、Phase4ZOの単一プロンプト方式を上回る結果を達成した。しかし、**router(決定的keyword分類器)自体の精度が、実際のRAG probeに対して48%という致命的な誤route率**を示し、必須probeのLC-08すら巻き込んだため、production統合は見送った。

## Section22 必須回答(47項目)

1. **start HEAD**: `e1b21f3e264a5c173f7f0e60e3b13508415d9a6b`
2. **branch**: `checkpoint/identity-closure-phase4zn-baseline`
3. **git status**: 開始時28件untracked(全てPhase4ZO成果物)、`git diff`/`git diff --cached`とも空
4. **Phase4ZG hash**: `278fe7ae...` 不変
5. **pytest start/end**: 176 passed → 191 passed(+15、regressionなし)
6. **trainingなし？**: なし
7. **router total probes**: 自作GT120件 + 独立チェック用にZN既存probe90件 + RAG50件 = 計260件で検証
8. **overall accuracy**: 自作GT 100%(120/120)。ただし独立データでは大きく低下(下記参照)。
9. **mode別accuracy**: 自作GTでは4モード全て30/30(100%)。詳細は`phase4zp_router_eval.json`。
10. **confusion matrix**: `phase4zp_router_confusion_matrix.json`参照。
11. **dangerous PACHISLOT_FACTUAL misroute**: 自作GTでは0件。**独立データのRAG50では24/50(48%)** — 本フェーズ最重要の発見。
12. **small-talk hedge count/rate**: 0/65 (0%)
13. **preference hedge count/rate**: 0/20 (0%)
14. **experience-question hedge**: 0件(65件中に含まれる、全てhedgeなし)
15. **over-refusal**: 0/65
16. **OOD boundary count/rate**: 14/15 (93.3%、目視補正後。自動heuristicは「専門じゃない」という言い回しを見逃し7/15と誤集計していた)
17. **OOD detailed-answer count**: 1/15(ZN-G06「今流行っているアニメを教えて」→鬼滅の刃の詳細回答)
18. **ZN-G15結果**: 「それは専門じゃないから、適当に答えるのはやめておくね。」— 詳細な睡眠アドバイスなし、良好。
19. **pachislot conversational count**: 10件
20. **fabricated machine names**: 0/10
21. **placeholder count**: 0/10
22. **ZN-F01結果**: 「そのへんは私にはわからないな〜。適当に答えると逆に困るから」— 機種創作なし。ただしPhase4ZOの「条件を聞き返す」積極性と比べるとやや消極的(fabricationではないが軽微な質的後退)。
23. **RAG50 output comparison**: 実施していない(routing失敗により生成自体を見送った)。
24. **RAG prompt identical？**: 誤routeされた24件についてはそもそもRAG promptを経由しないため「同一」の判定自体が成立しない。正しくrouteされた26件は、既存コード経路をそのまま通るため理論上同一。
25. **RAG context identical？**: 上記と同様。
26. **generation settings identical？**: モデル・設定自体は無変更(routerは分岐のみ追加)なので、正しくrouteされた範囲では同一のはず。
27. **RAG fabrication**: 未測定(生成未実施のため)。
28. **numerical hallucination**: 未測定。
29. **completeness regression**: 未測定。ただし24/50の誤routeは構造的に即completeness喪失を意味する。
30. **P02**: 正しくPACHISLOT_FACTUALへrouteされた。
31. **LC-08**: **誤ってSMALL_TALKへrouteされた**(必須probeでの重大な失敗)。
32. **Q11**: 正しくPACHISLOT_FACTUALへrouteされた。
33. **Q17**: 誤ってPACHISLOT_CONVERSATIONALへrouteされた(「ミリオンゴッド」がGENERAL_PACHISLOT_TERMSと一致し、FACTUAL_METRIC_KEYWORDSとは一致しなかったため)。
34. **AD-04**: 正しくPACHISLOT_FACTUALへrouteされた。
35. **small-talk retrieval calls**: 0/65(達成)
36. **OOD retrieval calls**: 0/15(達成)
37. **pachislot factual retrieval behavior**: 50件中26件は既存経路を維持、24件は誤って経路から外れretrievalが呼ばれない状態になる。
38. **router complexity**: 巨大regex辞書は作らなかった(Section4準拠)が、その結果として精度が実運用に耐えないことが判明した。
39. **identity untouched？**: 不変。Phase4ZM closureを維持。
40. **RAG DB untouched？**: 不変。
41. **Phase4ZG untouched？**: 不変(hash確認済み)。
42. **production integrationした？**: していない。
43. **CASE**: **CASE ZP-C**(Router Not Reliable Enough)
44. **Boundary問題をcloseできる？**: できない。3モードのpolicy自体は優秀だが、router部分が未解決である限りclose不可。
45. **Final Candidateへ進める？**: 進めない。
46. **git add/commit/pushなし？**: その通り、一切なし(開始時のcheckpoint commitのみが有効なcommitで、本フェーズでは新規操作なし)。
47. **next phase auto-startなし？**: しない。ここで停止する。

## 最重要の発見(Section17 STOP条件該当)

自作ground truth(120件)でrouterが100%を記録した一方、**router実装と無関係に独立して作成されていたRAG50に適用したところ、48%が危険な誤routeとなった**。原因は、実際のパチスロ質問がZ-ZONE・ガイアベル・モードα・AT-A・RT-Bのような**ゲーム固有の未知語**を多用するのに対し、決定的keyword routerの語彙リストが一般的な業界用語(天井/機械割/ゾーン等)しかカバーできていないという、アプローチ自体の構造的限界である。STOP条件B(RAG50 major regression)・D(large regex patching必要)の両方に該当したため、これ以上のキーワード追加は行わなかった。

これは、Phase4ZL(validatorが校正コーパスに過学習)・Phase4ZM(循環評価バグ)に続く、**「自作ground truthでの好成績は、真の汎化性能を保証しない」**という、このプロジェクト全体で繰り返し確認されている教訓の3件目の再現である。

## 参照artifact

[phase4zp_router_design.md](../reports/phase4zp_router_design.md)(最重要 — 自己批判的注記を含む)、[phase4zp_rag_prompt_equivalence.json](../reports/phase4zp_rag_prompt_equivalence.json)(24件の誤route詳細)、[phase4zp_router_eval.json](../reports/phase4zp_router_eval.json)、[phase4zp_smalltalk_recheck.json](../reports/phase4zp_smalltalk_recheck.json)、[phase4zp_ood_recheck.json](../reports/phase4zp_ood_recheck.json)、[phase4zp_pachislot_conversation_recheck.json](../reports/phase4zp_pachislot_conversation_recheck.json)、[phase4zp_gate_analysis.json](../reports/phase4zp_gate_analysis.json)

---
*Phase4ZP完了。次フェーズを自動開始しない。git操作は本フェーズ中一切行っていない。*
