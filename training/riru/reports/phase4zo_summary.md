# Phase4ZO 完了報告: Small-talk / Specialist Boundary Product Fix

## 結論

**CASE ZO-B — Prompt Improvement Partial**

system prompt(3-mode policy)のみでsmall-talk/specialist boundary問題を改善できるかを検証した。**Causal isolation・OOD境界・パチスロ会話・矛盾解消は明確に成功**したが、**RAG50 regressionで必須probe(P02)を含む複数件のcompleteness regression/未検証claim追加を確認**し、Section12のgateを満たさなかったため、production統合は見送った。`config/prompts/system.jinja2`は無変更。

## Section22 必須回答(40項目)

1. **checkpoint commit HEAD**: `e1b21f3e264a5c173f7f0e60e3b13508415d9a6b`
2. **push結果**: 成功(`origin/checkpoint/identity-closure-phase4zn-baseline`、force pushなし)。※mainへ直接commitせず新規branchを作成した(harness標準方針: デフォルトブランチ上では先にbranchを切る)。指示書はbranch指定がなかったための安全側判断。
3. **Phase4ZG hash**: `278fe7ae...` 不変(preflight・全過去phaseと一致)
4. **pytest start/end**: 162 passed → 176 passed(+14、regressionなし)
5. **trainingなし？**: なし。GPUはPhase4ZG読み取り専用推論にのみ使用。
6. **baseline preference hedge**: 12/20 (60%)
7. **minimal prompt preference hedge**: 0/20 (0%)
8. **3-mode prompt preference hedge**: 1/20 (5%、soft partial hedgeで完全拒否ではない)
9. **causal conclusion**: hedge挙動はadapter/model自体ではなく、現行system promptの厳格なRAG指示文言に強く由来する(prompt-layer dominant)。minimal promptが0/20を達成したことがこれを裏付ける。
10. **small-talk total**: 65件
11. **small-talk hedge count/rate**: 自動集計7/65(10.8%)、目視補正後4/65(6.2%) — 補正の内訳: 1件false positive(ZN-A01)、1件soft partial(ZN-C14)、1件ambiguous-but-reasonable(ZN-D06)
12. **small-talk over-refusal**: 0件
13. **preference hedge count/rate**: 1/20 = 5%(目標達成)
14. **greeting regression**: 実質0(1件はheuristicのfalse positiveのみ)
15. **emotional regression**: 0/15
16. **OOD total**: 15件
17. **correct boundary count/rate**: 自動集計13/15、目視補正後15/15(100%) — 2件は「専門外」の完全一致検索に起因するheuristicの見逃しで、実際は「専門分野じゃない」「専門家じゃない」という同義表現で正しく境界を表現していた
18. **ZN-G15どうなった？**: 「良い睡眠をとるコツ」に対し、詳細な睡眠アドバイスの継続はせず、専門外である旨を述べた上で一言だけの軽いコメントに留まった。preflightで懸念されたunder-refusalパターンは解消。
19. **pachislot conversational total**: 10件
20. **fabricated machine names**: 0/10(目標達成)
21. **ZN-F01どうなった？**: 「パチスロ〇〇」のような具体機種の創作はせず、機種タイプ・狙い目条件を聞き返す応答に変化した。
22. **RAG50 fabrication**: 未検証の新規claim追加をQ11・Q17の2件で確認(fabrication疑い、0ではない)
23. **numerical hallucination**: 明確な誤数値そのものは確認されなかったが、Q11/Q17は根拠未確認の新規記述であり要注意
24. **completeness regression**: あり。P02(必須probe、数値が全て欠落)、Q15(2つ目の点灯パターン説明が完全欠落)、Q17(説明欠落+新規claim)、P04(個別設定値の省略)の計4件+Q11
25. **P02**: 重大なcompleteness regression。baselineは設定1〜5の具体的確率を提示していたが、NEWは「設定1〜5までの確率だね」とのみ述べ数値を一切示さなかった。
26. **LC-08**: baselineと完全同一(modified対象外)、regressionなし
27. **Q11**: 未検証の新規claim(「ループストックが1%+Z-ZONEだとGG前兆以上濃厚」)がbaselineに存在せず新規追加された
28. **AD-04**: 実質同一(「情報がない」という結論は同じ、表現のみ変化)
29. **H01 contradiction解消？**: 解消した。「パチスロのデータしか知らない」→「パチスロ以外ならもっと詳しく答えられる」という自己矛盾は消え、短く一貫した「わからない」応答になった(explicit_contradiction 0/10)
30. **system promptだけで十分？**: 不十分。small-talk/OOD/pachislot会話/矛盾解消は成功したが、RAG50で複数のregressionが残った。
31. **router必要？**: 次phaseで軽量routerまたはprompt文言のさらなる調整のいずれかを検討する必要がある(本フェーズのスコープ外)。
32. **model-level training必要？**: 不要と判断。Stage A(causal isolation)がhedge挙動をprompt-layerの問題と強く裏付けたため、trainingへの回帰は推奨しない。
33. **Phase4ZG unchanged？**: 不変(hash確認済み)
34. **identity unchanged？**: 不変。Phase4ZM closureを維持し、本フェーズでidentity guard/validatorには一切触れていない。
35. **RAG DB unchanged？**: 不変
36. **GGUF/Q8/Q5なし？**: なし
37. **CASE ZO-A/B/C/D/U**: **CASE ZO-B**(Prompt Improvement Partial)
38. **Final Candidateへ進める？**: 進めない。RAG50のregression(特にP02)が解消されるまでFinal Candidate総合評価には進まない。
39. **commit/pushはcheckpoint以外でしていない？**: していない。checkpoint commit(`e1b21f3`)1回のみ、それ以降は一切のgit操作を行っていない。
40. **next phase auto-startなし？**: 自動開始しない。ここで停止する。

## 参照artifact

[phase4zo_causal_analysis.json](../reports/phase4zo_causal_analysis.json)、[phase4zo_smalltalk_recheck.json](../reports/phase4zo_smalltalk_recheck.json)、[phase4zo_ood_recheck.json](../reports/phase4zo_ood_recheck.json)、[phase4zo_pachislot_conversation_recheck.json](../reports/phase4zo_pachislot_conversation_recheck.json)、[phase4zo_rag50_recheck.json](../reports/phase4zo_rag50_recheck.json)(最重要 — P02等のregression詳細)、[phase4zo_ambiguous_recheck.json](../reports/phase4zo_ambiguous_recheck.json)、[phase4zo_gate_analysis.json](../reports/phase4zo_gate_analysis.json)、[phase4zo_prompt_change.json](../reports/phase4zo_prompt_change.json)、[phase4zo_end_integrity.json](../reports/phase4zo_end_integrity.json)

---
*Phase4ZO完了。次フェーズを自動開始しない。checkpoint commit以外のgit操作は行っていない。*
