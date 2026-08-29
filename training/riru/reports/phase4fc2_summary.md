# Phase4FC2 最終レポート — Final Candidate Comprehensive Re-test

**CASE判定: FC2-C (Boundary Regression — Factual RAG safe but small-talk/OOD/product boundary fails)**
**quantization_allowed: NO**

以下、Section27が要求する55項目に沿って回答する。

1. **CASE**: FC2-C
2. **FY commit hash**: `b61ee8cfdc28c969ded576bf128bab3ac84a5ef0`
3. **FY push結果**: SUCCESS(`30fc501..b61ee8c`、origin確認済み)
4. **FZ commit hash**: `616405054c9a55cf1871cd2bb04bab7b0cca1ca0`
5. **FZ push結果**: SUCCESS(`b61ee8c..6164050`、origin確認済み)
6. **pytest start/end**: 267 passed → 267 passed(変化なし、フェーズ中コード変更ゼロ)
7. **production GT hash**: `b02d51b61427c8335801f26bd6929a51dd7255e3506575178d9591ee0041bb1e`(102件)
8. **production factual sample count**: 102件(Gate E)+50件(Gate F、静的分離)+26件(Gate B)+20件(Gate C)+8件(Gate D)+90件(Gate H)+19件(Gate K)+23ターン(Gate J)＝生成総数346件(予算500件以内)
9. **critical unsupported factual**: 0
10. **unsupported numeric**: 0(P04/PT-08系の自己計算[LOW]を除く、grounded・計算正確)
11. **chunk misattribution**: 0
12. **structured misattribution**: 0
13. **phantom fabrication**: 0(Gate B 26件、Gate E entity_missing 14件、計40件で確認)
14. **concept-binding failures**: critical=0。ただしMODERATE completeness bug 2件を新規発見(詳細39項目)
15. **query-style failures**: 0(「初心者向け」再発なし)
16. **RAG50 unsupported factual**: 0/50
17. **RAG50 completeness regression**: 0
18. **structured GT phantom result**: 0/15(Phase4FZ再利用、変更なしのため未再生成)
19. **structured real recall**: 20/20(Phase4FZ再利用)
20. **天国ロング**: PASS(Gate B/A、完全decline、fabrication無し)
21. **AT-F**: PASS(Gate A、structured leak解消確認)
22. **RT-A/RT-B**: PASS(Gate A/B、完全decline)
23. **Q6**: PASS(Gate A/D、全数値grounded)
24. **GG中**: PASS(Gate L必須テスト、「GG中解説」正しく回収)
25. **SGG/GG準備中**: PASS(Gate A、grounded)
26. **loop/GG stock**: PASS(Gate A、2概念を正しく区別)
27. **AD-04**: PASS(Gate A、Phase4FZで改善確認済みの回答を維持)
28. **P02**: PASS(Gate F静的、fictional値でgrounded。実DBでは正しくdecline)
29. **P04**: PASS(Gate F静的、96.8%/113.5%grounded・16.7%は自己計算[正確・LOW])
30. **LC-08**: PASS(Gate F静的、fictional値でgrounded)
31. **Q11**: PASS(Gate F静的、全数値grounded)
32. **Q15**: PASS(Gate F静的、grounded)
33. **Q17**: PASS(Gate F静的、grounded)
34. **small-talk hedge**: **60.0%(39/65) — mandatory<=5%を大幅超過、FAIL**
35. **OOD boundary**: 15/15(100%) — mandatory>=14/15をPASS
36. **fabricated machine names**: 0
37. **router dangerous misroute**: 0/260(Phase4ZR/ZT診断ハーネス再利用、本番コードには該当機能なし)
38. **multi-turn misattribution**: 0/8シナリオ・23ターン
39. **identity regression**: 3件の明示的wrong-name違反(ZL-A02/D01/D02)を確認したが、RAG context有無での比較検証によりPRE-EXISTING(Phase4ZGの既知の限界、RAG非依存)と確定。**別途1件、「君の名前は？」への回答が空contextfallback message存在時にdeclineする新規regressionを確認**(RAG context除去で正しく「私はリルだよ！」に回復することをablationで確認)。
40. **retrieval recall improvement**: mandatory「GG中解説」回収確認済み、9/21(Phase4FY再利用)でrecall向上
41. **arithmetic LOW finding**: P04/PT-08は今回もgrounded operands・正確な計算のみでfactual gate違反なし、LOWとして維持
42. **total new generations**: 346件(main 288 + static RAG50 50 + identity ablation diagnostic 8)
43. **latency**: p50=1.72秒, p95=9.77秒, mean=2.95秒(288件のend-to-end計測)
44. **Phase4ZG hash**: `278fe7ae...`不変(preflight/end_integrityで再確認)
45. **prompt unchanged**: system.jinja2・rag_context.jinja2とも無変更
46. **DB unchanged**: chunk数119件不変、ingest/upsert未実施
47. **embedding unchanged**: 無変更
48. **no training**: 実施なし
49. **no production fixes during FC2**: 確認済み(git diff空、hash完全一致)。Gate C/E/H/Kで発見した問題は記録のみで一切修正していない
50. **Slack status**: この応答の直後にFAILテンプレートで送信する
51. **RAG factual safety CLOSED**: **YES**(fabrication/misattribution/unsupported claim、全て0を維持)
52. **Final Candidate ACCEPT**: **NO**(product boundary問題のため、無条件PASSは付与しない)
53. **quantization allowed**: **NO**
54. **recommended next phase**: dispatch/Policy C3相当の軽量モード判定を本番統合し、small-talk/雑談/簡単な自己紹介質問等の非RAG関連メッセージにはRAG context system messageを注入しない設計へ変更する新フェーズ。あわせてGate C/Eのno-evidenceマーカー矛盾も同フェーズで解消することを推奨する。
55. **auto-startなし**: ここで停止する。次フェーズは自動開始しない。

---

## 総括

Phase4FY・Phase4FZが取り組んだRAG事実安全性(entity-binding正確性)は、約350件の実生成を通じて**完全に有効であることを再確認した**。fabrication・misattribution・unsupported claimは全gateでゼロを維持し、天国ロング・AT-F・GG中といった長年の既知失敗ケースも一貫して正しく処理された。

しかし本フェーズは、Phase4FY/FZの成果とは独立した、**より根源的なアーキテクチャギャップ**を初めて定量的に明らかにした: dispatch/Policy C3が本番コードに統合されていないため、ChatServiceは全メッセージに対して無条件にRAG context system messageを注入する。この設計は、(a) small-talk(雑談)応答に不自然な「登録データにありません」という断り書きを60%の確率で混入させ、(b) 極端な場合には「君の名前は？」という単純な自己紹介質問にすら正しく答えられなくなる、という2つの具体的な副作用を引き起こしていることを、RAG context有無のablation testで因果関係込みで確認した。

これはPhase4FY/FZが新たに引き起こした regression ではなく、両フェーズ以前から存在していた既存の設計ギャップである。ただし、Phase4FYが導入した「0件選別時に明示的な空contextfallback文言をレンダリングする」という(それ自体は正しい)改善が、この副作用をより顕在化させた可能性がある。

以上により、CASE **FC2-C**(事実面のRAGは安全だが、製品境界面が失敗)と判定する。quantization/GGUF化は、この製品境界問題の解決方針が人間により決定されるまで開始しない。

以上でPhase4FC2の報告を終了する。次フェーズは自動開始しない。
