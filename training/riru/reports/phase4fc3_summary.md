# Phase4FC3 最終レポート — Production Boundary Dispatch & Evidence Arbitration Integration

**CASE判定: FC3-C (Boundary Still Polluted — 文字通りの数値基準に照らした判定。ただし重要な限定あり、下記参照)**

以下、Section32が要求する51項目に沿って回答する。

1. **CASE**: FC3-C
2. **implementation summary**: `src/pachislot_ai/dispatch/`(新規、conservative dispatch)+ `src/pachislot_ai/rag/evidence_arbitration.py`(新規)+ `pipeline.py`/`chat_service.py`への最小限の統合(各1箇所)。Phase4ZR/ZPの既存診断ハーネスの意味論を移植、IDENTITY_PERSONAを新規追加。
3. **production modes**: SMALL_TALK / IDENTITY_PERSONA / PACHISLOT_FACTUAL / PACHISLOT_CONVERSATIONAL / OOD_FACTUAL / UNKNOWN の6モード実装。
4. **UNKNOWN policy**: 常に既存RAG pipelineへ委譲(is_emptyでも省略しない)。開発中のablation testで「GGプラスとは何か説明して」を無条件でcontext省略すると架空の詳細説明を創作することを実証したため、安全側に倒す設計へ確定。
5. **small-talk65 hedge**: 26.2%(17/65)。FC2の60.0%から大幅改善したが、mandatory目標<=5%は未達。
6. **small-talk RAG injection count**: 0/40(SMALL_TALK確信モードで注入0件)。残存hedge全17件は`rag_context_injected=false`。
7. **identity result**: 「君の名前は？」→「私はリルだよ！」に修正確認(FC2 regression解消)。IDENTITY_PERSONA判定7件全てでRAG注入なし。
8. **identity RAG injection count**: 0/7
9. **OOD15 result**: 適切な専門外境界15/15(100%、mandatory>=14/15達成)
10. **OOD RAG injection count**: 0/11(確信モード)
11. **conversational10 fabricated names**: 0
12. **GT260 dangerous misroute**: 0/260(開発中に1件の潜在的危険パターンを発見・修正済み)
13. **UNKNOWN rate**: 21.5%(56/260、GT260)
14. **天井とヤメ時関係結果**: 修正確認済み(以前は矛盾するno-evidenceマーカーにより完全decline、修正後は両方の情報を含むgrounded応答)
15. **contradictory no-evidence count**: 0/102(体系的チェック)
16. **chunk-only cases**: 全件PASS(FC3のGate G内訳参照)
17. **structured-only cases**: 全件PASS
18. **both-evidence cases**: 全件PASS
19. **no-evidence cases (phantom)**: 8/8正しくdecline
20. **Q6**: PASS、grounded
21. **AT-F**: PASS、正しくdecline
22. **RT-A/RT-B**: PASS、正しくdecline
23. **SGG/GG準備中**: PASS、grounded
24. **GG中**: PASS、grounded
25. **天国ロング**: PASS、正しくdecline
26. **AD-04**: PASS、grounded(「GG終了後…G-ZONE終了後、32G消化」)
27. **P02**: PASS、正しくdecline
28. **P04**: PASS(LOW自己計算、grounded operand・正確な計算)
29. **LC-08**: PASS、正しくdecline
30. **Q11**: PASS、grounded
31. **Q15**: PASS、grounded
32. **Q17**: PASS、grounded
33. **critical factual regression count**: 0
34. **multi-turn mode leakage**: 0/5シナリオ、10ターン
35. **pytest start/end**: 267 passed → 294 passed(0 regression)
36. **tests added**: 27件(unit 21 + integration 5、+1回帰テスト後日追加分含む)
37. **added latency**: dispatch平均0.005ms、arbitration平均0.0003ms(目標<2msを大幅に下回る)
38. **generation count**: 170(main)+ 生成前のablation診断8 = 178件(予算300、目標220以内に収まった)
39. **Phase4ZG hash**: `278fe7ae...`不変
40. **prompt unchanged**: system.jinja2無変更
41. **FY entity binding unchanged**: entity_attribution.py無変更(hash一致確認済み)
42. **FZ structured binding unchanged**: structured_lookup.py無変更(hash一致確認済み)
43. **DB unchanged**: 確認済み
44. **embedding unchanged**: 確認済み
45. **trainingなし**: 確認済み
46. **moderation untouched**: 確認済み(本フェーズのスコープ外として一切触れていない)
47. **quantizationなし**: 確認済み
48. **Slack status**: この応答の直後にFAILテンプレートで送信する
49. **boundary architecture ACCEPT/REJECT**: **ACCEPT(アーキテクチャとしては)** — dispatch/evidence-arbitration統合は完全に検証済みで意図通り機能している(アーキテクチャ由来hedge=0/65、dangerous misroute=0/260、evidence矛盾=0/102)。ただしmandatory数値ゲート(hedge<=5%)は、本フェーズのスコープ外にあるPhase4ZG自身の学習済み挙動により未達成。
50. **recommended next phase**: 人間が残存hedge(26.2%、100%がモデル自身の学習済み挙動に起因)を受容するか、Phase4ZGの追加学習を検討する別フェーズを起こすかを判断すること。dispatch/evidence-arbitrationの実装自体はレビュー後にcommit可能な状態にある。
51. **auto-startなし**: ここで停止する。次フェーズは自動開始しない。

---

## 重要な限定条件(必読)

本フェーズの最終CASEはFC3-C(mandatory数値基準の文字通りの判定)だが、これを「dispatch統合が失敗した」と解釈するのは不正確である。以下を明確に区別すること:

- **本フェーズが実際にスコープとした問題(RAG context injectionアーキテクチャによる雑談・自己紹介・専門外への汚染)は、ablation testによる因果関係の実証を含む厳密な検証により完全に解消された**(アーキテクチャ由来hedge: 60%相当→0%)。
- **残存する26.2%のhedgeは、100%がRAG context非注入状態で発生しており**、Phase4ZG自身の学習済み挙動(個人的な質問への「登録データにない」という定型応答パターン)に起因する、別種の、本フェーズのスコープ外の問題である。
- dangerous factual misroute・evidence arbitration・factual regressionは全てmandatory基準を満たしている。

## 未commitの変更(人間確認待ち)

- `src/pachislot_ai/dispatch/`(新規)
- `src/pachislot_ai/rag/evidence_arbitration.py`(新規)
- `src/pachislot_ai/rag/pipeline.py`(修正)
- `src/pachislot_ai/services/chat_service.py`(修正)
- `tests/unit/test_conservative_dispatch.py`・`tests/integration/test_dispatch_integration.py`(新規)
- `training/riru/reports/phase4fc3_*.json`・`.md`(本フェーズの全成果物)

Phase4FC2の診断成果物は既にcheckpoint commit(`f7b4239`)としてpush済み(production codeは含まない)。

ここで停止する。
