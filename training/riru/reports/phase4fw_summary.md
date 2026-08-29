# Phase4FW 完了報告: Product-side Grounded Claim Verification Feasibility

## 結論

**CASE FW-E — Verification Fundamentally Insufficient（concept-binding等を安定検出できない）**

生成後の回答をretrieved contextと照合するproduct-side grounded verificationの実現可能性を診断した。deterministic verifier(entity grounding + evidence binding + numeric/symbol exact match)、Phase4ZG自身による構造化model verifier、両者のhybrid(OR結合)の3構成を評価した結果、**Mandatory Safety Gate(Section18)の「SGG/GG準備中confusion検出」カテゴリで未達**が判明し、加えてself-verification bias(Section11の懸念が的中)、V3 regenerationの新規捏造リスク、hybrid構成のレイテンシ問題、境界的なRAG50 false positive率という4つの追加問題が複合的に確認された。**production統合は行わず、次フェーズでのアーキテクチャ再検討を推奨する。**

## Section24 必須報告項目(41項目)

1. **CASE**: FW-E
2. **recommended verifier architecture**: 本フェーズで検証したいずれの構成(deterministic単独/model単独/hybrid)も即座には推奨しない。次フェーズでの実装修正(subject抽出の改善、自己検証バイアス回避の仕組み)を条件とした再評価が必要。
3. **GT hash**: `fc5ebb86a074035018d92d7b6a4ffd255b9ba2034fcb5ab988450fb94480dc80`（84応答・129 atomic claim、frozen_before_verifier_construction=true）
4. **total atomic claims**: 129件(SUPPORTED 78、MISATTRIBUTED 32、NON_FACTUAL 10、UNSUPPORTED 6、AMBIGUOUS 3)
5. **extraction accuracy**: E2(既存モデルによる構造化抽出)は24件サンプルで100%のJSON parse成功率、かつE1(deterministic)が比較文のsubject分割に失敗した14/24probeで正しく分解に成功。E1はE2に対し一度も上回らなかった(0/24)。E2を主要抽出方式として推奨。
6. **Q6 detection**: PASS（hybridで主要unsafe claim4件中4件検出）
7. **AT-F detection**: PASS（hybridで3/3件検出。ただしmodel verifier単独では0/3=完全に見逃し）
8. **RT-A/RT-B detection**: PASS（hybridで2/2件検出。model単独では0/2）
9. **phantom entity detection**: PASS（22probe中、deterministicのNOT_FOUND経路により概ね安定検出）
10. **concept reversal detection**: PASS（FV-C03のループストック/GGストック関係逆転はhybridで2/2検出。model側の意味理解が寄与した数少ない成功例）
11. **SGG/GG準備中 detection**: **FAIL**（10件中4件、ガイアベル・SU4・SGG・GG当選が、deterministic・model・hybridいずれでも検出できなかった）
12. **critical FN**: 4件（ガイアベル[FU-E02]、SU4[FU-D05]、SGG[FU-D03]、GG当選[FV-C05]。全てconcept-binding系。Section18のMandatory Gateを満たせない直接原因）
13. **deterministic precision/recall**: precision=0.4507, recall=0.8421（critical set 129claim中126件、AMBIGUOUS3件除く）
14. **model verifier precision/recall**: precision=0.7500, recall=0.4737（54probe分のclaimで評価）
15. **hybrid precision/recall**: precision=0.6667, recall=0.8947（54probe分、OR結合）
16. **self-verification failures**: 4/11件（36%）。既知unsafe claim(Q6/AT-F/RT-A・RT-B/ループストック・GGストック)をmodel verifierに判定させたところ、AT-F(3/3)とRT-A(1/1)が誤ってSUPPORTED判定された。Section11の懸念が明確に的中。
17. **RAG50 FP**: model_alone=12.5%(3/24)、deterministic_alone=50.0%(GT構築上の欠陥により水増し、詳細後述)、hybrid=58.3%(同欠陥の影響を含む)
18. **RAG50 FN**: 0（サンプル20probe中、そもそもGT上のunsafe claimが存在しないため測定対象なし）
19. **V1結果**: 34件の危険probe中67.6%(23/34)が破棄・fallback対象となった。安全性は担保されるが、拒否率は高い。
20. **V2結果**: unsafe claimを除去しsupported claimsのみで再構成。既存生成済みテキストの部分利用のみで新規生成なし、安全かつ部分的completeness維持を両立する妥当な挙動を確認(例: Q6は捏造部分を除きSGGゲーム数抽選の正しい情報だけを残した)。
21. **V3結果**: **4件中2件(50%)で、除外指示したはずのunsafe claimの代わりに全く新しい捏造を生成する重大な副作用を確認**（AT-F: 1480G→1200Gという新たな誤数値、1〜4G→1〜3Gという新たな誤範囲。RT-A/RT-B: 「ランプ点滅/点灯」という完全な創作、contextに存在しない語彙まで使用）。残り2件は正しく情報不足を申告。
22. **best product action**: V2(supported claimsのみで再構成)を推奨。V3(regeneration)は新規捏造リスクが高く不採用。V1は最も安全だが拒否率が高すぎる。
23. **fallback rate**: V1で67.6%(34probe中)。V2は0%(常に何らかの応答を返す、ただし空になった場合のみfallback)。
24. **completeness impact**: V2はcompletenessを部分的に維持(例: Q6でSGGの正しい情報を残す)しつつ安全性を確保できる、最もバランスの良い挙動を確認。
25. **added latency**: E2抽出(平均6.22秒) + model verifier(平均7.01秒) ≒ 13秒/response(hybrid構成)。deterministic単独なら実質0秒。
26. **GPU impact**: 追加のGPUメモリロードは実質なし(既存Phase4ZGプロセスを再利用可能)。
27. **new generation count**: 82件（E2抽出24 + model verifier54 + V3サンプル4。予算150件・目安100件未満を達成）
28. **pytest start/end**: 233 passed → 233 passed（regressionなし）
29. **Phase4ZG hash**: `278fe7aedc5f302b9966689c9e92c8363fea246db71aab7cc959ce9609dcc9dc`（不変）
30. **production prompt unchanged**: `e859e2aa443160f8b4e8c897f2d9af1e7d310599047ce484864b578984ebddd7`（不変）
31. **routing unchanged**: conservative dispatch `80dbb4469a201030de1ee7ec6f1d57b69a990d62e7e741638a275f5772a018ad`（不変）
32. **Policy C3 unchanged**: `cb9f904bb02d9b109e1a6b6f773b976d2699d14b39833ce5876a4ba6c1963caf`（不変）
33. **DB/retriever/embedding unchanged**: 変更なし(本フェーズはPhase4FU/4FVの既存生成データの再利用が中心で、新規retrievalは実施していない)
34. **trainingなし**: 実施なし
35. **production integrationなし**: 全てoffline prototype(`training/riru/guard/`配下)に留まる
36. **git opsなし**: commit/push一切なし
37. **Slack status**: 送信試行あり、成功
38. **production guard feasible YES/NO**: **NO（現時点では見送り）**
39. **quantization可能か**: いいえ（引き続き保留。CASE FC-C→FU-H→FV-C→FW-Eの系譜すべて未解決）
40. **recommended next phase**: (1) claim抽出のsubject決定ロジックをprobe IDのような無意味な文字列ではなく実際のcontext由来の内容語から行うよう実装修正、(2) 自己検証バイアスを避けるための独立した検証手段(同一モデルに依存しない仕組み)の検討、(3) V3ではなくV2方式(supported claimsのみでの再構成)を軸としたproduct action設計、(4) レイテンシと精度を両立する経路の探索。
41. **next phase auto-startなし**: しない。ここで停止する。

## 最重要の発見

1. **『存在しないentityへの誤紐付け(phantom entity)』は概ね安定して検出できるが、『実在するentity同士の関係の取り違え(concept-binding)』は、deterministic・model・hybridのいずれでも一部が検出漏れになる** — これはPhase4FUの根本原因分析、Phase4FVのprompt-only対策の結果と完全に一致する、3つ目の独立した検証による同一パターンの再確認である。
2. **同一モデルによる自己検証(self-verification)には明確なバイアスがある** — 既知の誤り11件中4件を、生成に使ったのと同じモデルが誤ってSUPPORTEDと判定した。特にAT-Fは3件全てで見逃された。
3. **『危険な回答を検出したら答え直させる』という直感的に妥当に見える対策(V3)は、実際には新たな捏造を生む危険がある** — 4件中2件で、除外を指示したはずの内容の代わりに、全く新しい不支持な具体的数値・語彙が生成された。
4. **hybrid構成は精度面では最良のバランスを示したが、レイテンシ(約13秒/response)がリルの『軽量・高速』という製品目標と衝突する。**

## 次への申し送り

Phase4FC→FU→FV→FWと4フェーズにわたり、RAG factual safety問題への対策を(1)何もしない診断、(2)prompt強化、(3)生成後検証、という3つの異なるレイヤーで検証してきたが、いずれも単独では「未サポートの統合を一切許さない」という製品要件を完全には満たせていない。次フェーズでは、本フェーズで発見した具体的な実装上の欠陥(subject抽出方法、自己検証バイアス、V3の危険性)を修正した上での再評価、または全く異なるアプローチ(retrieval/context assembly段階でのentity-level source-attribution等)を検討することを推奨する。GGUF量子化は引き続き保留とする。

---
*Phase4FW完了。次フェーズを自動開始しない。*
