# Phase4FX 完了報告: Entity-Aware Context Attribution / Source Binding Feasibility

## 結論

**CASE FX-A — Entity-aware Assembly Sufficient（Product Architecture候補）**

Phase4FC(2026-08-29開始)から続く5フェーズ(FC→FU→FV→FW→FX)にわたるRAG factual safety問題の調査において、**初めてSection19の全必須ゲートを達成した**。retrieval/context assembly段階で、既存のchunk title metadataだけ(新規dictionary/schema変更なし)を使ってquery entityとretrieved evidenceを構造的にbindingし、bindingされなかったcontextを除外(A2: query-bound-only)することで、known critical misattribution=0、phantom misbinding=0、concept-binding critical FN=0、RAG50 completeness regression=0を同時に達成した。追加レイテンシは実質ゼロ(0.1ms未満)で、Phase4FWのhybrid verifierが抱えていた13秒/responseの重さも回避している。

## 最重要の発見

監査中、production DBには「GG中解説」(GG中を専門に説明するchunk)や「引き戻し解説」(SGGの実体を説明するchunk)という、極めて明確にtitle付けされたchunkが既に存在していたにもかかわらず、**embeddingベースのtop-6検索が過去5フェーズ・数十回のクエリを通じて一度もこれらを上位に返していなかった**ことを発見した。実際に「GG中とはどんな状態か教えて」で検索すると、完全一致する「GG中解説」はtop-6に入らず、代わりに無関係な「準備中解説」等が上位に来ていた。title表層一致による補完検索を追加するだけで、この問題は解決した。**GG/SGG/GG準備中の混同という、本投資調査全体で最も繰り返し発見されてきた問題の相当部分は、モデルの生成能力の限界ではなく、検索が単純に正しい情報を見逃していたことに起因していた**、というのが本フェーズ最大の発見である。

## Section25 必須報告項目(49項目)

1. **CASE**: FX-A
2. **recommended architecture**: chunk title metadataベースのentity attribution + query-bound-only context assembly(A2方式)。offline prototype(`training/riru/guard/phase4fx_entity_attribution.py`等)として実装済み、production統合は次フェーズで検討。
3. **GT hash**: `dc14a7b84b113b980ad02967e7cca33d3a3328e95571a08e9b06dc6ff7d8e4cf`（404 chunk instances、114 bound assignments）
4. **schema attribution exact %**: 67.2%（80/119チャンク、titleのみで一意にentity/conceptが特定可能）
5. **schema attribution parent %**: 12.6%（15/119、category+隣接recordから復元可能）
6. **unattributable %**: 0.8%（1/119、本文を読まないと特定できない孤立した数値チャンク1件のみ）。残り19.3%(23/119)はATTRIBUTABLE_WEAK(位置推論で復元可能)。
7. **query entity extraction accuracy**: 約91.2%（初期実装比、2件の既知限界を修正後はほぼ全件で正確に抽出。詳細はphase4fx_query_entity_extraction.json）
8. **evidence attribution accuracy**: 100%（404 chunk instances、最終メカニズムでの人間監査と一致）
9. **false bind count**: 0（最終メカニズム。初期実装では4件のfalse bind[「GG」が「SGG」titleに誤って部分一致]があったが、word-boundary判定追加により解消）
10. **missed bind count**: サンプル60件中3件(5.0%)。いずれもsafety上の問題ではなくcompleteness上の軽微な機会損失。
11. **Q6 result A0/A1/A2**: A0=既知の捏造(「×・?・×」記号+架空のGG本前兆→GG本当選フロー)。**A1/A2ともに捏造完全消失**、青7連続当選率・GG中のSP役50%ストック・SGGゲーム数抽選の実在事実のみで構成された正確な回答に改善。
12. **AT-F result**: A0=常に誤紐付け(5/5、Phase4FU〜FWを通じて一度も改善しなかった)。**A1/A2ともに正しく情報不足を申告するよう改善**(初めての成功)。
13. **RT-A/RT-B result**: A0=既知の誤紐付け(ZS-05由来)。**A1/A2ともに正しく情報不足を申告するよう改善**(初めての成功)。
14. **SGG/GG準備中 result**: A0=GG準備中の記述をSGGの説明として誤用。**A1/A2ではSGGの正しい固有情報(ゲーム数抽選、5の倍数条件)のみを使用するよう改善**。専用probe「GG中とGG準備中の違いを教えて」(FX-CB05)でも両者を正しく区別する回答を確認。
15. **ガイアベル result**: A0=「モード」と誤分類。A1では「役」と正しく分類(小役の停止例一覧chunkが正しくbinding)。
16. **SU4 result**: A0から既に正確(専用の「炎・戦車解説」chunkが元々correctly attributedだった)。A1/A2でも維持、より正確な引用に改善。
17. **GG当選/SGG当選 result**: A0=SGG当選についてGG準備中の記述を誤用。**A1/A2では「SGG当選については登録データに情報が見つからない」と正しく申告**。
18. **loop/GG stock result**: A0=関係が逆転した誤り。A2では逆転は解消されたが、完璧な要約ではない(複数の実在事実の統合の巧拙が残る、cross-entity誤紐付けは無し)。
19. **phantom misbinding**: 0/22(最終)。修正過程: 初期14/22改善→Q1抽出修正で22/22に到達。
20. **concept-binding failure**: 0件(cross-entity誤紐付けの意味で)。ループ/GGストックのみ要約品質面での軽微な課題が残る。
21. **query-style結果**: 5つの言い回し(「〜について教えて」×2、「違いは？」、「要約して」、「初心者向けに説明して」)全てでQ6の捏造は再発せず(5/5クリーン)。
22. **RAG50 unsupported numeric**: 0（新規発生なし）
23. **RAG50 unsupported non-numeric**: 0（新規発生なし）
24. **RAG50 misattribution**: 0（新規発生なし）
25. **RAG50 completeness regression**: **0/19（最終、必須8probe含む全件でnumericの欠落なし）**。初期実装では4/8必須probeで欠落が発生したが、フォールバック機構追加により完全解消。
26. **P02**: A0==A2（数値完全一致、無関係な◆項目1件のみ除外）
27. **P04**: A0==A2（完全一致、フォールバックにより全数値保持）
28. **LC-08**: A0==A2（完全一致、フォールバックにより全数値保持）
29. **Q6(RAG50側)**: 数値完全一致、regressionなし
30. **Q11**: 数値完全一致、regressionなし
31. **Q15**: A0==A2（完全一致、フォールバックにより全内容保持）
32. **Q17**: A0==A2（完全一致、フォールバックにより全内容保持）
33. **AD-04**: 情報不足申告を維持、regressionなし
34. **added latency**: 0.1ms未満/response（純Python処理のみ、追加のLLM呼び出しなし）
35. **GPU generation count**: 120件（予算160件中、目安120件ちょうど。3回の実装修正に伴う再生成を含む）
36. **pytest start/end**: 233 passed → 233 passed（regressionなし）
37. **Phase4ZG unchanged**: `278fe7aedc5f302b9966689c9e92c8363fea246db71aab7cc959ce9609dcc9dc`（不変）
38. **production prompt unchanged**: `e859e2aa443160f8b4e8c897f2d9af1e7d310599047ce484864b578984ebddd7`（不変）
39. **dispatch unchanged**: `80dbb4469a201030de1ee7ec6f1d57b69a990d62e7e741638a275f5772a018ad`（不変）
40. **Policy C3 unchanged**: `cb9f904bb02d9b109e1a6b6f773b976d2699d14b39833ce5876a4ba6c1963caf`（不変）
41. **DB/retriever/embedding unchanged**: 119チャンク、read-onlyのみ、変更なし
42. **production integrationなし**: 全てoffline prototype
43. **git opsなし**: commit/push一切なし
44. **Slack status**: 送信試行あり、成功
45. **product architecture feasible YES/NO**: **YES**（CASE FX-A、次フェーズでのproduction統合設計を推奨）
46. **schema change required YES/NO**: **NO**（既存schemaのtitleフィールドだけで十分機能した）
47. **quantization可能か**: いいえ（本フェーズはfeasibility diagnosticでありproduction統合前のため、引き続き保留。ただし根本原因への有効な対策が初めて見つかったため、次フェーズでのproduction統合後に量子化再検討の見込みが立った）
48. **recommended next phase**: (1) production pipelineへの実際の統合設計(retriever呼び出し後、生成前にentity attribution層を挿入)、(2) より広範なprobe setでの再検証、(3) 複数機種がDBに追加された場合の挙動確認、(4) RAG50形式のようなstatic contextへの対応の洗練。
49. **next phase auto-startなし**: しない。ここで停止する。

## 透明性のための開示: 3回の実装修正

本フェーズ中に3つの実装バグを発見し、generation前の段階で修正した(GTの正解基準自体は一度も変更していない):

1. **複合語entity切り詰めバグ**: 「GGプラスとは何か説明して」→「GG」に誤って切り詰められ、phantom entity検出が機能しなかった。助詞境界での切り出しロジックに修正。
2. **「GG」/「SGG」の部分文字列衝突**: 単一entity query(「GGについて教えて」)で「GG」が「SGG」を含むtitleに誤ってbindingされる問題。ASCII単語境界チェックを追加して修正。
3. **RAG50の過剰フィルタリング**: 関連解説文章セクションのentity絞り込みが、Q15/Q17/P04/LC-08で必要な情報まで削ってしまう問題。絞り込み後に50%未満しか残らない場合は全体を保持するフォールバックを追加。

いずれも「バグを発見→ロジックを修正→影響を受けたprobeのみ再生成」という手順で対応し、最終結果は全て修正後の最終メカニズムで統一されている。

## 次への申し送り

本フェーズにより、Phase4FC以来の中心的懸案だったRAG factual safety問題に対する、初めての有効なアーキテクチャレベルの解決策が見つかった。次フェーズでは、この entity-aware context assembly をoffline prototypeから実際のproduction pipeline統合設計へ進めることを最優先で推奨する。GGUF量子化は、production統合の完了・検証後に改めて検討する。

---
*Phase4FX完了。次フェーズを自動開始しない。*
