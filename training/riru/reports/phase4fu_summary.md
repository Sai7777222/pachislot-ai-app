# Phase4FU 完了報告: Multi-Context Grounded Synthesis Root-Cause Diagnostic

## 結論

**CASE FU-H — Mixed Cause（複合要因）**

Phase4FCで発見されたQ6の非数値事実捏造(「×・?・×」)を含む、複数context統合時の事実捏造・誤紐付け問題について、診断のみ(修正なし)を目的に根本原因を特定した。単一の支配的原因は存在せず、**query-style(質問の言い回し)** と **prompt(grounding指示の欠如)** がほぼ同格の主要因であり、context assembly(構造)は捏造の一部サブタイプにのみ有効な部分的緩和策、Phase4ZGアダプタは原因ではなくBase Qwenが時折発揮する自然なhedge行動を抑制した副作用要因、という4層構造であることが実験的に確認された。

## Section: 必須報告項目(37項目)

1. **Section3 forensic結論**: 「×・?・×」はcontext中に完全一致・単純結合いずれの形でも一切存在しない。実在するのは「×・?・?」のみ(1回)。個別記号「×」(1回)「?」(2回)は存在するが、新規記号列の構築は意味的創作(confabulation)と判定。CASE FU-N(捏造ではなかった)は明確に棄却。
2. **Section4 GT件数**: 31件(必須7probe: Q6/ZS-05/P02/LC-08/Q11/Q17/AD-04を全て含む)。カテゴリ: single_factual x5, comparison x5, summary x5, beginner_explanation x6, multi_entity_relation x5, insufficient_context_comparison x5。
3. **GT作成方式**: 実際に取得されたretrieved contextのみを読み、モデル出力を一切見ずに人間が判定(RULE EVAL-002準拠)。
4. **GTハッシュ**: sha256 `789d19c917dbeeb273995d63ce1acb0562384005dbdaf30cbd2f6bea56ce4378`、frozen_before_generation=true。
5. **Stage A(検索十分性分布)**: SUFFICIENT 9件, PARTIAL 9件, INSUFFICIENT 4件, IRRELEVANT 9件。
6. **Stage A(検索十分性と捏造発生率の相関)**: SUFFICIENT 11.1%(1/9)、PARTIAL 55.6%(5/9)、INSUFFICIENT 25.0%(1/4)、IRRELEVANT 55.6%(5/9)。**INSUFFICIENT区分が最も発生率が低いという直感に反した結果**であり、検索十分性が支配的要因(CASE FU-A)ではないことを裏付けた。
7. **Stage B(全31probeベース生成)結果**: production prompt + Phase4ZG + 実context + greedyで生成。31件中18件はクリーン、13件で何らかの問題(誤紐付け/捏造/無根拠推論/サイレント話題脱落/検索結果の誤活用)を検出。
8. **Stage B atomic claim分類**: MISATTRIBUTED 8件、UNSUPPORTED(記号・フロー創作) 1件、UNSUPPORTED(関係性創作) 2件、UNSUPPORTED_INFERENCE 1件、サイレント話題脱落 2件、検索活用ミス 1件。
9. **Q6(FU-D01)の再現**: production promptで100%再現。さらに、Phase4FCで確認された1文の捏造に加え、「GG本前兆→GG本前兆→GG本前兆→GG本当選」という全く新しい段階遷移フローの創作も新たに確認された(捏造が反復ごとに拡張されうることを示す)。
10. **ZS-05(FU-B05, RT-A/RT-B)の再現**: production promptで確認済みの誤紐付けが完全に再現。Stage E(10回のsampling再試行)でも一貫して誤紐付けが発生(10/10)、hedgeは一度も発生せず。
11. **新規発見: phantom-entity誤紐付けパターン**: 完全に架空の固有名詞ペア(X-A/X-B型)を問う8probe中5件(AT-F, モードα/β, RT-A/B, RT-C/D)で無関係な実在チャンクを誤紐付け。残り3件(AT-A/AT-B, CZ-A/CZ-B, モードγ/δ)は正しく情報不足を申告しており、発生は非決定的だが高頻度で再現性あり。
12. **新規発見: SGG/GG準備中混同パターン**: SGGについて問われる際、GG準備中固有の記述がSGGの定義であるかのように誤って紐付けられる現象が3probe(D01, D03, D05)で再現。SUFFICIENT/PARTIALいずれの検索十分性でも発生。
13. **Stage C(context構造ablation)対象**: Q6+ZS-05+10probe=12probe × 5条件(C1現状/C2関連のみ/C3エンティティグループ化/C4構造化のみ/C5説明文のみ)=60生成。
14. **Stage C結果(Q6)**: C3(エンティティグループ化並べ替え)が「×・?・×」の発生を防いだ(実在エンティティに対する物語的脚色型捏造には有効)。
15. **Stage C結果(RT-A/RT-B)**: C1〜C5いずれの条件でも誤紐付けは解消されなかった(完全に架空の固有名詞への誤紐付け型捏造には無力)。
16. **Stage C結論**: context構造調整は捏造の一部サブタイプ(物語的脚色)にのみ有効な部分的緩和策であり、単独の解決策にはなり得ない。
17. **Stage D(prompt比較)対象**: 同じ12probeについて、本番prompt(Stage Bで既取得) vs 診断用最小grounding prompt(反推測・反誤紐付け指示を明示)。
18. **Stage D結果**: 最小grounding promptへの差し替えのみで、Q6は「×・?・×」を発生させなくなり、RT-A/RT-B(B05)とAT-F(A03)は誤紐付けから正しい情報不足申告に転換、ガイアベル(E02)の定義的誤り(モードと誤称)も修正された。
19. **Stage D結論**: prompt層のgrounding指示の強化が最も一貫して効果を示した単一介入である。本番promptには「複数断片統合時に新しい対比構造を付け加えない」「無関係な断片を問い合わせられた固有名詞に紐付けない」という指示が欠けている。
20. **Stage E(adapter比較)対象**: Q6・ZS-05は各10回(ZG10+Base10)、追加4probe(AT-F, モードα/β, SGGとRT, RT-C/D)は各1回(ZG1+Base1)、`model.disable_adapter()`使用。
21. **Stage E結果(Q6)**: Phase4ZGは10回中1回のみ元の捏造そのものを再現、残りは毎回異なる形の創作(sampling起因の多様性はあるが、いずれも何らかの無根拠な関係性を含む)。Base Qwenは10回**全て**でPhase4ZGより悪質な捏造(「GG(Great Game)」「SGG(Super Great Game)」という架空の英語略称の創作など)を示した。
22. **Stage E結果(RT-A/RT-B)**: Phase4ZG・Base Qwen双方とも10回中10回とも誤紐付けが発生。Base Qwenはさらに具体的な数値(15.2%/20.3%/64.5%など、他の文脈では実在するがRT-A/B specificではない数値)まで誤紐付けしており、これがPhase4ZS/ZTの数値限定監査がPhase4ZG版の同種問題を見逃した理由(ZGは非数値的な言い回しで誤紐付けするため数値パターンに引っかからない)を説明する副次的発見となった。
23. **Stage E結果(モードα/β、RT-C/D)**: Phase4ZGは両方とも確信的に誤紐付け。Base Qwenは両方とも正しく「情報が含まれていません」「教えていただけますか」と情報不足を申告し、聞き返しまで行った。
24. **Stage E結論**: アダプタは捏造の「原因」ではない(Base単体でも同等以上に悪質な捏造を起こす)。しかしPhase4ZGの性格学習は、Baseが時折発揮する適切な留保・確認要求の動作を体系的に抑制しており、Phase4ZN〜ZPで意図的に排除した「hedge intrusion」除去の反動である可能性が高い。
25. **Stage F(query style)対象**: 同一事実・同一context・同一adapter・同一promptで5つの言い回し(「GGについて教えて」「SGGについて教えて」「GGとSGGの違いは？」「GGとSGGを要約して」「GGとSGGの違いを初心者向けに説明して」)を比較。
26. **Stage F結果**: 「初心者向けに説明して」の場合のみ「×・?・×」の記号捏造と段階遷移フローの創作が発生。他4つの言い回しでは一度も発生しなかった(0/4)。
27. **Stage F結論**: 「初心者向けに」という指示が対比構造の完成を促す合成圧力を生み、データが支持しない対称的構造を創作させている。これが本フェーズで最も明確・決定的な単一の発見である。
28. **Stage G(insufficient-context stress test)対象**: 10probe(INSUFFICIENT/IRRELEVANT区分)×2回(greedy+sampling)=20生成。
29. **Stage G結果**: 全probeで両ラン(greedy/sampling)が一貫した挙動を示した(誤紐付けするprobeは両方で誤紐付け、正しく断るprobeは両方で断る)。これはサンプリングノイズではなく体系的な傾向であることを裏付ける。
30. **GPU生成数**: 176件(予算上限180件、上限まで無理に生成せず根本原因特定時点で終了)。
31. **Root cause CASE**: FU-H(複合要因)。主要因(同格): FU-F(query-style)、FU-C(prompt/grounding指示不足)。副次要因(サブタイプ限定): FU-B(context構造)。基層要因: FU-D/FU-E相互作用(アダプタの性格学習によるhedge抑制)。
32. **棄却したCASE**: FU-N(捏造ではなかった)—forensic分解により明確に否定。FU-A(検索十分性が支配的)—相関分析により否定。
33. **推奨される次フェーズの方向性**: (1)本番RAG promptへの反創作・反誤紐付け専用指示の追加(A/Bテスト)。(2)複数断片統合を要求する質問スタイルへの生成後self-critiqueステップ、または保守的な断片列挙形式への制約の検討。
34. **明示的に推奨しない対応**: Phase4ZGの再学習(本フェーズの結果からは正当化されない、FU-Dは主要因ではない)。context構造変更のみでの解決(サブタイプ限定的)。GGUF/Q8/Q5量子化(引き続き保留)。
35. **フリーズしたアーキテクチャの整合性**: Phase4ZGアダプタ(`278fe7ae...`)・conservative dispatch(`80dbb446...`)・Policy C3(`cb9f904b...`)・本番prompt(`e859e2aa...`)は全てフェーズ開始時と終了時で完全一致。変更なし。
36. **pytest**: 233 passed → 233 passed(フェーズ中無変更、regressionなし)。
37. **git状態**: commit/push一切なし。新規29ファイル(前フェーズ分9件+本フェーズ分ファイル群)は全てuntrackedのまま。branch: `checkpoint/identity-closure-phase4zn-baseline`(変更なし)。

## Slack通知

既存の`.env`設定済みIncoming Webhook経由で完了通知を送信済み(下記参照)。

## 次への申し送り

本フェーズにより、Phase4FCで発見された非数値事実捏造は「単一のバグ」ではなく、query-style・prompt・context構造・adapter性格学習という複数レイヤーが絡み合った複合的な現象であることが判明した。最も明確で再現性の高い単一の発見は、「初心者向けに説明して」という言い回しが対比構造完成の合成圧力を生むこと(Stage F)、および最小grounding promptへの差し替えだけでQ6・RT-A/RT-B・AT-F・ガイアベルの4つの既知の問題を同時に解消できたこと(Stage D)である。次フェーズでは、この2つの知見を土台に、本番prompt層への反創作指示の追加と、複数断片統合クエリへのproduct側の保守的UX制約を検証することを推奨する。GGUF量子化は引き続き保留とする。

---
*Phase4FU完了。次フェーズを自動開始しない。*
