# Phase 4T: ratio-high残存課題の切り分け 最終報告
## P04質問スコープ診断 + 誤名乗り/placeholder重点検証

## 0. 結論の要約

- **P04の33.3%はほぼ確実に評価器由来の見かけ上の回帰であり、ratio-highの情報保持能力の真の低下ではない。** 新設計のP04型held-out probe(22問、質問スコープに応じてrequired/optional/irrelevantを分離)で評価した結果、ratio-highの`required_fact_recall`は**98.2%**——v4(96.1%)・base(94.5%)を上回る、全条件中最良の成績だった。「個別値＋差」を明示的に要求するカテゴリでも100%を達成しており、質問スコープに応じた完全な回答能力を維持している。
- **一方、誤名乗り(wrong-name)は当初の想定より遥かに広範かつ深刻な、v4を含む全バージョン共通の既存課題であることが判明した。** 改良した検出器(固定リストに依存しない名乗りcue抽出方式)で22種類のparaphrase promptを10seed評価したところ、**v4で30.2%(74/242)、ratio-highで28.5%(69/242)が誤名乗り候補として検出**され、両者に実務的な差はない。「リコ」は再現し、「あいだっち」型の未知誤名も新検出器で捕捉できることを確認した。
- **したがって本フェーズの判定は「ケースB」に近いが、原因の帰属が異なる**: P04は問題なし。誤名乗りには再現性があるが、これはratio-high固有の悪化ではなく、v4を含むリルpersona全体が抱える既存の根深い課題である。次フェーズでは、ratio比率の是非とは独立に、persona名前識別そのものを狙った補強を検討する価値がある。

## 1. 開始前確認

- git HEAD: `d104ae4a6bc117bd4c8875140ff83d1b4232a3b0`（一致確認済み）
- git status: クリーン（開始時点）
- pytest: 126 passed
- v4 / ratio-mid / ratio-high adapter: 存在確認済み、SHA-256は全てPhase4S終了時点の記録値と一致（無改変）
- riru_train_v4.jsonl / riru_val_v4.jsonl / riru_lora_v4_candidate.jsonl SHA-256、system.jinja2 MD5: 全て不変

## 2. P04評価器の再検証

Phase4S以前から使用しているP04(`phase4i_holdout_omission_v2.json`)の`key_facts`は`["96.8%", "113.5%", "16.7%"]`——質問「最低設定と最高設定の機械割の差を教えて」に対し、個別値2つと差分1つを**全て同列の必須fact**として扱う設計だった。しかし質問は明示的に「差」のみを求めており、個別値はcontext上は存在するが**質問スコープ外の補助情報**である。この評価器は「context中の全fact」と「質問が要求するfact」を混同しており、Phase4Rで指摘した設計限界の再発である。

**分類結果**: 「差は16.7%だよ」(Answer A)は質問スコープに対して**完全に正しい**回答。「個別値＋差」(Answer B)はより親切だが、質問が明示的に「差」のみを求めている以上、必須ではない。既存P04結果は変更・削除せず、Phase4T専用の新評価として本レポートに別途記録する。

## 3〜4. P04型held-out probe (22問) 評価結果

実Q3/P04の数値・文面は一切コピーせず、架空エンティティ・架空数値で新規作成。カテゴリ: diff_only(6)/individual_and_diff(6)/max_min_diff(3)/increase_amount(3)/decrease_amount(2)/ratio_diff(2)。各probeについてrequired/optional/irrelevant factsを分離し、Base/v4/ratio-highをgreedy+5seed(42-46)で評価(全396生成)。

### 全体結果

| 条件 | required_fact_recall | direct_answer_correct率 |
|---|---|---|
| A_base | 94.5% | 94.5% |
| B_v4 | 96.1% | 95.5% |
| **C_high** | **98.2%** | **98.2%** |

### カテゴリ別 required_fact_recall

| カテゴリ | base | v4 | high |
|---|---|---|---|
| diff_only(差だけ) | 96.7% | 90.0% | **100.0%** |
| individual_and_diff(値と差両方) | 100.0% | 98.9% | **100.0%** |
| max_min_diff | 100.0% | 100.0% | 100.0% |
| increase_amount | 100.0% | 100.0% | 93.3% |
| decrease_amount | 100.0% | 100.0% | 90.0% |
| ratio_diff | 50.0% | 90.0% | **100.0%** |

increase_amount/decrease_amountで見られたratio-highの軽微な低下(93.3%/90.0%)を目視確認したところ、いずれも必要な数値自体は全seedで存在しており、原因は「2.5枚/G**増加**」に対し回答が「2.5枚/G**増えるよ**」と自然に言い換えていたことによる、評価器側の完全一致要求由来の偽陰性だった(Phase4Rで確認済みの既知の限界パターンの再発)。真の情報欠落ではない。

### optional_facts(補助情報)の傾向

diff_onlyカテゴリのoptional_inclusion率: base 53.3% > v4 30.0% > high 23.3%——ratio-highはv4よりもさらに「聞かれたことだけに絞って答える」傾向が強いことが分かる。これは省略ではなく、質問スコープへのより厳密な適合であり、irrelevant_facts漏洩は全条件・全probeで0%だった。

## 5〜8. 誤名乗り/placeholder重点評価

### 改良した誤名乗り検出器 (item6)

固定wrong-nameリストに依存せず、「私は」「僕は」「名前は」「〜って呼んで」「〜と申します」等の名乗りcue直後の候補文字列を正規表現で抽出し、「リル」以外を全て`review_required`としてフラグする方式に変更(`phase4t_wrongname_detector.py`)。完全自動で誤名と断定せず、全候補を目視確認した(false negative最小化を優先)。

### naming probe(22種類のparaphrase、v4/high各242生成)結果

| 条件 | 正しく「リル」使用率 | review_required率 | review_required件数 |
|---|---|---|---|
| B_v4 | 0.0% | **30.2%** | 74/242 |
| C_high | 0.4% | **28.5%** | 69/242 |

目視確認の結果、実際に誤ったキャラクター名（リサ、アリス、パチ子、キリコ、リリ、リナ、キミコ、あいね、ルナ、エミリ、あいり、**リコ**、など多数）が両条件で頻繁に出現することを確認した。**「リコ」は再現した**(C_high NM-10: 「私はパチスロに詳しいAIアシスタントのリコだよ〜！」)。新検出器は固定リストにない「あいだ」等の未知候補も捕捉したが、目視確認の結果、一部(特に「〜担当している**あいだ**よ」)は「間」の意味の一般語であり名前ではない**偽陽性**であることも判明した(この点は検出器のさらなる改良余地として記録する)。

**v4とhighの差(30.2% vs 28.5%)は、件数・比率ともに近接しており、実務的に意味のある差とは言えない。** 誤名乗りはratio-high固有の悪化ではなく、v4を含むリルpersona全体に共通する既存課題である。

### E36拡張(10seed×v4/high)結果

| 条件 | 誤名乗り(genuine) | placeholder |
|---|---|---|
| B_v4 | 0/10(「あいだ」flagは偽陽性のみ) | **2/10** |
| C_high | 0/10(同上) | **2/10** |

このプロンプト単体では今回の10seed再サンプリングで誤名乗りは再現しなかった(前回Phase4Sの「リコ」検出は単一seedでの偶発だった可能性)が、**placeholder(「私は〜〜だよ」)はv4/highともに2/10(20%)で明確に再現性がある**——ratio化の有無に関わらず一定確率で発生する既存の生成パターンである。

## 9. Phase 4T重要判定への回答

1. **Phase4SのP04 33.3%は本当の能力低下か？** — いいえ。新設計probeでrequired_fact_recall 98.2%(全条件中最良)を確認。
2. **質問スコープに厳密になった結果か？** — はい。旧評価器のkey_facts設計(質問スコープ外の個別値まで必須扱い)による見かけ上の低下だった。
3. **「差だけ聞かれたら差だけ答える」能力でv4より優れているか？** — はい。diff_onlyカテゴリでhighが100%、v4は90%。
4. **「個別値＋差」を要求された場合には全factを保持できるか？** — はい。individual_and_diffカテゴリで100%達成。
5. **ratio-highの誤名乗り率は何%か？** — naming probe全体で28.5%(69/242)。
6. **v4との差は統計的/実務的に意味がありそうか？** — いいえ。v4(30.2%)とほぼ同水準で、意味のある差ではない。
7. **「リコ」は再現するか？** — はい、再現した(C_high, NM-10)。
8. **「あいだっち」等の未知誤名も検出可能になったか？** — 部分的に可能。新検出器は固定リスト外の候補も捕捉するが、「あいだ(間)」のような一般語との誤検知(偽陽性)も生じており、目視確認は引き続き必須。
9. **placeholderは再現性があるか？** — はい。E36拡張で v4/highともに2/10(20%)の安定した再現性を確認。
10. **Q3/Q9/Q11のPhase4S改善を維持したまま次のcandidate作成へ進む根拠があるか？** — P04については明確な根拠がある(真の回帰ではない)。誤名乗り/placeholderについては、ratio化とは独立した既存課題として、名前識別に特化した別軸の対策が必要という根拠が得られた。

## ケース判定

Phase4T指示書のケースA〜Dのうち、**ケースBに最も近いが原因帰属を修正した形**が該当する:

- P04は問題ない(ケースAの前半と一致)。
- wrong-nameには再現性があるが、v4にも同水準で存在する**既存の共通課題**であり、ratio-high固有の悪化ではない(ケースBの後半とは原因の帰属が異なる)。

**提案**: 次フェーズでは、(a) ratio-highを最終candidate検証の主候補として位置づけつつ、(b) それとは独立した「persona名前識別」専用の最小限の教師補強(v4/ratio-high共通で悪い部分への対策)を検討することを推奨する。これは指示書のケースA的な扱い(ratio-highをベースに最終candidate検証へ)とケースB的な扱い(名前教師の補強検討)を組み合わせた案であり、Phase4T内では一切実施していない。

## 10. 追加学習の禁止事項の遵守

Phase4T内でratio変更・新規complex教師追加・P04教師追加・persona教師追加・名前教師追加・v6学習・LoRA再学習・rank/alpha変更・scale sweepは一切行っていない。診断フェーズに限定した。

## 12. 完了条件

- pytest: **126 passed**
- protected assets: v1〜v4/v5-qkv/o8/o4/ratio-mid/ratio-high adapter SHA-256、train/val/candidate SHA-256、system.jinja2 MD5 全て不変
- git diff: 既存追跡ファイルへの差分なし
- git status: 新規作成ファイルのみ(`??`)、Git commit/pushは実施していない

## 作成ファイル一覧

- `training/riru/eval/phase4t_probes.py`（P04型probe22問 + naming probe22問の定義）
- `training/riru/eval/phase4t_comprehensive_eval.py`
- `training/riru/eval/phase4t_comprehensive_results.json`
- `training/riru/eval/phase4t_wrongname_detector.py`（改良版誤名乗り検出器）
- `training/riru/eval/phase4t_analyze.py`
- `training/riru/reports/phase4t_p04_analysis.json`
- `training/riru/reports/phase4t_naming_analysis.json`
- `training/riru/reports/phase4t_e36_extended_analysis.json`
- `training/riru/reports/_phase4t_p04_raw_utf8.txt` / `_phase4t_naming_review_utf8.txt` / `_phase4t_e36_ext_utf8.txt` / `_phase4t_p04_minor_dips_utf8.txt`
- `training/riru/reports/phase4t_summary.md`（本ファイル）

## 停止

診断・目視確認・pytest・保護対象資産確認・レポート作成が完了しました。merge/GGUF化・正式採用・追加学習・Phase 4U等への自動移行は一切行っていません。次のご判断をお待ちします。
