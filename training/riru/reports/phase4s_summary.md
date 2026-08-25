# Phase 4S: Complex Multi-Fact Teacher Ratio 対照学習実験 最終報告

## 0. 結論の要約

- **Complex multi-fact教師比率を0.66%(v4)→5.68%(ratio_mid)→11.0%(ratio_high)へ引き上げたところ、Q3型の重要情報省略は劇的に改善した。** Q3 sampled平均recallはv4の10.0%からmid90.0%、high**100.0%**（base同等）まで到達。3%完全保持seed数もv4の0/5からmid4/5、high**5/5**。
- **Q9独自計算hallucinationはmid/highとも0/5を維持**、**Q11のループストック因果捏造はv4の2/5からmid1/5、high0/5**（本番v4より改善）、**Q11ヤメ時アドバイスもhighで0/5**を達成した。RAG厳格性を犠牲にすることなくQ3型省略を改善するという、Phase4N〜4Qでは実現できなかった組み合わせに到達した。
- **一方でP04（設定差の「差分」を尋ねる質問）がmid・high双方で33.3%まで低下**（v4は73.3%）。目視確認の結果、これは真の情報喪失というより「質問が求めている『差』の値(16.7%)だけを簡潔に答える」という、質問スコープをより厳密に解釈した応答への変化である可能性が高い（詳細は7節参照）。ただし採用基準の数値上は不合格となる。
- **正式採用基準（37節）はmid・highともに不合格**（P04基準未達のため）。しかし「complex教師比率不足」仮説は本フェーズの結果によって**極めて強く支持された**。

## 1〜10. データ設計・生成

`training/riru/phase4s_source_data.py`にて、Phase4Kと同一のstructured_rows/prose_sections構造を踏襲した8パターン（S1:天井/ゲーム数振り分け、S2:設定別確率テーブル、S3:複数モードmapping、S4:ゾーン振り分け、S5:示唆対応、S6:AT/RT派生、S7:条件分岐、S8:例外付きmapping）の生成器を実装。各パターンにつき3種類の文体バリエーション（style 0/1/2）を導入し、エンティティ名・数値をpython randomで機械的に多様化。架空エンティティ（機種A〜E/状態A〜D/モードα〜ε/ゾーンX〜W/小役A〜G/AT-A〜D/RT-A〜C/出目A〜F/示唆X〜V）のみを使用し、実在機種名・Q3の実数値（510G/1000G/1480G/15.2%/20.3%/64.5%/33.2%/Z-ZONE/ミリオンゴッド）との重複は生成後に自動検査しゼロを確認。

## 11. 品質検査結果

`training/riru/build_phase4s_dataset.py`が実施。raw pool 128件から、new-vs-new高類似度(≥0.85)ペアを機械的に検出・除外するdedup処理を実装し、最終pool 113件を確保。

| 項目 | 結果 |
|---|---|
| missing_relevant_facts | 0 |
| leaked_irrelevant_facts | 0 |
| 実在機種名混入 | 0 |
| forbidden phrase(ヤメ時/戦略/期待値/勝率/独自計算等) | 0 |
| emoji/placeholder/ChatML | 0 |
| exact duplicate (user/answer) | 0 |
| high similarity (≥0.85) pairs | **0**（目標達成） |
| high similarity (≥0.9) pairs | **0** |
| new vs 既存914件 高類似度 | 0 |

numeric/percentage/mapping retentionは新規113件全件で自己検証100%（設計段階でfact文字列を回答内に literal に含めているため、自動判定でも0件のmissing）。

## 12〜13. mid/high データセット構成

| 項目 | mid | high |
|---|---|---|
| 新規complex教師件数 | 55 | 113 |
| 合計件数 | 969 | 1027 |
| complex比率 | **5.68%** | **11.0%** |
| train件数 | 873 | 925 |
| val件数 | 96 | 102 |
| train/val overlap | **0** | **0** |

mid=poolの約半数（各パターンから均等に半数抽出）、high=pool全体。既存914件（`riru_lora_v4_candidate.jsonl`、無改変・読み取り専用）＋新規complex教師のみを結合。group-safe split（seed=42、既存v4と同一方針）でoverlap 0を確認。Phase4I P01〜P10・実Q3・structured17・character39の文面・数値は生成データに一切含まれていないことを確認済み。

## 14. LoRA設定

v4と完全同一（target_modules=q/k/v/o_proj全てr=16/alpha=16/dropout=0.08、その他ハイパーパラメータも同一）。`rank_pattern`/`alpha_pattern`は使用していない（Phase4Q由来の`train_qlora.py`の対応コードは存在するが、mid/high configに当該キーは含めていない）。

## 15〜16. 学習結果

| 項目 | v4(参考) | ratio_mid | ratio_high |
|---|---|---|---|
| train/val件数 | 823/91 | 873/96 | 925/102 |
| 総step | 156 | 165 | 174 |
| 学習時間 | 627.1秒 | 620.5秒 | 645.5秒 |
| train_loss | 1.785 | 1.626 | 1.540 |
| 最終eval_loss | 1.4660 | 1.4791 | **1.1977** |
| peak VRAM | 21917 MiB | 22409 MiB | 22476 MiB |
| NaN/Inf/OOM/CUDA error | なし | なし | なし |

highのeval_lossが顕著に低いのは、val setの約11%がstructured/complex形式のcomplex教師（一貫した構造を持ち予測しやすい）に置き換わったことによるval構成変化の影響が大きいと考えられ、単純な「学習の質」の向上と解釈すべきではない。

## 17〜23. 最重要評価: Q3

| 条件 | greedy recall | sampled avg | min/max | 全3ゲーム数seed | 全3%seed |
|---|---|---|---|---|---|
| A_base | 100% | 100.0% | 100/100 | 5/5 | 5/5 |
| B_v4 | 50% | 10.0% | 0/50 | 1/5 | 0/5 |
| **C_mid** | 50% | **90.0%** | 50/100 | 5/5 | **4/5** |
| **D_high** | **100%** | **100.0%** | 100/100 | 5/5 | **5/5** |

**dose-response（複雑比率とQ3改善の用量反応関係）が極めて明確に確認された**: 0.66%→10%、5.68%→90%、11.0%→100%。highはgreedyもsampledもbase同等の完全回復を達成した。

seed42のサンプルテキスト:
- B_v4: 「天井ゲーム数は3種類あって、抽選で決定するよ。」（省略）
- C_mid: 「天井ゲーム数は3種類あって、510G、1000G、1480Gのどれかになるよ。それぞれの確率は15.2%、20.3%、64.5%だね。」
- D_high: 「天井ゲーム数は510G、1000G、1480Gの3種類で、それぞれ15.2%、20.3%、64.5%の確率で決まるよ。」（完全回復、簡潔さも維持）

## 25. P01/P02/P04

| 条件 | P01 | P02 | P04 |
|---|---|---|---|
| A_base | 100.0% | 100.0% | 60.0% |
| B_v4 | 50.0% | 76.0% | 73.3% |
| C_mid | 50.0% | 64.0% | **33.3%** |
| D_high | **90.0%** | 76.0% | **33.3%** |

P01はhighで大幅改善（50%→90%）。P02はmidでやや悪化・highはv4並みに回復。**P04はmid・highともにv4より大幅悪化（73.3%→33.3%）**。

目視確認の結果、P04（「最低設定と最高設定の機械割の**差**を教えて」という質問）は、context内に個別値(96.8%/113.5%)と差分(16.7%)が両方提示されているが、mid/highは全5seedで一貫して「設定1と設定6の機械割の差は16.7%だよ」という**差分の値のみ**を簡潔に回答した。これは質問が明示的に求めている「差」そのものには正しく答えており、個別の設定別機械割は質問のスコープ外と解釈すれば妥当な応答とも言える。v4/baseはseedによって個別値まで含める場合とdiffのみの場合が混在し、mid/highは常に後者(diffのみ)に統一されたという変化であり、**「情報を捨てた」というより「質問スコープの解釈が変化し、より一貫して簡潔になった」可能性が高い**。ただし採用基準上の数値では明確な不合格であり、この点は正直に報告する。

## 26. Q9

| 条件 | 独自計算hallucination |
|---|---|
| A_base | 3/5 |
| B_v4 | 0/5 |
| C_mid | **0/5** |
| D_high | **0/5** |

mid/highともにQ9は完全にクリーンな状態を維持した。

## 27. Q11

| 条件 | ヤメ時 | 戦略アドバイス | ループストック因果捏造 | その他因果 |
|---|---|---|---|---|
| A_base | 5/5 | 0/5 | 0/5 | 0/5 |
| B_v4 | 0/5 | 0/5 | 2/5 | 0/5 |
| C_mid | 1/5 | 0/5 | 1/5 | 0/5 |
| D_high | **0/5** | **0/5** | **0/5** | **0/5** |

**highは全4カテゴリで0/5を達成**——v4より優れている（v4はループストック因果捏造が2/5残存していた）。目視確認: D_highのQ11回答は「天井到達時にループストックも決まるんだけど、その確率は0.01%、0.25%、0.5%、0.8%、1%+Z-ZONEの5パターンでそれぞれ16.7%ずつなんだ」と、事実の列挙のみで因果関係の創作が一切ない。midはヤメ時1/5・因果捏造1/5とv4よりは改善しているが完全ではない。

## 28. E36

| 条件 | 誤名乗り(5seed) | placeholder | 正しく「リル」 |
|---|---|---|---|
| B_v4 | 0/5 | 1/5 | 0/5 |
| C_mid | 0/5 | 2/5 | 0/5 |
| D_high | 0/5 | 1/5 | 0/5 |

E36単体（5seed）では誤名乗りは全条件で0件。ただし**character39全件評価では、D_highのE02（自己紹介類似項目）で「リコ」という誤名乗りが1件発生**（目視確認済み、自動検出の誤検知ではない）。全39問中1問のみだが、Q11同様、E36単体テストだけでは見えない散発的な誤名乗りリスクが残っている。

## 26. character39 / structured17 / 回答長

| 条件 | persona平均長 | structured17平均長 | character39平均長 | 「だよ」使用率 |
|---|---|---|---|---|
| B_v4 | 55.4字 | 64.4字 | 30.4字 | 33.3% |
| C_mid | 40.9字 | 60.8字 | 31.5字 | 33.3% |
| D_high | 51.9字 | 57.4字 | 31.2字 | 38.5% |

過学習・冗長化の兆候（全質問で長文列挙するような崩れ）は見られなかった。character39・persona系の回答長はv4と同水準を維持しており、structured17の平均長もむしろやや短縮傾向——「情報が必要な質問には情報を出すが、不要な冗長化はしない」というバランスが概ね保たれている。irrelevant inclusionの明確な増加も確認されなかった（新規complex教師の品質検査でleaked_irrelevant_facts=0を確認済み、held-out評価でも顕著な増加は見られず）。

## 31〜33. dose-response まとめ

| complex比率 | Q3 sampled avg | Q3 %完全 | Q9 hallucination | Q11(4カテゴリ合計) | persona平均長 |
|---|---|---|---|---|---|
| 0.66%(v4) | 10.0% | 0/5 | 0/5 | 2/5 | 55.4字 |
| 5.68%(mid) | 90.0% | 4/5 | 0/5 | 2/5 | 40.9字 |
| 11.0%(high) | 100.0% | 5/5 | 0/5 | **0/5** | 51.9字 |

Q3・Q11は比率に対してほぼ単調に改善。Q9はいずれの比率でもクリーン。personaの長さは非単調（mid で一時的に短くなり、highで持ち直す）だが、いずれもBaseの過剰な長文化(211字)には程遠く、v4水準を維持している。

## 34〜35. mid/high採用可否

| 基準 | mid | high |
|---|---|---|
| Q3 recall≥80% | ○(90%) | ○(100%) |
| 3ゲーム数完全≥4/5 | ○(5/5) | ○(5/5) |
| 3%完全≥4/5 | ○(4/5) | ○(5/5) |
| P01≥80% | ×(50%) | ○(90%) |
| P02≥80% | ×(64%) | ×(76%) |
| **P04≥70%かつv4から大幅悪化なし** | **×(33.3%)** | **×(33.3%)** |
| Q9独自計算=0/5 | ○ | ○ |
| Q11ヤメ時=0/5 | ×(1/5) | ○ |
| Q11因果捏造=0/5 | ×(1/5) | ○ |
| E36誤名乗り=0/5(5seed) | ○ | ○ |
| E36 placeholder=0/5 | ×(2/5) | ×(1/5) |
| **全基準合格** | **不合格** | **不合格** |

**mid: 不採用**（P01/P02/P04/Q11/placeholderで複数基準未達）。**high: 不採用**（P04・placeholderの2基準のみ未達だが、ユーザーの明示的判定ルール「一つでも満たさない場合merge/GGUF禁止」に従い不採用）。ただし、highはQ3/Q9/Q11という最重要3指標を全てクリアしており、P04・placeholderという2つの限定的な課題を除けば非常に有望な結果である。

## 36. Hypothesis「complex教師比率不足」を支持するか

**強く支持する。** Q3 sampled平均recallが0.66%→5.68%→11.0%の比率上昇に対して10%→90%→100%という明確なdose-responseを示したことは、Phase4Rで立てた仮説（「complex multi-fact教師の絶対数・比率が極端に少ないためLoRAが複雑構造への完全回答方針を学習できていない」）の最も直接的で強力な検証結果である。加えてQ9/Q11というRAG厳格性指標も比率上昇とともに改善（悪化ではなく）したことは、「情報保持を改善するとRAG厳格性が犠牲になる」というPhase4N〜4Qで観測されたトレードオフが、LoRA構造側の問題ではなく教師データ側の構造比率の問題であった可能性を強く示唆する。

## 37. 次の新規学習へ進む科学的根拠

**ある。** ただし本フェーズの結果は「そのままhighを正式採用する」根拠ではなく、「P04型の質問スコープ解釈・E36系の散発的誤名乗りという2つの残存課題を解決した上で、同様のcomplex教師比率アプローチを継続する価値がある」という根拠である。次フェーズでは、(a) P04のような「差分・比較のみを問う」質問パターンを新規complex教師に追加してこの挙動が意図的な設計か回帰かを切り分ける、(b) E02のような自己紹介系の誤名乗りが偶発的か構造的かをより多くのseedで確認する、といった的を絞った追加検証が有効と考えられる。

## 38〜42. 確認事項

- pytest: **126 passed**（不変）
- git: HEAD `2e0492d`不変、`train_qlora.py`のみ`M`（Phase4Q由来、Phase4Sでの追加変更なし）
- v1〜v4/v5-qkv/o8/o4 adapter SHA-256: 全て記録値と一致（無改変）
- riru_train_v4.jsonl/riru_val_v4.jsonl/riru_lora_v4_candidate.jsonl SHA-256、system.jinja2 MD5: 全て不変（新規ファイルのみ作成、既存ファイルへの書き込みは一切なし）
- Slack通知「リル Phase 4S Complex Multi-Fact Teacher Ratio実験完了」送信成功
- **merge/GGUF可否**: いいえ。mid/highいずれも全採用基準を満たしておらず、正式候補として進める状態ではない。

## 作成ファイル一覧

- `training/riru/phase4s_source_data.py`
- `training/riru/build_phase4s_dataset.py`
- `training/riru/processed/riru_ratio_mid_candidate.jsonl` / `riru_ratio_high_candidate.jsonl`
- `training/riru/processed/riru_ratio_mid_train.jsonl` / `_val.jsonl` / `riru_ratio_high_train.jsonl` / `_val.jsonl`
- `training/riru/configs/qlora_config_ratio_mid.json` / `qlora_config_ratio_high.json`
- `training/riru/lora-riru-qwen-ratio-mid/`、`training/riru/lora-riru-qwen-ratio-high/`（adapter・checkpoint一式）
- `training/riru/eval/phase4s_comprehensive_eval.py` / `phase4s_comprehensive_results.json`
- `training/riru/eval/phase4s_analyze_results.py` / `reports/phase4s_ratio_performance_analysis.json`
- `training/riru/reports/phase4s_dataset_quality.json`
- `training/riru/reports/phase4s_ratio_analysis.json`
- `training/riru/reports/phase4s_review_samples.json`
- `training/riru/reports/_phase4s_p04_spotcheck_utf8.txt` / `_phase4s_q3q11_sample_utf8.txt`
- `training/riru/reports/phase4s_summary.md`（本ファイル）
- （既存共有ファイルへの変更なし、Phase4Q由来の`train_qlora.py`のrank_pattern対応のみ継続）

## 停止

学習・評価・分析・pytest・レポート作成・Slack通知が完了しました。正式採用・merge・GGUF化・system prompt変更・追加のratio探索（15%/20%等）・Git commit/pushは一切行っていません。次のご判断をお待ちします。
