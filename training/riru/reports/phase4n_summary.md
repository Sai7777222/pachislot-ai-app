# Phase 4N: LoRA影響度・適用強度・層別切り分け監査 最終報告

## 0. 結論の要約

- **安全な一時scale変更は達成できた**（peft 0.20.0の`LoraLayer.scaling[adapter_name]`を一時的に乗算・復元、ファイル書き換えなし）。
- **Q3型省略はLoRA適用強度に強く依存する**: v4は scale 0.25以下で recall 100%（base同等）に回復し、scale 0.5〜0.75で崩れ始め、production設定(scale=1.0)で sampled recall 10%まで低下する。v2も同型のカーブを示し、v4固有の問題ではない。
- **module-level ablationでo_projが単独最大の犯人**であることが判明: q/k/v_projをフル強度のまま o_projのLoRA寄与だけを0にすると、Q3 recallが100%まで完全復元し、かつ他module(q_proj/v_proj)単独off時に見られた「誤ったキャラクター名の幻覚」も出ない。
- **layer-level ablationでは特定の1/3層への局所化は見られず**、front/mid/backのどれを外しても（あるいはどれか1つだけ残しても）ほぼ全条件でsampled recallが100%へ回復する — 「全48層が揃って初めて」省略挙動が安定再現される、分散的・脆弱な現象である。
- **重要な留保**: scale=0.25でもo_off条件でも、Q3・P01・P02のrecallは完全回復し、Q9の派生計算幻覚も出ないが、**Q11で新種の禁止カテゴリ幻覚（ヤメ時アドバイス）が新たに出現する**。production v4はこの問題を起こさない代わりに別の幻覚（ループストック因果の捏造）を起こす。つまりどの条件も「クリーンな勝利」ではない。

## 1. scaleの安全な変更手法

`peft==0.20.0`の`LoraLayer.forward`実装を`inspect.getsource`で直接確認: `result = result + lora_B(lora_A(dropout(x))) * scaling` であり、`scaling`は`{adapter_name: float}`の単純な辞書。`training/riru/eval/phase4n_lora_scale_experiment.py`の`scaled_lora()`がcontextmanagerとして、対象レイヤーの`scaling[adapter_name]`を一時的に`factor`倍し、`finally`ブロックで必ず元の値へ復元する。**adapterファイルへの書き込みは一切ない**。`model.config.num_hidden_layers`から実測した層数は**48**（front/mid/back = 0-15/16-31/32-47）。

## 2. scale sweep 結果（v4主・v2副、詳細は`phase4n_scale_summary.json`）

| scale | greedy recall | sampled avg recall(5seed) | %出現seed | game count全3種seed | E36プレースホルダー | persona平均文字数 |
|---|---|---|---|---|---|---|
| 0.00 (=base) | 100% | 100% | 5/5 | 5/5 | なし | 146.7 |
| 0.10 | 100% | 100% | 5/5 | 5/5 | なし | 126.5 |
| 0.25 | 100% | 100% | 5/5 | 5/5 | なし | 89.8 |
| 0.50 | 100% | 80% | 3/5 | 5/5 | なし | 57.6 |
| 0.75 | 50% | 70% | 2/5 | 5/5 | なし | 41.4 |
| 1.00（本番v4） | 50% | 10% | 0/5 | 1/5 | **あり** | 30.4 |

v2は0.0/0.5/1.0の3点評価で 100%→80%→10% と、v4とほぼ同じカーブを描いた。**「どのscaleで具体値が消えるか」への回答**: 0.25までは完全維持、0.5から%情報が欠落し始め、1.0で壊滅的に欠落する。

想定外の発見として、**persona側は単調改善ではない**: E36（自己紹介）でscale=0.5「あいりちゃん」、scale=0.75「あいこ」という、リルの設定にない名前の幻覚が発生し、scale=1.0では逆に既知の「私は〜〜だよ」プレースホルダー欠落バグに戻る。E01（呼称に関する質問）ではscale=0.25でのみ「リルはパチスロ『モンキーターン』シリーズのキャラクター」という無関係な実機への誤紐付けが発生した。scaleを下げれば下げるほど安全、という単純な関係ではない。

## 3. module-level ablation（v4、詳細は`phase4n_module_results.json`）

**実装上の注意**: `full_v4`行は`module_types=None`が「全層に適用」と解釈されてしまい、実質scale=0.0(base相当)になっていたバグを発見・記録した（他の6条件は正しく機能）。真のfull_v4基準はscale sweepのscale=1.00の値を使用。

| 条件 | greedy recall | sampled(seed42) recall | E36 |
|---|---|---|---|
| full_v4 (scale=1.0, 正しい基準値) | 50% | 0% | プレースホルダー欠落 |
| q_off | 50% | 50% | 「リリ」と誤称 |
| k_off | 0% | 0% | プレースホルダー欠落（悪化） |
| v_off | 50% | 50% | 「リサ」と誤称 |
| **o_off** | **100%** | **100%** | 誤称なし、キュート口調維持 |
| qk_off | 50% | 50% | 「リリ」（q_offと同一） |
| **vo_off** | **100%** | **100%** | 誤称なし |

o_proj単体のLoRA寄与を無効化するだけで、q/k/vをフル強度のまま保ったままQ3 recallが完全復元する。これが本フェーズで最も明確・再現性の高い単一要因である。

## 4. layer-level ablation（v4、48層をfront/mid/back各16層に分割、詳細は`phase4n_layer_results.json`）

| 条件 | greedy recall | sampled(seed42) recall | E36誤称 |
|---|---|---|---|
| all_on（正しいfull v4基準） | 50% | 0% | プレースホルダー欠落 |
| front_off | 0% | 100% | なし |
| mid_off | 100% | 100% | 「リコ」 |
| back_off | 50% | 100% | 「あいり」 |
| front_only_on | 100% | 100% | なし |
| mid_only_on | 50% | 100% | 「ゆめぴょん」 |
| back_only_on | 100% | 100% | なし |

front/mid/backいずれの1/3を外しても（あるいはどれか1つだけ残しても）ほぼ全条件でsampled recallが100%まで回復する。特定層グループへの局所化は見られず、**全層の総和的な強度が揃って初めて省略挙動が安定再現される、分散的で脆い現象**である。これはscale実験の「強度を落とすと崩れる」結果と整合的。名前幻覚はmid/back関連条件でのみ出現し、front関連条件（front_off/front_only_on/back_only_on）では出現しなかった。

## 5. 有望候補の深掘り（scale=0.25 / o_off、詳細は`phase4n_promising_deepdive_results.json`）

| 指標 | full_v4 (scale=1.0) | scale=0.25 | o_off |
|---|---|---|---|
| P01 recall | 50% | 100% | 100% |
| P02 recall | 40% | 100% | 100% |
| Q9（派生計算幻覚） | なし | なし | なし |
| Q11（ループストック因果捏造） | **あり**（既知バグ） | なし | なし |
| Q11（ヤメ時アドバイス＝禁止カテゴリ） | なし | **あり（新規）** | **あり（新規）** |

Q3改善はP01/P02にも一般化しており、Q3固有の効果ではないことを確認した。Q9の派生計算幻覚はどの条件でも出なかった。しかし**Q11で重大な留保**が見つかった: scale=0.25・o_offのどちらも、production v4が持つ既知のループストック因果捏造は解消するが、代わりにPhase 4Kで明示的に禁止したカテゴリである「ヤメ時アドバイス」を新規に生成した。つまりどの条件も完全にクリーンではなく、「LoRA強度を下げる／o_projを切る」ことは、persona圧縮を抑える一方でBase自身が持つ「RAGにない助言をしてしまう」傾向を部分的に呼び戻してしまう。これはpersonaとrecallの2軸トレードオフではなく、**RAG厳格性を含む3軸のトレードオフ**であることを示す測定結果である。

## 6. Base能力の犠牲度の定量化

- Q3 sampled avg recall: base(scale=0.0) 100% → production v4(scale=1.0) 10%（約90ポイントの犠牲）
- held-out P01: 100%→50%、P02: 100%→40%、Q3と同程度の劣化幅
- Phase 4Mのlogits測定（参考）: base-adapter間の平均logit差(~2.1-2.2)は、adapter間差(~0.09-0.15)の約15倍。今回の結果は、この差が「adapter固有の学習内容の違い」よりも「LoRA適用強度・構造そのもの」に強く支配されていることと整合する。

## 7. v2とv4のscale感度の違い

ほぼ同一のカーブ（0.0/0.5/1.0で100%→80%→10%）。v4固有の脆弱性ではなく、v2/v3/v4に共通するLoRA適用設計（target_modules=q/k/v/o_proj、r=16/alpha=16）由来の現象であることを支持する。

## 8. Base-vs-persona LoRA干渉の確認

確認された。scale/module両方の実験で、LoRAの総寄与量（特にo_proj）がBaseの完全列挙能力を強く抑制していることが直接測定された。

## 9. 「persona LoRAが不必要にBaseのRAG能力を壊しているか」への回答

**壊している、と言える。** ただし「不必要に」かどうかは一概には言えない — o_projを完全に切ってもpersona自体は（名前を名乗らないという制約付きで）ある程度維持されるため、o_projの寄与の一部は「不必要な圧縮」である可能性が高い。一方、scale/o_off双方でQ11の新規幻覚が出た事実は、LoRAの寄与には「望ましくないBase癖の抑制」という正の役割もあったことを示唆する。したがって「一律に不要」という単純な結論ではなく、**o_proj（および全層合計の強度）が持つ複数の役割（persona圧縮／RAG厳格性の維持／欠落）が絡み合っている**、というのが測定に基づく結論である。

## 10. v5学習を進めるべきか、それともLoRA適用方式・人格アーキテクチャの再考が必要か

**両方の要素がある（B_primary_with_C_caveat）。**

- Hypothesis D（データ・学習目的自体が原因で、適用時の調整では直らない）は**明確に棄却**: scale低減・o_proj除外のいずれでもQ3/P01/P02は完全回復する。
- 最も明確で再現性が高く、次の一手として最も実行しやすいのは**module-level（Hypothesis C寄りの結果）**: o_projへのLoRA寄与を抑える（低rank化、あるいはtarget_modulesから除外、あるいはo_projのみ低いalpha/scaleを設定）設計変更をv5で検討する価値がある。
- ただし、Q11の深掘りで判明した「ヤメ時アドバイス」という新規幻覚は、**scale/module単体の調整だけではrecall・persona・RAG厳格性を同時に満たせない**ことを示しており、単純な推論時強度調整だけで本番投入可能な解が今回見つかったわけではない（Hypothesis Eの要素も残る）。

**次フェーズへの提案（本フェーズでは着手しない）**: v5で(a) o_projのrank/alphaを他moduleより下げる、または対象から除外する構成を試験学習し、(b) その上でQ3/P01/P02/Q9/Q11/persona-39全項目を同時に再評価する、という設計に絞ることを推奨する。「教師データを増やす」方向の結論には至っていない（今回の主要因はLoRA適用構造であり、データ量ではない）。

## 11. pytest / 変更なし確認

- pytest: **126 passed**（ベースラインと同数、warningは既存のfastapi/starlette非推奨警告のみ）
- git: HEAD = `2e0492d`（不変）、`git status --porcelain`は新規ファイル(`??`)のみ、コミットなし
- 4種adapter（v1/v2/v3/v4）の`adapter_model.safetensors` SHA-256: Phase 4M記録値と完全一致（無改変）
- `config/prompts/system.jinja2`: MD5 `f3ea72a9ea9a400fcfae0018896350b8`（不変）
- v5 adapter・merge・GGUFは一切作成していない
- Slack通知「リル Phase 4N LoRA影響度・適用強度監査完了」送信済み

## 作成ファイル一覧

- `training/riru/eval/phase4n_lora_scale_experiment.py`
- `training/riru/eval/phase4n_module_ablation.py`
- `training/riru/eval/phase4n_layer_ablation.py`
- `training/riru/eval/phase4n_analyze_scale.py`
- `training/riru/eval/phase4n_promising_deepdive.py`
- `training/riru/eval/phase4n_scale_results_v4.json` / `phase4n_scale_results_v2.json` / `phase4n_scale_meta.json` / `phase4n_scale_summary.json`
- `training/riru/eval/phase4n_module_results.json`
- `training/riru/eval/phase4n_layer_results.json`
- `training/riru/eval/phase4n_promising_deepdive_results.json`
- `training/riru/reports/phase4n_tradeoff_analysis.json`
- `training/riru/reports/phase4n_summary.md`（本ファイル）
