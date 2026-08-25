# Phase 4O: o_proj除外LoRA対照学習実験 (v5-qkv) 最終報告

## 0. 結論の要約

- **v5-qkv (target_modules=q/k/v_projのみ、o_proj除外)をv4と厳密に対照学習した。**
- Q3・P01・P02など「重要情報省略」の指標は**明確に改善**した（特にP01/P02はbase同等の100%まで回復、Q3も0-50%の不安定な回復から全seed一貫50%へ安定化）。
- しかし**Q11（ヤメ時アドバイス）とQ9（独自倍率計算）で新規hallucinationが再発**した。v4はこの2問を安定して回避していたが、v5-qkvは両方で退行した。
- **E36で「リル」以外のキャラクター名を名乗る誤称が複数seedで再発**した（あいり/あいこ/ココ）。Phase 4Nの推論時o_proj OFF実験で見られたのと同じ名前群が、学習時除外でも再発した。
- **採用判定: 不採用（正式候補としない）**。ユーザー指定の採用条件（Q11/Q9でhallucinationを起こさないこと、誤名乗りなしを含む）を満たさない。「o_projを後から切る」ことと「最初から学習しない」ことは学習力学として異なる挙動を示したが、いずれも同種の副作用（persona名前識別とRAG厳格性の喪失）を伴うという点で収束した。

## 1. v4/v5-qkv config差分

`target_modules`（v4: q/k/v/o_proj → v5-qkv: q/k/v_proj）のみが意図的差分。その他はドキュメント文言（`_comment`、`base_model.note`、`loss.description`、`data.format`、`lora.mlp_modules_note`）と出力先パスのみで、ハイパーパラメータ（r/alpha/dropout/epoch/lr/optim/batch/grad_accum/seed等）は完全一致。`training/riru/phase4o_pretrain_checks.py`による自動検証で確認済み（`overall_status: READY`、詳細は`phase4o_pretrain_checks.json`）。

## 2. train/val SHA一致確認

`riru_train_v4.jsonl` SHA-256 `d331fef4...`、`riru_val_v4.jsonl` SHA-256 `3df5c2a8...`、`riru_lora_v4_candidate.jsonl` SHA-256 `341c44d0...` — いずれもPhase 4K/4L記録値と完全一致。新しいcandidate/splitは作成していない。件数 train=823 / val=91 確認済み。

## 3. target_modules / o_proj不在の確認

- v5-qkv `adapter_config.json`: `target_modules: ["q_proj", "v_proj", "k_proj"]`
- v5-qkv `adapter_model.safetensors`: 全288 tensor（q_proj 96 / k_proj 96 / v_proj 96 / **o_proj 0**）。o_proj関連tensorは1件も存在しない。

## 4. 学習結果

| 項目 | v4 | v5-qkv |
|---|---|---|
| 総step | 156 | 156 |
| 学習時間 | 627.1秒(10.45分) | **470.8秒(7.85分)** |
| train_loss(Trainer集計) | 1.785 | 1.968 |
| eval_loss推移 | (Phase4L記録) | 2.679→1.967→1.812→1.722→1.656→1.620→1.598→1.595 |
| 最終eval_loss | 1.4660 | **1.5946**（v4よりやや高い） |
| peak VRAM | 21917 MiB | 21403 MiB |
| NaN/Inf/OOM/CUDA error | なし | **なし** |
| adapterサイズ | 100.7 MB | **69.2 MB**（約69%、moduleが3/4のため） |
| checkpoint | 120/140/156 | 120/140/156 |

v5-qkvのeval_lossがv4よりやや高いのは、学習対象パラメータが少ない（表現力が下がる）ことと整合的で、異常ではない。

## 5. Q3 5seed評価（sampled, temperature=0.3）

| 条件 | seed42 | seed43 | seed44 | seed45 | seed46 | 平均recall |
|---|---|---|---|---|---|---|
| A_base | 100% | 100% | 100% | 100% | 100% | 100.0% |
| B_v4 | 0% | 50% | 0% | 0% | 0% | 10.0%（不安定） |
| **C_v5qkv** | **50%** | **50%** | **50%** | **50%** | **50%** | **50.0%（全seed一貫）** |

v5-qkvはv4より明確に改善し、かつ**全5seedで完全に同じ50%recall**という高い安定性を示した（v4は0-50%の間で不安定）。ただし510G/1000G/1480Gのゲーム数は毎回回復する一方、15.2%/20.3%/64.5%の確率情報はどのseedでも回復しなかった（%系のみ一貫して欠落）。

## 6. Q3 greedy評価

- A_base: recall 100%（全facts）
- B_v4: recall 50%（ゲーム数のみ）
- C_v5qkv: recall 50%（ゲーム数のみ、v4と同一パターン）

「%まで回復するか」という問いへの回答: **回復しなかった**。greedy条件ではv4と同じ水準に留まる。

## 7. held-out省略評価（Phase4I P01〜P10）

| ID | A_base | B_v4 | C_v5qkv |
|---|---|---|---|
| P01 | 100% | 50% | **100%** |
| P02 | 100% | 40% | **100%** |
| P03 | 100% | 100% | 100% |
| P04 | 33.3% | **100%** | 33.3%（v4より後退、base水準） |
| P05 | 33.3% | 33.3% | 33.3% |
| P06 | 100% | 100% | 100% |
| P07 | 33.3% | 33.3% | 33.3% |
| P08 | 66.7% | 66.7% | 66.7% |
| P09 | 100% | 100% | 100% |
| P10 | 100% | 100% | 100% |
| **平均** | **76.7%** | **72.3%** | **76.7%（base水準まで回復）** |

P01/P02は明確に改善（base同等）。全体平均もv4(72.3%)からbase水準(76.7%)まで回復した。ただしP04は逆にv4の改善(100%)を失いbase水準(33.3%)に後退しており、**均一な改善ではない**（o_prójがP04のようなケースでは正の役割を果たしていたことを示唆）。

## 8. Q11 5seed hallucination検査（最重要）

- **A_base**: 全5seedで長大なヤメ時アドバイス（「ヤメ時の考え方」節を含む解説）を生成。
- **B_v4**: 全5seedでヤメ時アドバイスの明示的節は**出現せず**。代わりに「ループストックが大きいほど〜」という毎seed内容が微妙に異なる（大きいほど当たりやすい/大きいほど遠くなる、と矛盾する主張すらある）因果捏造が一貫して出現（既知の問題）。
- **C_v5qkv**: **全5seedでヤメ時アドバイスが再発**（「ヤメ時は、設定が変わったタイミングで止めるのが一般的です」「初心者の方は、天井に達したら一旦ヤメてみることをおすすめします」等）。Phase 4Kで明示的に禁止したカテゴリのhallucinationが、v4では抑制されていたにもかかわらずv5-qkvで系統的に再発した。

**Q11 hallucination率**: v4=0/5（ヤメ時型）だがループストック因果捏造が5/5、v5-qkv=5/5（ヤメ時型）。どちらも0件ではないが、**種類が異なる**: v4はhallucinationの種類が1つに収束している一方、v5-qkvはbaseとほぼ同じ「網羅的だが未確認の助言」パターンに回帰している。

## 9. Q9 5seed検査

- A_base: 全5seedで「約1.8倍」「約17.4%」「約0.57%」等の派生計算hallucinationが出現。
- B_v4: 全5seedで派生計算なし（生値の比較表現のみ）。
- **C_v5qkv**: **seed44/45/46の3/5で派生計算hallucinationが再発**（「約2.4倍」「約1.8倍」「約2倍」「27.4ポイント」— 一部は数値自体が誤り: 正しい機械割差は17.4ポイントだが、v5-qkvのseed46は「27.4ポイント」と誤って計算している）。seed42/43は清潔。

v4が完全に抑制していたQ9の独自計算hallucinationが、v5-qkvでは過半数のseedで再発し、しかも一部は数値自体が誤っている（新種の計算エラー）。

## 10. E36 5seed検査

| 条件 | 誤名乗り | プレースホルダー欠落 | 傾向 |
|---|---|---|---|
| A_base | ピコ/ピッコロ/ぴよこ/パティ（毎seed別名、AIディスクレーマーあり） | なし | persona未学習のため当然 |
| B_v4 | なし（「あい」という部分的名乗りが1件） | 1/5（seed42のみ） | 概ね安定 |
| **C_v5qkv** | **3/5で誤名乗り**（あいり/あいこ/ココ） | 0/5 | **名前識別が不安定** |

v5-qkvはプレースホルダー欠落バグ（v4の既知問題）は出さない代わりに、Phase 4Nの推論時o_proj OFF実験で見られたのと**同じ系統の誤名**（あいり/あいこ、Phase4Nではリリ/リサ/リコ/ゆめぴょん）が再発した。これは「o_prójを後から切る」でも「最初から学習しない」でも収束的に同じ副作用が出ることを示す強い証拠である。

## 11. character39（自動集計、詳細E01/E20-E22/E36は目視確認）

自動集計（`phase4o_aggregate_stats.json`）: 平均文字数 v4=30.4字 / v5-qkv=32.1字（ほぼ同等）。「だよ」使用率 v4=33.3% vs v5-qkv=12.8%（v5-qkvの方が語尾のリルらしさがやや弱い）。プレースホルダーはv5-qkv 0件（v4は1件）。

目視確認（E01/E20/E21/E22/E36）: **E01/E20/E21/E22はv4と同水準の自然な口調を維持**しており、単純にBaseへ戻っただけではない（カジュアルな語尾・パチスロへの言及・キャラクター性は概ね保持）。問題は**E36のような「自己紹介・名乗り」を要求する場面に限定的に集中**しており、それ以外の一般会話ではpersonaは大きく損なわれていない。

## 12. structured17

自動集計: 平均文字数 v4=64.4字 / v5-qkv=107.5字（v5-qkvの方がbase寄りに長い＝Q3/P01/P02の改善と整合）。「だよ」等の語尾がv5-qkvでは0%（v4は41.2%）——**structured17セット全体でも語尾のリルらしさが明確に後退**している。数値精度・登録外拒否・反復については明示的な悪化パターンは自動集計上見られなかったが、詳細な項目別（Q1/Q2/Q4/Q9数値精度、Q7/Q12-14登録外拒否等）の全文目視は今回のQ9/Q11/P01/P02/E36ほど深くは行っておらず、今後の詳細確認の余地がある。

## 13. logits比較（Q3生成開始位置）

| ペア | max_abs_diff | mean_abs_diff |
|---|---|---|
| base_vs_v4 | 18.81 | 2.182 |
| base_vs_v5qkv | 17.44 | **1.804**（v4よりbaseに近い） |
| v4_vs_v5qkv | 4.79 | 0.506 |

v5-qkvはbaseとの平均logit差がv4より小さく（1.80 vs 2.18）、**構造的にbase寄りの分布を持つ**ことが定量的にも裏付けられた。v4-v5qkv間の差（mean 0.506）は、Phase4Mで見たv2/v3/v4間の差（0.09〜0.15）よりずっと大きく、これはtarget_modules自体が異なる「アーキテクチャレベルの違い」であるため妥当な結果である。top1トークンはbase「「」→v4/v5qkv共に「天」に変化しているが、v5qkvの確信度(0.885)はv4(0.932)よりやや低い。

## 14. v4との純粋な構造比較としての意味

同一データ・同一split・同一seed・同一epoch・同一lr・同一rank・同一alphaで、o_proj有無**だけ**を変えた対照実験として意味のある結果が得られた。「o_projを後から推論時に切る」(Phase4N)と「最初から学習しない」(Phase4O)は、**Q3型省略の改善度合いが異なる**（Phase4N推論時OFF: 100%完全回復 vs Phase4O学習時除外: 50%の部分的だが安定した回復）ことが分かった。一方で、**両者はQ11/E36の副作用の系統がほぼ一致する**（ヤメ時アドバイス、誤ったキャラクター名）。これは、o_prójの寄与が(a)全体的な出力量・具体性の抑制と(b)persona名前識別・RAG厳格性の維持、という複数の役割を担っていることを示しており、この2つの役割は学習時除外・推論時無効化のどちらの方法でも同時に失われる、という構造的な結論を支持する。

## 15. pytest / 既存資産の無変更確認

- pytest: **126 passed**（不変）
- git: HEAD=`2e0492d`不変、`git status --porcelain`で`??`以外の行なし（コミットなし）
- v1〜v4 adapter SHA-256: 4本ともPhase4M記録値と完全一致（無改変）
- `riru_train_v4.jsonl`/`riru_val_v4.jsonl`/`riru_lora_v4_candidate.jsonl` SHA-256: Phase4K/4L記録値と完全一致（無改変・再分割なし）
- `config/prompts/system.jinja2` MD5: 不変
- structured.db / RAG・Vector DB: 未変更（本フェーズで一切アクセスしていない）
- merge/GGUF: 未実施
- Slack通知「リル Phase 4O qkv-only LoRA対照実験完了」送信済み

## 16. 最終判定

**33. v5-qkvはv4より優れているか**: 部分的にyes、全体としてno。Q3/P01/P02/held-out平均では明確に優れる。しかしQ9/Q11のhallucination抑制とE36の名前識別ではv4の方が優れており、v4が持っていた「RAG厳格性・persona名前の安定性」という重要な特性を失っている。単純な優劣ではなく**別方向のトレードオフへ移動しただけ**というのが正確な表現である。

**34. q/k/v-onlyを今後のLoRA設計として採用すべきか**: **現時点では採用すべきではない**。ユーザーが事前に定義した採用条件（Q11で新hallucinationを起こさない、Q9独自計算なし、誤名乗りなし）のいずれにも抵触しており、「Q3だけ100%でもQ11/E36が壊れるなら正式候補にしない」という明示的な基準に照らして不採用となる。ただし、Q3/P01/P02の改善とP01/P02の完全回復は本物であり、「o_prójの寄与を弱める」方向性自体は無駄ではなかった。今後の設計候補としては、(a) o_projを完全除外ではなく低rank/低alphaで部分的に残す、(b) o_projのみ別のrank/alphaを個別設定する、といった「中間的な扱い」を試す価値がある。

**35. merge/GGUFへ進める状態か**: **いいえ**。今回の対照実験はあくまで診断目的であり、Q11/Q9/E36で新規の禁止カテゴリhallucinationが確認された以上、本番適用可能な状態ではない。

## 作成ファイル一覧

- `training/riru/configs/qlora_config_v5_qkv.json`
- `training/riru/phase4o_pretrain_checks.py` / `reports/phase4o_pretrain_checks.json`
- `training/riru/lora-riru-qwen-v5-qkv/`（adapter・checkpoint一式）
- `training/riru/audit/phase4o_tensor_check.py` / `reports/phase4o_tensor_check.json`
- `training/riru/eval/phase4o_comprehensive_eval.py` / `phase4o_comprehensive_results.json`
- `training/riru/eval/phase4o_logits_compare.py` / `reports/phase4o_logits_compare.json`
- `training/riru/reports/phase4o_aggregate_stats.json`
- `training/riru/reports/phase4o_summary.md`（本ファイル）

## 停止

評価終了。指示どおりmerge/GGUF化・本番適用・system prompt変更・Git commit/pushは一切行っていません。v5-qkvは対照実験用の一時的なadapterであり、正式なv5採用を意味するものではありません。次フェーズの判断は、この結果をご確認いただいた上でお願いします。
