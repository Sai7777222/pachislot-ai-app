# Phase 4Q: o_proj低rank/低alpha対照学習実験 (o8/o4) 最終報告

## 0. 結論の要約

- **o8・o4のどちらも採用基準（18節の全項目）を満たさなかった。**
- **想定外の重大発見**: o_projのrankを16→8→4と縮小すると、Q3情報保持が改善するどころか、**Q11のヤメ時アドバイスhallucinationがv4:0/5→o8:2/5→o4:3/5と段階的に悪化**した。Q9独自計算hallucinationもv4:0/5→o8:1/5→o4:1/5で悪化。これはPhase4Pで発見した「o_prójのRAG厳格性への寄与はscale=1.0近傍でのみ成立する崖」が、rank（学習容量）を減らす方向でも同型の脆さとして再現されたことを示す。
- **Q3情報保持自体も改善しなかった**: o8のQ3 sampled平均recallはv4と同じ10.0%（改善ゼロ）。o4は40.0%まで上昇したが、%情報（15.2%/20.3%/64.5%）は3条件（v4/o8/o4）とも全seedで0/5——ゲーム数のみの部分改善に留まる。
- P02/P04はo8/o4いずれもv4より悪化（P04: v4=73.3%→o8=33.3%→o4=46.6%）。
- **総合結論**: 「o_projを消す」でも「o_projの学習容量を減らす」でも、情報保持とRAG厳格性を同時に改善する方向は見つからなかった。むしろ容量を減らすほど、RAG厳格性側の副作用がなだらかに悪化する一貫した傾向が確認された。alpha分離実験（rank固定・alphaのみ変更）へ進む根拠は、この結果からは支持されない。

## 1〜6. 実装・検証

- **意図的な既存ファイル変更（開示）**: `training/riru/train_qlora.py`の`build_model_and_tokenizer()`に、config側`lora.rank_pattern`/`lora.alpha_pattern`が存在する場合のみ`LoraConfig`へ渡す後方互換コードを追加した（キーがない場合は空dict=完全に従来通り）。v1〜v4/v5-qkvのconfigにはこのキーが存在しないため、既存adapterの学習ロジックへの影響はゼロ。pytest 126 passed、既存4 adapter+v5-qkvのSHA-256が全て記録値と一致することで無影響を確認済み。
- **PEFT rank_pattern/alpha_pattern実装確認**: `peft==0.20.0`の`get_pattern_key()`が`re.match(rf"(.*\.)?({key})$", current_key)`でモジュール名の末尾一致を行うことをソース確認。`rank_pattern={"o_proj": 8}`が48層全てのo_projにのみマッチし、q/k/v_projには一切影響しないことを、実モデル(Qwen2.5-14B)へ実際に適用した`phase4q_peft_pattern_verify.py`で実測確認（`verification_passed: true`、q/k/v: r=16/alpha=16/scaling=1.0、o_proj: r=8/alpha=8(o8)・r=4/alpha=4(o4)、scalingはいずれも1.0で一致）。
- **Preflight**: train=823/val=91件、train/val/candidate SHA-256全てPhase4K/4L記録値と一致、system.jinja2 MD5一致、v1〜v5-qkv adapter SHA-256全て一致、config差分がo_projのrank_pattern/alpha_patternとドキュメント文言のみであることを自動検証（`overall_status: READY`、o8/o4双方）。

## 7〜8. 学習結果

| 項目 | v4(参考) | o8 | o4 |
|---|---|---|---|
| 総step | 156 | 156 | 156 |
| 学習時間 | 627.1秒 | 519.6秒(8.67分) | 521.7秒(8.7分) |
| train_loss | 1.785 | 1.849 | 1.902 |
| 最終eval_loss | 1.4660 | 1.5126 | 1.5474 |
| peak VRAM | 21917 MiB | 21662 MiB | 21622 MiB |
| NaN/Inf/OOM/CUDA error | なし | なし | なし |

eval_lossはv4(r16)=1.466 < o8(r8)=1.513 < o4(r4)=1.547 < v5-qkv(r0, Phase4O)=1.595と、o_proj容量の縮小に対して単調に増加しており、容量削減が意図通りモデルの表現力を減らしていることを裏付ける。

## 9. tensor監査

両candidateとも`adapter_model.safetensors`を直接開いて確認: o8はq/k/v_proj全て lora_A shape (16, 5120)、o_prójのみ(8, 5120)。o4はq/k/v_proj全て(16, 5120)、o_prójのみ(4, 5120)。configだけでなくtensor実体で一致を確認済み（`verification_passed: true`）。

## 10〜13. Q3結果

| 条件 | greedy recall | sampled avg | min/max | 全3ゲーム数seed | 全3%seed |
|---|---|---|---|---|---|
| A_base | 100% | 100.0% | 100/100 | 5/5 | 5/5 |
| B_v4 | 50% | 10.0% | 0/50 | 1/5 | 0/5 |
| C_o8 | **0%** | 10.0%（v4と同一） | 0/50 | 1/5 | 0/5 |
| D_o4 | **0%** | **40.0%（改善）** | 0/50 | **4/5** | 0/5 |

o8はQ3改善なし（sampled平均は v4と完全に同値）、greedyはむしろv4より悪化（0%）。o4はsampled平均・ゲーム数保持ではv4より改善したが、greedyは同じく0%に悪化。**%情報（15.2%/20.3%/64.5%）はv4/o8/o4のいずれも全5seedで一度も完全には揃わなかった**——rank縮小だけでは%情報の保持を全く解決できていない。

## 14〜15. P01/P02/P04

| 条件 | P01 | P02 | P04 |
|---|---|---|---|
| A_base | 100.0% | 100.0% | 60.0% |
| B_v4 | 50.0% | 76.0% | **73.3%** |
| C_o8 | 50.0% | **0.0%（悪化）** | **33.3%（大幅悪化）** |
| D_o4 | 50.0% | 60.0%（悪化） | 46.6%（悪化） |

P01はv4/o8/o4で変化なし（50%のまま）。**P02/P04はo8・o4のどちらもv4より悪化**しており、特にo8のP02は0%まで落ち込んだ。「o_prójが正方向に働く代表例」として重視していたP04も、rank縮小によって回復するどころか一貫して悪化した——o_prójの容量縮小は、Q9/Q11の副作用を軽減しないまま、v4が持っていた長所（P04等）だけを損なう結果となった。

## 16〜19. Q9/Q11/E36 hallucination（最重要）

| 指標 | B_v4 | C_o8 | D_o4 |
|---|---|---|---|
| Q9独自計算 | 0/5 | 1/5 | 1/5 |
| Q11ヤメ時アドバイス | **0/5** | **2/5** | **3/5** |
| Q11ループストック因果捏造 | 2/5 | 3/5 | 0/5 |
| Q11その他因果 | 0/5 | 0/5 | 0/5 |
| E36誤名乗り | 0/5 | 0/5 | 1/5 |
| E36 placeholder/未完成 | 1/5 | 2/5 | 0/5 |

手動全文確認で、o8 seed42「天井に近づくほどヤメ時としては考えた方がいいかな」、o4 seed43/44/46の複数箇所で明示的な「ヤメ時は〜」節を確認し、自動検出が誤検知でないことを確認した。**v4が0/5に抑えていたヤメ時アドバイスが、rank縮小に応じてなだらかに悪化する**（0→2→3、Phase4Oのv5-qkvでは5/5）——これはPhase4Pのscale sweepで見た「o_prójの寄与が閾値的」という発見が、rank軸でも(より緩やかながら)再現されたことを意味する。一方でo4はループストック因果捏造を0/5まで解消しており、**「v4と違うhallucinationへ置き換わっただけ」という懸念が部分的に成立している**（ループストック捏造は減るがヤメ時アドバイスが増える）。

また目視確認で、既知の誤名リストにない新規の名乗り「**あいだっち**」（o8 seed46・o4 seed46で共通して出現）を発見した。自動集計の誤名乗り0/5という数字は、この新規パターンを検出できていない点に留意が必要（既知リストでの機械集計は過小評価しうる）。

## 20. E01/E20/E21/E22

v4・o8・o4のいずれも一般会話パーソナは大きく破綻していない（目視確認）。character39自動集計でも、「だよ」使用率はv4=33.3%/o8=28.2%/o4=25.6%と緩やかに低下する程度で、致命的な崩壊はない。

## 21〜23. structured17 / character39 / 回答長

平均回答長: persona系はv4=55.4字→o8=44.9字→o4=47.4字（base参考=211.2字）。structured17系はv4=64.4字→o8=63.6字→o4=78.4字。character39平均文字数はv4=30.4字/o8=29.8字/o4=31.7字とほぼ横ばい。回答長からは「Base寄りの大幅な長文化」は見られず、Q3/P01等の改善（あった場合）が単純な長文化由来ではないことを示唆するが、そもそもQ3改善自体がo4でも部分的（ゲーム数のみ）に留まっている。

## 24. logits比較

| ペア | mean_abs_diff | max_abs_diff |
|---|---|---|
| base vs v4 | 2.182 | 18.81 |
| base vs o8 | 2.063 | 18.50 |
| base vs o4 | 1.999 | 18.13 |
| v4 vs o8 | **0.170** | 1.21 |
| v4 vs o4 | **0.274** | 2.25 |
| o8 vs o4 | 0.135 | 1.31 |

v4-o8間・v4-o4間の差はPhase4Mで見たv2/v3/v4間の差(0.09〜0.15)と同程度の小ささであり、**first-token logitsだけでは、この後に観測されたQ9/Q11の顕著な挙動差（hallucination率の変化）を予見できない**——300トークンにわたる自己回帰生成での複利的な分岐が支配的である、というPhase4Pからの知見と整合する。

## 25. o16→o8→o4→o0の傾向

| 指標 | o16(v4) | o8 | o4 | o0(v5-qkv, 参考) |
|---|---|---|---|---|
| eval_loss | 1.466 | 1.513 | 1.547 | 1.595 |
| Q3 sampled avg | 10.0% | 10.0% | 40.0% | 50.0% |
| Q11ヤメ時 | 0/5 | 2/5 | 3/5 | 5/5 |
| P04 | 73.3% | 33.3% | 46.6% | 33.3% |

eval_loss・Q11ヤメ時は容量低下に対してほぼ単調に悪化する。Q3 sampled avgは非単調（o8で足踏みし、o4で急伸）——**閾値的な変化**であり、緩やかな線形応答ではない。P04はo16で最良、o8で急落、o4でやや回復、o0で再度低下という非単調な挙動を示し、単純な「容量が多いほど良い/少ないほど良い」という説明では捉えられない。

## 26. sweet spotの有無

**なし。** o8/o4いずれも18節の採用基準（全10項目）を満たさなかった。

## 27. Pareto frontier

- Q3改善重視: o4（sampled avg 40%、全3ゲーム数4/5）——ただしQ11ヤメ時3/5・P02/P04悪化
- hallucination最小重視: v4自身（Q9=0/5、Q11ヤメ時=0/5）——ただしQ3型省略は未解決のまま
- P04（v4の長所）維持重視: v4自身（73.3%、o8/o4は共に悪化）

o8・o4のどちらも「他条件より優れる」単一の軸を持たず、**v4からの移行を正当化する明確な優位性はどの軸にも見出せなかった**。

## 28〜30. 比較まとめ

- v4 scale=1.0（Phase4Pの基準）との比較: 本フェーズのB_v4はPhase4P/4Oのv4と同一adapterであり、数値もPhase4P (sampled avg 10.0%, greedy 50%) とほぼ整合（greedyのみ今回50%→今回のB_v4欄は上表で50%と記載、一致）。
- o0(v5-qkv)との比較: Q11ヤメ時はo16:0→o8:2→o4:3→o0:5と単調悪化の系列上にo8/o4がきれいに乗る。これは「o_prójの学習容量」という単一の軸が、少なくともQ11ヤメ時アドバイスの抑制に関して、scale(強度)軸とは別に、独立して有効な連続変数として機能していることを示す——ただし「有効」なのは"v4に近いほど良い"という自明な方向のみで、中間点に利点はなかった。
- Phase4N全module scale sweepとの違い: Phase4Nはq/k/v/o全体を均等に動かしたため崖の位置が0.25〜1.0にかけて分散していた。Phase4Q（o_prójのrankのみ）は、v4に対するどの程度の縮小でも、Q3改善が乏しいか不完全なまま、Q11/Q9の副作用だけ確実に発生するという、より悪い意味で明確な結果となった。

## 31. 「o_prójは何を担っていると考えられるか」の更新結論

Phase 4N〜4Pまでの仮説（A圧縮/B hallucination抑制/C名前識別/D一部情報選択）に対し、Phase4Qは以下を追加で示した:

- o_prójの**RAG厳格性維持機能（B）は、scaleだけでなくrank（表現力）に対しても閾値的**であり、r=16→r=8のような「半分」の削減でも即座に効果が損なわれ始める。「容量を少し残せば厳格性も少し残る」という単純な比例関係ではない。
- o_prójの**情報選択機能（D、P04で代表）も同様に脆弱**で、半分の容量では逆に大きく損なわれ、さらに容量を減らすとやや持ち直すという非単調な挙動を示した。これは、o_prójが単一の機能ではなく、複数の異なる情報方向を同じ低次元空間内で競合的に符号化している可能性を示唆する。

## 32. 推論時scale調整だけで解決可能か（再確認）

Phase4Pの結論を追認。加えて今回、学習時のrank調整でも同様に解決できないことが確認された。

## 33. 次にalpha分離実験へ進む価値があるか

**現時点では推奨しない。** rank8/alpha8・rank4/alpha4はいずれもscaling=1.0(alpha/rank比が常に1)で揃えているため、alpha単体の効果は今回分離できていないが、そもそも「容量を減らす」という基本方向自体がQ11/Q9の副作用を悪化させる一貫した傾向を示しており、alphaだけを変えても同じ機構（低次元表現力の不足）が働く可能性が高い。まず容量削減方向自体が筋の悪いアプローチである可能性を、これ以上の細かい条件を追加する前に人間側で検討することを推奨する。

## 34. pytest

**126 passed**（不変）。

## 35〜37. protected asset確認・Git状態

- v1/v2/v3/v4/v5-qkv adapter SHA-256: 全て記録値と一致（無改変）
- train_v4/val_v4/candidate_v4 SHA-256、system.jinja2 MD5: 不変
- git HEAD `2e0492d`不変、commit/push等の操作なし
- **開示**: `training/riru/train_qlora.py`に1件の意図的・後方互換な変更あり（rank_pattern/alpha_pattern対応、上記1〜6節参照）。protected assetsのリストには含まれていなかったが、既存学習ロジック自体（v1〜v4/v5-qkv）への影響がないことをpytest・全adapter SHA-256一致で実証済み。

## 38. Slack通知結果

「リル Phase 4Q o_proj低rank対照実験完了」送信成功（`notification_sent: True`）。

## 39. merge/GGUF可否

**いいえ。** o8・o4のどちらも18節の採用基準を満たしておらず、正式候補として進める状態ではない。

## 作成ファイル一覧

- `training/riru/phase4q_peft_pattern_verify.py` / `reports/phase4q_peft_pattern_verify.json`
- `training/riru/configs/qlora_config_o8.json` / `qlora_config_o4.json`
- `training/riru/phase4q_pretrain_checks.py` / `reports/phase4q_pretrain_checks_o8.json` / `_o4.json`
- `training/riru/lora-riru-qwen-o8/`、`training/riru/lora-riru-qwen-o4/`（adapter・checkpoint一式）
- `training/riru/phase4q_tensor_check.py` / `reports/phase4q_tensor_check_o8.json` / `_o4.json`
- `training/riru/eval/phase4q_comprehensive_eval.py` / `phase4q_comprehensive_results.json`
- `training/riru/eval/phase4q_logits_compare.py` / `reports/phase4q_logits_compare.json`
- `training/riru/eval/phase4q_analyze_results.py` / `reports/phase4q_aggregate_analysis.json`
- `training/riru/reports/_phase4q_q11e36_spotcheck_utf8.txt`
- `training/riru/reports/phase4q_summary.md`（本ファイル）
- （既存ファイルへの意図的変更）`training/riru/train_qlora.py`（rank_pattern/alpha_pattern対応、後方互換）

## 停止

評価・解析・レポート作成・pytest・Slack通知が完了しました。正式採用・merge・GGUF化・本番組み込み・system prompt変更・追加学習（o6/o10/alpha単体変更/rank32/layer別rank等）・追加scale/rank探索・Git commit/pushは一切行っていません。次の判断をお待ちします。
