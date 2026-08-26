# Phase 4ZC: Forward Numerical Drift Localization — 完了報告

## 0. 目的

Phase4ZBで「全579テンソル・147億要素がHF↔GGUF間で完全にbit一致」することが証明され、weight差説明が完全に否定された。本フェーズは、その先の問い——**transformer forward計算のどのlayer・どの演算(RMSNorm/RoPE/QK-matmul/softmax/attention-output/o_proj/MLP gate-up/SiLU/down_proj/residual-add)で、PyTorch(HF)とllama.cppの数値計算が最初に乖離し、それがどう増幅して最終的な「リ」(36723) vs「ル」(32610)のlogit順位反転に至るか**——を局在化することを目的とする。

**重要な前提(繰り返し厳守)**: 全579テンソルが完全一致した以上、「GGUFが壊れた」「量子化で情報が落ちた」「weightが違う」という説明へは絶対に戻らない。本報告書もこの原則を一貫して守っている。

---

## 1-3. 事前安全確認

- `git branch --show-current`: main
- `git rev-parse HEAD`(開始時・終了時とも): `a61d664f1d6af087b69056eb718fafeab7892401`(不変)
- `git status --short`: 開始時65行→終了時77行(本フェーズで新規作成した未追跡ファイルの増加のみ。全てgitignore対象の`training/riru/reports/_*`または新規eval/reportsファイルで、コミットは一切行っていない)
- `git diff` / `git diff --cached`: 空(追跡ファイルへの変更なし)
- `pytest`: 126 passed(開始時・終了時とも)
- Freeze Manifest記載のprotected asset hash(candidate/train/val/config/adapter/adapter_config/system.jinja2/merged_hf/bf16_gguf): 開始時・終了時とも全て一致、不一致0件
- llama.cpp canonical source (`D:/AI/tools/llama.cpp`) HEAD: `5d5cb4c3a4ea8769490d39a275ee49a45184774d`(不変)、`git status --short`は`.venv-gguf/`のみ(Phase4Y-Rから既存、本フェーズでの新規変更なし)

---

## 4-8. instrumentation調査とツール方針(Section5-8への回答)

**Q1. 新規ツール作成前にsourceを検索したか。** はい。`cb_eval`, `ggml_backend_sched_eval_callback`等のキーワードでcanonical source全体をgrep検索した。

**Q2. 既存のtensor inspection機構は見つかったか。** はい。`ggml_backend_sched_eval_callback`という公式コールバック機構と、それをラップした`common/debug.cpp`の`common_debug_cb_eval`、さらにそれを使う既存example `examples/debug`(ビルド後の実行ファイル名`llama-debug`)が既に存在した。`--tensor-filter`(正規表現)、`--save-logits`等のオプションを持つ。

**Q3. 既存機構だけで十分だったか。** 一部不十分だった。既存の`common_debug_print_tensor`は先頭/末尾3要素+sumのみをテキストログ出力する仕様であり、5120次元ベクトル全体をPython側で数値比較するには不足していた。

**Q4. そのため何をしたか。** `common/debug.cpp`に、環境変数`LLAMACPP_TENSOR_DUMP_DIR`が設定されている場合にmatchしたtensorの生バイトをバイナリファイルへダンプする処理を追加した(既存のログ出力機能はそのまま維持)。加えて`examples/debug/debug.cpp`のトークナイズ呼び出しを`parse_special=true`に変更し、既にrender済みのchat-template文字列中の特殊トークンを正しくパースできるようにした(HF側と同じトークンID列を得るために必須)。

**Q5. この変更はどこに行ったか。canonicalに影響したか。** canonical(`D:/AI/tools/llama.cpp`)には一切変更していない。全ての変更は`D:/AI/tools/llama.cpp-phase4zc-debug`(canonicalと同一commit `5d5cb4c3a4ea8769490d39a275ee49a45184774d`からのgit clone)上でのみ実施した。canonical側は`git status`/`git rev-parse HEAD`で不変を確認済み。

**Q6. 変更行数は。** 2ファイル、+59/-4行(`git diff`で確認、`_phase4zc_debug_copy_diff.txt`に保存)。

---

## 9-10. CMake/MSVCビルドツールの再確認(重要な訂正)

**Q7. Phase4Y-Rの「ビルドツールなし」は正しかったか。** **誤りだった。** 本フェーズで`vswhere.exe`を用いて再確認したところ、Visual Studio 2022 Communityが実際にインストール済みで、C++デスクトップ開発ワークロード(`Microsoft.VisualStudio.Component.VC.Tools.x86.x64`)、MSVC(`cl.exe` 14.43.34808)、CMake(VS同梱)、Ninja(VS同梱)、さらにCUDA Toolkit v12.8(`nvcc.exe`)まで全て揃っていることが判明した。Phase4Y-Rの判定は、単に`cmake`/`cl`がPATH環境変数に直接通っていなかったことによる誤判定だったと考えられる。この訂正を`phase4zc_llamacpp_instrumentation_feasibility.json`に明記した。

**Q8. ビルドは実施したか。CPU/CUDAどちらか。** 実施した。CPU-only(`GGML_CUDA=OFF`)でビルドした。CUDAツールチェーンは利用可能だったが、CPU-onlyビルドで本フェーズの目的(layer-wise比較による乖離の局在化)を十分達成できると判断し、コンパイル時間の大幅増加を招くCUDAビルドは見送った(Phase4ZAで「narrow backend-sensitive margin」であることは既に確認済みであり、CPU/CUDA間の大きな挙動差は想定していない)。

**Q9. ビルドは成功したか。** 成功した。`vcvars64.bat`でMSVC開発者環境を初期化後、`cmake -G Ninja`で構成し、`llama-debug`ターゲットのみをビルドした(全ターゲットのフルビルドは不要と判断し、対象を絞った)。

---

## 11. HF/PyTorch側 layer-wise hidden state抽出

**Q10. どの精度・attention実装で実施したか。** `torch_dtype=torch.bfloat16`、`attn_implementation="eager"`、bitsandbytes量子化なしの素のHFロード(`training/riru/eval/phase4zc_hf_hidden_dump.py`)。

**Q11. 重要な発見はあったか(従来のHF基準値について)。** **重要な発見があった。** 従来Phase4Z/4ZAで使われていた「HF」基準値は、実は`BitsAndBytesConfig(load_in_4bit=True)`による4bit NF4量子化ロードで測定されたものであり、真のフル精度BF16フォワードではなかったことが、`phase4z_logits_compare.py`/`phase4z_identity_eval_hf.py`のコード確認で判明した。本フェーズのHF側は全てbitsandbytes量子化なしの素のBF16/float32ロードで統一した。

**Q12. どうやってlayer毎のhidden stateを取得したか。** 当初`output_hidden_states=True`のタプルを使おうとしたが、**バグを発見**: HFの実装ではタプルの最後の要素が「最終layerの生出力」ではなく「post-final-norm済みの値」であることが判明した(`layer_47_output`と`post_final_norm`がbit-for-bit完全一致することを実測で確認)。これを避けるため、各decoder layerモジュール(`model.model.layers[i]`)に直接forward hookを登録し、llama.cppの`cb(cur, "l_out", il)`と同じ意味(layer ilの生の残差ストリーム出力)を確実に捕捉する方式に修正した。

**Q13. 固定prefixの手法は。** Phase4Z/4ZAと同一のE36 original(system+user)をHFの`apply_chat_template(add_generation_prompt=True)`でrenderし、forced prefix「こんにちは〜！私はパチスロの専門アシスタントの」を追記した677文字/440トークンのテキストを使用。プロンプト/トークンIDはPhase4Z Section7で既にbyte-identicalと確認済みの方式を踏襲。

**Q14. リ/ルのtoken IDは。** リ=36723、ル=32610。本フェーズでも`tokenizer.encode()`で再確認し(推測ではない)、Phase4ZBと同一であることを確認した。

---

## 12. llama.cpp側 layer-wise hidden state抽出

**Q15. どのツールで実施したか。** 本フェーズでビルドした`llama-debug.exe`(CPU-only)。BF16 GGUF(`training/riru/gguf/riru-qwen-final-bf16.gguf`、Phase4ZBでHFと100% bit一致証明済み、再変換なし)に対し、`--tensor-filter`で`l_out-<il>`(全48層)、`result_norm`、`result_output`をダンプした。

**Q16. なぜllama-serverではなくllama-debugを使ったのか。Phase4ZAの決定と矛盾しないか。** Phase4ZAで確立した「テキスト生成比較はllama-server使用」という決定と、本フェーズの「内部tensor値の直接ダンプにはllama-debug(cb_eval機構を持つ唯一のツール)を使う」という決定は、目的が異なるため矛盾しない。llama-serverにはtensorダンプ機構が存在しない。ただし本フェーズの最終logits測定(llama-debugのresult_outputダンプ)は、Phase4ZAのllama-server測定値(CPU: リ=0.083, ル=0.087)と定性的に完全に一致した(本フェーズ: リ=12.533, ル=12.592、順位・方向とも一致)ため、CLI/server間の既知の不一致(Phase4ZAで記録済み、未解決)が本フェーズの主結論に影響していないことも確認できた。

**Q17. 何層分のデータを取得したか。** 48層全ての`l_out-<il>`(0-47)、`result_norm`(post-final-norm)、`result_output`(logits、152064次元)。

---

## 13-15. Layer-wise比較結果

**Q18. 最初の乖離層(first divergence layer)はどこか。** `first_nonzero_diff_layer`=layer 0(最初の層から既に非ゼロの乖離が存在)。`first_10x_jump_layer`=なし(単一の急激な10倍ジャンプは存在しない)。`first_cosine_drop_below_0.9999_layer`=layer 27(cosine類似度がそこで緩やかに0.9999を下回り始める)。

**Q19. 乖離のパターンは「単一層への集中」か「分散」か。** **分散型**。layer0から既にl2_relative_error約1.1-1.6%の乖離が存在し、48層を通じて緩やかに、しかし単調ではなく変動しながら拡大していく(絶対誤差はlayerが進むにつれ大きくなるが、これはhidden stateのノルム自体が深い層ほど大きくなることと概ね比例しており、相対誤差は概ね1-1.7%のレンジ内に収まっている)。

**Q20. 各層のcosine類似度は。** 全層で0.9999前後の非常に高い値を維持しており(layer27以降やや低下するが最終層でも0.99996)、方向としてはHFとllama.cppのhidden stateはほぼ同一方向を向いている。乖離は「大きな破綻」ではなく「小さいが系統的なノイズ」の性質を持つ。

**Q21. 最終層(layer47)・post_final_norm・logitsでの乖離は。** layer47: max_abs=4.51, cos=0.99996, l2_rel=0.89%(直前の層と同程度、破局的な拡大なし)。詳細は`phase4zc_layerwise_hidden_diff.json`参照。

**Q22. リ/ルのlogit値そのものの比較は。** HF(bf16, eager): リ=12.5625, ル=12.5625(完全同値)。llama.cpp(CPU GGUF): リ=12.533, ル=12.592(ル優勢、差0.059)。

---

## 15. layer0 sub-block分解

**Q23. なぜlayer0を対象にしたか。** layer-wise解析で単一の急激なジャンプ層が見つからなかったため、Section15の指示に従い、最初に乖離が観測される層(layer0)を対象に、どのoperation区間で乖離が最も拡大するかを調べた。

**Q24. sub-block分解の結果は。** `attn_norm-0`(RMSNorm直後): l2_rel=0.23%(極めて小さい)。`ffn_inp-0`(self-attentionブロック通過後): l2_rel=1.46%(**約6.4倍に拡大、layer0内で最大のジャンプ**)。`ffn_norm-0`(2つ目のRMSNorm後): l2_rel=2.23%(RMSNormによる二次的増幅)。`ffn_out-0`(MLP出力): l2_rel=1.17%(むしろ低下、MLPは新たな独立した乖離源ではない)。

**Q25. どの演算が支配的か特定できたか。** attention計算ブロック全体(QKV projection・RoPE・QK-matmul・softmax・attention-output合成・o_proj)が、layer0における乖離の主要な発生源であることが判明した。ただしGQA構造(40 query heads, 8 KV heads)を考慮したQcur/Kcur/Vcur個別のhead単位分解までは実施しておらず、attention内のどの単一演算(QK-matmul単体、softmax単体、o_proj単体)が最終的に支配的かの完全な切り分けには至っていない。

**Q26. なぜそれ以上の細分化をしなかったか。** Section9で明示的に警告されていたGQAレイアウト整合の複雑さ(40 query headsを8 KV headsで共有するrepeat/broadcastパターン)を考慮すると、Q/K/V個別tensorの正確な対応付けには追加の慎重な設計が必要であり、既に得られている3つの独立した証拠(layer-wise解析・sub-block解析・precision-sensitivityテスト、下記)が一貫した結論を支持していたため、時間予算とのバランスの上で、このレベルの深さで十分な説明力が得られたと判断した。

---

## 21-22. Precision-sensitivityテスト(重要な補完実験)

**Q27. どのような実験を行ったか。** llama.cppを一切介さず、HF/PyTorch単体で、重み・トークナイズ・プロンプトを完全固定した状態のまま、(a)attention実装をeager→sdpaに変更、(b)計算精度をbfloat16→float32に変更、の2つの独立した軸で、リ/ルの優劣がどう変化するかを検証した。

**Q28. 結果はどうだったか。** 決定的な結果が得られた。
- HF bf16+eager(基準): リ=ル(**完全な同点**、logit両方とも12.5625)
- HF bf16+SDPA: ル(12.5625) > リ(12.5) — **ル優勢に反転**
- HF float32+eager: ル(12.566) > リ(12.518) — **ル優勢に反転(より明確な差)**
- llama.cpp CPU GGUF(本フェーズ実測): ル(12.592) > リ(12.533) — ル優勢
- llama.cpp CUDA GGUF(Phase4Z再利用): ル優勢
- llama.cpp CPU GGUF via llama-server(Phase4ZA再利用): ル優勢
- (参考)HF 4bit NF4量子化(Phase4Z/4ZAの従来「HF」基準): リ優勢 ← これのみ量子化ノイズが乗った値

**Q29. この結果は何を意味するか。** GGUF/llama.cpp側に欠陥があるという単純な図式ではないことを示している。このトークン対(リ/ル)は、モデル自身の中で極めて僅差であり、bfloat16+eagerという「素朴で厳密な」計算経路がたまたま丸め誤差の範囲内で完全な同点に収まっているに過ぎない。それ以外のほぼ全ての実務的な計算経路(SDPA、float32、GGUF/llama.cpp)は、同じ僅かな「ル優勢」の潜在的傾向を(それぞれ異なる丸め方向・累積順序により)顕在化させている。

**Q30. Flash Attentionでのテストは実施したか。** 実施していない。既存の`.venv-qlora`環境に`flash_attn`パッケージがインストールされていないことを確認し(`ModuleNotFoundError`)、Section22の「新規パッケージインストールなし」の制約に従いスキップした。

---

## 25-26. Logits比較テーブルと分岐点の関係性

| 条件 | リlogit | ルlogit | 差(リ-ル) | 優勢 |
|---|---|---|---|---|
| HF bf16+eager | 12.5625 | 12.5625 | 0.0 | 同点 |
| HF bf16+SDPA | 12.5 | 12.5625 | -0.0625 | ル |
| HF float32+eager | 12.518 | 12.566 | -0.048 | ル |
| llama.cpp CPU GGUF(本フェーズ) | 12.533 | 12.592 | -0.059 | ル |
| llama.cpp CUDA GGUF(Phase4Z) | prob 0.083 | prob 0.088 | - | ル |

**Q31. layer-wise hidden state driftと最終logit反転はどう繋がるか。** layer0のattentionブロックで生じた小さな数値差(l2_rel約1.5%)が、以降48層を通じて劇的に増幅されることなく、ほぼ同程度の相対誤差(1-1.7%)を保ったまま伝播し、最終層のlogit計算(post_final_norm→lm_head projection)に到達する。この累積した小さな差が、そもそもモデル自身がリ/ル間でほぼ完全な同点(margin=0)という極めて脆弱な分岐点において、最終的な順位を反転させるのに十分な大きさ(約0.05-0.06 logit)になっている。すなわち「巨大な計算バグによる崩壊」ではなく「小さいが一貫した数値ノイズが、たまたま極端に薄い氷の上にある分岐点を押し倒した」という説明が、全ての実測データと整合する。

---

## 27-28. CASE ZC選定

**選定: CASE ZC-G(mixed / 分散型ドリフト、attention計算ブロックに集中)**

**Q32. なぜ単一のletter(A-F)ではなくGを選んだか。** 3つの独立した診断軸——(1)layer-wise解析(分布型、単一層への集中なし)、(2)layer0 sub-block解析(RMSNormやMLPではなくattentionブロック全体が支配的)、(3)precision-sensitivityテスト(HF自身のeager/sdpa切替・bf16/fp32切替でも同方向の反転が再現される)——が全て、「単一の名指し可能な演算(RMSNorm単体/RoPE単体/softmax単体/GEMM単体/MLP単体)への完全な帰属はできないが、attention計算ブロック内の複数演算の数値実装差が主要因である」という一貫した像を示したため。CASE F(全層に完全に一様)ではない理由は、layer内でattentionブロックとRMSNorm/MLPとの間に明確な偏り(attentionが支配的)が存在するため。

**Q33. 確信度は。** 中〜高。3つの独立したアプローチが同じ方向を示している点で高い一貫性があるが、GQAレイアウトを考慮したQ/K/V個別の完全な切り分けには至っていないため、「attentionブロック内のどの単一演算が最終的な支配要因か」は完全には特定できていない。

---

## 30-31. デバッグビルドの記録

`phase4zc_llamacpp_debug_build.json`に、debug copyのパス・commit・変更ファイル・diff・ビルドコマンド・コンパイラ・CPU/CUDA区分・canonical不変確認を全て記録した。

---

## 32. RAGスモークテスト

**Q34. 実施したか。** 実施していない。本フェーズで再ビルドしたのは診断専用の`llama-debug.exe`のみであり、production側のllama-server/llama-cli/GGUFファイル/RAGパイプラインは一切変更していないため、Section32の「モデル再構築時のみ必須」の条件に該当せず、対象外と判断した。

---

## 35-38. 最終確認

- `pytest`: 126 passed(不変)
- protected asset hash: 全て一致(不一致0件、開始時・終了時とも)
- `git status`/`git diff`: HEAD不変(`a61d664f1d6af087b69056eb718fafeab7892401`)、追跡ファイルへの変更なし、commit/push一切なし
- llama.cpp canonical: HEAD不変、変更なし

**Q35. training/QLoRA/dataset/adapter/merged HF/GGUF/quantization/system prompt/tokenizer/chat template/productionへの変更はあったか。** 一切なし。

**Q36. git commit/pushは行ったか。** 行っていない。

**Q37. 次フェーズへの自動移行はしたか。** していない。人間の判断待ちで停止する。

---

## 39. Slack通知

パイプライン全体(instrumentation調査→debug build→layer-wise比較→sub-block分解→precision-sensitivityテスト→CASE選定)が完全に完了したため、下記内容でSlack通知を送信する:
- CASE ZC-G(mixed、attention計算ブロックに集中した分散型ドリフト)
- first divergence layer: layer 0(分布型、単一ジャンプ層なし。layer27付近で緩やかな加速)
- 疑われる演算: attention計算ブロック(QKV projection/RoPE/QK-matmul/softmax/attention-output/o_proj)
- 重要な補完知見: HF自身のeager/SDPA切替・bf16/fp32切替でも同方向の反転が再現され、根本原因はllama.cpp固有のバグではなく、モデル自身の極めて僅差な分岐点(bfloat16+eagerでは丸めにより完全同点)にあることが判明
- production変更: なし(引き続き禁止)
- protected assets: 変更なし
- git操作: なし

---

## 42. 最終原則の再確認

Phase4ZBで証明された「全579テンソルが完全にbit一致」という事実は、本フェーズの全ての分析を通じて一貫して尊重された。本報告書のいかなる箇所も、「weightが違う」「GGUF変換で情報が失われた」という説明には一切言及・示唆していない。本フェーズが特定したのは、100%同一のweightに対して、PyTorch(HF)とllama.cppという異なる計算エンジンが、attention計算ブロックを中心とした数値実装の違い(累積和の順序・カーネル実装差)により、極めて僅差(bfloat16+eagerでは完全同点)な分岐点で異なる方向に丸められている、という現象である。

---

## まとめ

| 項目 | 結果 |
|---|---|
| CASE | ZC-G(mixed、attention計算ブロックに集中した分散型ドリフト) |
| First divergence layer | Layer 0(分布型、単一ジャンプ層なし) |
| 疑われる演算 | attention計算ブロック(QKV/RoPE/QK-matmul/softmax/attn-output/o_proj) |
| 重要な補完知見 | HF自身のeager/SDPA・bf16/fp32切替でも同方向反転が再現(llama.cpp固有バグではない) |
| Weight差説明 | 引き続き完全に否定(Phase4ZBの証明を継承・強化) |
| Production変更 | なし |
| Protected assets | 変更なし(全hash一致) |
| Git操作 | なし(HEAD不変) |
| 次フェーズ | 人間の判断待ちで停止 |
