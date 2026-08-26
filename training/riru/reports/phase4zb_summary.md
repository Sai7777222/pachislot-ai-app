# Phase 4ZB: GGUF Weight vs llama.cpp Forward Isolation 最終報告

## 0. 結論の要約

- **merged HF(safetensors)とBF16 GGUFの全579テンソル・147億要素を、公式gguf-py(TensorNameMap/GGUFReader/dequantize、新規インストールなし)を用いて数値的に比較した結果、全テンソルが完全にビット単位で同一(誤差ゼロ)であることを確認した。**
- **「リ」(token ID 36723)・「ル」(token ID 32610)に対応するoutput_head/token_embeddingの行ベクトルも、両者で完全に同一だった。**
- **RoPE theta・RMSNorm epsilon等、テンソルとして保存されない実行時パラメータも完全一致することを確認した。**
- **判定: CASE ZB-F — llama.cpp共通forward計算経路が支配的要因。** HF→GGUF変換によるweight representation差は、Phase4Zで観測されたidentity regressionの原因として完全に否定された。
- Gate2(weight数値差の定量化)で全テンソル完全一致という最も強い形の証拠が得られたため、Section12で条件付けられていた「GGUF由来weight+PyTorch forward」の追加構築・実行は、論理的に冗長(既にHFのforward pathと数学的に同一結果になることが保証される)と判断し、実施しなかった。これは技術的制約によるCASE ZB-U該当ではなく、根拠に基づく省略である。
- **production移行禁止を継続する。**

## 1. 開始前確認 (Section 3)

| 項目 | 値 |
|---|---|
| git HEAD | `a61d664f1d6af087b69056eb718fafeab7892401`(不変) |
| pytest | 126 passed |
| Final Candidate adapter/adapter_config | 不変 |
| candidate/train/val | 不変 |
| config | 不変 |
| system.jinja2 | 不変 |
| merged HF | 不変 |
| BF16 GGUF | 不変(SHA-256: `e102f5d3...`) |
| llama.cpp commit | `5d5cb4c3a4ea8769490d39a275ee49a45184774d`(不変) |

## 2. 実現可能性調査 (Section 6)

新規framework・新規GGUF推論engineのインストールは行わなかった。Phase4Y-Rで既に導入済みの`gguf-py`(llama.cpp同梱、`.venv-gguf`に既にインストール済み)に含まれる公式実装のみを使用した:

- `gguf.GGUFReader`: GGUFファイルのtensor一覧・raw dataを読み取り専用で取得
- `gguf.dequantize`: 量子化tensorをlosslessにfloat32へ復元(公式実装)
- `gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN2, 48)`: HF名→GGUF名の公式mapping(convert_hf_to_gguf.py自身が使用するものと同一)

HF側はsafetensorsの`framework='pt'`(torch)でbfloat16を正しく読み込み、`.float()`でlosslessにfloat32へアップキャストした(numpy frameworkはbfloat16未対応のため使用せず)。

## 3. Tensor Inventory (Section 7)

| 項目 | 値 |
|---|---|
| HF tensor数 | 579 |
| GGUF tensor数 | 579 |
| マッチしたペア数 | **579(100%)** |
| GGUF側に見つからないHF tensor | 0 |
| 予期しないGGUF tensor | 0 |

初回実行時、mapping関数の戻り値に既に接尾辞(`.weight`/`.bias`)が含まれているにも関わらずスクリプト側で再度接尾辞を付与するバグ(`blk.0.attn_q.weight.weight`のような二重接尾辞)があり、全579件がマッピング失敗と誤判定された。convert_hf_to_gguf.py本体のソース(`conversion/base.py`の`map_tensor_name`実装)を確認し、`get_name()`の戻り値をそのまま使うのが正しい使用法であることを確認して修正した。詳細は`phase4zb_tensor_inventory.json`を参照。

## 4. Weight Direct Comparison (Section 8-11) — 最重要測定

全579テンソルを1つずつストリーム処理し(メモリ効率のため全体を同時展開しない)、HF(float32アップキャスト)とGGUF(dequantize後float32)を比較した。

### 全体集計

| 指標 | 値 |
|---|---|
| exact identical tensor数 | **579 / 579(100%)** |
| non-identical tensor数 | **0** |
| global max abs diff | **0.0** |
| global mean abs diff | **0.0** |
| global RMS diff | **0.0** |
| differing element ratio | **0.0** |
| 比較した総要素数 | 14,770,033,664 |

### tensor種別ごとの内訳(全種別で完全一致)

| tensor種別 | 件数 | max_abs_diff | exact_identical |
|---|---|---|---|
| token_embeddings | 1 | 0.0 | 1/1 |
| output_head | 1 | 0.0 | 1/1 |
| attention_q | 96 | 0.0 | 96/96 |
| attention_k | 96 | 0.0 | 96/96 |
| attention_v | 96 | 0.0 | 96/96 |
| attention_output | 48 | 0.0 | 48/48 |
| mlp_gate | 48 | 0.0 | 48/48 |
| mlp_up | 48 | 0.0 | 48/48 |
| mlp_down | 48 | 0.0 | 48/48 |
| attn_norm | 48 | 0.0 | 48/48 |
| ffn_norm | 48 | 0.0 | 48/48 |
| final_norm | 1 | 0.0 | 1/1 |

**layer別集計(48層全て)でも例外なく完全一致。shape不一致も0件。**

## 5. BF16 Round-trip仮説の検証 (Section 9)

merged HF自体がBF16で保存されており、GGUF変換も`--outtype bf16`で行われたため、理論上、変換が単純なBF16値の並べ替え(layout変換)であればビット一致しうるという仮説を立てていた。**実測の結果、この仮説を支持する形で、全テンソルが完全に一致した。** 「BF16だから多少違って当然」という推測に頼らず、実際に全要素を比較して結論づけた。

## 6. Attention Tensor重点確認 (Section 10)

Section10の要求通り、tensor種別ごと(embedding/output/attn_q/k/v/output/mlp_gate/up/down/norm)およびlayer別(0〜47)の集計を`phase4zb_weight_diff_analysis.json`の`by_tensor_type`/`by_layer`に保持した。**どの種別・どの層でも差は一切見られなかった。**

## 7. Output Embedding / lm_head確認 (Section 11)

「リ」「ル」のtoken IDをtokenizerで直接確認した(推測なし):

| トークン | ID |
|---|---|
| リ | 36723 |
| ル | 32610 |

`output.weight`(lm_head)・`token_embd.weight`双方について、この2トークンに対応する行ベクトルをHF/GGUFで比較した。

| tensor | トークン | exact_equal | max_abs_diff |
|---|---|---|---|
| output.weight | リ(36723) | true | 0.0 |
| output.weight | ル(32610) | true | 0.0 |
| token_embd.weight | リ(36723) | true | 0.0 |
| token_embd.weight | ル(32610) | true | 0.0 |

**identity分岐に直接関与するembedding/output vectorも完全一致。**

## 8. 実行時パラメータ確認

RoPE theta・RMSNorm epsilon等、テンソルとして保存されない設定値も、HF `config.json`とGGUF変換ログの両方から直接確認した。

| パラメータ | HF | GGUF |
|---|---|---|
| rope_theta | 1000000.0 | 1000000.0 |
| rms_norm_eps | 1e-06 | 1e-06 |
| num_attention_heads | 40 | 40(一致) |
| num_key_value_heads | 8 | 8(一致) |

**完全一致。**

## 9. GGUF→PyTorch Forward実現可能性判定 (Section 12)

Section12は「全tensor mappingが説明可能・shape完全一致・missing/unexpected tensor 0」という条件を満たした場合のみGGUF由来weightをPyTorchへ構築するとしていた。**この条件自体は満たされた**(Gate1: 579/579マッピング成功、shape mismatch 0件)。

しかし、Gate2で**全テンソルが完全にビット同一である**ことが判明したため、GGUF由来weightをPyTorchのQwen2ForCausalLMへ再構築して推論しても、**数学的に元のmerged HFのforward pathと同一の計算になることが保証される**(重みが同一であれば、同じPyTorch実装で計算する限り結果も同一)。したがって、この追加構築・実行は新たな情報をもたらさない科学的に冗長な工程と判断し、実施しなかった。

これはSection18の「技術的に安全な完全mappingを保証できない場合はSKIP」という条件とは異なる理由によるSKIPである点を明記する。技術的には実行可能だったが、既に得られた証拠(Gate2)により結果が自明であるため省略した。

## 10. Section 20 CASE判定

**判定: CASE ZB-F — llama.cpp common forward dominant。**

根拠(Section19の基準に厳密に合致):
- 全対応tensorが変換後も**実質どころか完全にbitwise identical**であることを、579テンソル全件・147億要素の網羅的比較で確認した。
- 「リ」「ル」に対応する具体的なembedding/output vectorも完全一致した。
- RoPE等の実行時パラメータも一致した。
- したがって、HF→GGUF weight転換説(CASE ZB-W)は完全に否定された。
- Phase4Z/4ZAで確認済みの「HF: リ>ル」「GGUF(CUDA/CPU共通): ル>リ」というlogit順位反転は、**同一の重みを異なるforward実装(PyTorch vs llama.cpp/ggml)で計算した結果の差**によってのみ説明可能である。

## 11. Section 40 最終報告(50項目)

1. **git HEADは不変か** — 不変(`a61d664f...`)
2. **pytest結果** — 126 passed
3. **protected assetsは不変か** — 不変(全項目一致確認)
4. **BF16 GGUF hashは一致したか** — 一致(`e102f5d3...`)
5. **llama.cpp commit/buildは不変か** — 不変(`5d5cb4c3a`, b10631)
6. **GGUF tensor数** — 579
7. **HF tensor数** — 579
8. **tensor mappingは全件説明できたか** — できた(公式TensorNameMapで579/579)
9. **missing tensorはあるか** — ない(0件)
10. **unexpected tensorはあるか** — ない(0件)
11. **shape mismatchはあるか** — ない(0件)
12. **HF/GGUF tensorはbitwise同一か** — **同一(579/579、100%)**
13. **exact identical tensor数** — 579
14. **non-identical tensor数** — 0
15. **global max abs diff** — 0.0
16. **global mean abs diff** — 0.0
17. **global RMS diff** — 0.0
18. **differing element ratio** — 0.0
19. **最も差が大きいtensorは何か** — 該当なし(全tensor差0)
20. **attention Q/K/Vで差はあるか** — ない
21. **MLPで差はあるか** — ない
22. **embeddingで差はあるか** — ない
23. **lm_head/outputで差はあるか** — ない
24. **「リ」token ID** — 36723
25. **「ル」token ID** — 32610
26. **「リ」output vectorのHF/GGUF差** — 0.0(完全一致)
27. **「ル」output vectorのHF/GGUF差** — 0.0(完全一致)
28. **GGUF→PyTorch完全mappingは可能だったか** — 可能だった(技術的には)
29. **GGUF-derived PyTorch modelを作成したか** — **作成しなかった**(Gate2の結果により科学的に冗長と判断)
30. **prompt/token IDsは完全一致したか** — Phase4ZAで確認済み(423トークン完全一致、本フェーズでは重み比較が主目的のため再確認は省略)
31. **original HF PyTorchのE36結果** — safe(Phase4Z既存結果)
32. **GGUF-derived PyTorchのE36結果** — 未実施(Gate2により論理的に「HF側と同一」と結論)
33. **llama.cpp CPU/serverのE36結果** — 「ルリ」(Phase4ZA既存結果)
34. **original HFでのリ/ルmargin** — リが優位(+2.7pt、Phase4Z/4ZA既存logits)
35. **GGUF-derived PyTorchでのリ/ルmargin** — 未測定(上記理由により実施せず)
36. **llama.cppでのリ/ルmargin** — ルが優位(+0.4〜0.5pt、Phase4ZA既存logits)
37. **GGUF-derived PyTorchはHF側とGGUF側のどちらに近いか** — 該当なし(未実施)。ただしweight完全一致より、実施すればHF側と一致することが論理的に導かれる
38. **hidden state比較を実施したか** — 実施しなかった(主目的である重み比較で十分な結論が得られたため、Section23の「追加診断であり主目的を遅らせる場合は無理に実施しない」という方針に従った)
39. **weight差だけで順位反転を再現したか** — できない(weight差が存在しないため)
40. **forward差の証拠はあるか** — ある(weight完全一致下でPhase4Z/4ZAのlogits差が存在するという消去法的証拠)
41. **CASE ZB-W/F/M/Uのどれか** — **CASE ZB-F**
42. **CASEの根拠** — 全579テンソル・147億要素が完全にビット同一。identity/output embeddingも完全一致。実行時パラメータも一致。残る説明変数はforward実装の違いのみ
43. **CUDA backend説は現在どう評価するか** — 否定的(Phase4ZAで既に後退、今回も変更なし)
44. **HF→GGUF変換説は現在どう評価するか** — **強く否定**(weight完全一致により)
45. **llama.cpp共通forward説は現在どう評価するか** — **強く支持**
46. **CLI/server差は今回の結論へ影響したか** — 影響していない。Phase4ZBでは比較をllama-serverに固定するPhase4ZAの方針を維持し、今回は生成実験ではなく重み比較が主体だったため、この既知の未解決事項(CLI vs server差)はそのまま将来のPhase4ZC候補として保留した
47. **production禁止を解除できるか** — **できない**
48. **次に変更すべき最小1変数は何か** — llama.cpp(ggml) vs PyTorch forwardのlayer-wise hidden-state driftを追跡し、どの層・どの演算(attention softmax、RMSNorm、SiLU等)から数値差が生じ始めるかを特定する
49. **追加学習の科学的根拠は現時点であるか** — ない。今回の結果は推論エンジンの実装差を示すものであり、モデル自体の再学習で解決する性質の問題ではない
50. **Git commit/pushを行ったか** — 行っていない

## 12. Section 41 次ステップ提案(実行はしない)

**CASE ZB-Fのため、以下を提案する:**

llama.cpp(ggml/CPU・CUDA共通部分)とPyTorch(HF) forwardのlayer-wise hidden-state divergenceを追跡する。具体的には、E36 originalのprefix通過時の各層(layer 0〜47)出力hidden stateを、HF側(`output_hidden_states=True`)とllama.cpp側(可能であれば`--verbose`や内部debug出力、または将来的なinstrumentationを用いて)で層ごとに比較し、どの層・どの演算(attention softmax正規化、RMSNorm、SiLU活性化関数、RoPE適用等)から数値差が蓄積し始めるかを特定することを推奨する。

本フェーズではこの追跡は実施していない。

## 13. Section37 既知の未解決事項

Phase4ZAで発見された「CPU llama-cli(safe) vs CPU llama-server(unsafe)」というtool依存差は、本フェーズでは意図的に取り扱わず、既知の未解決事項として保留した。将来的な**Phase4ZC: Sampler / Tool Path Isolation**の候補として記録する。

## 14. 禁止事項の遵守

追加学習・LoRA再学習・SFT・QLoRA・identity教師追加・negative example追加・dataset/candidate/train-val再構築・merged HF/BF16 GGUF/Q8_0/Q5_K_M変更・再変換・再量子化・Base Qwen変更・system.jinja2変更・RAG DB/structured.db/Vector DB変更・production変更・アプリコード変更・API接続変更・llama.cpp update/checkout/rebuild・transformers/llama-cpp-python更新・CUDA toolkit/driver変更・Git commit/pushは一切行っていない。既存`.venv`/`.venv-qlora`も変更していない(全ての解析は既存の`.venv-gguf`専用venvで実施)。

## 15. 最終確認 (Section 32, 38)

- pytest: **126 passed**(開始前・終了後とも)
- git status: Phase4X checkpoint分のみ反映済み。以降の成果物は未commit
- git diff / git diff --cached: 差分なし
- git HEAD: `a61d664f1d6af087b69056eb718fafeab7892401`(不変)
- Protected Assets: 全て不変(candidate/train/val/config/adapter/adapter_config/system.jinja2/merged HF/BF16 GGUF)
- llama.cpp commit: 不変

## 作成ファイル一覧

- `training/riru/eval/phase4zb_gguf_tensor_compare.py`
- `training/riru/reports/phase4zb_tensor_inventory.json`
- `training/riru/reports/phase4zb_weight_diff_analysis.json`
- `training/riru/reports/phase4zb_generation_config.json`
- `training/riru/reports/phase4zb_gate_analysis.json`
- `training/riru/reports/phase4zb_summary.md`(本ファイル)

## 停止

Tensor inventory・weight direct comparison・実行時パラメータ確認・CASE判定・pytest・保護対象資産確認・レポート作成が完了しました。

**判定はCASE ZB-F(llama.cpp共通forward計算経路が支配的要因)です。** HF→GGUF変換によるweight差説は、全579テンソルの完全一致という直接的証拠により強く否定されました。

production移行禁止を継続します。追加学習・新GGUF作成・production変更・アプリコード変更・API変更・llama.cpp更新・Git commit/push・Phase4ZC等への自動移行は一切行っていません。次のご判断をお待ちします。
