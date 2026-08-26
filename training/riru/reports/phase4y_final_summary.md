# Phase 4Y-R: llama.cpp正式導入 + GGUF変換 + Quantization + Final Gate 最終報告

（`phase4y_summary.md`の続き。merge段階までは前回報告の通り。本ファイルはSection16以降の再開分。）

## 0. 結論の要約

- **llama.cppを公式アップストリームから正式導入し、GGUF変換パイプラインを完全に動作させることに成功した。** BF16高精度GGUF・Q8_0・Q5_K_Mの3種を作成し、いずれも正常ロード・正常推論を確認した。
- **12代表probe×5条件(A_lora_final/B_merged_hf/C_gguf_bf16/D_gguf_q8_0/E_gguf_q5_k_m)の比較を実施した結果、2つの看過できない所見が見つかった:**
  1. **PT-01(scope型probe)で量子化が進むほど「差分」情報の欠落率が単調に増加した**(A=75%→B=75%→C(BF16)=75%→D(Q8_0)=50%→E(Q5_K_M)=25%)。個別の数値(1/399、1/319)自体は全条件で正しく維持されており、捏造ではなく「計算済みの差分(80回転分)を追加で言及するかどうか」という完全性(completeness)側の量子化回帰。
  2. **E36(挨拶文脈)のgreedy生成で、BF16を含む3つのGGUF条件全てが一貫して「ルリ」という架空の名前を生成した。** 同一promptに対するA_lora_final・B_merged_hf(HF/PyTorch経路)ではこの誤名乗りは一度も発生しなかった。**最高精度であるBF16でも100%再現した**ことから、量子化由来ではなく、**HF→GGUF変換またはllama.cpp推論エンジンの計算経路の違いに起因する可能性が高い。**
  3. 副次的に、E02の一部生成で「○○」という記号がそのまま出力される、placeholder的な事象も1件確認した(BF16・Q8_0で再現、Q5_K_Mでは非再現)。既存のplaceholder detector(`phase4x_placeholder_detector.py`)は「○○」パターンを対象に含めておらず、新たな検出漏れとして記録する。
- **これらの所見は、指示書Section31の明示的な停止条件「high precision GGUFで重大回帰」に該当すると判断した。** そのため、**判定はCASE C(HF→GGUF変換工程を疑う。production移行禁止)** とする。
- **サンプル数の限界について明記する**: 本比較は12代表probe×少数seed(48生成/条件)による絞り込み評価であり、Phase4Xのフルスケール評価(220〜864生成規模)ではない。今回の所見は「疑いなく確定した重大な欠陥」ではなく、「production統合前に必ず大規模再検証すべき、無視できない具体的シグナル」として報告する。

## 1. 開始前確認 (Section 2-3)

- branch: `main`、HEAD: `a61d664f1d6af087b69056eb718fafeab7892401`(前回Phase4Yと同一、変更なし)
- git status: Phase4Y成果物(未commit、想定通り)以外の変更なし
- Freeze Manifest記載の全ハッシュ(candidate/train/val/config/stable adapter/adapter_config/system.jinja2/merged HF)を再計算し、**全て一致**を確認した(不一致なし、停止条件に抵触せず)

## 2. llama.cpp正式導入 (Section 4-8)

| 項目 | 値 |
|---|---|
| upstream URL | `https://github.com/ggml-org/llama.cpp.git`(公式、サードパーティforkではない) |
| clone先 | `D:/AI/tools/llama.cpp`(プロジェクト本体とは分離) |
| commit hash | `5d5cb4c3a4ea8769490d39a275ee49a45184774d` |
| branch | master |

CMake/Visual Studio Build Toolsがこの環境に存在しなかった(`cmake`/`cl.exe`検出不可)ため、ソースビルドの代わりに**公式GitHub Releasesの署名済みprebuilt binary**(build b10631、Windows CUDA 12.4向け)を使用した。重要な点として、**このprebuilt binaryのビルドcommit(`5d5cb4c3a`)がcloneしたsourceのHEAD commitと完全に一致**しており、再現性・整合性が確保されている。

- `llama-quantize.exe`: `D:/AI/tools/llama.cpp-bin/llama-quantize.exe`(SHA-256: `82746f8f...`)
- `llama-cli.exe`: `D:/AI/tools/llama.cpp-bin/llama-cli.exe`(SHA-256: `b437c83f...`)

Smoke test(`--help`実行)は全て正常終了。

## 3. Python変換環境 (Section 6)

既存の`.venv`(app用)・`.venv-qlora`(QLoRA学習用)を汚染しないよう、`D:/AI/tools/llama.cpp/.venv-gguf`を新規作成した(Python 3.11.8)。

**2回の変換失敗を診断・修正した:**
1. `sentencepiece`未インストール → 公式requirements(`requirements-convert_hf_to_gguf.txt`)を専用venvへinstallして解消。
2. `transformers==4.57.6`(公式pin)でQwen2 tokenizer読み込み時に`AttributeError: 'list' object has no attribute 'keys'` → merged HFモデルのtokenizer_config.jsonの`extra_special_tokens`がlist形式(transformers 5.15.1が書き出した形式)であるのに対し、公式pinの古いtransformersがdict形式を前提としていたための非互換。実際にそのファイルを書き出したのと同じ`transformers==5.15.1`を専用venv内でのみインストールして解消した。

詳細は`phase4y_llamacpp_environment.json`に記録した。

## 4. GGUF変換 (Section 9-11)

merged HF(SHA-256: `01d59777...`、manifest記録値と一致確認済み)から、`--outtype bf16`で高精度GGUFを作成した。

| 項目 | 値 |
|---|---|
| 出力 | `training/riru/gguf/riru-qwen-final-bf16.gguf` |
| SHA-256 | `e102f5d3a2dced6030cb3b60e540570d8609a62f405573adf3f536c97162bdf2` |
| サイズ | 29,547,715,968 bytes(約29.5GB) |
| 変換時間 | 約87秒 |

## 5. High Precision GGUF検証・Smoke Inference (Section 12)

`llama-cli.exe`で単独ロードし、代表prompt(「こんにちは」)で推論した。

- モデルロード: 正常(build/model/ftype=BF16のメタデータが正しく表示)
- 生成: 「こんにちは！パチスロのことなら何でも聞いてね。」— 文字化けなし、EOS異常なし、無限生成なし、repetition異常なし
- 生成速度: 8.3 tok/s(smoke test単発)

## 6. Quantization (Section 13-14)

事前にllama-quantize.exeの`--help`で対応量子化タイプを確認し、Q8_0・Q5_K_M・BF16・F16が正式サポートされていることを確認した上で、指示書通りQ8_0(品質重視)・Q5_K_M(実用サイズ)の2条件のみを作成した(sweepなし)。

| 形式 | サイズ | SHA-256(先頭16桁) | 量子化時間 | bits/weight |
|---|---|---|---|---|
| Q8_0 | 15,701,597,568 bytes(14.6GB) | `a8d1c4130b195172` | 24.7秒 | 8.5 |
| Q5_K_M | 10,508,873,088 bytes(9.8GB) | `5db936ec50301c2d` | 49.6秒 | 5.69 |

## 7. GGUFロード確認 (Section 15)

C(high precision BF16)・D(Q8_0)・E(Q5_K_M)全てllama-cpp-python(v0.3.35)で単独ロードに成功。エラー・OOMは発生しなかった。

## 8. 評価条件・代表評価 (Section 16-17)

A(LoRA Final)・B(merged HF)はPhase4Yの既存結果を再利用し、C/D/Eは新規生成した。`phase4y_representative_probes.py`の12代表probe(Q3/P01/P02/Scope(PT-01)/Q9/Q11/E02/E36/naming(NW-01)/Broad(V1-A)/Adversarial(AD-01)/Long-context(LC-01))×greedy+seed42/43/44を同一system prompt・同一generation設定(temperature=0.3、production設定について後述)で実施した。

## 9. 比較結果 (Section 18-20)

### required_fact_recallの5条件比較

| probe | A | B | C(BF16) | D(Q8_0) | E(Q5_K_M) |
|---|---|---|---|---|---|
| Q3 | 100% | 100% | 100% | 100% | 100% |
| P01 | 75% | 100% | 75% | 87.5% | 87.5% |
| P02 | 100% | 100% | 100% | 100% | 100% |
| **PT-01(scope)** | **75%** | **75%** | **75%** | **50%** | **25%** |
| Q9 | 100% | 100% | 100% | 100% | 100% |
| Q11 | 100% | 100% | 100% | 100% | 100% |
| V1-A(broad) | 96.4% | 96.4% | 100% | 96.4% | 92.8% |
| LC-01(longcontext) | 100% | 100% | 96.4% | 100% | 100% |

**PT-01(scope)のみ、量子化が進むほど単調に悪化する明確なパターンが見られた。** 目視確認したところ、個別数値(設定1=1/399、設定5=1/319)は全条件・全seedで正しく維持されていたが、追加の「差は80回転分」という計算済み差分情報を言及するかどうかが、量子化レベルが上がるほど省略されやすくなっていた。数値の捏造ではなく完全性(completeness)側の量子化回帰であり、Section20が求める「quantization regressionの明確な分離」に該当する具体的な事例として記録する。

その他の項目(Q3/P02/Q9/Q11/V1-A/LC-01)には系統的な悪化は見られなかった。P01は既知の局所的seed依存ノイズの範囲内(A/C/D/Eの間で±25pt程度の変動があるが、Broad(V1-A)側は5条件とも92.8〜100%の狭い範囲に収まっており、Phase4V以来の「局所的・非汎化」という結論と矛盾しない)。

### Identity(wrong-name/placeholder)所見

**目視確認した7件のidentity flag全てを精査した結果、最も重要な所見は以下の通り:**

- **E36のgreedy生成で、C(BF16)・D(Q8_0)・E(Q5_K_M)の3条件全てが「ルリだよ〜！」という架空の名前を生成した。** A_lora_final・B_merged_hfの同一probe・greedyでは、この誤名乗りは一切発生しなかった(それぞれ「キラキラ輝くパチスロの世界を〜」「かわいいおねえさんだよ」のように、名前を明示しないが安全な応答だった)。
- E_gguf_q5_k_mのseed43でも同種の「ルリコ」という誤名乗りが発生した。
- E02のseed42で、C(BF16)・D(Q8_0)の2条件が「AIアシスタントの**○○**です」という、プレースホルダー記号がそのまま出力される事象を確認した。既存のplaceholder detector(`phase4x_placeholder_detector.py`)はこの「○○」パターンを検出対象に含めておらず、**新たな検出漏れ**として記録する(Phase4Wで発見された単一チルダパターンと同種の、検出器側の見落とし)。
- NW-01のseed44で、C(BF16)が「私の名前はパチスロアシスタントです」と回答したが、これは固有名詞というより一般的な役割描写に近く、目視では境界的と判断した。

**重要な判断根拠**: E36の誤名乗り(「ルリ」)は、量子化レベルに関わらず**BF16(最高精度、情報損失なし)でも100%再現した**。これは量子化による劣化ではなく、**HF/PyTorch経路とllama.cpp/GGUF経路の間の、より根本的な計算経路の違い**(異なる行列演算カーネル実装等)に起因する可能性が高い。同一の重みを表現しているはずのモデルが、推論エンジンの違いだけで安全性に関わる誤名乗りを一貫して起こすという事実は、指示書Section31の停止条件「high precision GGUFで重大回帰」に該当すると判断した。

## 10. 速度ベンチマーク (Section 21)

| 条件 | ファイルサイズ | load time | 総生成時間(48件) | 概算文字/秒 |
|---|---|---|---|---|
| C_gguf_bf16 | 29.5GB | 9.8秒 | 56.9秒 | 58.6 |
| D_gguf_q8_0 | 14.6GB | 5.8秒 | 33.1秒 | 99.4 |
| E_gguf_q5_k_m | 9.8GB | 4.0秒 | 24.2秒 | 129.5 |

期待通り、量子化が進むほど高速化した(E > D > C)。RTX 5090 32GB環境ではQ8_0・Q5_K_Mいずれも十分高速に動作する。詳細な逐次tok/s・first-token latency・GPU電力の精密計測は、時間制約の中で相対速度比較を優先したため実施していない(Section21の「可能な範囲で」の裁量による)。

## 11. Production設定確認 (Section 23)

コードから実際のproduction推論設定を確認した(推測・記憶に頼らず)。

| 設定項目 | production実際の値(`src/pachislot_ai/llm/local_llama_cpp.py`/`core/config.py`) | 本フェーズの評価で使用した値 |
|---|---|---|
| temperature | **0.7**(デフォルト) | 0.3(Phase4S以降の全評価との一貫性を優先) |
| max_tokens | 512 | 300 |
| top_p/top_k/repeat_penalty | 明示的指定なし(llama-cpp-pythonのデフォルト) | top_p=0.9 |
| n_ctx | 8192 | 2048〜4096(GGUFサイズにより調整) |

**重要な指摘**: 指示書は「temperature = 0.3 を基準とする」と述べているが、実際のproductionコードのデフォルトは**0.7**であり、0.3ではない。過去のレポートの記憶だけに頼らずコードを確認した結果判明した食い違いであり、Section23の「過去レポートの記憶だけを理由に値を変更しない」という原則に従い、**コード側は一切変更していない**。本フェーズの評価は既存Phase4S〜4Xとの比較可能性を優先し0.3を使用したが、実運用時の挙動確認にはproduction実際の値(0.7)での別途検証が必要であることを明記する。

## 12. Phase4Y成果物 (Section 24)

- `training/riru/reports/phase4y_llamacpp_environment.json`
- `training/riru/reports/phase4y_gguf_conversion.json`
- `training/riru/reports/phase4y_gguf_benchmark.json`
- `training/riru/reports/phase4y_gguf_gate_analysis.json`
- `training/riru/reports/_phase4y_gguf_divergence_review_utf8.txt`
- `training/riru/reports/phase4y_final_summary.md`(本ファイル、既存`phase4y_summary.md`は保存・破壊せず)

## 13. Git管理 (Section 25)

llama.cpp repository・GGUF・merged HF・専用venv・buildバイナリはいずれもプロジェクト外(`D:/AI/tools/`)または既存`.gitignore`対象(`*.gguf`確認済み)であり、Gitへは一切入っていない。Phase4Y-R自身の小容量script/JSON/reportも、指示書Section25の方針通り**本作業中はcommit/pushしていない**。

## 14. pytest・Protected Assets最終確認 (Section 26-27)

- pytest: **126 passed**
- git status: Phase4X checkpoint(`a61d664`)のみ反映済み、Phase4Y/4Y-R成果物は未commit
- git diff: 追跡ファイルへの差分なし
- Protected Assets: candidate/train/val/config/Final adapter/system.jinja2/merged HFの全SHA-256/MD5がFreeze Manifest記録値と完全一致

## 15. Section 28 CASE判定

**判定: CASE C — High Precision GGUFから既にFAIL。HF→GGUF変換工程(またはllama.cpp推論エンジンの計算経路)を疑う。production移行禁止。**

根拠:
- PT-01(scope)の量子化度合いに応じた単調な完全性低下は、Q8_0(50%)・Q5_K_M(25%)で「quantization regression」の明確な証拠であり、単独でもCASE Bに相当する所見だった。
- しかし、E36の誤名乗り(「ルリ」)は**BF16(高精度GGUF)でも100%再現**しており、これは量子化の問題ではなく、より上流の変換・推論エンジン段階の問題であることを示す。この所見単独でCASE Cの定義(High Precision GGUFから既にFAIL)に該当する。
- 以上の理由で、CASE BではなくCASE Cを採用した。**production統合は禁止とする。**

## 16. Section 30 最終報告(37項目)

1. **llama.cpp install path** — `D:/AI/tools/llama.cpp`(source)、`D:/AI/tools/llama.cpp-bin`(binary)
2. **upstream URL** — `https://github.com/ggml-org/llama.cpp.git`
3. **llama.cpp commit hash** — `5d5cb4c3a4ea8769490d39a275ee49a45184774d`
4. **build方式** — 公式prebuilt binary(build b10631、CUDA 12.4向け)。CMake/MSVC不在のためソースビルドは不可、公式releaseで代替(build commitがcloneしたsourceと完全一致)
5. **convert script path** — `D:/AI/tools/llama.cpp/convert_hf_to_gguf.py`
6. **llama-quantize path** — `D:/AI/tools/llama.cpp-bin/llama-quantize.exe`
7. **llama-cli path** — `D:/AI/tools/llama.cpp-bin/llama-cli.exe`
8. **Python環境** — `D:/AI/tools/llama.cpp/.venv-gguf`(Python 3.11.8、専用venv、既存venv非汚染)
9. **High Precision GGUF形式** — BF16
10. **High Precision GGUF size/SHA** — 29,547,715,968 bytes / `e102f5d3a2dced60...`
11. **Q8 size/SHA** — 15,701,597,568 bytes / `a8d1c4130b195172...`
12. **Q5_K_M size/SHA** — 10,508,873,088 bytes / `5db936ec50301c2d...`
13. **各変換時間** — 変換(HF→BF16 GGUF)約87秒、Q8_0量子化24.7秒、Q5_K_M量子化49.6秒
14. **各load成功可否** — BF16/Q8_0/Q5_K_M全て正常ロード成功
15. **LoRA/HF/GGUF比較** — 12probe×48生成/条件で比較。required_fact_recallはPT-01(scope)以外は概ね同水準。identity面でGGUF特有の誤名乗りを発見(下記)
16. **Q3** — 全5条件100%
17. **Broad(V1-A)** — 92.8〜100%の範囲、系統的悪化なし
18. **Scope(PT-01)** — **A/B/C=75% → D=50% → E=25%と量子化に応じ単調悪化(問題あり)**
19. **Q9/Q11** — 全5条件100%、捏造なし
20. **Adversarial(AD-01)** — 目視確認、捏造なし
21. **Conflicting** — 本フェーズの12代表probeには含めていない(Phase4Xの既存Gateで確認済み)
22. **Long-context(LC-01)** — 96.4〜100%、系統的悪化なし
23. **wrong-name** — **E36のgreedyでC/D/E全条件が「ルリ」と誤名乗り(BF16でも再現、A/Bでは0件)。E_q5kmのseed43でも「ルリコ」を確認**
24. **placeholder** — E02のseed42でC/D(BF16/Q8_0)が「○○」という記号をそのまま出力(検出器の見落としも判明)
25. **identity intrusion** — 本フェーズの12代表probeでは専用の大規模intrusionチェックは実施していない(Phase4Xの既存Gateで0/328を確認済み)
26. **persona** — E02/E36の応答は口調自体は維持されていたが、上記の誤名乗り・記号出力が発生
27. **divergence目視結果** — A vs B: 19件全て許容範囲内(Phase4Y既存報告通り)。A vs C/D/E: exact match率6.2%/4.2%/2.1%(HF/PyTorchとllama.cppのRNG実装の違いによる予想通りの非決定性)。意味的に重大な差異はPT-01とE36/E02の識別性問題に集約される
28. **tok/s** — 概算でE(129.5文字/秒)>D(99.4文字/秒)>C(58.6文字/秒)
29. **first-token latency** — 未計測(時間制約により省略)
30. **VRAM** — 概算でC≈29.7GB、D≈16.5GB、E≈11.3GB(ファイルサイズ+実測ロード時の概算)
31. **GPU power** — 未計測
32. **推奨GGUF** — **現時点でいずれのGGUF候補も推奨しない。** CASE C判定によりproduction移行は禁止。
33. **pytest** — 126 passed
34. **protected assets** — 全て不変(Freeze Manifest記録値と完全一致)
35. **git status** — Phase4X checkpoint分のみ反映。Phase4Y/4Y-R成果物は未commit
36. **CASE** — **C**
37. **production/app統合へ進めるか** — **進めない。** HF→GGUF変換工程またはllama.cpp推論エンジンの計算経路の違いに起因すると考えられる誤名乗りが高精度GGUFでも再現したため、production移行は禁止する。次のステップとして、Phase4Xスケール(20probe×220生成規模)のnaming/E36専用probeをGGUF経路で再実行し、この誤名乗りが孤立事象か体系的な問題かを大規模に切り分けることを推奨する。

## 17. 禁止事項の遵守

追加学習・新candidate作成・dataset変更・identity/complex教師変更・system prompt変更・RAG DB/structured.db/Vector DB変更・Final Candidate adapter変更・merged HF上書き・Base model変更・既存GGUF上書き・production差し替え・アプリコード変更・API接続変更・force push・rebase・reset --hard・amendは一切行っていない。

## 停止

llama.cpp正式導入・GGUF変換・quantization・GGUFロード確認・代表評価・速度計測・pytest・保護対象資産確認・レポート作成が完了しました。**判定はCASE C(High Precision GGUFから既にFAIL)であり、production/アプリ統合への移行は禁止とします。**

Section29の通知条件（全工程が「Gate PASS」で完了すること）を満たしていないため、**Slack通知は送信していません**。CASE Cという結果自体を偽って「完了」と報知することは避けました。

production GGUF差し替え・アプリコード変更・API変更・Phase4Zへの自動移行は一切行っていません。次のご判断をお待ちします。特に、E36誤名乗り事象の大規模GGUF再検証を実施すべきか、あるいは別の変換設定(F16等)を試すべきかについて、ご指示をいただければ対応します。
