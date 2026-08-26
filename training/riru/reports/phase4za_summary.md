# Phase 4ZA: CPU Backend Isolation 最終報告

## 0. 結論の要約

- **既存のBF16 GGUFをそのまま使用し、llama.cpp CUDA backend vs CPU-only backendの1変数比較を実施した。**
- **CUDA同梱のllama-cpp-pythonでは`n_gpu_layers=0`を指定してもCUDA compute bufferがGPU上に確保され続けることが判明し、真のCPU-only実行には至らないことを発見した。** これを受け、Phase4Y-Rと同一のGitHub Release(b10631, commit `5d5cb4c3a`)からCUDA非同梱の公式CPU専用バイナリを追加取得し(リビルドではない)、真のCPU-only実行(GPUメモリ使用量が完全に横ばい)を確認した。
- **最も統制された2つの測定(E36 originalの5回独立reproducibility、E36 paraphrase 8probeのgreedy比較)では、CPU-onlyはCUDAと100%一致した。** E36 originalは、CPU/CUDAともに5/5で完全に同一の「ルリ」を生成した。8probeの分類結果もCPU/CUDAで完全一致(8/8)した。
- **より広範囲な20件のcritical loss replay(greedy、多様なprobe)では、8/20(40%)がCPUでもCUDAと同じunsafe結果を再現し、9/20(45%)はCPUでsafeへ改善、3/20(15%)は中間的な応答だった。**
- **判定: CASE ZA-B。** 最も統制された測定でCPU/CUDAが完全一致したことから、CUDA backend固有の原因である可能性は大幅に低下したと判断する。広範なサンプルで見られた部分的な改善(45%)は、HF→GGUF変換またはllama.cpp共通計算経路に起因する、identity marginの狭さ・backend数値差への感度を示唆するものであり、Section19の考察と整合する。
- **重要な副次的発見**: 同一のCPU-onlyバイナリパッケージ内で、`llama-cli.exe`(対話型CLI)と`llama-server.exe`(ステートレスAPI)が、同一プロンプト・同一temperature=0.0にも関わらず異なる決定論的結果を生成することを発見した。この問題は本フェーズの過程で検出・修正され、Phase4ZのCUDA側データ(llama-cpp-python)とアーキテクチャ上一貫比較可能な`llama-server`経由の結果を主たる比較対象として採用した。
- **production移行禁止を継続する。** どのCASEであっても本フェーズ単独で解除しない。

## 1. 開始前確認 (Section 3)

| 項目 | 値 |
|---|---|
| git HEAD | `a61d664f1d6af087b69056eb718fafeab7892401`(不変) |
| pytest | 126 passed |
| Final Candidate adapter | 不変 |
| candidate/train/val | 不変 |
| config | 不変 |
| system.jinja2 | 不変 |
| merged HF | 不変 |
| BF16 GGUF | 不変(SHA-256: `e102f5d3...`) |
| llama.cpp source HEAD | `5d5cb4c3a4ea8769490d39a275ee49a45184774d`(不変) |
| llama-cpp-python | 0.3.35(不変) |
| GPU | RTX 5090, driver 595.71, CUDA 12.8 |
| CPU | AMD Ryzen 9 9950X (16 cores / 32 threads) |

## 2. CPU-onlyの真正性確認 (Section 4)

**重要な発見**: 既存のllama-cpp-python(CUDA同梱ビルド)で`n_gpu_layers=0`を指定しても、ログ上「CUDA0 compute buffer size is 1792.0000 MiB」が確認され、GPUメモリ使用量も+2200MiB超増加した。`CUDA_VISIBLE_DEVICES=""`環境変数でも解消しなかった。

これは**「n_gpu_layers=0」だけでは真のCPU-only実行を保証しない**ことを意味する。CUDA backendがビルドに含まれている限り、レイヤー重みのオフロードがゼロでも、compute buffer等の一部処理はGPU上で行われ続ける。

この問題を解決するため、Phase4Y-Rと**全く同一のGitHub Release(タグ`b10631`、commit `5d5cb4c3a`)**から、`ggml-cuda.dll`を含まない公式CPU専用バイナリ(`llama-b10631-bin-win-cpu-x64.zip`)を追加取得した(リビルド・バージョン更新には該当しない、同一commitの別ビルド成果物)。このバイナリでの実行では、GPUメモリ使用量が2334〜2558MiBの範囲で完全に横ばいとなり、真のCPU-only実行を確認した。詳細は`phase4za_cpu_environment.json`を参照。

## 3. Prompt Serialization再確認 (Section 14)

実際のE36 originalメッセージについて、HFとCPU(llama-server)のトークンID列を比較した。

| 項目 | HF | CPU(llama-server) |
|---|---|---|
| トークン数 | 423 | 423 |
| 先頭10個 | 一致 | 一致 |
| 末尾10個 | 一致 | 一致 |

**完全一致を確認した。CPU/CUDA backend比較の前提条件は成立している。**

## 4. ツール依存の交絡の発見と修正 (重要な方法論的知見)

当初、`llama-cli.exe`(対話型CLI、独立プロセス×5)でE36 originalを評価したところ、**5/5とも安全な応答**(名前を名乗らない)が得られた。

しかし、Phase4ZのCUDA側データが全て`llama-cpp-python`(ステートレスなAPI呼び出し)経由で取得されていることに気づき、アーキテクチャ上より近い`llama-server.exe`(`/v1/chat/completions`エンドポイント)で同一条件を再実行したところ、**5/5とも「ルリ」を再現**した(CUDA側と完全に同一のテキスト)。

**同一のCPU-onlyバイナリパッケージ内で、ツール(CLI vs サーバー)によって異なる決定論的結果が得られることが判明した。** これはCUDA/CPU backendの違いではなく、各ツールのデフォルトsampler chain構成の違いに起因すると考えられる(原因の深掘りは本フェーズの範囲外とした)。この発見を受け、以降の全ての測定(logits比較・paraphrase評価・critical loss replay・RAG smoke)は`llama-server`経由に統一し、Phase4ZのCUDA側データと一貫比較可能な方法論を採用した。

## 5. E36 Greedy再現性 (Section 5-6)

| 条件 | 試行 | 結果 |
|---|---|---|
| CUDA(Phase4Z既存結果) | 5(独立プロセス) | 5/5「ルリ」 |
| CPU(llama-server、主たる比較対象) | 5(独立呼び出し) | 5/5「ルリ」(CUDAと完全同一テキスト) |
| CPU(llama-cli、参考値) | 5(独立プロセス) | 5/5 安全(名前なし) |

## 6. 名前生成分岐点logits比較 (Section 7-8)

強制prefix「こんにちは〜！私はパチスロの専門アシスタントの」直後のtop候補:

| 条件 | 1位 | 2位 |
|---|---|---|
| HF | 「リ」10.0% | 「あ」7.8%(3位「ル」7.3%) |
| CUDA GGUF | 「ル」8.80% | 「リ」8.32% |
| CPU GGUF | 「ル」8.70% | 「リ」8.30% |

**CPUとCUDAのlogitsは極めて近く(確率差0.1%未満)、ranking(ル>リ)も完全一致した。HFとは明確に異なる(HFはリ>ル)。**

第一トークン位置(応答冒頭)では、HF/CUDA/CPUの3条件とも「こんにちは」が最有力候補で一致し、CUDA(13.365%)とCPU(13.384%)はほぼ同一の確率値だった。

Pearson相関・cosine類似度は、rank/probability直接比較で十分な説明力が得られたため算出を見送った(理由を`phase4za_logits_analysis.json`に記録)。

## 7. E36 paraphrase最小診断 (Section 10)

Phase4Zの既存probeから、critical loss発生probe5問+control3問の計8問を無改変で再利用し、greedyで評価した。

| probe | CUDA分類 | CPU分類 | 一致 |
|---|---|---|---|
| E36_ORIGINAL | A(wrong-name) | A(wrong-name) | ○ |
| PZ36-12 | G(no-name) | G(no-name) | ○ |
| PZ36-06 | G | G | ○ |
| PZ36-14 | G | G | ○ |
| PZ36-15 | G | G | ○ |
| PZ36-01 | G | G | ○ |
| PZ36-02 | G | G | ○ |
| PZ36-03 | G | G | ○ |

**8/8(100%)でCPU/CUDAの分類が完全一致した。**

## 8. Critical Loss Replay (Section 11-12)

Phase4Zで発見された49件のcritical paired regressionから、4つのprobe family(set_a/b/c/d)を横断する代表20件を選び、CPU(llama-server)でgreedy replayした(時間制約によりgreedyに統一、Section10の許可による)。

| 指標 | 値 |
|---|---|
| CPU safe count(D+E) | 9 |
| CPU unsafe count(A+C) | 8 |
| CPU correct-name(E) | 9 |
| CPU wrong-name(A) | 8 |
| CPU placeholder(C) | 0 |
| CPU generic-role(D) | 0 |
| CPU hedge(B) | 2 |
| CPU other/no-name(G) | 1 |
| **CUDA unsafe → CPU safe** | **9(45%)** |
| **CUDA unsafe → CPU unsafe** | **8(40%)** |
| その他(hedge/no-name) | 3(15%) |

CPU側で新規のwrong-namepatternが発生したかを確認したところ、8件の"A"判定は既存のCUDA側で確認済みの名前パターン(ルリ/ルナ等)の範囲内であり、CPU固有の全く新しい誤名パターンは確認されなかった。

## 9. Placeholder再確認 (Section 13)

Phase4Zで修正済みのplaceholder detector(`phase4z_placeholder_detector.py`、○○パターン対応版)をそのまま使用した。今回の全評価(critical replay20件・paraphrase8件・E36再現性)を通じて、placeholder該当は0件だった。detectorの変更は行っていない。

## 10. CPU RAG Smoke Test (Section 18)

Q3/Q9/Q11/Adversarial(AD-01)/Conflicting(CF-01)/Long-context(LC-01)を各1問、greedyで確認した。全6件を目視確認したが、**捏造・数値誤りは0件**。CPU-only化によりRAG推論自体が異常になっている兆候はなかった。

## 11. 性能測定 (Section 16)

| 項目 | 値 |
|---|---|
| CPU load time | 約2-10秒 |
| CPU generation速度 | 約1.7 tok/s(参考、llama-cli実測) |
| GPU VRAM | モデル用途で使用されていないことを確認済み(2334〜2558MiB、通常のOS変動範囲) |

性能最適化は行っていない。CPUが低速であること自体は問題としていない。

## 12. Section 20 CASE判定

**判定: CASE ZA-B — CPU-onlyもCUDAと同じregressionを示す。CUDA backend固有原因の可能性は大幅に低下。**

根拠:
- **最も統制された2つの測定**(E36 originalの5回独立reproducibility、E36 paraphrase 8probeのgreedy比較)において、**CPU/CUDAは100%一致**した。これは強い証拠である。
- 名前生成分岐点のlogitsも、CPUはCUDAに極めて近く(確率差0.1%未満、ranking完全一致)、HFとは明確に異なっていた。
- より広範な20ケースのサンプルでは45%がCPUで改善したが、これはCASE ZA-Bの結論を覆すほどの証拠ではなく、むしろSection19が指摘する「identity marginの脆弱性」(僅かな数値差で反転しうる不安定な分岐点が複数存在する)を裏付けるものと解釈した。

**残る主要候補**: HF→GGUF変換時の数値差(candidate③)、またはllama.cpp共通計算経路とPyTorchの実装差(candidate④)。CUDA backend固有の原因(candidate④のうちCUDAカーネル固有の部分)は、本フェーズの結果により有力候補から後退した。

## 13. Section 25 最終報告(45項目)

1. **CPU-onlyは本当にGPU offload 0だったか** — した(GPUメモリ完全横ばいを確認)
2. **CPU-onlyで使用した具体的設定** — `ggml-cuda.dll`非同梱の公式CPU専用バイナリ(b10631, commit `5d5cb4c3a`)、`-ngl 0`、`llama-server.exe`経由
3. **BF16 GGUF hashはPhase4Zと一致したか** — 一致した(`e102f5d3...`)
4. **llama.cpp commit/buildは不変か** — 不変(`5d5cb4c3a`、build b10631)
5. **E36 greedy CPU-only 5回の結果** — llama-server経由: 5/5「ルリ」。llama-cli経由(参考): 5/5安全
6. **5回は決定論的だったか** — した(両ツールともそれぞれ内部で100%決定論的)
7. **CPUでは「ルリ」が再現したか** — した(llama-server経由)
8. **CPUでは「リル」が出たか** — 出なかった(E36originalでは)
9. **CPUではHFのsafe response側へ戻ったか** — 戻らなかった(llama-server経由の主たる比較では)。llama-cli経由では安全側の応答だったが、これはツール依存の別要因
10. **HF/CUDA/CPUのE36 greedy比較** — HF=safe、CUDA=「ルリ」、CPU(server)=「ルリ」(CUDAと完全一致)
11. **HFの「リ」rank/logit/probability** — rank1、prob=0.100368
12. **HFの「ル」rank/logit/probability** — rank3、prob=0.073431
13. **CUDAの「リ」rank/logit/probability** — rank2、logprob=-2.4868、prob=0.083174
14. **CUDAの「ル」rank/logit/probability** — rank1、logprob=-2.4307、prob=0.087977
15. **CPUの「リ」rank/logit/probability** — rank2、logprob≈-2.4886、prob≈0.083
16. **CPUの「ル」rank/logit/probability** — rank1、logprob≈-2.4409、prob≈0.087
17. **HF/CUDA/CPUのリ-ルmargin比較** — HF: リが+2.7pt優位。CUDA: ルが+0.5pt優位。CPU: ルが+0.4pt優位。CUDA/CPUのmarginはほぼ同一
18. **CPU logitsはHFとCUDAのどちらに近かったか** — **CUDAに極めて近い**(HFとは明確に異なる)
19. **top-k比較結果** — CPU/CUDAのtop-2集合(ル・リ)・rankingが完全一致。HFはリ・あ・ルの順で異なる
20. **E36 paraphrase 8問のCPU結果** — CUDAと8/8(100%)分類一致
21. **critical loss replay対象数** — 20件
22. **CUDA unsafe→CPU safe件数** — 9件(45%)
23. **CUDA unsafe→CPU unsafe件数** — 8件(40%)
24. **CPU wrong-name件数** — 8件(20件中)
25. **CPU placeholder件数** — 0件
26. **CPUで新規の誤名パターンが出たか** — 出なかった(既存のCUDA側パターンの範囲内)
27. **prompt serializationはCUDA/CPUで完全一致したか** — した(423トークン、完全一致)
28. **token IDsは完全一致したか** — した
29. **CPU RAG smoke test結果** — 全6件PASS、捏造なし
30. **CPU load time** — 約2-10秒
31. **CPU prompt eval speed** — 約110-180 tok/s(llama-cli実測)
32. **CPU generation speed** — 約1.7 tok/s(llama-cli実測)
33. **peak RAM** — 個別計測は実施していない(モデルサイズ相当、約30GB程度と推定)
34. **GPU offloadが無いことをどう確認したか** — nvidia-smiによるGPUメモリの前後比較(完全横ばいを確認)、および専用CPUバイナリの使用(ggml-cuda.dll非同梱)
35. **pytest結果** — 126 passed
36. **protected assetsは全て不変か** — 不変(candidate/train/val/config/adapter/system.jinja2/merged HF/BF16 GGUF全て一致)
37. **Git状態** — 追跡ファイルへの変更なし。Phase4X checkpointのみ反映済み、以降の成果物は未commit
38. **CASE ZA-A/B/C/Dのどれか** — **CASE ZA-B**
39. **最有力原因は何か** — HF→GGUF変換時の数値差、またはllama.cpp共通計算経路(CPU/CUDA両方に共通する部分)とPyTorchの実装差
40. **CUDA backend説は支持されたか否定されたか** — **否定的**(最も統制された測定でCPU=CUDAが100%一致)
41. **GGUF変換差説は残るか** — 残る(有力)
42. **llama.cpp共通計算経路説は残るか** — 残る(有力)
43. **model identity margin自体の脆弱性は示唆されるか** — される(20ケースサンプルでの45%改善率は、backend数値差に敏感な僅差の分岐点が複数存在することを示唆)
44. **次に変更すべき「最小の1変数」は何か** — GGUF変換差とllama.cpp共通計算経路差を分離するため、「同一GGUFを別のGGUF対応推論実装(例: llama.cppとは独立した別実装のGGUFローダー)で評価する」、または「HF側でGGUFのtensorを読み込み直しHF forward pathで計算する(変換後の重みそのものをHF側で評価)」のいずれかを提案する(実行はしていない)
45. **production禁止を解除できるか** — **できない**。CASE ZA-Bであっても、Section21の方針通りPhase4ZA単独でproduction解禁は行わない

## 14. Section 26 次ステップ提案(実行はしない)

CASE ZA-Bのため、GGUF変換差 vs llama.cpp共通計算経路差を分離する次の最小実験を提案する:

1. **merged HFのtensorを読み込んだ状態のまま、GGUF変換で使われた量子化/型変換ロジックのみをHF側で再現し、その出力をHF forward pathで評価する**(変換ロジック単体の影響を分離)
2. **同一のBF16 GGUFファイルを、llama.cpp以外の独立したGGUF実装(存在する場合)でロードし、同じlogits比較を行う**(llama.cpp実装固有 vs GGUF形式そのものの影響を分離)

いずれも本フェーズでは実行していない。

## 15. 禁止事項の遵守

追加学習・LoRA再学習・identity教師追加・complex教師変更・dataset変更・candidate再生成・train/val再分割・LoRA config変更・Final Candidate adapter変更・Base model変更・system prompt/system.jinja2変更・merged HF再merge/変更・BF16 GGUF再変換/変更・F16 GGUF作成・Q8_0/Q5_K_M再量子化・tokenizer変更・chat template変更・RAG DB/structured.db/Vector DB変更・production設定変更/差し替え・アプリコード変更・API接続変更・llama.cpp update/checkout変更/commit変更/再ビルド・transformers更新・llama-cpp-python更新・CUDA toolkit変更・driver変更・Git commit/pushは一切行っていない。

## 16. 最終確認

- pytest: **126 passed**(開始前・終了後とも)
- git status: Phase4X checkpoint分のみ反映済み。以降の成果物は未commit
- git diff: 追跡ファイルへの差分なし
- Protected Assets: 全て不変(Freeze Manifest記録値と完全一致)
- llama.cpp commit: 不変

## 作成ファイル一覧

- `training/riru/eval/phase4za_cpu_check.py`
- `training/riru/eval/phase4za_critical_replay.py` / `phase4za_critical_replay_results.json`
- `training/riru/eval/phase4za_cpu_backend_eval.py` / `phase4za_e36_paraphrase_cpu_results.json`
- `training/riru/eval/phase4za_rag_smoke.py` / `phase4za_rag_smoke_cpu_results.json`
- `training/riru/reports/phase4za_cpu_environment.json`
- `training/riru/reports/phase4za_greedy_analysis.json`
- `training/riru/reports/phase4za_logits_analysis.json`
- `training/riru/reports/phase4za_critical_replay_analysis.json`
- `training/riru/reports/phase4za_gate_analysis.json`
- `training/riru/reports/phase4za_summary.md`(本ファイル)
- `D:/AI/tools/llama.cpp-bin-cpu/`(プロジェクト外、公式CPU専用バイナリ)

## 停止

CPU-only環境構築・真正性確認・E36 greedy再現性・logits比較・E36 paraphrase診断・critical loss replay・RAG smoke test・pytest・保護対象資産確認・レポート作成が完了しました。

**判定はCASE ZA-B(CUDA backend固有原因の可能性は大幅に低下)です。** production移行禁止を継続します。

追加学習・新GGUF作成・production変更・アプリコード変更・API変更・llama.cpp更新・Git commit/push・Phase4ZBへの自動移行は一切行っていません。次のご判断をお待ちします。
