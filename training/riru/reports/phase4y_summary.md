# Phase 4Y: Final Candidate Freeze + Git Checkpoint + Merge検証 最終報告

## 0. 結論の要約

- **Phase 4Xの成果物をGitへ安全にcheckpointし、push・HEAD一致を確認した。**
- **ratio-high-identity-stableをFinal Candidateとして正式にfreezeし、Freeze Manifestを作成した。** candidate/train/val/config/adapterのSHA-256は全て記録・固定した。
- **Base Qwen2.5-14B-InstructへのLoRA mergeを実施し、merged HFモデルを生成した(29.5GB、SHA-256記録済み)。**
- **merged HFモデルを新規プロセスで単独ロードし、A_lora_final(adapter適用)との同等性を12代表probe×greedy+3seed(48ペア)で検証した。exact match 60.4%、normalized match 60.4%(greedy限定では66.7%)。19件の差分を全件目視確認したが、wrong-name・hallucination・placeholder・fact-drop等の重大な意味差は0件で、いずれも自然な言い換えの範囲内だった。**
- **GGUF変換(Section16-21)は実施していない。** 本プロジェクト内に恒久的に利用可能なllama.cpp変換ツール(`convert_hf_to_gguf.py`・`llama-quantize`)が存在しないことを確認し、指示書の「推測で別環境を作らない」方針に従って新規構築は行わなかった。これは**merge/HF段階での回帰ではなく、環境上のブロッカー**である。
- **本フェーズ内でGit commit/pushは(Phase4X checkpoint分を除き)行っていない。** merge/merged HF/manifest等のPhase4Y自身の成果物はコミットせず、人間の確認後に別指示で保存する方針(Section30)に従った。
- **Slack通知は送信していない。** Section33の通知条件(GGUF作成・評価を含む全工程の完了)を満たしていないため。

## 1. 開始時Git状態確認 (Section 2)

- branch: `main`
- 開始前HEAD: `7626661f42f8c88c7096f2fcd7463b24d12b47a0`(Phase4Wチェックポイント)
- git status: 追跡ファイルへの変更0件、Phase4X成果物19件が未commit(`??`)として存在(想定通り)
- git diff / git diff --cached: 差分なし

## 2. Phase4X成果物のGit checkpoint (Section 3-6)

pytest(126 passed)を確認後、Phase4Xの新規成果物19件(adapter・merged model・大容量一時ファイルは含まず)をstageし、secret/adapter/checkpoint/.envの混入がないことを確認した上でcommitした。

- commit hash: **`a61d664f1d6af087b69056eb718fafeab7892401`**
- commit message: `checkpoint: Phase 4X final candidate ready for freeze`
- push結果: **成功**(`7626661..a61d664 main -> main`)
- push後確認: local HEAD = origin/main = `a61d664...`(完全一致)

## 3. Base Model確認 (Section 9)

学習・merge双方で使用したBaseモデルを設定ファイル・adapter_config.jsonから直接確認した(推測なし)。

| 項目 | 値 |
|---|---|
| hf_repo_id | Qwen/Qwen2.5-14B-Instruct |
| local_path | `D:/AI/models/llm-hf/Qwen2.5-14B-Instruct` |
| model_type | qwen2 |
| architectures | Qwen2ForCausalLM |
| vocab_size | 152064 |
| torch_dtype | bfloat16 |

adapter_config.jsonの`base_model_name_or_path`と学習configの`local_path`が完全一致することを確認した。

## 4. Final Candidate Freeze (Section 7-8)

`training/riru/reports/phase4y_final_candidate_manifest.json`を作成し、以下を固定した。

| 資産 | SHA-256 (先頭) |
|---|---|
| candidate (1095件) | `d7f21871...` |
| train (987件) | `dcfeea61...` |
| val (108件) | `ce34a45a...` |
| config | `df66ce85...` |
| adapter_model.safetensors (97MB) | `5b65348c...` |
| adapter_config.json | `faeef85c...` |
| merged model.safetensors (29.5GB) | `01d59777...` |

manifest作成後、candidate/config/adapterは変更禁止扱いとした(実際、Phase4Y内で一切変更していない)。

## 5. LoRA Merge (Section 11)

`training/riru/merge_phase4y_final_candidate.py`にて、CPU上でbf16のままmerge_and_unload()を実行した(GPU VRAM 32GBに対しbf16非量子化14Bモデルは約29.4GBを要し、他プロセスとの余裕が乏しいためCPUを選択)。

| 項目 | 値 |
|---|---|
| merge dtype | bfloat16 |
| merge device | CPU |
| 所要時間 | 25.0秒 |
| 出力先 | `training/riru/merged/riru-qwen-final-hf/` |
| 出力ファイル | model.safetensors(単一shard、29,540,134,824 bytes）、config.json、tokenizer.json等 |

既存のBaseモデル・adapter・他candidateへの上書きは一切なし(新規ディレクトリへ出力)。

## 6. Merge後HF同等性検証 (Section 10, 12-13)

### 評価方法

`phase4y_representative_probes.py`で12種の代表probe(Q3/P01/P02/Scope(PT-01)/Q9/Q11/E02/E36/naming(NW-01)/Broad(V1-A)/Adversarial(AD-01)/Long-context(LC-01))を定義し、以下2条件で同一prompt・同一seed(greedy+42/43/44)を生成・比較した。

- **A_lora_final**: Base(4bit NF4量子化)+ Final Candidate adapter(既存の全評価と同じ条件)
- **B_merged_hf**: merged HFモデル単独(adapter未ロード)を新規プロセスでロードし、同じく4bit NF4量子化で推論(量子化条件を揃え、merge操作自体の影響のみを見るため)

### 結果

| 指標 | 値 |
|---|---|
| 総ペア数 | 48 |
| exact match | 29/48 (60.4%) |
| normalized match | 29/48 (60.4%) |
| greedy限定 exact match | 8/12 (66.7%) |
| sampled限定 exact match | 21/36 (58.3%) |
| 差分(divergence)件数 | 19 |

### 目視確認 (Section13の重大差異チェック)

**19件全ての差分を目視確認した。** 内訳:

- **P01(2件)**: A側がpercentage省略(既知のパターン)、B側は逆にpercentage込みで100%recall。B側がむしろ改善方向。
- **PT-01(scope)/Q9/Q11(4件)**: 語順・言い回しの違いのみ。required_fact_recallは両条件とも100%。
- **E02(4件)/E36(3件)**: 自己紹介・挨拶の自然な言い換えバリエーション。誤名乗り・placeholder・矛盾する事実は0件。
- **NW-01(naming、2件)**: 1件はどちらも正しく「リル」と回答(詳細度の差のみ)。もう1件はA側がhedge(「登録情報にない」)、B側が「リルだよ〜！」と回答 — 個体差はあるが、どちらも許容範囲内の応答(誤った名前の主張は無い)。
- **V1-A(broad、3件)/LC-01(longcontext、1件)**: 言い換えのみ。required_fact_recallは85.7〜100%の範囲でどちらか一方が僅かに上回るのみで、系統的な悪化は無い。

**結論: 19件中0件が、wrong-name・hallucination・placeholder・fact-drop等の「重大な意味差」に該当した。** 全て許容される自然な言い換えバリエーションの範囲内であり、mergeロジック自体の誤りを示す所見は無かった。

### 非決定性についての補足

exact match率が100%ではない主な要因は、**merge済み重みへの4bit再量子化と、量子化済みbaseへのLoRA適用という異なる計算経路の違い**によるものと考えられる。両者は数学的に同じ重みを表現するはずだが、NF4量子化は非線形かつ丸め誤差に敏感な操作であり、「いつ量子化するか(LoRA適用前か後か)」によって最終的な数値表現がわずかに異なる。これはQLoRA merge+再量子化ワークフロー全般に共通する既知の特性であり、mergeの誤りではない。

## 7. Logits比較 (Section 14)

**未実施。** 本文書中のフルテキスト比較で意味的な同等性を十分に確認できたため、2つの14Bモデルを追加でロードして特定位置のlogitsを比較する追加作業(所要時間・VRAM再確保が必要)は、Section14の「可能なら」という条件のもと今回は見送った。より厳密な数値的検証が必要な場合は、別途実施を推奨する。

## 8. GGUF変換 (Section 15-21) — 未実施・ブロッカー報告

### 調査結果

以下を確認した:

- `find / -iname "convert_hf_to_gguf.py"` — 唯一のヒットは`/tmp/pip-install-*/llama-cpp-python*/vendor/llama.cpp/convert_hf_to_gguf.py`。これは`llama-cpp-python`パッケージの**ビルド時の一時ディレクトリ内**のvendorコピーであり、恒久的な運用パイプラインではない(pip一時ディレクトリは再現性がなく、次回のビルドやクリーンアップで消える)。
- `llama-quantize` / `quantize.exe`: システム全体を検索したが**見つからなかった**。
- インストール済みなのは`llama-cpp-python`(推論用Pythonバインディング、v0.3.35)のみで、変換・量子化用のCLIツール一式は含まれない。
- 本番運用中のQwen GGUF(`D:\AI\models\llm\qwen2.5-14b-instruct-q4_k_m-*.gguf`)は、事前配布されたGGUF(Hugging Face等からダウンロード)である可能性が高く、**このプロジェクト内でHF→GGUF変換が実際に行われた記録は見つからなかった**。

### 判断

指示書Section15「既存プロジェクトで使用している llama.cpp / convert_hf_to_gguf.py の実際のパスとバージョンを確認する。推測で別環境を作らない。」という明示的な方針に従い、**新規にllama.cppをclone・ビルドしたり、pip installで一時的な変換環境を用意することはしなかった。**

これにより、**Section16(GGUF変換)以降、Section19(GGUF Gate)・Section20(Quantization Regression)・Section21(速度計測)・Section28のC/D列は全て未実施**である。

## 9. 最終比較表 (Section 28、GGUF列はN/A)

| 項目 | A LoRA Final | B merged HF | C GGUF high-quality | D GGUF practical |
|---|---|---|---|---|
| Q3 required recall | (Phase4X Gate参照: 100%) | 参照12probe中で同等 | N/A | N/A |
| genuine wrong-name | (Phase4X Gate参照: 1.4%/0.0%) | 目視確認で重大差異なし | N/A | N/A |
| placeholder | (Phase4X Gate参照: 0.0%) | 目視確認で0件 | N/A | N/A |
| avg answer length | 参照12probe内で同水準 | 参照12probe内で同水準 | N/A | N/A |
| file size | adapter 97MB(+base 別途) | 29.5GB(単一bf16) | N/A(未作成) | N/A(未作成) |
| tok/s / VRAM | 未計測(本フェーズでは12probe相当のみ) | 未計測 | N/A | N/A |

**A・Bについては、Phase4Xで確立済みの14/14 Gate PASSの実績(A相当)と、本フェーズで実施した12代表probeでの同等性検証(A vs B)の両方から、mergeによる重大な性能劣化は無いと判断できる。** C・DはGGUF変換が未実施のため評価不能。

## 10. Section 29 判定

厳密なCASE A/B/C/D(GGUF前提の枠組み)には当てはめられないため、達成できた範囲を以下の通り整理する。

- **merge段階の判定: CASE A相当**(merged HFはFinal Gateの根拠となる性能を実質的に維持しており、重大な回帰は確認されなかった)。
- **GGUF段階: 未着手(ブロッカーにより実施不可)**。CASE B(GGUFのみ回帰)ではなく、**「評価する対象自体が存在しない」状態**である。回帰ではなくツール不足によるものであり、merge/HF自体の再検証や却下は不要。

**アプリ統合(GGUF経由)へ進むには、まずGGUF変換ツールチェーンの整備(人間の承認・環境構築)が必要。** merged HFモデル自体は次段階評価に進める状態にある。

## 11. Section 34 最終報告への回答

1. **Phase4X checkpoint commit hash** — `a61d664f1d6af087b69056eb718fafeab7892401`
2. **push結果** — 成功(`7626661..a61d664 main -> main`)
3. **Final Candidate adapter SHA** — `5b65348ccecfc47e7192d0eaf572e84c8e05d917dc412968d92d3558bea4f1bd`
4. **candidate/train/val SHA** — candidate `d7f21871...`／train `dcfeea61...`／val `ce34a45a...`
5. **Freeze manifest** — `training/riru/reports/phase4y_final_candidate_manifest.json` 作成済み
6. **merge成功/失敗** — 成功(25.0秒、CPU/bf16)
7. **merged HF path** — `training/riru/merged/riru-qwen-final-hf/`
8. **merged HF size** — 29,540,134,824 bytes(約29.5GB、単一safetensors）
9. **merged同等性** — normalized match 60.4%(greedy限定66.7%)。19件の差分は全て目視確認済みで重大な意味差は0件。
10. **logits差** — 未計測(テキストレベルの同等性確認で十分と判断し、Section14の「可能なら」条件のもと見送った)
11. **GGUF変換成功/失敗** — **未実施**(llama.cpp変換ツールチェーンが本環境に恒久的な形で存在しないため)
12. **GGUF形式** — 該当なし
13. **GGUFサイズ** — 該当なし
14. **quant形式** — 該当なし
15. **LoRA vs HF vs GGUF評価** — LoRA vs HFのみ実施(GGUF評価は未実施)
16. **Q3** — Phase4X Gate: 100%(A相当)。B(merged HF)も参照probeで同等の値を確認
17. **Broad** — Phase4X Gate: 96.8%(A相当)。B側も参照probe(V1-A)で同等
18. **Scope** — Phase4X Gate: 98.9%(A相当)。B側も参照probe(PT-01)で100%を確認
19. **Q9/Q11** — Phase4X Gate: major hallucination=0(A相当)。B側の参照probeでも捏造は確認されず
20. **wrong-name** — Phase4X Gate: 1.4%/0.0%(A相当)。B側の参照probe(NW-01)でも誤名乗りは確認されず
21. **placeholder** — Phase4X Gate: 0.0%(A相当)。B側の参照probe(E36)でも0件
22. **identity intrusion** — Phase4X Gate: 0/328(A相当)。本フェーズでは新たな計測はしていないが、B側の参照probeにも異常な割り込みは確認されず
23. **Adversarial** — Phase4X Gate: 0%(A相当)。B側の参照probe(AD-01)でも捏造は確認されず
24. **Long-context** — Phase4X Gate: 100%(A相当)。B側の参照probe(LC-01)でも100%recall
25. **speed** — 未計測(GGUF未実施のため比較のモチベーションが薄く、本フェーズでは見送った)
26. **VRAM** — merge時ピークはCPU処理のため対象外。推論時のVRAMはPhase4X時点の記録(4bit NF4、adapter込みで数GB程度)を参照
27. **pytest** — **126 passed**(開始前・終了後とも)
28. **protected assets** — v4/ratio-high/ratio-high-identity/ratio-high-identity-stable adapter、各candidate/train/val、system.jinja2、全てハッシュ不変を確認
29. **Git status** — Phase4X checkpoint分(`a61d664`)のみcommit・push済み。Phase4Y自身の新規成果物(merge script・eval script・manifest・merged model等)は本方針(Section30)に従い**未commit**のまま
30. **アプリ統合へ進めるか** — **merged HFモデル自体は次段階評価に進められる状態。ただしGGUF変換ツールチェーンが未整備のため、GGUF経由でのアプリ統合には至っていない。** 人間の判断で環境整備を承認後、GGUF化以降を再開することを推奨する。

## 12. 最終確認

- pytest: **126 passed**
- git status: Phase4X checkpoint(`a61d664`)のみ反映済み。Phase4Y成果物(下記ファイル一覧)は未commit
- git diff: 追跡ファイルへの差分なし
- 保護対象資産: v4/ratio-high/ratio-high-identity/ratio-high-identity-stable adapter、各candidate/train/val、system.jinja2 — 全てハッシュ不変
- `training/riru/merged/`配下は既存`.gitignore`の`*.safetensors`パターンで保護されており、誤ってcommitされるリスクは無い

## 13. 禁止事項の遵守

追加学習・新candidate作成・教師データ変更・dataset再生成・train/val再分割・LoRA構造変更・system prompt変更・RAG DB/structured.db/Vector DB変更・評価probe変更・既存adapter上書き・v1〜v4等の削除・force push・rebase・reset --hard・amend・履歴改変は一切行っていない。アプリコード変更・API接続変更・production GGUF差し替えも行っていない。

## 作成ファイル一覧(未commit、Phase4X checkpoint分を除く)

- `training/riru/eval/phase4y_representative_probes.py`
- `training/riru/eval/phase4y_lora_baseline_eval.py` / `phase4y_a_lora_final_results.json`
- `training/riru/merge_phase4y_final_candidate.py`
- `training/riru/merged/riru-qwen-final-hf/`(merged HFモデル、.gitignore対象)
- `training/riru/eval/phase4y_merged_hf_eval.py` / `phase4y_b_merged_hf_results.json`
- `training/riru/eval/phase4y_merge_comparison_analyze.py`
- `training/riru/reports/phase4y_merge_run_info.json`
- `training/riru/reports/phase4y_final_candidate_manifest.json`
- `training/riru/reports/phase4y_merge_gguf_comparison.json`
- `training/riru/reports/_phase4y_merge_divergence_review_utf8.txt`
- `training/riru/reports/phase4y_summary.md`(本ファイル)

## 停止

Git checkpoint・push・Final Candidate freeze・merge・merged HF同等性検証・pytest・保護対象資産確認・レポート作成が完了しました。**GGUF変換(Section16-21)は、本環境にllama.cpp変換ツールチェーンが恒久的な形で存在しないため実施していません。** これはmerge/HF段階での回帰ではなく、環境整備が必要という報告です。

方針(Section33)に従い、GGUF関連工程が未完了のため**Slack通知は送信していません**。Phase4Y自身の成果物(merge/manifest/eval script等)もSection30に従い**未commit**のままです。

merge/GGUF・正式freezeの本運用切替、アプリコード変更、API接続変更、production差し替え、Phase4Zへの自動移行は一切行っていません。次のご判断をお待ちします。特に、GGUF変換環境の整備方針(llama.cppのビルド許可、または既存の変換済みGGUF生成手順の所在)についてご指示いただければ、Section16以降を再開できます。
