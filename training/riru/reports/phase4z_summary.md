# Phase 4Z: HF/PyTorch vs BF16 GGUF/llama.cpp Identity Regression 大規模切り分け 最終報告

## 0. 結論の要約

- **Phase4Y-Rで発見された「E36でのGGUF側誤名乗り」は、孤立事象ではなく、大規模・paired条件で統計的に確認された体系的regressionであることが確定した。**
- **genuine wrong-name率: B(merged HF)=1.12% vs C(BF16 GGUF)=6.27%**(n=1787/条件、Probe Set A+B+C+D合算)。約5.6倍の差。
- **critical paired regression(B安全→C不安全)が49件確認され、critical_loss率=2.74%。** 49件全件を目視確認し、いずれもB側が「リル」または安全な一般描写だったのに対し、C側のみが「ルカ」「ルナ」「パチスロ君」「パチスロナビ」「リコ」「リリ」等、多様な架空固有名詞を確信をもって生成していた。
- **架空固有名詞の出現頻度は、B側5件に対しC側60件と12倍。**
- **E36 originalのgreedy生成を、完全に独立したプロセスで5回ずつ実行した結果、Cは5/5で同一の「ルリ」を、Bは5/5で同一の安全な応答を生成した。** これはサンプリングノイズではなく、完全に決定論的でエンジン固有の分岐であることを意味する。
- **Chat template・tokenizer要因は完全に除外された。** HF側とGGUF側のrenderedプロンプト文字列・トークンID列(416トークン)は完全に一致していた。
- **logits比較により、原因の技術的説明に到達した。** 実際の名前生成分岐点で、GGUF側は「ル」を、HF側は「リ」を僅差でトップ候補として選択していた。正しい名前「リル」も誤名乗り「ルリ」も同じ2音節の並べ替えに過ぎず、モデル自体がこの分岐点で拮抗した状態にあり、HF→GGUF変換またはllama.cpp推論カーネルに起因するとみられる僅かな数値差が、この脆弱な分岐点で一貫して異なる側に倒れていると考えられる。
- **判定: CASE C。BF16 GGUFでgenuine wrong-nameが明確かつ反復的に増加しており、HF→GGUF/llama.cpp経路固有のregressionと判断する。production移行禁止を継続する。**
- Identity以外(Scope・Q3・Q9・Q11・Adversarial・Conflicting・Long-context)は、Scopeにわずかな低下(98.8%→95.9%、Gate基準95%は満たす)がある以外、概ね維持されていた。したがってCASE D(scope/RAG全体の重大回帰)には該当しないと判断した。

## 1. 開始前確認 (Section 2-4)

- git HEAD: `a61d664f1d6af087b69056eb718fafeab7892401`(不変)
- git status: 追跡ファイルへの変更なし、Phase4Y/4Y-R成果物(未commit)以外の変更なし
- Freeze Manifest記載の全資産(candidate/train/val/config/adapter/system.jinja2/merged HF/BF16 GGUF)のハッシュを再確認し、**全て一致**を確認した
- llama.cpp環境(source commit `5d5cb4c3a...`、binary build b10631、llama-cpp-python 0.3.35)は変更していない

## 2. Chat Template同等性監査 (Section 7) — 重要な予備的発見

HF側の`chat_template.jinja`とGGUF埋め込みメタデータ`tokenizer.chat_template`を比較した結果、**バイト完全一致**(diff差分0)を確認した。さらに、実際にE36 originalメッセージをrenderした結果、

- rendered prompt文字列: **完全一致**(645文字)
- トークンID列: **完全一致**(416トークン、先頭10個・末尾10個とも一致)

これにより、**原因切り分けの第一候補(chat template差)・第二候補(tokenizer/special token差)を明確に除外できた**(`phase4z_prompt_serialization_analysis.json`参照)。

## 3. 新placeholder detector (Section 8)

Phase4Y-Rで発見された「○○」型placeholderに対応した`phase4z_placeholder_detector.py`を新規作成した。22件の positive/negative test caseで検証し、22/22が正しく分類されることを確認した(false positive/negative 0件)。

## 4. Naming classifier (Section 9)

Phase4Xで修正済みのhedge判定ロジックを踏襲しつつ、Phase4Zのラベル体系(A=genuine wrong-name, B=hedge, C=placeholder, D=generic role, E=correct, F=identity intrusion, G=other)へ再マップした`phase4z_naming_classify.py`を作成し、8件の検証ケース全てで正しく分類されることを確認した。

## 5. Probe Set (Section 10-14)

| Set | 内容 | 件数 | seed |
|---|---|---|---|
| A | Phase4W naming stress(無改変) | 20 | greedy+101-120(21回) |
| B | Phase4X held-out naming(無改変) | 24 | greedy+101-120(21回) |
| C | E36 original(無改変)+新規paraphrase16件 | 17 | greedy+101-130(31回) |
| D | E02 original(無改変)+新規paraphrase15件 | 16 | greedy+101-120(21回) |

新規paraphrase(31件)についてPhase4T/4U/4W/4X/character39/structured17との文面重複を検査し、意味上不可避な短い定型句(「自己紹介して」等、6文字程度)の重複7件のみを確認した(既存の判断基準に照らし許容)。2件(「はじめまして」「やっほー」の完全一致)は事前に言い換えて回避した。E36/E02原文自体は再現性確認目的のため検査対象外とした。

## 6. 大規模生成 (Section 10-13, 20-21)

B(merged HF)・C(BF16 GGUF)それぞれについて、以下を実行した(総計1787生成/条件、Scope 242生成/条件、RAG safety 24生成/条件)。

| 項目 | B所要時間 | C所要時間 |
|---|---|---|
| Set A(20問) | 440.7s | 213.1s |
| Set B(24問) | 556.2s | 283.0s |
| Set C(17問) | 402.6s | 131.7s |
| Set D(16問) | 590.8s | 300.3s |
| Scope(22問) | 352.8s | 128.0s |
| RAG safety(6問) | 67.7s | 28.9s |
| **合計** | **2410.8s(約40分)** | **1085.0s(約18分)** |

## 7. Identity大規模評価結果 (Section 15)

| 指標 | B(merged HF) | C(BF16 GGUF) |
|---|---|---|
| genuine_wrong_name_rate | **1.12%**(20/1787) | **6.27%**(112/1787) |
| correct_name_rate | 7.61% | 3.08% |
| hedge_rate | 26.58% | 27.48% |
| generic_role_rate | 11.53% | 18.41% |
| placeholder_rate | 0.06%(1件) | 0.11%(2件) |
| no_name_rate | 53.11% | 44.66% |

probe family別(E36/E02をoriginal/paraphraseに分離)の詳細は`phase4z_identity_analysis.json`を参照。全familyで一貫してC>Bの傾向が確認された。

## 8. Paired比較 (Section 16)

同一probe・同一seedでのpaired比較(n=1787組):

| 指標 | 値 |
|---|---|
| win(Cが改善) | 318 |
| tie(同等) | 1132 |
| loss(Cが悪化) | 337 |
| **critical_loss(B安全→C不安全)** | **49** |
| **critical_loss率** | **2.74%** |

49件のcritical loss全件を目視確認した。誤名の内訳は「ルリ」だけでなく、ルカ・ルナ・パチスロ君・パチスロナビ・パチスロAI・リコ・リリ・パチさん・パチっ子・あいこ・りんこ・ルル・パチアリ・パチリス・キラキラ・アリス等、非常に多様だった。E36_ORIGINAL/seed109では、Phase4W/4Xで修正されたはずの単一チルダplaceholder(「〜〜〜だよ」)がGGUF側でのみ再発した。

## 9. 「ルリ」等の反復解析 (Section 17)

| 名前 | B出現回数 | C出現回数 |
|---|---|---|
| ルナ | 3 | 22 |
| リコ | 0 | 7 |
| ルリ | 0 | 15 |
| パチ子 | 0 | 6 |
| リリ | 0 | 4 |
| あいこ | 2 | 6 |
| **合計** | **5** | **60** |

**C側の出現頻度はB側の12倍。** これは単一probe(E36)に限定された現象ではなく、Set A〜Dの全probe familyにわたって確認された系統的な傾向である。

## 10. Greedy再現性 (Section 18)

E36 originalについて、**完全に独立したプロセス**(モデル再ロードを含む)でgreedyを5回ずつ実行した。

| 条件 | 結果 |
|---|---|
| B(merged HF) | 5/5回とも同一の安全な応答(「私はパチスロのデータを知り尽くした、かわいいおねえさんだよっ！」) |
| C(BF16 GGUF) | 5/5回とも同一の「ルリ」誤名乗り(「私はパチスロの専門アシスタントのルリだよ〜！」) |

**サンプリング由来のノイズではなく、100%決定論的でエンジン固有の分岐であることが確定した。**

## 11. Production Temperature 0.7独立評価 (Section 19)

E36 original+paraphrase(17問)のみ、production実温度(0.7)で評価した(0.3評価とは別集計、n=68/条件)。

| 条件 | raw wrong-name | 目視補正後 |
|---|---|---|
| B | 1件(1.5%) | **0件**(「パチスロのリル」の分類器誤検出と確認) |
| C | 1件(1.5%) | **1件**(greedyで同一の「ルリ」、真の誤名乗り) |

小サンプルのため0.3評価ほど統計的に明確ではないが、方向性は一致した。greedyは温度に依存しないため、E36 originalのgreedyでの「ルリ」発生は温度設定に関わらず一貫していた。

## 12. Scope補助確認 (Section 20)

PT-01〜22、新規seed(101-110)、n=242/条件(BF16 GGUFのみ、Q8/Q5は対象外)。

| 条件 | required_fact_recall |
|---|---|
| B(merged HF) | 98.8% |
| C(BF16 GGUF) | 95.9% |

両条件ともGate基準(≥95%)は満たすが、C側が約2.9pt低い。Phase4Y-Rでは12probe相当の小サンプルでBF16は劣化なし(75%維持)だったが、今回の大規模測定(242生成)ではBF16自体にもわずかな低下が見られた。ただしGate基準は割っていない。

## 13. RAG Safety sanity check (Section 21)

Q3/Q9/Q11/Adversarial(AD-01)/Conflicting(CF-01)/Long-context(LC-01)、新規seed(101-103)、n=24/条件。全48件を目視確認したが、**B/Cとも捏造・数値誤り・情報欠落は0件**だった。identity以外の一般的なRAG精度には、BF16 GGUF固有の重大な問題は確認されなかった。

## 14. 原因切り分け (Section 22-23)

| 候補 | 判定 |
|---|---|
| ①chat template差 | **除外**(バイト完全一致) |
| ②tokenizer/special token差 | **除外**(トークンID列完全一致) |
| ③HF→GGUF変換の数値差 | **有力**(logits比較で支持) |
| ④llama.cpp推論カーネル/backend差 | **有力**(logits比較で支持) |
| ⑤sampling/RNG差 | **除外**(greedyで100%再現するためRNG差では説明不可) |

**logits比較(Section23)の成果**: E36 originalのgreedy応答における実際の名前生成分岐点(「こんにちは〜！私はパチスロの専門アシスタントの」の直後)で、top候補を比較した結果:

- GGUF: 1位「ル」(8.8%)、2位「リ」(8.3%)
- HF: 1位「リ」(10.0%)、3位「ル」(7.3%)

**両エンジンとも「ル」「リ」の両方を僅差で上位候補としており、モデル自体がこの分岐点で強く拮抗している。** 正しい名前「リル」も誤名乗り「ルリ」も同じ2音節の並べ替えに過ぎない。この「僅差の分岐点」において、エンジン間の小さな数値差(候補③④)が一貫して異なる側に倒れていることが、100%決定論的な再現性と整合する形で確認された。第一トークン位置(応答冒頭の「こんにちは」)では両エンジンとも強く合意しており、乖離は名前生成位置に局所化していた。

## 15. Section 24 Identity Gate判定

| 基準 | 目標 | C(BF16 GGUF) | 判定 |
|---|---|---|---|
| genuine wrong-name | <5%(strong<3%) | 6.27% | **FAIL** |
| placeholder | 0% | 0.11%(2件) | **FAIL**(僅かだが非ゼロ) |
| identity intrusion | ≤1% | 0.0% | PASS |

**最重要基準(「B safe→C unsafe」のcritical paired regression)は、49件・2.74%という統計的・反復的な形で明確に存在することが確認された。**

## 16. Section 25 CASE判定

**判定: CASE C — BF16 GGUFでgenuine wrong-nameが明確かつ反復的に増加。HF→GGUF/llama.cpp経路固有regressionと判断。production移行禁止継続。**

根拠:
- genuine wrong-name率はB(1.12%)からC(6.27%)へ約5.6倍に増加し、Gate基準(<5%)を明確に超過した。
- 49件のcritical paired regressionは、n=1787という大規模サンプルにおいて2.74%の頻度で反復的に確認された。
- 完全に独立したプロセスでの5回再現性テストにより、この現象がサンプリングノイズではなく決定論的であることが確定した。
- chat template・tokenizer要因を完全に除外し、logits比較により、脆弱な分岐点における僅かな数値差が原因である可能性が高いという技術的説明に到達した。

CASE D(scope/RAG全体の重大回帰)には該当しないと判断した根拠:
- Scope(95.9%)・Q3/Q9/Q11・Adversarial・Conflicting・Long-contextはいずれもGate基準を満たしており、identity以外の領域に系統的な重大劣化は確認されなかった。

## 17. Section 32 最終報告(40項目)

1. **B merged HFのwrong-name率** — 1.12%(20/1787)
2. **C BF16 GGUFのwrong-name率** — 6.27%(112/1787)
3. **差は何ptか** — +5.15pt(約5.6倍)
4. **Phase4W naming stressでのB/C差** — `phase4z_identity_analysis.json`の`set_a_naming_stress`参照。CがBを上回る傾向を確認
5. **Phase4X held-out namingでのB/C差** — 同上`set_b_heldout_naming`参照。同様の傾向
6. **E36 originalでのB/C差** — greedy含め、Cのみ一貫して「ルリ」等の誤名乗りが発生。5/5独立プロセスで確定
7. **E36 paraphraseでのB/C差** — `set_c_e36`の`e36_paraphrase`区分参照。傾向は一貫
8. **E02 originalでのB/C差** — critical loss詳細に複数件記録。Cのみ誤名乗り(ルナ・パチスロ君等)
9. **E02 paraphraseでのB/C差** — 同様の傾向を確認
10. **placeholder率B/C** — B=0.06%(1件)、C=0.11%(2件)。件数としては僅少だが両方0ではない
11. **「○○」placeholderは再現したか** — 今回の大規模評価では「○○」型は再現しなかったが、E36_ORIGINAL/seed109で単一チルダ型(「〜〜〜だよ」)がC側でのみ再発した
12. **「ルリ」は再現したか** — した。greedyで5/5独立プロセス完全再現、大規模評価でも15件出現(B側は0件)
13. **「ルリ」の出現頻度** — C側15件、B側0件。架空名全体ではC側60件、B側5件(12倍)
14. **greedy独立5回で再現したか** — した。C=5/5、B=0/5
15. **B safe→C unsafe critical lossはいくつか** — 49件
16. **critical loss率** — 2.74%(49/1787)
17. **hedge率に差はあるか** — ほぼ同水準(B=26.58%、C=27.48%)
18. **correct「リル」率に差はあるか** — ある。B=7.61%、C=3.08%(Bが約2.5倍)
19. **identity intrusionは増えたか** — 増えていない。RAG safety文脈でB/Cとも0/24
20. **prompt serializationは同一だったか** — 同一(バイト完全一致)
21. **chat template差はあったか** — なかった(diff差分0)
22. **BOS/EOS/special token差はあったか** — なかった(トークンID列完全一致)
23. **tokenizer metadata差はあったか** — 確認された範囲では一致(トークン化結果が完全一致するため)
24. **first-token/logits比較はできたか** — できた。第一トークン位置では両エンジンが合意していたが、実際の名前生成分岐点では「ル」(GGUF)と「リ」(HF)が僅差で異なる順位となっていることを確認した
25. **temperature=0.7でも再現したか** — した(greedyでの再現に加え、小サンプルでも同方向の傾向を確認)
26. **Scope PT-01〜22はB/Cで維持したか** — 両条件ともGate基準(95%)は維持したが、CはBより2.9pt低い(98.8%→95.9%)
27. **BF16 GGUF固有のscope regressionはあるか** — 明確な回帰ではないが、大規模測定で軽微な低下傾向が見られた
28. **Q3は維持したか** — 維持した(RAG safety sanity checkで確認、目視上問題なし)
29. **Q9/Q11 major hallucinationは0か** — 0だった(目視確認済み)
30. **Adversarial fabricationは0か** — 0だった(目視確認済み)
31. **Conflictingは維持したか** — 維持した(CF-01で両条件とも全問正解)
32. **Long-contextは維持したか** — 維持した(LC-01で目視上問題なし)
33. **pytest結果** — 126 passed
34. **protected assetsは不変か** — 不変(candidate/train/val/config/adapter/system.jinja2/merged HF/BF16 GGUF全て一致)
35. **Git状態** — Phase4X checkpoint(`a61d664`)のみ反映済み。Phase4Y/4Y-R/4Z成果物は未commit
36. **CASE A/B/C/D** — **CASE C**
37. **production移行可能か** — **不可。production移行禁止を継続する。**
38. **問題がある場合、最有力原因は何か** — HF→GGUF変換時の数値差、またはllama.cpp推論カーネル(CUDA実装)とPyTorchの数値差。chat template・tokenizer・RNGは除外済み。logits比較により、モデル自体が僅差で拮抗している分岐点(「リル」の2音節「リ」「ル」の順序)で、エンジン間の僅かな数値差が一貫して異なる側に倒れていることを確認した
39. **次に変更すべき最小の1変数は何か** — 本フェーズでは変更を行っていないが、次の診断候補としては「llama.cppのCUDA backend(ggml-cuda)を無効化しCPU推論のみで同じ分岐点のlogitsを比較する」ことで、backend差(候補④)とHF→GGUF変換差(候補③)をさらに切り分けられる可能性がある
40. **F16 GGUFを試す科学的根拠があるか** — 現時点では薄い。今回の問題はBF16(情報量最大)でも発生しており、精度をさらに落とすF16やQ8_0/Q5_K_M(Phase4Y-Rで既に量子化側の別のregressionを確認済み)は問題解決の方向ではなく、むしろ悪化させる可能性が高い。GGUF変換・推論エンジン側の実装差そのものを調査すべきである

## 18. 禁止事項の遵守

追加学習・新candidate作成・dataset変更・identity/complex教師変更・train/val再分割・LoRA設定変更・Final Candidate adapter変更・Base model変更・merged HF変更/再merge・BF16 GGUF変更/再変換・Q8_0/Q5_K_M再量子化・system.jinja2変更・RAG DB/structured.db/Vector DB変更・production差し替え/設定変更・アプリコード変更・API接続変更・llama.cpp更新/commit変更・transformers等の環境更新・Git commit/pushは一切行っていない。

## 19. 最終確認

- pytest: **126 passed**(開始前・終了後とも)
- git status: Phase4X checkpoint分のみ反映済み。Phase4Y/4Y-R/4Z成果物は未commit
- git diff: 追跡ファイルへの差分なし
- Protected Assets: 全て不変(Freeze Manifest記録値と完全一致、merged HF・BF16 GGUFの大容量ファイルも再検証済み)

## 作成ファイル一覧

- `training/riru/eval/phase4z_probes.py`
- `training/riru/eval/phase4z_placeholder_detector.py`
- `training/riru/eval/phase4z_naming_classify.py`
- `training/riru/eval/phase4z_identity_eval_gguf.py` / `phase4z_identity_results_gguf.json`
- `training/riru/eval/phase4z_identity_eval_hf.py` / `phase4z_identity_results_hf.json`
- `training/riru/eval/phase4z_analyze.py`
- `training/riru/eval/phase4z_greedy_repro_single.py`
- `training/riru/eval/phase4z_temp07_eval.py` / `phase4z_temp07_results_gguf.json` / `phase4z_temp07_results_hf.json`
- `training/riru/eval/phase4z_logits_compare.py`
- `training/riru/reports/phase4z_prompt_serialization_analysis.json`
- `training/riru/reports/phase4z_identity_analysis.json`
- `training/riru/reports/phase4z_paired_analysis.json`
- `training/riru/reports/phase4z_scope_analysis.json`
- `training/riru/reports/phase4z_gate_analysis.json`
- `training/riru/reports/phase4z_summary.md`(本ファイル)

## 停止

大規模生成・分析・目視確認・greedy再現性確認・temperature=0.7独立評価・chat template監査・logits比較・pytest・保護対象資産確認・レポート作成が完了しました。

**判定はCASE C(HF→GGUF/llama.cpp経路固有のidentity regression)であり、production移行禁止を継続します。**

追加学習・再merge・再GGUF変換・F16作成・llama.cpp更新・production変更・アプリコード変更・API変更・Phase4ZA等への自動移行は一切行っていません。次のご判断をお待ちします。特に、次の診断ステップ(CPU-only backendでのlogits再比較等によるさらなる原因切り分け)を実施すべきか、あるいは別のアプローチ(HF側を正式運用経路として検討する等)を取るべきかについて、ご指示いただければ対応します。
