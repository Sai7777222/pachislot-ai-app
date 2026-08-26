# Phase 4ZD: True-BF16 Baseline Recalibration — 完了報告

## 0. 目的

Phase4Zで報告された「HF wrong-name 1.12% vs GGUF 6.27%」という差の一部が、実は
HF側の評価がbitsandbytes 4bit NF4量子化ロードで行われていた(BF16 GGUFとの精度不一致という
未制御の交絡変数)ことによるものではないかを検証し、CASE C(Phase4Z)/ZB-F(Phase4ZB)/
ZC-G(Phase4ZC)の帰属を再較正する。診断専用フェーズであり、追加学習・GGUF再変換・
production変更は一切行っていない。

---

## 1. Git Checkpoint(Section1)

- 開始前確認: branch=main, HEAD=`a61d664f...`(不変), `git status --short`=78行(Phase4Y〜4ZCの未commit成果物), pytest 126 passed
- secret scan: `hooks.slack.com`/`api_key`/`password`等のパターンで全未追跡ファイルを検査し、実際の漏洩なし(1件のヒットは"secret"という単語を含む説明文のみで誤検知)を確認
- commit対象: Phase4Y〜4ZCのPython source(23ファイル)・JSON analysis/results(48ファイル)・summary.md(6ファイル)・manifest(1ファイル)、計78ファイル
- 除外: `training/riru/merged/`(merged HF本体、model.safetensors含む全ファイル。`.gitignore`に新規ルール追加)、`*.gguf`、adapter/checkpoint本体(既存パターンで除外済み)、`.venv-qlora/`、temporary binary dump(`training/riru/reports/_*`、既存パターンで除外済み)
- commit: `76f68b5c420944d62f5f750d67d06e7dd20406c2` ("checkpoint: Phase 4ZC forward drift diagnosis before Phase 4ZD")
- push: 成功。`git fetch origin`後、local HEAD == origin/main == `76f68b5c...` を確認
- **Phase4ZD自身のcommit/pushは行っていない**(以降の全成果物は未commit)

---

## 2-6. 事前確認・Protected Assets

Freeze Manifest記載のcandidate/train/val/config/adapter/adapter_config/system.jinja2/merged_hf/bf16_ggufの
hashは、checkpoint前後・Phase4ZD開始時・終了時の計4回全て一致(不一致0件)。

llama.cpp canonical(`D:/AI/tools/llama.cpp`)HEADは`5d5cb4c3a4ea8769490d39a275ee49a45184774d`で不変。
Phase4ZCで作成したdebug copy(`D:/AI/tools/llama.cpp-phase4zc-debug`)は新規パッチを一切行わず、
既存instrumentationの結果を参照利用のみ。

---

## 7-8. 比較条件・環境matrix(Section5, 37 Q6-Q9)

| 条件 | ロード方式 | 実際のattn_implementation |
|---|---|---|
| A_LEGACY_4BIT | merged HF, `BitsAndBytesConfig(load_in_4bit=True, nf4, bnb_4bit_compute_dtype=bfloat16)`, attn指定なし | **sdpa(デフォルト)** — 明示的に指定していなかったが実際にはSDPAが使われていたことが判明 |
| B_HF_BF16_EAGER | merged HF, `torch_dtype=bfloat16`, quantizationなし, `attn_implementation="eager"` | eager |
| C_HF_BF16_SDPA | merged HF, `torch_dtype=bfloat16`, quantizationなし, `attn_implementation="sdpa"` | sdpa |
| D_LLAMA_BF16_CPU | 既存BF16 GGUF, 公式CPU-onlyバイナリ(b10631, `D:/AI/tools/llama.cpp-bin-cpu`), llama-server, GPU使用ゼロ | N/A(GGML CPU backend) |
| E_HF_FP32_EAGER(補助) | merged HF, `torch_dtype=float32`(runtime cast、weight自体はBF16由来), quantizationなし, `attn_implementation="eager"`, device_map=cpu | eager |

全条件でn_tokens=440、input_ids先頭10/末尾10が完全一致することを確認(Section23、Gate1)。

---

## 9-10. E36 Forced-Prefix Margin比較(Section8-9, Q10-Q28)

| 条件 | リlogit | ルlogit | margin(リ-ル) | 勝者 |
|---|---|---|---|---|
| A(4bit) | 12.875 | 12.5625 | +0.3125 | **リ** |
| B(BF16 eager) | 12.5625 | 12.5625 | 0.0 | **TIE(完全同点)** |
| C(BF16 SDPA) | 12.5 | 12.5625 | -0.0625 | **ル** |
| D(llama.cpp BF16 CPU) | 12.533010 | 12.592121 | -0.059111 | **ル** |
| E(float32 eager) | 12.518037 | 12.565989 | -0.047953 | **ル** |

4bit(A)のみリ優勢。BF16 eager(B)は完全同点。SDPA(C)/llama.cpp(D)/float32(E)は全てル優勢。
**llama.cppを一切介さないHF単体の条件変更(SDPA化 or float32化)だけで、旧4bit baselineとは
逆方向、かつllama.cppと同方向の結果が再現された。**

---

## 11. 実際の非拘束生成(E36 original repro5, Section10, Q12-Q15)

forced-prefix logit分析だけでなく、実際の非拘束greedy生成(5回、決定論性確認)も実施。

- **A(4bit)**: 「こんにちは〜！私はパチスロのデータを知り尽くした、かわいいおねえさんだよっ！」— 名前を名乗らない(no-name)。5/5同一。
- **B(BF16 eager)**: 「こんにちは〜！私はパチスロのことを知ってる、あいこだよっ！」— **「あいこ」という誤った名前**。5/5同一。
- **C(BF16 SDPA)**: Bと完全同一テキスト「あいこ」。5/5同一。
- **D(llama.cpp)**: 「こんにちは〜！私はパチスロの専門アシスタントのルリだよっ！」— **「ルリ」という誤った名前**(既知の旗艦事例)。5/5同一。

**重要な発見**: forced-prefix「こんにちは〜！私はパチスロの専門アシスタントの」は、D(llama.cpp)の
自然なgreedy軌道は実際に通過するが、A/B/C(HF)いずれの自然なgreedy軌道もこの厳密な文言を
通過しない(HFは「パチスロのことを知ってる、」という異なる言い回しを取る)。したがって
forced-prefix margin比較(Section8-9)は『仮想的な分岐点』での比較であり、実際の非拘束生成結果とは
必ずしも1対1で対応しない。それでも最終結果は一貫していた: **B/C/Dはいずれも何らかの誤った名前を
生成し、Aのみ無回答**。

初回自動分類器は「あいこだよっ」を、名前が読点を挟んで登場するパターンを検出できず
D(generic)と誤判定したが、目視確認により手動でA(genuine wrong-name)へ補正した。

---

## 12. E36 Paraphrase 8問(Section11)

Phase4ZAの8問(無改変)を4条件で評価。E36_ORIGINAL以外の7問はいずれも名前を尋ねない
カジュアルな挨拶バリエーションであり、4条件全てで名前への言及なし(安全)。E36_ORIGINALのみ
上記repro5と同一の結果(A=no-name, B/C=「あいこ」, D=「ルリ」)。

---

## 13-14. Stage1 Naming Stress(Section12-13, 最重要, Q32-Q37)

**Probe Set A(Phase4W naming stress, 20問、無改変) x (greedy + seed101-110) = 220生成/条件**、
B_HF_BF16_EAGER と D_LLAMA_BF16_CPU のみ実施。

### 生の自動分類結果
- B: genuine wrong-name = 20/220 (9.09%)
- D: genuine wrong-name = 37/220 (16.82%)

### 手動レビュー(Section24、全件目視確認)

既存`classify_naming()`の2つの既知の抜けを発見した:
1. `NAME_CUE_PATTERNS`正規表現は「私は&lt;name&gt;だよ」のように名前が直接続くパターンのみ検出し、
   読点を挟む自己紹介文(「私は&lt;説明&gt;、&lt;name&gt;だよ」)では名前候補を検出できない。
2. `HEDGE_PATTERN`は「名前はまだ決まってない」「名前は言わないでおこうね」等の言い回しを
   網羅していない。

これらの抜けにより、B/Dとも一部の自動判定が誤っていたことが判明。全A判定候補を目視確認し
手動補正した(共有ファイル`phase4z_naming_classify.py`自体は変更していない):

| 条件 | 自動A件数 | 補正内容 | 補正後A件数 | genuine wrong-name率 |
|---|---|---|---|---|
| B | 20 | 1件→G(名前なし)、1件→E(正しい名前+修飾語)、4件→B(hedge) | 14 | **6.36%** |
| D | 37 | 1件→E(正しい名前+修飾語)、7件→B(hedge) | 29 | **13.18%** |

### Paired比較(補正後)

- n=220ペア, WIN(B不安全→D安全)=7, TIE=194, LOSS(B安全→D不安全)=19, CRITICAL_LOSS(B安全→D genuine wrong-name)=3
- **absolute_diff_pt = 6.82pt**, **ratio(D/B) = 2.07倍**, **critical_loss_pct = 1.36%**
- Stage2判定基準(absolute_diff≥2.0pt または critical_loss≥1.0%)を両方満たしたため、Stage2実施。

---

## 15. Stage2(Section14, Q38)

**Probe Set B(Phase4X held-out naming, 24問、無改変) x (greedy + seed101-103) = 96生成/条件**。

- B: genuine wrong-name = 1/96 (1.04%)、全1件を目視確認、正しくA判定(補正不要)
- D: genuine wrong-name = 5/96 (5.21%)、全5件を目視確認、正しくA判定(補正不要)
- absolute_diff_pt = 4.17pt, ratio(D/B) = 5.0倍

独立した第2のprobe setでもD>Bの方向性が安定して再現された(絶対率はprobe setにより変動するが、
比率2〜5倍でD側が一貫して高い)。paired差は安定していると判断し、Set A+B+C+D全体への
機械的な拡張(過去の1787件への回帰)は実施しなかった。

---

## 16. Section6の核心質問(Q6): 旧gapのどの程度がprecision mismatchで説明されるか

Set A(220サンプル)ベースで、旧4bit-HF baseline(Phase4Zの同一probe+seed部分集合を本フェーズの
classifierで再分類)と比較:

| 指標 | 値 |
|---|---|
| 旧4bit-HF(Set A部分集合, 本フェーズclassifierで再分類) | 0.91% (2/220) |
| 新BF16-HF eager(本フェーズB, 補正後) | 6.36% (14/220) |
| 新BF16-llama.cpp(本フェーズD, 補正後) | 13.18% (29/220) |
| precision-pathのみで説明されるgap(4bit→BF16, engine=HF固定) | **5.45pt** |
| 4bit-HF→BF16-llama.cppの全gap | 12.27pt |
| **precision-pathが占める割合** | **約44.4%** |
| 残存するllama.cpp/attention実装差の割合 | 約55.6%(6.82pt) |

**4bit→BF16というprecision path差だけで、旧来Phase4Zが報告した「HF vs GGUF」gap(5.15pt)と
同等かそれ以上の規模のgapが説明される。** 従来「GGUF/llama.cppがHFより著しく悪い」という
結論の相当部分(Set A基準で約44%)は、実際には4bit量子化ロードという未制御の交絡変数に
よるものだった可能性が高い。ただし残り約56%は、真にBF16同士のHF-vs-llama.cpp間でも
実在する差であり、llama.cpp固有の(あるいは少なくともHFのeagerパスとは異なる)forward計算
実装差に起因すると考えられる。

---

## 17-18. Full Naming Gate注記、P01等の確認(Section17-18, Q39-Q40)

**注意**: 本フェーズのB_HF_BF16_EAGER Stage1結果(6.36%)は、Phase4X Final Gateの1.4%/0.0%とは
測定条件が異なる(Phase4X GateはLoRAアダプタ+4bit量子化ベースモデルでの測定であり、本フェーズは
merged HF+BF16フル精度)。**両者を直接同一視してはならない。**

Q3/P01/Q9/Q11/PT-01/AD-01/CF-01/LC-01の8問をB/Dでgreedy+seed101-103(4生成/問)確認した結果、
**全問でB/Dの回答内容(数値・事実)は完全に一致**しており、RAG/事実想起能力への影響は
確認されなかった。本フェーズで観測された脆弱性は、identity(名乗り)に限定される。

---

## 19-20. HF内部でのattention実装感度(Section19-20, Q41-Q42)

Phase4ZCで取得済みのHF eager/SDPA hidden state dumpを再利用し、**llama.cppを一切使わない**
HF eager vs HF SDPA(同一BF16 weight)のlayer-wise比較を実施(`phase4zd_hf_backend_diff.py`)。

- `first_nonzero_diff_layer = 0`, `first_10x_jump_layer = None`, `first_cosine_drop_layer = 20`
- Phase4ZCのHF-vs-llama.cpp比較(layer0から乖離、単一ジャンプなし、layer27でcosine低下開始)と
  **ほぼ同種の分散型ドリフトパターン**が、llama.cppを全く使わない純粋なHF内部のattention実装
  切替だけで再現された。

この観測は、Phase4ZCで見られたdrift patternがllama.cpp固有の現象ではなく、attention計算の
数値実装差一般に対するモデルの感度を反映したものであることの独立した追加証拠である。

---

## 21. Float32計算精度感度(Section21, Q28)

E_HF_FP32_EAGER(weight自体はBF16由来、実行時floatキャスト)でも margin=-0.048(ル優勢)。
bfloat16の丸め粒度が、この特定のトークン対の同点/僅差を生み出している可能性が示唆される。

---

## 22-24. 生成パラメータ・プロンプト同一性・分類器品質(Q10-Q11, 43省略先行)

全条件でgreedy(temperature=0相当)を主評価、sampled評価は既存フェーズと同一のtemperature=0.3/top_p=0.9・
同一seed(101-110または101-103)を使用。全条件でprompt/token IDが完全一致することを確認済み(Gate1)。
分類器の既知の抜け2件を発見・手動補正し、自動率のみに依存しない結論とした(Section24遵守)。

---

## 25-26. CASE定義・選定(Section25-27, Q49-Q50)

**選定: CASE ZD-E (Mixed: precision-path + attention implementation sensitivity + 残存するllama.cpp固有差)**

根拠:
1. 4bit→BF16のprecision path切替だけで、旧来のHF-vs-GGUF gapと同等規模(Set A基準で44%)の
   gapが生じる → precision-path要因は無視できないどころか支配的な部分を占める。
2. しかし4bit→BF16切替だけでは説明しきれない残差(56%、6.82pt)が、真にBF16同士のHF-vs-llama.cpp
   比較でも残る → 純粋なllama.cpp/attention実装差要因も同時に実在する。
3. llama.cppを全く使わないHF内部でも、attention実装(eager→SDPA)切替だけで、GGUF-likeな結果への
   反転およびPhase4ZCと同種のlayer-wise分散型driftパターンが再現される → 「llama.cppのバグ」という
   単純な説明ではなく、attention計算の数値実装一般に対するモデル自身の感度の高さが本質的要因の一部。

これら3つの独立した証拠系列が、いずれも単独では全体を説明できず、複合的に寄与していることを
示しているため、単一原因のCASE(ZD-A/B/C/D)ではなくCASE ZD-Eを選択した。確信度は高い
(3つの独立した制御実験が相互に補強し合う一貫した結果を示しているため)。

---

## 27. 過去CASEの再評価(Section26, Q43-Q47) — 新証拠が旧結論を弱めた場合は無理に守らない

| フェーズ | 元のCASE | 再評価 |
|---|---|---|
| Phase4Z | CASE C(identity-only re-regression、HF/GGUF engine差に主に帰属) | **精緻化**。原因の相当部分(Set A基準で約44%)が、HF側測定に使われていた4bit NF4量子化という未制御の交絡変数によるものだったと判明。engine差自体は実在するが、Phase4Zが報告した5.15ptより狭い可能性が高い。 |
| Phase4ZA | CASE ZA-B(CUDA固有原因は否定、narrow backend-sensitive marginで説明可能) | **維持・強化**。本フェーズの結果は「識別マージンが極めて狭く、様々な数値実装差に敏感」というZA-Bの結論と完全に整合し、その「数値実装差」がattention計算の実装に由来することを明確化した。 |
| Phase4ZB | CASE ZB-F(llama.cpp forward実装がweight差より支配的) | **維持**(撤回の根拠なし)。全579テンソルのbit完全一致という証拠は本フェーズでも一切揺らいでいない。 |
| Phase4ZC | CASE ZC-G(mixed、attention計算ブロックに集中した分散型ドリフト) | **維持・強化**。Section19-20で、Phase4ZCと同種のdrift patternがllama.cpp不使用でも独立に再現され、頑健性が大きく向上した。 |

**総括**: 一連の調査(Phase4Z〜4ZC)は「weight差ではない」ことを正しく確定させたが、
「HF(4bit)は安全、GGUF(BF16)は危険」という当初の対比自体が、量子化経路の違いという
未制御の交絡変数によって誇張されていたことが、本フェーズで初めて明らかになった。真の対比は
「BF16 HF(eager)も、BF16 llama.cppも、共に(4bit量子化されたHFより)genuine wrong-name率が
有意に高く、両者の間にも残存する有意差がある」というものであり、識別マージンの本質的な
脆弱性(razor-thin、時にbit-exact tie)が根本原因であるという理解がより正確である。

---

## 28-31. 禁止事項の遵守確認(Q51-Q54)

- 追加学習(LoRA/identity教師追加/DPO/SFT等): 一切実施していない
- GGUF再変換・再量子化: 一切実施していない(既存BF16 GGUF固定)
- llama.cpp変更: canonical(`5d5cb4c3a...`)は不変。debug copyへの新規patchなし(Phase4ZCの既存instrumentationを参照利用のみ)
- production変更: 一切実施していない
- **Phase4ZDでGit commit/pushしたか**: Section1のcheckpoint(Phase4ZC以前の成果物、コミットハッシュ`76f68b5`)のみ。Phase4ZD自身の成果物は未commit。

---

## 32. 次フェーズ案(Section38、1つのみ提示)

**ZD-B: eager vs SDPA vs ggml attention numerical isolation**

本フェーズで「attention計算ブロックが主要因」であることの状況証拠は3つの独立した経路で
確認されたが、attention内部のどの具体的な演算(QK-matmul単体、softmax単体、o_proj単体、
またはRoPE)が支配的かはPhase4ZCのGQAレイアウト複雑性により未特定のまま残っている。
次に着手すべき最小の1変数は、この特定である。ただし、これは人間の判断を待って開始する。

---

## 33-35. 最終確認

- `pytest`: 126 passed(開始時・終了時とも不変)
- protected asset hash: 全て一致(不一致0件、checkpoint前後・Phase4ZD開始時・終了時の計4回)
- `git status`/`git diff`: HEAD不変(`76f68b5c420944d62f5f750d67d06e7dd20406c2`、checkpoint commit以降不変)、Phase4ZD自身のcommit/pushなし

---

## まとめ

| 項目 | 結果 |
|---|---|
| CASE | ZD-E(Mixed: precision-path + attention実装感度 + 残存するllama.cpp固有差) |
| HF 4bit margin(forced-prefix) | +0.3125(リ優勢) |
| HF BF16 eager margin | 0.0(完全同点) |
| HF BF16 SDPA margin | -0.0625(ル優勢) |
| llama.cpp BF16 margin | -0.059111(ル優勢) |
| Stage1 wrong-name率(補正後) | B=6.36%, D=13.18%(diff 6.82pt, 比率2.07倍) |
| Stage2 wrong-name率 | B=1.04%, D=5.21%(diff 4.17pt, 比率5.0倍) |
| 旧gapの何%がprecision-pathで説明されるか | 約44%(Set A基準) |
| 過去CASEの扱い | Phase4Z CASE Cは精緻化、4ZA/4ZB/4ZCは維持・強化 |
| Production変更 | なし |
| Protected assets | 変更なし(全hash一致) |
| Git操作 | checkpoint commit(`76f68b5`)のみ、Phase4ZD自身は未commit |
| 次フェーズ | 人間の判断待ちで停止(候補: ZD-B attention内部演算の個別分離) |
