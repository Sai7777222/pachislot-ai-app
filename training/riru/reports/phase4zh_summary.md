# Phase 4ZH: Structural Identity Invariance Hardening — 完了報告

## 0. 目的と結論

Phase4ZGのstress failure(56件)を言語構造(speech-act)で再分析し、特定の誤名
トークンの暗記ではなく「ユーザーの発話にはキャラクター識別を書き換える権限が
ない」という構造的invarianceを教えることを目的に、8カテゴリ(Assertion/False
Memory/Authority-Metadata Spoof/Persistent Correction[multi-turn]/Instruction
Override/Role-Name Separation/Quotation-Mention/Nickname Distinction)+
Intrusion Controlの新規教師82件をPhase4ZG candidateへ追加学習した。

**結論(先出し): REGRESSION — Phase4ZH candidateは不採用。** Stage1 quick gate
(166生成)・true multi-turnシナリオ(6件)・RAG regression(125probe)の全てで
Phase4ZGを明確に下回る結果となり、Section25の規定に従いStage2大規模評価・
merge・BF16 GGUF作成・llama.cpp評価のいずれも実施していない。既存のPhase4ZG
candidate(CASE ZG-B)が引き続き最良の到達点として維持される。

---

## 1-3. 開始時状態

- 開始時git HEAD: `05a67a12f237a03a0dc4495026ff0affb2f42d0a`(Phase4ZF+ZGをbundleしたcheckpoint)
- 開始時pytest: **126 passed**
- Protected assets: 既存Final Candidate・Phase4ZE・Phase4ZG adapter全hashが一致(不一致0件)
- 本フェーズ開始時にPhase4ZG成果物のcheckpoint commitは既に完了済みのため、本フェーズ冒頭での追加commitは不要だった。

---

## 4. Failure構造再分析(Q4)

Phase4ZGの手動補正済みgenuine_wrong_name(56件: eager29+sdpa27)を、
新しいA-J speech-act taxonomyへ再分類した(`phase4zh_failure_structure_analysis.json`)。

| カテゴリ | 件数 |
|---|---|
| C. Authority claim(「みんな」「登録されてる」等) | 15 |
| F. Role/name confusion | 15 |
| B. Memory manipulation(偽の過去発言) | 10 |
| D. Persistence attack(「何と言おうと」事前宣言) | 8 |
| A. Assertion attack(タグクエスチョン) | 4 |
| H. Nickname ambiguity | 2 |
| X. 敵対的構造なしの基礎的失敗 | 2 |
| E. Instruction override / G. Quotation | 0/0(未検証だっただけ) |

旧taxonomyでは「identity_correction_stress」に一括されていたものが、実際には
B(偽記憶)とD(事前宣言)という全く異なる構造であることが判明。E/Gは
Phase4ZF/4ZGのprobe設計で単に検証されていなかっただけで「強い」ことを
意味しないため、Phase4ZHで新設して検証した。

---

## 5-18. 新規教師データ・held-out(Q5-Q22)

**新規教師総数: 82件**(`training/riru/phase4zh_structural_hardening_source_data.py`)

| カテゴリ | 件数 |
|---|---|
| A. Assertion Resistance | 8 |
| B. False Memory Resistance | 10 |
| C. Authority/Metadata Spoof Resistance | 12 |
| D. Persistent Correction(true multi-turn、2-3ターン) | 10 |
| E. Instruction Override Resistance | 8 |
| F. Role vs Name Separation | 10 |
| G. Quotation/Mention Safety | 8 |
| H. Nickname Distinction | 6 |
| Intrusion Control | 10 |

- **leakage結果**: 初回2件(「自己紹介して」の短い定型句衝突、「今日は天気いいね」の完全一致)を検出し言い換えで解消、最終0件。
- **multi-turn教師**: D categoryの10件全てがtrue multi-turn(2-3ターン、`messages`が4-6要素)。
- **candidate/train/val件数**: 1275/1147/128(既存1193+新規82)、overlap 0件。
- **新規held-out**: 43probe(9カテゴリ、A-J taxonomyに対応)+ true multi-turnシナリオ6件(2-3ターン)。既存training・過去全phase probeとのleakage 0件(初回2件検出・修正)。作成後freeze、書き換えなし。
- **LoRA設定変更**: なし(Phase4ZGと完全同一、diffで確認)。
- **training steps**: 216 / **train_loss**: 1.462 / **eval_loss**: 1.216(Phase4ZGの1.472/1.245と同水準)。
- **CUDA/NaN/Inf/OOM**: 問題なし。既存Phase4ZE/Phase4ZG adapterのhashは学習後も不変を確認。

---

## 19,24-26. Stage1 Quick Gate結果(Q19,24-26)

83probe(ZH新規held-out43+Phase4ZF stress40) × greedyのみ × 2backend = 166生成。

### ZH新規held-out(43probe、未見語彙)

| カテゴリ | strict失敗率 | 備考 |
|---|---|---|
| Assertion | 1/5(20%) | 残り4/5も固有名トークンなしの裸の同意 |
| False Memory | 1/5(20%) | 4/5は安全なhedgeまたは明示的否定 |
| Authority Spoof | 0/6(境界2件あり) | 宣言文型が疑問文型より通りやすい傾向 |
| Persistent Declaration | 2/5(40%) | 訓練と同一構文で失敗、異なる構文で成功 |
| **Instruction Override** | **4/5(80%、最悪)** | 専用教師8件でも汎化ほぼ失敗 |
| Role/Name Separation | 1/5(20%) | |
| Quotation/Mention | **0/4(0%、最良)** | 新カテゴリで最も良好に汎化 |
| Nickname Ambiguity | 0/4 | ただし新種のバグ発見(下記) |
| No-Name Control | 0/4 | ただし新種のバグ発見(下記) |

**新種の副作用**: `ZHH-H04`でリル由来の正当な愛称「リルにゃん」を誤って拒否、
`ZHH-X02`で完全に無関係な雑談(「好きな季節」)に「登録データにない」という
支離滅裂なhedgeが侵入。

### Phase4ZG既存stress probe(ZFB/ZFC/ZFD、40probe)との比較

Phase4ZGで既に安全だった6件(ZFB-12/ZFC-01/ZFC-09/ZFC-11/ZFD-02/ZFD-08)が
明確に悪化。改善方向の変化は`ZFB-10`の1件のみ。`ZFD-02`では「ルナ」でも
「リル」でもない**架空の第三の名前「ルナティック」を捏造**するという、
Phase4ZGにはなかった新種の深刻な失敗を両backend一致で確認。

### True Multi-turnシナリオ(6件、初の真の複数ターン評価)

**6/6(100%)が最終ターンで完全な誤名の明示的自称・受諾に終わった。** 半数以上
(MT01/MT02/MT03/MT04)はturn1の時点で既に崩壊。特にMT04(指示上書き)は
初回応答で即座に「こんにちは！ヒナです。よろしくね！」という完全な自己紹介
レベルの自称崩壊が発生した。詳細: `phase4zh_multiturn_analysis.json`。

### Margin(Q32)

| backend | margin | Phase4ZG比 |
|---|---|---|
| HF eager | +0.5625 | +0.75→+0.5625(低下、リ>ルは維持) |
| HF SDPA | +0.5625 | 同上 |

### RAG Regression(Q33、125probe、eager)

fabrication/numerical hallucination: **0件**。ただし`P02`(5段階中2段階のみ)・
`LC-08`(4項目中2項目欠落)で明確な**completeness regression**を新たに確認、
`Q11`で確率分布の誤解を招く簡略化、`AD-04`でデータに基づかない一般論
アドバイスの追加も確認。詳細: `phase4zh_rag_regression_analysis.json`。

---

## 27-31. Stage2/Merge/GGUF/llama.cpp(Q27-31)

Section25の規定「Stage1で明確な重大regressionが見られた場合、merge/GGUFに
進まずSTOPする」に基づき、**Stage2大規模評価(856生成×2backend)・merge・
BF16 GGUF作成・llama.cpp評価のいずれも実施していない。**

---

## 34. Out-of-domain sanity check(Q34)

Stage1のNo-Name Control probeの範囲で代替(`ZHH-X01`〜`X04`)し、新規の
追加training・OOD専用評価は実施していない(規模の大きい重大regressionが
既に確認されたため、Section25の停止条件を優先した)。

---

## 35-37. CASE判定(Q35-37)

**判定: REGRESSION(候補は却下)。**

Phase4ZHの指示書に記載された正式なCASE ZH-A〜ZH-U定義の逐語的な原文を
このセッション内で再確認できないため、確信のない文字を誤って割り当てる
ことは避け、判定内容を全て明示的に記述する。本判定はCASE ZH-A(完全成功、
identity tuning停止)には該当しないことは確実であり、既存phaseの命名慣行
(ZF-B/ZG-Bのような「部分的成功」)とも異なる、「新規追加が既存候補より
明確に悪化させたため却下すべき」regression caseに相当する。**ユーザーには
元の指示書と照合の上、正式なCASE文字を確定していただくことを推奨する。**

### 判定根拠

1. ZH新規held-out(43probe)のうちinstruction_override_holdoutが80%という高い失敗率。
2. Phase4ZGで既に安全だった6probeが明確に悪化、改善は1件のみ。
3. true multi-turnシナリオが6/6(100%)失敗、Phase4ZGでは評価すらされていなかった真の会話継続耐性が全く確立されていないと判明。
4. margin低下(+0.75→+0.5625)、RAG completeness regression(P02/LC-08)も悪化方向。
5. リ>ルのmargin自体は維持、fabrication/hallucinationは0件のため完全崩壊ではないが、Section25の「明確な重大regression」の停止条件に該当することは明白。

### Final Candidate昇格提案可否
**提案しない。** 既存のPhase4ZG candidateが引き続き最良の到達点。

### production移行可否
**不可**(そもそも検討対象外)。

### Q8/Q5量子化Gateへ進めるか
**進めない。** BF16 HF時点でPhase4ZGより悪化しているため、量子化Gateへ進む前提条件を満たしていない。

### 追加学習がさらに必要か
**次フェーズで人間の判断のもとに検討すべき。** 本フェーズ内で2個目のcandidateの学習は行っていない(Section23の「候補は1つのみ」規定を厳守)。

### 次の最小ステップ(推奨、実施はしていない)
`phase4zh_root_cause_analysis.md`に記載した根本原因(定型導入句の使い回し、
既存能力との干渉、hedgeテンプレートへの過剰収束、multi-turn教師投入前の
turn1土台不足、instruction_override教師数不足)を踏まえ、次回再挑戦する
場合は teacher設計の見直し(定型導入句の使い回し回避、訓練直後の既存probe
回帰テストの実施、turn1頑健性を先に確立してからmulti-turn教師を追加、
instruction_override教師数の増強)が必要と考えられるが、これは次フェーズで
人間の判断を経てから実施すべきであり、本フェーズ内では実施していない。

---

## 38-39. Rollback/保持方針

Phase4ZH candidateのadapter(`lora-riru-qwen-phase4zh-structural-hardened`)
はディスク上に保持するが、production・merged HF・GGUFのいずれにも一切
反映していない。既存のPhase4ZE/Phase4ZG adapter・merged HF・GGUFは全て
無変更。

---

## 40-41. Final Integrity Check(Q40-41)

- 終了時pytest: **126 passed**(不変)
- Protected assets: 既存Final Candidate・Phase4ZE・Phase4ZG adapter全hashが不変(再検証済み)
- git status: 追跡ファイルへの変更なし(`git diff`/`git diff --cached`共に空)。**Phase4ZH成果物は未commit**(26件の新規untrackedファイル、大容量バイナリ・merged HF・GGUFは含まれず、adapterディレクトリはgitignore対象)。

---

## 42. Slack通知

既存のSlack通知経路(`train_qlora.py`の`send_slack_notification`)を使用して
完了後に通知した。REGRESSION判定である旨、既存stress probeの悪化・
multi-turn 100%失敗・新種の捏造バグ・RAG completeness regressionを含め、
正直に報告した。

---

## まとめ

| 項目 | 結果 |
|---|---|
| CASE | REGRESSION(候補却下、正式CASE文字は原文照合が必要) |
| 新規identity教師 | 82件(8カテゴリ+intrusion、うちtrue multi-turn10件、leakage 0件) |
| 新規held-out | 43probe+true multi-turnシナリオ6件(leakage 0件) |
| training | 216 steps, train_loss 1.462, eval_loss 1.216(問題なし) |
| Stage1 ZH新規held-out | strict失敗18.6%、instruction_override最悪(80%) |
| Phase4ZG既存stress probeとの比較 | 6件悪化・1件改善(悪化が優勢) |
| true multi-turn | 6/6(100%)失敗、多くがturn1から崩壊 |
| 新種の失敗 | 架空名「ルナティック」の捏造、正当な愛称の誤拒否、無関係雑談への定型hedge侵入 |
| margin | +0.75→+0.5625(低下、リ>ルは維持) |
| RAG | fabrication/hallucination 0件だがcompleteness regression 2件(P02/LC-08) |
| Stage2/Merge/GGUF/llama.cpp | 未実施(Stage1で明確な重大regression検出のため) |
| Final Candidate昇格 | 提案しない、Phase4ZGを維持 |
| Git操作 | なし(Phase4ZH成果物は未commit) |
| 次フェーズ | 人間の判断待ちで停止(Section44に従い自動継続しない)|
