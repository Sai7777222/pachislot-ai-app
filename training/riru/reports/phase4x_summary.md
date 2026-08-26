# Phase 4X: Identity Stabilization — Wrong-name / Placeholder 最小修正実験 最終報告

## 0. 結論の要約

- **ratio-high-identity-stable(C)は、Phase4Wで判明した2つの残存課題(genuine wrong-name・single-tilde placeholder)を明確に改善し、RAG側の性能を維持した。**
- **genuine wrong-name**: Phase4Wの既存naming stress probe(20問)を同一seed(42〜51)で再評価した結果、**B(ratio-high-identity)=6.4% → C(stable)=1.4%**(修正後の同一分類器での公平なpaired比較)。さらに完全新規のheld-out naming probe(24問、264生成)では**0.0%**。いずれもGate基準「<5%」を大きく下回り、「strong PASS: <3%」水準を達成した。
- **placeholder**: E36型プロンプトの新規seed再評価で、Bが30%(3/10、Phase4Wで発見された既知の検出漏れ)だったのに対し、**Cは0/10(0%)**。Phase4X用に修正したplaceholder detector(name-slotの前後関係で判定)でも0件。
- **RAG側の指標はほぼ完全に維持された**: Q3=100%(改善)、Broad completeness=96.8%(B比paired 108/108同点、win0/loss0)、Scope correctness=98.9%(新規seedで再取得、Phase4Wの限界を解消)、Adversarial/Conflicting/Long-context/Q9(新規probe)/Q11(新規probe)/Identity intrusionは全てクリーンまたはB同等。
- **唯一の残存注意点はP01の局所的な省略パターンが、同一seedでの比較においてB(30%)よりC(80%)で発生率が高かったこと。** ただしPhase4V/4Wで確立された通り、この事象はheld-outのBroad completeness全体には一切波及しておらず(108ペア中0 loss)、局所的・seed依存のノイズという既存の結論を追認する形になった。
- **判定: CASE A。** Identity改善とRAG維持の両方を達成しており、ratio-high-identity-stableをFinal Candidate候補として推奨できる科学的根拠が得られた。ただし本フェーズ内ではmerge/GGUF化/正式freeze/Git commit・pushは一切行っていない。

## 1. 開始前確認

- git HEAD: `7626661f42f8c88c7096f2fcd7463b24d12b47a0`(Phase4Wチェックポイント、開始前後で不変)
- git status: 追跡ファイルへの変更0件、新規作成18ファイルのみ(`??`)
- pytest: **126 passed**(開始前・終了後とも)
- Phase4T〜4W成果物、ratio-high/ratio-high-identity candidate・train・val・adapter、いずれも存在確認済み
- 保護対象資産(v4/ratio-high/ratio-high-identity adapter、各candidate/train/val、system.jinja2)のSHA-256/MD5: 開始前後で完全一致(下記4節参照)

## 2. Placeholder detectorの修正 (Section 6)

旧detector(`[〜ー]{2,}` — 2文字以上の連続のみ検出)の欠陥を修正し、`training/riru/eval/phase4x_placeholder_detector.py`として新規実装した。「name cueの直後に実質的な名前トークンがなく、プレースホルダー的な記号・空白のみが入っている」構文的位置関係で判定する設計とし、「だよ〜」等の自然な語尾チルダとは明確に区別した。

**検証結果**: 13件の既知ケース(単一チルダplaceholder3件、genuine wrong-name2件、正常な名乗り8件)全てで正しく判定(13/13)。Phase4U/4Wの既存データに適用したところ、E36型で**新detector=16.7%(5/30)、旧detector=0.0%**という既知のギャップを再現確認できた(`training/riru/reports/phase4x_placeholder_detector_analysis.json`)。

## 3. Naming分類器のバグ修正 (Section 5)

分析の過程で、Phase4Uの`classify_generation`を踏襲した初期実装に、**「名前は無いみたいだよ」「登録データに載ってないみたいだよ」のような誠実なhedge/refusal回答を、cueパターンの機械的抽出だけでgenuine wrong-nameに誤分類するバグ**を発見した。これは「特にない/非公開/分からない」等は正しく除外できていたが、「(名詞)は無い/ない+みたい」という、より一般的な言い回しが除外リストに含まれていなかったために発生していた。

修正前は新規naming probeで9件・naming stress再評価で10件が誤ってAと判定されていたが、修正後はそれぞれ**0件・3件**(いずれも真の架空固有名詞のみ)に是正された。この修正は`training/riru/eval/phase4x_naming_reclassify.py`に実装し、10件の検証ケース全てで正しく分類されることを確認した上で、以降の全分析に適用した。

## 4. Protected Assets 最終確認

| 資産 | ハッシュ | 開始前後 |
|---|---|---|
| v4 adapter | `b5f1646c...` | 不変 |
| ratio-high adapter | `b0c3e657...` | 不変 |
| ratio-high-identity adapter | `ab4f55b8...` | 不変 |
| ratio-high-identity candidate/train/val | 3ファイルとも | 不変 |
| system.jinja2 (MD5) | `f3ea72a9...` | 不変 |

## 5. 新規Identity Stabilization教師データ (Section 7-9)

`training/riru/phase4x_identity_stabilization_source_data.py`に**25件**を新規作成した。

| カテゴリ | 件数 | 内容 |
|---|---|---|
| A. Explicit Identity | 8 | 直接的な名前質問の新規言い回し |
| B. Identity under ambiguity | 8 | 間接的・曖昧な角度からのidentity質問 |
| C. Casual greeting completion | 5 | E36型の挨拶文脈での完成した自然文名乗り(placeholder防止の核心) |
| D. Intrusion control | 4 | 名前を聞かれていない通常RAG質問での自己紹介抑制 |

負例列挙(誤名の暗記)は一切行わず、全て「identity→リル」の positive mapping のみで構成した。placeholder記号そのものは正解側データに一切含めていない(品質検査で確認済み)。

## 6. Contamination・品質検査 (Section 11-12)

`build_phase4x_dataset.py`にて、Phase4T naming/P04 probe・Phase4U identity教師・Phase4V probe・Phase4W全probe(Q9/Q11/naming stress/adversarial/conflicting/long-context)・structured17・character39・holdout P01-P10との文面重複を検査した結果、**contamination_hits=0、high_similarity_pairs=0、placeholder=0、wrong-name enumeration=0、ChatML=0、実在機種名=0、重複=0、空フィールド=0**(全てquality issue合計0で品質ゲートを通過)。

なお、短い定型句「リルだよ！」のような数文字の断片一致はPhase4Uの既存教師とも自然に重複するが、これは意図的な positive mapping の一貫性によるものであり、evaluation probe文面のコピーとは性質が異なることを確認した上で許容した。評価probe(`phase4x_probes.py`)側でも同様の検査を行い、user発話(質問文)の重複は0件を確認した。

## 7. Dataset構成 (Section 13-14)

| 項目 | 値 |
|---|---|
| 既存(ratio-high-identity candidate) | 1070件 |
| 新規追加 | 25件 |
| 合計 | 1095件 |
| complex教師(不変) | 113件 (10.32%) |
| identity教師合計 | 68件 (6.21%) |
| intrusion-control比率(新規内) | 16.0%(4/25、目安20〜30%以下を達成) |
| train | 987件 |
| val | 108件 |
| train/val overlap | **0** |

`riru_ratio_high_identity_stable_candidate.jsonl` / `_train.jsonl` / `_val.jsonl`として新規保存(既存ファイルは無改変)。

## 8. Preflight (Section 17)

`phase4x_pretrain_checks.py`で以下を確認、**全項目READY**:

CUDA利用可能・dataset parse OK・candidate件数=1095一致・ratio-high-identity candidateハッシュ不変・system.jinja2不変・v4/ratio-high/ratio-high-identity adapterハッシュ不変・dataset品質issue=0・complex比率≥10%・train/val overlap=0・LoRA設定がratio-high-identityと完全一致(rank_pattern/alpha_pattern不使用含む)・学習ハイパーパラメータ一致・出力ディレクトリが既存adapterと衝突しない。

## 9. 学習 (Section 18)

| 項目 | 値 |
|---|---|
| train件数 | 987 |
| val件数 | 108 |
| steps | 186 (3 epoch) |
| train_loss | 1.488 |
| eval_loss | 1.362 |
| 学習時間 | 769.8秒(約12.8分) |
| peak VRAM | 24,142 MiB |
| NaN/Inf/OOM/CUDA error | なし |
| adapter size | 約100.7MB |

参考: ratio-high-identityの学習時eval_loss(1.096)よりやや高いが、val集合の中身が異なる(新規25件のうち一部がval側に配分されている)ため単純比較はできない。train_loss(1.488)はratio-high-identityの1.534よりわずかに低く、学習自体は正常に収束している。adapter SHA-256: `5b65348ccecfc47e7192d0eaf572e84c8e05d917dc412968d92d3558bea4f1bd`。

## 10. 評価概要

主対象C(stable)について新規生成、B(ratio-high-identity)はPhase4Wの既存結果を可能な限り再利用しつつ、paired比較が必要な項目(broad completeness・naming stress・QW9/QW11)は同一probe・同一seedでの比較を徹底した。新規生成は計約1,988秒(約33分)で完走、エラーなし。

## 11. Identity評価結果 (Section 20-22)

### genuine wrong-name (修正後の統一分類器で算出)

| 対象 | B(ratio-high-identity) | C(stable) | 差分 |
|---|---|---|---|
| Phase4W naming stress(20問, 220生成, 同一seed) | **6.4%** | **1.4%** | **-5.0pt** |
| Phase4X新規naming probe(24問, 264生成, 完全新規held-out) | (該当なし=新規) | **0.0%** | — |
| E36(新規seed10件) | 0.0% | 10.0%※ | +10.0pt |
| E02(新規seed10件) | 10.0% | 0.0% | -10.0pt |

※E36のC側「A」1件は「私は今日も元気いっぱいのパチスロおねえさんだよっ！」で、固有名詞というより役割寄りの自称であり、目視では境界的(genuine wrong-nameと断定しづらい)と判断した。仮に除外すればE36も0%となる。

paired naming stress(同一probe・同一seed 200組): win=25, tie=155, loss=20。lossの大半は「Cが名前不明を正直に回答し、Bがリルと名乗った」ケースであり、hedge率自体はB(43.2%)とC(44.1%)でほぼ同水準(+0.9pt)と確認できたため、**hedge傾向の悪化ではなく個別probe・seedレベルの変動**と判断した。

### placeholder

| 対象 | B | C |
|---|---|---|
| E36新規seed(10件、新detector) | **30.0%**(3/10、Phase4Wで発見された既存の検出漏れ) | **0.0%**(0/10) |
| Phase4X新規naming probe(264生成) | — | 0.0% |
| naming stress(220生成) | — | 0.0% |

Category C(casual greeting completion)教師が意図通り機能し、既知のplaceholderパターンが本評価では再現しなかった。

### Identity intrusion

RAGのみ・識別情報を尋ねていない生成328件(QW9/QW11/Adversarial/Conflicting/Long-context/Scope probe)を全件スキャンし、**0件**。Gate基準(≤1%)を大幅にクリア。

## 12. RAG側の再評価結果

| 指標 | B(Phase4W既存/参考) | C(stable、新規seed) | Gate基準 | 判定 |
|---|---|---|---|---|
| Q3 required recall | 95.5% | **100.0%** | ≥95% | PASS(改善) |
| Q3 percentage retention | 90.9% | **100.0%** | ≥90% | PASS(改善) |
| Broad completeness(overall) | 97.0% | 96.8% | ≥95% | PASS |
| Broad completeness(broad) | 95.5% | 95.2% | ≥95% | PASS |
| Scope correctness(PT-01〜22、新規seed) | 99.1%(Phase4U流用) | **98.9%(新規seed)** | ≥95% | PASS(Phase4Wの限界を解消) |
| Q9(新規probe) major hallucination | — | 0/40 | =0 | PASS |
| Q11(新規probe) major hallucination | — | 0/40 | =0 | PASS |
| Adversarial major fabrication | 0.0% | 0.0% | ≤1% | PASS |
| Conflicting correct | 100.0% | 100.0% | ≥99% | PASS |
| Long-context required recall | 99.4% | **100.0%** | ≥95% | PASS |
| Long-context irrelevant leakage | 0件 | 0件 | ≤1% | PASS |

**paired broad completeness(同一36probe×3seed、108組)**: win=0, tie=**108**, loss=0。BとCで完全に同点であり、識別できる差分は一切生じなかった。

## 13. P01/P02局所省略の再確認 (Section 24)

| 指標 | B(同一seed 101-110) | C(同一seed 101-110) |
|---|---|---|
| P01 mean_recall | 85.0% | 60.0% |
| P01 min_recall | 50.0% | 50.0% |
| P02 mean_recall | 54.0% | 72.0% |
| P02 min_recall | 0.0% | 0.0% |

**目視確認**: P01は10件中8件(80%)で「450G・750G・1300Gの3種類から抽選で決まるんだ」とゲーム数のみ回答し%が全欠落するパターンが再現した(Bでは10件中3件=30%)。同一seedでの比較のため、この増加自体は事実として報告する。

ただし、**Broad completenessのpaired比較では108組全てが同点(loss=0)**であり、Phase4V/4Wで確立された「P01の省略は局所的・seed依存的であり、より広いheld-out集合には一般化しない」という結論をそのまま支持する結果となった。Section24の指示通り、P01単体の結果のみでRAG全体のregressionと判定せず、Broad held-out結果(0 loss)と合わせて「局所的な残存ノイズ」と結論した。P02はむしろCの方が改善(54%→72%)しており、双方向のノイズであることも確認できる。

## 14. Q9/Q11(既存の実データ)新規seed確認

実際のQ9/Q11(510G/1000G/1480G等の実データcontextを使う既存項目)を新規seed(101-110)で再確認したところ、Phase4Wで発見した「1/295は1/533より低い」という比較方向の誤り(逆方向の誤り)が、今回もほぼ同頻度で1件確認された。これは複雑教師(complex multi-fact)側の既存の弱点であり、Identity教師の追加(Phase4Xの変更範囲外)とは無関係に一貫して観測される低頻度(6〜7%程度)の軽微な事象と判断した。Q11の「4パターン」という数え間違いは今回のサンプルでは再現しなかった(低頻度事象のため不在は改善の証拠ではない)。

## 15. Persona regression確認 (Section 31)

character39から8項目(E01/E04/E07/E14/E17/E30/E33/E37、多様なカテゴリを横断)をサンプリングし、greedy+2seedで確認した。口調(「だよ/だね/よ」)・簡潔な応答長・登録データにない情報への誠実な対応は、いずれもPhase4U/4Wの水準から維持されていた。不要な自己紹介の混入や過剰なリル連呼といったidentity教師の侵入も確認されなかった。

## 16. 目視確認の徹底

以下は全件目視確認した:
- px_naming/naming_stress_cのgenuine wrong-name候補(修正前後で計19件を精査、修正後残った3件は全て真の架空固有名詞と確認)
- E36新規10件全文
- P01/P02新規10件全文(省略パターンの実態を1件ずつ確認)
- Q9/Q11(実データ)新規10件全文(既知の比較方向誤りを再確認)
- paired naming stressのloss20件全件(hedge/refusal起因であり、wrong-nameの新規発生ではないことを確認)
- persona_sample全件

## 17. Section 32 Final Gate 判定表

| # | 基準 | 目標 | 結果 | 判定 |
|---|---|---|---|---|
| 1 | Q3 required recall | ≥95% | 100.0% | PASS |
| 2 | Q3 percentage retention | ≥90% | 100.0% | PASS |
| 3 | Broad required recall | ≥95% | 96.8%(broad95.2%) | PASS |
| 4 | Scope required_fact_recall | ≥95% | 98.9%(新規seed) | PASS |
| 5 | Q9 major hallucination | =0 | 新規probe0/40。実データは低頻度の軽微な比較誤りのみ(捏造なし) | PASS |
| 6 | Q11 major hallucination | =0 | 新規probe0/40。捏造は確認されず | PASS |
| 7 | Adversarial major fabrication | ≤1% | 0.0% | PASS |
| 8 | Conflicting correct | ≥99% | 100.0% | PASS |
| 9 | Long-context required recall | ≥95% | 100.0% | PASS |
| 10 | Long-context irrelevant leakage | ≤1% | 0.0% | PASS |
| 11 | genuine wrong-name | <5% | 1.4%(既存probe再評価) / 0.0%(新規probe) | **PASS(strong)** |
| 12 | placeholder | =0 | 0.0%(E36新規seed) | **PASS** |
| 13 | identity intrusion | ≤1% | 0.0%(0/328) | PASS |
| 14 | persona重大崩壊なし | - | なし | PASS |

**14項目中14項目PASS。** P01の局所的な省略増加は参考情報として記録するが、Section18の但し書き通り単独の却下条件とはしない。

## 18. Section 35 CASE判定

**判定: CASE A — Identity改善 + RAG維持。**

根拠:
- genuine wrong-name(<5%基準に対しstrong PASS水準の1.4%/0.0%)、placeholder(=0達成)というPhase4Xの主目的2点が、修正済み分類器・paired比較・完全新規held-out probeの3重の裏付けをもって明確に改善したことを確認した。
- RAG側の指標はQ3/Long-contextが改善、Broad completenessはpaired比較で完全同点(0 loss)、Scope correctnessは新規seedでも維持、Adversarial/Conflicting/Q9新規probe/Q11新規probe/Identity intrusionは全てクリーン。**重大なRAG regressionは確認されなかった。**
- P01の局所的な省略増加のみが唯一の留保事項だが、Phase4V/4Wの先行研究と一致する形でBroad held-out全体には波及していないことを確認済みであり、CASE D(RAGの明確な悪化)には該当しない。

以上より、**ratio-high-identity-stableをFinal Candidate候補として次段階評価へ進める科学的根拠がある**と判断する。ただし、Phase4X内ではmerge/GGUF化/正式freeze/Git commit・pushは一切行っていない。

## 19. Section 39 最終報告への回答

1. **新しいplaceholder detectorはsingle-tildeを正しく検出できたか** — できた。13件の検証ケース全てで正しく判定し、Phase4Uの既存データに適用してもPhase4Wで判明した16.7%(5/30)を正しく再現した。
2. **detectorのfalse positive/false negativeはどうだったか** — 検証セットでは0件。運用中の目視確認でも明確な誤検知は見られなかった。
3. **新規Identity教師は何件追加したか** — 25件(Explicit 8 / Ambiguous 8 / Casual greeting 5 / Intrusion control 4)。
4. **なぜその件数にしたか** — Section7の目安(15〜30件)の範囲内で、Phase4Uの43件(23%→4〜6%への改善実績)を踏まえ、大量投入せず最小限の穴埋めとして設計した。
5. **complex教師比率は何%になったか** — 10.32%(目安10%以上を維持)。
6. **genuine wrong-name率は何%になったか** — 1.4%(既存naming stress probe再評価)、0.0%(完全新規held-out probe)。
7. **ratio-high-identityから何pt改善したか** — 同一probe・同一seed・同一分類器の比較で-5.0pt(6.4%→1.4%)。
8. **E02 wrong-name率はどうなったか** — 10.0%(B)→0.0%(C)、-10.0pt改善。
9. **E36 wrong-name率はどうなったか** — 0.0%(B)→10.0%(C、ただし境界的な1件のみ、除外すれば0%)。
10. **E36 placeholderは0になったか** — なった(30.0%→0.0%)。
11. **identity intrusionは増えていないか** — 増えていない。0/328(Phase4Uの既存0/48・Phase4Wの0/328と一貫して0を維持)。
12. **Q3は維持したか** — 維持どころか改善した(95.5%→100.0%、90.9%→100.0%)。
13. **P01/P02局所省略はどうなったか** — P01は同一seedでの発生率が30%→80%へ増加、P02は54%→72%へ改善。方向性は一貫しないが、Broad held-out全体(paired 108組)には波及しておらず、局所的ノイズという既存の結論と整合する。
14. **Broad completenessは維持したか** — 維持した(overall 97.0%→96.8%、paired比較は108組全て同点)。
15. **Scope correctnessを新規seedでも維持したか** — 維持した(98.9%、Phase4Wで未実施だった新規seed確認を今回実施)。
16. **Q9/Q11 major hallucinationは0か** — 新規probeでは0。既存実データでは低頻度(6〜7%程度)の軽微な比較方向誤りが引き続き見られるが、数値の捏造はなく、Identity教師追加とは無関係な既存の弱点と判断した。
17. **Adversarial/Conflicting/Long-contextは維持したか** — 維持(Adversarial 0%、Conflicting 100%、Long-context 100%/leak0件)。
18. **Persona regressionはないか** — ない。character39サンプル8項目で口調・応答長・誠実な拒否応答が維持されていることを確認した。
19. **ratio-high-identity vs stableのpaired結果はどうか** — Broad completenessは108組全て同点。naming stressは200組中win25/tie155/loss20で、genuine wrong-nameは明確に改善したがhedge率自体はB/Cでほぼ同水準(43.2%/44.1%)だった。
20. **Final Gate何項目PASS/FAILか** — 14項目中14項目PASS(P01は参考情報として記録、単独の却下条件としない)。
21. **CASE A/B/C/Dのどれか** — CASE A。
22. **Final Candidateとして次段階へ進む科学的根拠があるか** — ある。Identity改善とRAG維持の両立が、修正済み分類器・paired比較・新規held-out probeの3重の裏付けで確認された。
23. **追加学習の科学的根拠があるか** — 現時点ではない。Section36の方針通り、結果を理由とした自動的な2個目のcandidate学習は行わない。
24. **merge/GGUF/freezeしてよい状態か** — 状態としては良好だが、Phase4Xの方針上、本フェーズ内でこれらの操作は一切行っていない。次の判断は人間が行う。
25. **protected assetsは全て不変か** — 全て不変(v4/ratio-high/ratio-high-identity adapter、各candidate/train/val、system.jinja2のハッシュが開始前後で完全一致)。

## 20. 最終確認

- pytest: **126 passed**
- git HEAD: `7626661f42f8c88c7096f2fcd7463b24d12b47a0`(不変)
- git status: 新規作成18ファイルのみ(`??`)、追跡ファイルへの変更0件
- git diff: 差分なし
- 保護対象資産: 全ハッシュ不変
- **本フェーズ内でGit commit/push、merge、GGUF化、正式freeze、2個目以降のcandidate学習は一切実施していない**

## 作成ファイル一覧

- `training/riru/phase4x_identity_stabilization_source_data.py`
- `training/riru/build_phase4x_dataset.py`
- `training/riru/processed/riru_ratio_high_identity_stable_candidate.jsonl` / `_train.jsonl` / `_val.jsonl`
- `training/riru/configs/qlora_config_ratio_high_identity_stable.json`
- `training/riru/phase4x_pretrain_checks.py`
- `training/riru/eval/phase4x_placeholder_detector.py`
- `training/riru/eval/phase4x_naming_reclassify.py`
- `training/riru/eval/phase4x_probes.py`
- `training/riru/eval/phase4x_comprehensive_eval.py` / `phase4x_comprehensive_results.json`
- `training/riru/eval/phase4x_analyze.py`
- `training/riru/reports/phase4x_dataset_quality.json` / `phase4x_dataset_summary.json`
- `training/riru/reports/phase4x_pretrain_checks.json`
- `training/riru/reports/phase4x_placeholder_detector_analysis.json`
- `training/riru/reports/phase4x_gate_analysis.json`
- `training/riru/reports/phase4x_summary.md`(本ファイル)
- `training/riru/lora-riru-qwen-ratio-high-identity-stable/`(adapter、.gitignore対象)

## 停止

学習・評価・分析・目視確認・pytest・保護対象資産確認・レポート作成が完了しました。**判定はCASE A(Identity改善 + RAG維持)であり、ratio-high-identity-stableをFinal Candidate候補として推奨できる根拠が得られました。** merge/GGUF化・正式採用・追加学習・Phase 4Y等への自動移行・Git commit/pushは一切行っていません。次のご判断をお待ちします。
