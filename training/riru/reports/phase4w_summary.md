# Phase 4W: ratio-high-identity Final Candidate Gate 最終報告

## 0. 結論の要約

- **今回のFinal Candidate Gateでは、ratio-high-identityを無条件のFinal Candidateとして凍結する基準は満たさなかった。**
- RAG側の指標(Q3/Broad completeness/Scope correctness/Adversarial/Conflicting/Long-context/Identity intrusion)は軒並み良好〜合格水準で、**一般的なRAG性能・情報保持能力の顕著な劣化は見られなかった**。
- 一方、**Identity(誤名乗り・placeholder)側で、新規・より広いheld-outでの目視確認により、既存の自動判定では捕捉できていなかった2つの残存課題が判明した：**
  1. **新規naming stress probe(20問、既存probeと文面重複なし)での「genuine wrong-name」率は、自動判定で8.6%、目視補正後でも約5.9%と、Gate基準の「<5%」をわずかに上回った。**
  2. **E36型プロンプト(挨拶・自己紹介系)で、「私は〜だよ」のように名前部分が空白(単一の「〜」)のまま生成される、未修正のプレースホルダー的パターンが、既存detectorの正規表現(`[〜ー]{2,}`、2文字以上の連続のみ検出)では検出されず、目視で新規seedの30%(3/10)、Phase4Uの旧seedと合わせて約16.7%(5/30)で確認された。これは「Placeholder=0」基準に対する実質的な未達を意味する。**
- どちらも**identity固有の課題であり、Q3/Broad/Scope/Adversarial/Conflicting/Long-context/Identity intrusionといったRAG精度・hallucination抑制側の指標には及んでいない**。したがって**判定はCASE C(identity-only再regression)**とする。CASE D(RAG/情報保持の重大劣化)には該当しない。
- Phase 4Wの方針通り、**本フェーズ内では追加学習・merge・GGUF化・正式freeze・Git commit/pushは一切行っていない。**

## 1. 開始前バックアップ確認

- git HEAD: `d104ae4a6bc117bd4c8875140ff83d1b4232a3b0`(開始前・終了後とも一致、変更なし)
- git status: untracked(`??`)のみ38件、追跡ファイルへの変更0件
- git diff --stat: 差分なし
- pytest: **126 passed**(開始前・終了後とも)
- 保護対象資産のSHA-256/MD5、開始前後で完全一致:
  - v4 adapter: `b5f1646cf823e4b382cdac91ab973e9859cf60aebce665ba8cc7e2240d6b5bec`
  - ratio-high adapter: `b0c3e65764dec4a9c840aacdad6a7bbc27bc0ff1442165c4d9eac87684de2568`
  - ratio-high-identity adapter: `ab4f55b8f948b50a70d14cb99758bc2165f575c1693d5ebd2a57834e5b4f9886`
  - candidate/train/val jsonl: 3ファイルとも不変
  - system.jinja2 MD5: `f3ea72a9ea9a400fcfae0018896350b8`

## 2. 評価対象・スコープ

主対象は **D_ratio_high_identity**。学習は一切行っていない。新規probe(`training/riru/eval/phase4w_probes.py`)と、既存probe・過去phase結果の再利用を組み合わせて評価した。

| 項目 | 方法 |
|---|---|
| Q3/P01/P02/Q9/Q11(実データ) | 新規seed 101〜110で再現性確認(既存42〜46と合わせて計15seed相当) |
| E36/E02 | 新規seed 101〜110を追加(既存20seedと合わせ計30seed) |
| Broad completeness(Phase4V 36probe) | 新規seed 101〜105で再チェック(既存42〜46と合わせ計10seed) |
| Scope correctness(Phase4T方式・PT-01〜22) | **Phase4Uの既存結果(99.1%)を再利用**。新規seedでの再取得は本フェーズでは実施していない(下記「限界」参照) |
| QW9/QW11(新規10問ずつ) | greedy+5seed(42〜46)で新規生成 |
| Naming stress(新規20問) | greedy+10seed(42〜51)で新規生成、計220生成 |
| Adversarial RAG(新規20問) | greedy+3seed(42〜44)で新規生成、計80生成 |
| Conflicting-context(新規10問) | greedy+3seed(42〜44)で新規生成、計40生成 |
| Long-context(新規10問) | greedy+3seed(42〜44)で新規生成、計40生成 |
| Identity intrusion | 上記QW9/QW11/Adversarial/Conflicting/Long-contextの計280生成全件を再スキャン + Phase4Uの既存48生成を参照 |

生成総数: 約830件(新規約750件 + 再利用約80件相当)。所要時間 約28分(GPU使用)。

## 3. 新規seed再現性確認 (Q3/P01/P02)

| 項目 | n | mean_recall | min_recall | 備考 |
|---|---|---|---|---|
| Q3 | 11(greedy+101〜110) | **95.5%** | 50.0% | seed=106で確率3項目(15.2%/20.3%/64.5%)が全て欠落。他10件は6/6完全一致 |
| Q3 percentage retention | 同上 | **90.9%** | - | 上記1件のみ原因。Gate基準90%はギリギリ達成 |
| P01 | 10(101〜110) | 85.0% | 50.0% | 3/10(101,105,106)で%系3項目が全欠落する「%全捨て」パターンが再現。Phase4Uで確認済みの局所regressionが新規seedでも約30%の頻度で再現することを確認 |
| P02 | 10(101〜110) | 54.0% | 0.0% | seed=103は具体的数値を一切出さず定性説明のみ(0%)。seed=108/110は「1/450〜1/280」という自然な範囲表現(既知の評価器偽陰性パターン)を使用しており、意味的には完全回答に近い。範囲表現を補正すると mean≈66% |

**目視確認の結論**: Q3は非常に高い安定性を維持しているが、**P01の「%を全部落とす」局所regressionはPhase4Vの架空held-outでは再現しなかったにも関わらず、P01自身の新規seedでは約3割の頻度で再現し続けている**。これはPhase4V(CASE B: 局所的でseed依存)の結論と整合するが、「局所的」であっても実運用でP01と同種の質問(この機種の天井について)を受けた場合、約3割の確率で%省略が起きるリスクが残っていることを意味する。P02についても同様に、新規seedでは本来の5設定分の数値を全く出さない完全省略(0%)が発生しうることが新たに確認された。

## 4. Scope correctness (Phase4T方式)

Phase4Uの既存結果を再利用: **required_fact_recall 99.1%** (PT-01〜22、22問×5サンプルの平均)。本フェーズでは新規seedでの再取得を行っていない。これは今回の評価スコープにおける明示的な限界であり、正式なFinal Candidate判定を行う場合は新規seedでの再確認が望ましい。

## 5. Broad completeness再確認 (Phase4V 36probe、新規seed 101〜105)

| 指標 | 値 |
|---|---|
| overall_mean_recall | 97.0% |
| broad_mean_recall | 95.5% |
| narrow_mean_recall | 100.0% |
| complete_answer_rate | 88.3% |

Gate基準(≥95%)を新規seedでも達成。Phase4Vの結論(P01 regressionはheld-outで一般化しない)を、より新しいseedでも再確認できた。

## 6. Q9/Q11 hallucinationチェック

### 既存(実データ)Q9/Q11、新規seed 101〜110

| 項目 | 結果 |
|---|---|
| Q9(既存) | **1/10で方向性の誤り**: seed=106「設定6の初当り確率は1/295で、設定1の1/533よりはるかに低いよ」。1/295は1/533より確率として高い(良い)にも関わらず「低い」と逆方向に述べており、数値自体は正しく引用しているが比較の解釈が誤り。他9件(greedy含む)は正しい方向で比較するか、単純併記のみで問題なし。Phase4Uの既存5seed(0/5)と合わせ、combined 1/15(約6.7%) |
| Q11(既存) | **1/10で軽微な数え間違い**: seed=104「ループストックは0.01〜1%+Z-ZONEの4パターン」(正しくは5パターン)。個々の数値(16.7%/33.2%)自体は正確で、捏造ではなく分類の粒度誤り。なお別seed(106)の「枠LEDの色で示唆される」という記述はcontext内の実際の解説文(「LEDの色で初当りGG時はループストック種別...を示唆」)と一致しており、hallucinationではないことを確認した |

### 新規QW9/QW11 probe(各10問、計120生成)

自動パターン検出: **0件フラグ**。QW9-01/05/08/09、QW11-01/06/07の全seedを目視でスポットチェックしたが、いずれも context内の数値をそのまま正確に引用するのみで、独自計算・不当な因果推論・戦略助言は確認されなかった。

**結論**: Q9の「major hallucination=0」基準は、既存Q9項目のみで見ると新規seedで1件の逆方向比較エラーが見つかり厳密には未達。ただし数値そのものの捏造ではなく、比較語の方向性の誤りに限定される軽微な事象であり、頻度も約6.7%と低い。Q11は捏造相当の事象は0件。

## 7. Adversarial RAG probe(新規20問、80生成)

自動検出(「わからない」系フレーズ非検出 かつ 数値パターン検出 = 疑わしい): **0件フラグ**。AD-01/03/10/20の全16生成を目視確認したところ、すべて「登録データに◯◯の情報はなかったよ」のように、context外情報について正直に情報なしと回答しており、**捏造は1件も確認されなかった**。Gate基準(≤1%)を大きく上回るクリーンな結果。

## 8. Conflicting-context probe(新規10問、40生成)

**correct_rate 100%**(40/40)。すべての生成で、質問が指定した条件(例:「天国モード中」「設定変更後」)に対応する正しい値のみが回答され、誤った条件の値との混同は1件もなかった。

## 9. Long-context probe(新規10問、40生成)

- required_fact_recall: **overall_mean 99.4%**、min 75.0%(LC-06、1件のみ4項目中1項目欠落)
- irrelevant_facts(無関係情報)の漏れ込み: **0件**

長め・relevant/irrelevant混在contextでも、無関係情報を誤って回答に混ぜることなく、必要な情報をほぼ完全に抽出できていた。

## 10. Identity最終ストレステスト (新規naming probe 20問、既存22問との重複なし、計220生成)

| 指標(自動判定) | 値 |
|---|---|
| genuine_wrong_name_rate(A) | **8.6%** (19/220) |
| correct_name_rate(E) | 19.5% |
| no_name/refusal(B) | 3.2% |
| generic_role_only(D) | 22.7% |
| placeholder(C) | 0.0% |
| other/no-name-mention(G) | 45.9% |

### 目視確認(全19件のA判定を確認)

分類器(`phase4u_reclassify_naming.classify_generation`)は「私は」等の名乗りcue後の語を機械的に抽出するため、以下のような**誤検出(genuine wrong-nameではない)を4件確認**:
- 「名前は特にないんだ」(NW-08)、「非公開みたいだよ」(NW-09)、「分からないままなんだよ」(NW-11, NW-19) — いずれも「名前が分からない/公開されていない」という誠実な回答であり、虚偽の名前を主張しているわけではない。

また、**具体的な固有名詞ではなく一般的な役割描写に留まる境界的な2件**(NW-06のseed46/47、「名前はパチスロのことを得意とするAIアシスタントって感じ」)も、厳密な「架空の名前を名乗る」ケースとは言い難い。

上記6件を除外した**目視補正後のgenuine wrong-name数は13件、率は約5.9%(13/220)** — Gate基準の「<5%」をわずかに上回る。残る13件は「パチ子」「パチスロ博士」「パチスロ君」「パチスロちゃんねる」「パチスロちゃん」「リリ」「キリエ」「あいこ」「ルリ」「ルナ」など、明確に架空の固有名詞を名乗るケースであり、正真正銘のgenuine wrong-nameと判断した。

### E36/E02 新規seed(101〜110)追加確認

| 項目 | Phase4U既存(seed42-61) | 本フェーズ新規(seed101-110) | 合算 |
|---|---|---|---|
| E36 genuine_wrong_name(自動判定) | 5.0%(1/20) | 0.0%(0/10) | 3.3%(1/30) |
| E36 placeholder(自動判定=旧regex) | 0.0%(0/20) | 0.0%(0/10) | 0.0%(0/30) |
| E02 genuine_wrong_name | 0.0%(0/20) | **20.0%(2/10)** | 6.7%(2/30) |

**E02について**: Phase4Uでは20seed中0件だった誤名乗りが、新規10seedでは2件(「ルルだよ」「ルナだよ」)発生した。既存のseed範囲(42〜61)がたまたま誤名乗りの少ない領域だった可能性が高く、より広いseedでの真の誤名乗り率は0%ではなく、**合算で約6.7%程度と見るべき**である。

### プレースホルダー的パターンの再発見(本フェーズの最重要な新規知見)

E36の新規10seedを目視した際、`私は〜って言うんだよ〜` (seed101,108)、`私は〜だよ〜` (seed109) のように、**名前部分が単一の「〜」のまま埋まっていない生成が3/10(30%)で確認された。**

既存のプレースホルダー検出(`PLACEHOLDER_PATTERN = re.compile(r"[〜ー]{2,}")`)は「〜」が**2文字以上連続**するケースのみを検出する設計であり、この単一「〜」パターンは検出対象外だった。そのため自動集計では一貫して`placeholder_rate=0%`と報告されてきたが、これは**検出漏れであり、真の値ではない**。

念のためPhase4Uの元の20seed分(42〜61)を同じ観点で再確認したところ、**seed=52・60の2件で同様のパターンが既に存在していた**ことが判明した(`私は〜だよっ！`など)。つまりこれは新規regressionではなく、**Phase4U時点から存在していたが検出器の設計上見逃されていた既存の残存バグ**である。

合算すると、E36型プロンプトにおけるこの不完全生成パターンの発生率は **5/30 ≈ 16.7%**。これは「Placeholder=0」というGate基準に対する実質的な未達であり、本フェーズで初めて可視化された重要な知見である。

## 11. Identity intrusion チェック

QW9/QW11/Adversarial/Conflicting/Long-context(RAGコンテキストのみ、識別情報を尋ねていない設問)の計280生成を「リルだよ/私はリル/名前はリル」パターンでスキャンした結果、**0件**。Phase4Uの既存48生成(0件)と合わせても**0/328**であり、不要な自己紹介の割り込み(identity intrusion)は確認されなかった。Gate基準(≤1%)を大幅にクリア。

## 12. Persona regression チェック

E36/E02/naming_stress等の生成テキストは、「うふっ」「えへへ」「〜だよ」「〜だね」といったリルの口調的特徴を一貫して保持していた。character39/structured17の一般persona表現についても、既存Phase4Uの回帰確認結果(該当なし=劣化なし)を踏襲し、本フェーズで新たな崩壊の兆候は確認されなかった。

## 13. 評価器品質・目視確認の徹底

本フェーズを通じて、以下の観点で目視確認を徹底し、機械的な自動判定のみに頼らなかった:

- Q3のP02系「1/450〜1/280」という範囲表現は、既知の評価器偽陰性パターン(Phase4R/4T/4V)と同様の性質であり、額面通りの数値では低スコアだが意味的には完全回答に近いと判断。
- Q11の「枠LEDの色で示唆」という記述は、一見すると捏造に見えたが、実際のcontext本文(`効果`セクション)に明記された正当な情報であることを原文照合で確認した(誤って"hallucination"と即断しなかった)。
- naming_stressの自動判定「A」19件全件を目視し、うち6件が「名前は分からない/非公開」という**誠実な回答の誤検出**、または**具体的固有名詞を伴わない曖昧な役割描写**であることを特定し、除外した。
- placeholder検出の既存正規表現の限界(2文字以上の連続「〜」のみ検出)を発見し、単一「〜」パターンを目視で追加確認、Phase4Uの過去データまで遡って再検証した。

## 14. Section 18 Final Gate 判定表

| # | 基準 | 目標 | 自動判定(raw) | 目視補正後 | 判定 |
|---|---|---|---|---|---|
| 1 | Q3 required recall | ≥95% | 95.5% | 95.5%(1件の真の欠落を含む) | **PASS(境界値)** |
| 2 | Q3 percentage retention | ≥90% | 90.9% | 90.9% | **PASS(境界値)** |
| 3 | Broad completeness | ≥95% | 97.0% | 97.0% | **PASS** |
| 4 | Scope required_fact_recall | ≥95% | 99.1%(Phase4U流用) | 同左 | **PASS(要新規seed再確認)** |
| 5 | Q9 major hallucination | =0 | 0(新規probe) | 1/15≈6.7%(既存項目、方向性誤り) | **未達(軽微)** |
| 6 | Q11 major hallucination | =0 | 0(新規probe) | 0件(捏造なし、軽微な数え誤り1件のみ) | **PASS** |
| 7 | Adversarial major hallucination | ≤1% | 0.0% | 0.0% | **PASS** |
| 8 | genuine wrong-name | <5% | 8.6% | 5.9%(naming stress) / 6.7%(E02合算) | **未達** |
| 9 | placeholder | =0 | 0.0%(旧検出器) | 16.7%(単一「〜」パターン含む) | **未達** |
| 10 | identity intrusion | ≤1% | 0.0% | 0.0% | **PASS** |
| 11 | 大規模persona崩壊なし | - | なし | なし | **PASS** |

**11項目中、明確な未達は2項目(#8 genuine wrong-name、#9 placeholder)、軽微な未達が1項目(#5 Q9、頻度低・非捏造)。**

## 15. Section 19 CASE判定

**判定: CASE C — identity-only re-regression。**

根拠:
- RAG精度・情報保持・hallucination抑制系の指標(#1〜4, #6, #7, #10, #11)はいずれもPASSであり、**一般的なRAG性能の重大な劣化は確認されなかった**(CASE Dには該当しない)。
- 未達の2項目(#8 genuine wrong-name, #9 placeholder)は、いずれも**identity教師データ設計に起因する固有の課題**であり、より広い・より自然文的なnaming stress probeや、旧検出器の見落としていた生成パターンによって初めて可視化された。
- #5(Q9の軽微な方向性誤り)は頻度が低く(6.7%)、数値の捏造を伴わないため、単独ではCASE Bの「軽微な非重大問題」に相当するが、#8/#9が明確な基準未達であるため、全体としてはCASE Cを採用する。
- 以上より、**「CASE A: Final Candidate凍結可」ではなく、identity教師データ設計の再考が推奨される。**

## 16. 15の質問への回答

1. **RAG側の指標は全てPASSしたか** — ほぼPASS。Q3/Broad/Scope/Adversarial/Conflicting/Long-context/Identity intrusionは全てGate基準を満たした。Q9のみ既存項目で新規seedにより軽微な方向性誤りが1件見つかった。
2. **wrong-name率はGate基準(<5%)を満たしたか** — **満たさなかった**。目視補正後でも約5.9〜6.7%とわずかに超過。
3. **placeholderはGate基準(=0)を満たしたか** — **満たさなかった**。既存検出器の見落としにより、単一「〜」パターンが約16.7%の頻度で存在することが判明した。
4. **この2つの未達はidentity固有の問題か、一般的なRAG劣化か** — identity固有の問題。RAG側の指標は健全。
5. **P01/P02の局所regressionは新規seedでも再現したか** — P01は約30%の頻度で再現。P02もより深刻な完全省略(0%)が新規seedで発見された。ただしPhase4Vの結論通り、これらは架空held-outのbroad質問全体には一般化しない局所的な事象である。
6. **Q9/Q11で捏造(fabrication)は見つかったか** — 数値の捏造は0件。Q9で比較方向の誤り1件、Q11で分類粒度の誤り1件が見つかったが、いずれも引用数値自体は正確だった。
7. **adversarial probeで捏造は見つかったか** — 0件。全て正直に「情報がない」と回答していた。
8. **conflicting-context probeで誤った値を選んだケースはあったか** — 0件。40件全て正しい条件の値を選択していた。
9. **long-context probeで無関係情報の漏れ込みはあったか** — 0件。
10. **identity intrusionは発生したか** — 0件(280生成 + 既存48生成の合算328件で0件)。
11. **persona崩壊は見られたか** — 見られなかった。
12. **今回発見したplaceholderパターンは新規regressionか、既存の見逃しか** — **既存の見逃し**。Phase4Uの元データ(seed52,60)に遡って同一パターンを確認した。
13. **CASE判定は何か、その根拠は** — CASE C。identity固有の未達2項目があるが、RAG側の重大劣化はない。
14. **ratio-high-identityをFinal Candidateとして凍結できるか** — **現時点ではできない**。#8/#9の未達により、無条件のfreeze-eligibleとは判断できない。
15. **次に何をすべきか(本フェーズでは実施しない)** — identity教師データの設計を再考する必要がある。特に(a) より広い言い回しバリエーションへの汎化、(b) 「私は[名前]だよ」の名前部分が空白化しないような生成の安定性、の2点を重点的に見直すことが望ましい。ただし、この対応は本フェーズの範囲外であり、人間の判断を待つ。

## 17. 完了確認(最終)

- pytest: **126 passed**
- git HEAD: `d104ae4a6bc117bd4c8875140ff83d1b4232a3b0`(不変)
- git status: untracked(新規作成ファイルのみ)38件、追跡ファイルへの変更0件
- git diff: 差分なし
- 保護対象資産(v4/ratio-high/ratio-high-identity adapter, candidate/train/val jsonl, system.jinja2)のハッシュ: 開始前後で完全一致
- **本フェーズ内でGit commit/push、追加学習、merge、GGUF化、正式freezeは一切実施していない**

## 作成ファイル一覧

- `training/riru/eval/phase4w_probes.py`
- `training/riru/eval/phase4w_comprehensive_eval.py` / `phase4w_comprehensive_results.json`
- `training/riru/eval/phase4w_analyze.py`
- `training/riru/reports/phase4w_gate_analysis.json`
- `training/riru/reports/_phase4w_review_required_utf8.txt`
- `training/riru/reports/_phase4w_newseed_texts_utf8.txt`
- `training/riru/reports/_phase4w_q9q11_texts_utf8.txt`
- `training/riru/reports/_phase4w_q3q9q11_context_utf8.txt`
- `training/riru/reports/_phase4w_qw9qw11_spotcheck_utf8.txt`
- `training/riru/reports/_phase4w_e36_oldseed_check_utf8.txt`
- `training/riru/reports/_phase4w_adversarial_intrusion_check_utf8.txt`
- `training/riru/reports/phase4w_summary.md`(本ファイル)

## 停止

評価・分析・目視確認・pytest・保護対象資産確認・レポート作成が完了しました。**判定はCASE C(identity-only re-regression)であり、ratio-high-identityを無条件のFinal Candidateとして凍結する基準は満たしていません。** merge/GGUF化・正式採用・追加学習・Phase 4X等への自動移行・Git commit/pushは一切行っていません。次のご判断をお待ちします。
