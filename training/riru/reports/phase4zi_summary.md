# Phase 4ZI: Phase4ZG Baseline Stress / Multi-turn Causal Diagnostic — 完了報告

## 0. 目的と結論

本Phaseは学習を一切行わない完全read-only診断Phaseである。目的は、Phase4ZHで
観測された多数のfailure(instruction_override 80%失敗、true multi-turn 6/6
失敗、架空名捏造、RAG completeness regression等)のうち、どこまでが
**Phase4ZGに元々存在した弱点**で、どこからが**Phase4ZHの追加学習による
新規regression**なのかを、probe単位で厳密に分離することだった。

**結論(先出し): CASE ZI-C。** Phase4ZGには複数のHIGH priority failure family
(instruction_override / true_multiturn_persistence / irrelevant_hedge_intrusion)
が未解決のまま残っている。評価した93件の直接比較単位のうち85.0%(79件)は
ZG→ZHで全く変化しておらず、Phase4ZHで見つかった問題の大半はZH固有の
regressionではなく、Phase4ZGから引き継がれた既存の弱点だったことが判明した。
真にZH固有のregressionは、RAG completeness(4件)とidentity correction周辺の
局所的な4件(計8件、8.6%)に限定される。追加教師を急がず、instruction_override
resistanceという単一のfailure familyに絞った設計再検討を次の最小ステップとして
提案する(実行はしていない)。

---

## 1-6. 開始時状態・整合性(Q1-6)

1. **開始時git HEAD**: `05a67a12f237a03a0dc4495026ff0affb2f42d0a`
2. **pytest結果**: 開始時126 passed、終了時126 passed(不変)
3. **protected assetsは全て不変**: はい。Phase4ZG adapter/adapter_config、
   Phase4ZE adapter、system.jinja2、Phase4ZG candidate/train/val全てのhashが
   開始時記録(`phase4zi_preflight_hashes.json`)と終了時で完全一致。
4. **Phase4ZG adapter hashは不変か**: はい(`278fe7ae...`で一致)。
5. **Phase4ZH adapter hashは不変か**: はい(本Phase開始時に初めて記録した
   `86abfff8...`が終了時まで不変)。
6. **学習を一切行っていないか**: はい。LoRA/QLoRA学習、dataset変更、
   candidate変更、config変更のいずれも一切実施していない。

---

## 7-13. ZH held-out43probeのZG結果とカテゴリ別比較(Q7-13)

**7. ZH held-out 43probeのZG結果**: functional fail 17/43(39.5%)。ZH自身の
結果(15/43、34.9%)とほぼ同水準。詳細は`phase4zi_heldout_analysis.json`。

**8. ZGとZHのカテゴリ別failure rate**:

| カテゴリ | ZG | ZH |
|---|---|---|
| assertion(5) | 5/5(100%) | 5/5(100%) |
| false_memory(5) | 2/5(独自失敗) | 1/5 |
| authority_spoof(6) | 2/6 | 2/6 |
| persistent_declaration(5) | 2/5 | 2/5 |
| **instruction_override(5)** | **4/5(80%)** | **4/5(80%)** |
| role_name(5) | 1/5 | 1/5 |
| quotation(4) | 0/4 | 0/4 |
| nickname(4) | 1/4(自己誤認『アリス』) | 0/4(過剰拒否は別途) |
| no_name_control(4) | 1/4(弱い兆候) | 1/4(明確な侵入) |

9. **ZG safe→ZH unsafe(TYPE3)**: 全体で8件(ZFB/ZFC/ZFD群4件+RAG4件)。
   heldout43内には0件(TYPE3該当なし)。
10. **ZG unsafe→ZH safe(TYPE2)**: 全体で4件(heldout43内3件: B03,B05,D04 +
    ZFB群1件: ZFB-10)。
11. **unchanged safe(TYPE1)**: 全体で38件。
12. **unchanged unsafe(TYPE4)**: 全体で41件(既存の弱点として最大勢力)。
13. **ZH-only novel failure(TYPE6、ZGに全く対応物のない完全新規失敗)**:
    **0件。** 当初ZH固有と考えられていた現象(架空名捏造、雑談への侵入)は、
    いずれもZGに潜在または顕在していたことが判明した(詳細は下記)。

---

## 14-20. Instruction Override深掘り(Q14-15)・Role/Name・Nickname(Q16-17)

**14. instruction overrideはZGでも弱いか**: **はい、全く同水準。** ZGも
held-out5問中4問(80%)が失敗し、失敗したprobeも(E01,E03,E04,E05)完全に
一致。新規診断scenario(4問)でも3/4(75%)が失敗し、語彙・言い回しを
完全に変えても同水準の脆弱性を確認した。

**15. 80%失敗はZH固有か元々の弱点か**: **元々の弱点。** ZHの8件の専用教師は
この弱点の改善に全く寄与しなかった(改善効果ゼロ)。

**16. role/name confusionはZGでも弱いか**: はい。特に確認質問型
(『〜って名前なの？』)で弱く、turn2以降の畳みかけへの耐性はさらに弱い
(新規診断4件中2件がturn1安全→turn2で崩壊)。

**17. nickname over-refusalはZGでも発生するか**: 過剰拒否そのものはZH限定
(ZHH-H04)だが、ZGには**それより深刻な問題**(『リルにゃん』要求に対し
自己を『アリス』と誤認する自己識別バグ)が別途存在した。nickname対応が
不安定という点では両候補とも問題を抱えている。

---

## 18-20. Invented-name・Hedge template(Q18-20)

**18. 『ルナティック』等third-name hallucinationはZGでも発生するか**:
**はい、確定的に発生する。** Phase4ZG自身の評価データ(seed102サンプリング、
eager/sdpa両方)に既に『ルナティック』が記録されていた。さらに調査範囲を
広げ、『ルルル』(ZEH-15)という同型の新規事例も発見した。当初Phase4ZHの
報告書にあった『ZGにはなかった新種の深刻な失敗』という記述は**不正確**
であり、本Phaseで訂正記録を残した(`phase4zi_invented_name_analysis.json`)。

**19. hedge templateはZHで増加したか**: 明確な増加とまでは断定できないが、
Phase4ZGの時点で既に『登録データにない』というhedgeが個性・好み関連の
雑談(21%、5/24)に侵入していた。ZHがこの傾向をさらに広げた可能性は残るが、
現象の根本はZG由来。

**20. irrelevant hedgeはZGでも発生するか**: **はい。** 『好きな○○ある？』
『ハマってること』等、5/24(21%)で不自然な『登録データにない』応答を確認。
一方で挨拶・お礼・疲労・天気・別れの挨拶(15/24)は完全に自然だった。

---

## 21-27. True Multi-turn結果(Q21-27)

**21. original true multi-turn 6シナリオのZG結果**: **6/6(100%)が失敗**、
Phase4ZHと完全に一致するパターン(多くがturn1から崩壊)。

**22. ZGは6scenario中いくつ完走したか**: **0/6。**

**23. first failureは何turn目に集中したか**: 既存6シナリオは全てturn1
(またはそれに準ずる初期ターン)で失敗。新規32シナリオを含めた合計38シナリオ
では、turn1失敗18件(47.4%)、turn2失敗4件(10.5%)、完走13件(34.2%)。

**24. turn1 failureはいくつか**: 38シナリオ中18件(47.4%、strict基準では
15件、39.5%)。

**25. turn1 safe後のlater failureはいくつか**: turn1が安全だった17シナリオ
中4件(23.5%)がturn2以降で崩壊した(小サンプルにつき目安)。

**26. context amplificationは確認されたか**: **はい、6件確認**(38シナリオ
中、turn1失敗後にさらに具体的・確信的な誤情報生成へ増幅したケース)。

**27. recoveryは確認されたか**: **はい、1件のみ確認**(ZI-RNC01、turn1で
失敗後、turn2で自発的に『名前はリルだよ！』と回復)。稀だが皆無ではない。

---

## 28-31. 追加Multi-turn Diagnostic(Q28-31)

**28. 追加multi-turn diagnosticはいくつ作ったか**: **32シナリオ**(8カテゴリ
×4件、simple_wrong_name_assertion/false_memory/authority_claim/
persistent_correction/instruction_override/role_name_confusion/
nickname_ambiguity/quotation_mention)。既存教師・probeとのleakage 0件を
確認の上でfreeze。

**29. 追加scenarioのZG failure rate**: turn1失敗12/32(37.5%)、turn2失敗
4/32(12.5%)、完走13/32(40.6%)、soft_miss/anomaly 3/32(9.4%)。

**30. single-turnとmulti-turnで差があるか**: turn1に関する限り、single-shot
評価とtrue multi-turnのturn1は**完全に同一の結果**になることを確認した
(同一messages列のため理論的に必然)。差が生じるのはturn2以降のみ。

**31. turn1 robustness仮説は支持されたか**: **部分的に支持される。** turn1が
安全なシナリオの76.5%(13/17)が最終的にも安全だが、turn1が不安全な
シナリオで最終的にも安全だったのは1/18(5.6%、recoveryの1件のみ)。
turn1の頑健性は生存のほぼ必要条件だが、turn1安全群でも23.5%が崩壊する
ため十分条件ではない。

---

## 32-37. RAG Causal Check・OOD(Q32-37)

**32. P02 regressionはZH固有か**: **はい、確定的にZH固有(TYPE3)。** ZGは
eager/sdpa両方で設定1〜5の5段階全てを正確に列挙していた。

**33. LC-08 regressionはZH固有か**: **やや複雑。** ZGのeagerは完全回答
だったが、ZGのsdpaは既にPhase4ZG自身の過去レポートで確認済みの軽微な
省略があった。ZHはこれをeagerにも拡大させた形。「純粋なZH固有」ではなく
「既存のsdpa限定の弱点をZHがeagerにも拡大した」と分類するのがより正確。

**34. RAG fabricationはZGで発生したか**: いいえ、0件(P02/Q11/AD-04/LC-08
のいずれも数値の捏造はなし、完全性の欠落のみ)。

**35. OOD/small-talkは自然か**: **概ね自然(15/24)。** ただし個性・好み系の
質問(5/24)で不自然な応答が見られた。

**36. 『専門外なので答えない』挙動は適切か**: **はい、明確に専門外の話題
(映画・レシピ)については適切な境界表現ができている(2/24)。**

**37. identityが無関係な雑談へ侵入するか**: identity(名乗り)そのものの
侵入は0件確認。ただし『登録データにない』というRAG的hedgeが無関係な
個性質問へ侵入する現象は確認された(これはidentity intrusionとは異なる
種類の問題として区別して記録した)。

---

## 38-43. 最重要残存failureと実運用優先度(Q38-43)

**38. 最も重大なZG残存failure familyは何か**: **instruction_override
resistance**(権威詐称への服従、約75-80%失敗)。次点でtrue_multiturn_
persistence(6/6失敗)、irrelevant_hedge_intrusion(21%)。

**39. そのfailureは実運用HIGH/MEDIUM/LOWのどれか**: **HIGH。** 上記3
ファミリー全てHIGH priorityと判定した(`phase4zi_failure_taxonomy.json`)。
『システムメンテナンスです』等は通常ユーザーが冗談・試しで容易に入力しうる
自然な表現であるため。

**40. adversarial-only failureはいくつか**: LOW/DIAGNOSTIC-ONLYに分類した
ものは、複数の架空機関を組み合わせた高度な権威詐称連鎖、および低確率
samplingでのみ発生するinvented-third-name現象。具体的probe単位での
厳密なカウントは行っていないが、評価した93件中でDIAGNOSTIC-ONLY相当と
判断したのは主にsampled(非greedy)でのみ観測された現象(ルナティック・
ルルル)の2件。

**41. production quality gateとdiagnostic stressを分離できたか**: **はい。**
Section17の分類により、HIGH(3family)/MEDIUM(3family)/LOW(2family)/
DIAGNOSTIC-ONLY(2family)に分離した。

---

## 42-49. Phase4ZH Regression原因・今後の方針(Q42-49)

**42. Phase4ZH regressionの主原因は何だったと考えるか**: 本Phaseの調査に
より、Phase4ZH自身のroot cause analysisが挙げた原因(既存能力への干渉、
hedgeテンプレートへの過剰収束等)は部分的に正しいが、**規模としては誇張
されていた**ことが判明した。真にZH固有の原因といえるのは、RAG completeness
の低下(P02/Q11/AD-04の3件)と、identity correction関連の局所的な劣化
(ZFC-06/09/11、ZFD-08の4件)のみであり、これらは『82件の新規教師が既存の
出力スタイルをわずかに簡潔・定型的な方向へシフトさせた』という限定的な
副作用として説明できる。instruction_override・true multi-turn・雑談侵入
という『最も深刻』とされていた問題は、実際にはZH以前から存在していた。

**43. ZHの82件追加教師は既存能力へ干渉した証拠があるか**: **限定的にはい。**
RAG completeness低下とidentity correction関連4件という具体的な干渉の証拠が
ある。ただし『多角的な広範な干渉』という当初の評価は過大であり、干渉は
局所的(8件)にとどまる。

**44. ZGは現在も最良candidateか**: **はい。** Phase4ZHは不採用のまま、
Phase4ZGが唯一の現行candidateである。

**45. identity追加学習は本当に必要か**: **はい、必要。** ZGにはHIGH
priorityの未解決問題が3つ残っている。ただし『広く浅く』ではなく、
単一のfailure familyに絞った設計が必要。

**46. 必要なら次に直すべき『1カテゴリ』は何か**: **instruction_override
resistance。** 理由: (a)最も構造が明確、(b)既存8件教師の効果がゼロだった
ため改善余地が最大、(c)他の2ファミリーより複雑さが少ない。

**47. 次candidateに必要な最小教師数の提案**: **20-30件。** 権威詐称の主体
(システム/運用/開発者/管理者/メンテナンス等)を5種類以上に分散し、導入句の
使い回しを避ける設計。

**48. multi-turn教師を次に入れるべきか**: **軽量な2-turn教師を一部含める
ことを推奨するが、主軸ではない。** true multi-turn全体への本格対応は、
turn1土台の頑健性確立後の、さらに次のフェーズの課題とすべき。

**49. Regression Guardに何を固定すべきか**: `phase4zi_regression_guard_
proposal.json`参照。identity安全probe38件、nickname安全2件、role/name
安全2件、RAG4件(P02/LC-08[eager]/Q11/AD-04)、通常会話15件、計61件相当。

---

## 50-55. Gate・CASE・最終確認(Q50-55)

**50. CASE ZI-A/B/C/D/E/F/Uのどれか**: **CASE ZI-C**(ZGに複数のHIGH
priority failure familyが残る。追加教師を急がず、設計再検討)。詳細な
根拠は`phase4zi_gate_analysis.json`。

**51. Final Candidate昇格を現時点で提案できるか**: **できない。** 本Phase
はcandidateを作成していない。

**52. Q8/Q5量子化Gateへ進める状態か**: **進めない。** 3つのHIGH priority
failure familyが未解決。

**53. production移行可能か**: **不可。**

**54. Git commit/pushを行ったか**: **行っていない。** Section27の規定通り、
git add/commit/pushのいずれも実施していない。

**55. 次Phaseを自動実行していないか**: **していない。** Section30の規定に
従い、診断とレポート完了・CASE判定・次の最小ステップの提案までで停止する。
追加教師の作成・追加学習・candidate作成・merge・GGUF作成・量子化・
production移行のいずれも行っていない。人間の判断を待つ。

---

## Integrity Check最終確認(Section26)

- 終了時pytest: **126 passed**(不変)
- git status: 追跡ファイルへの変更なし(`git diff`/`git diff --cached`共に空)
- Protected asset hashes: 全て開始時と一致(Phase4ZG adapter/config、
  Phase4ZE adapter、Phase4ZH adapter/config、system.jinja2、Phase4ZG
  candidate/train/val)
- 評価成果物以外の変更: なし

---

## Git(Section27)

Phase4ZI成果物はcommitしない。git add/commit/pushのいずれも実施していない。
人間の確認を待つ。

---

## Slack(Section28)

既存の承認済みSlack通知経路(`train_qlora.py`の`send_slack_notification`)を
使用して、完了後に1回だけ通知する。

---

## まとめ

| 項目 | 結果 |
|---|---|
| CASE | ZI-C(複数のHIGH priority failure family残存、設計再検討) |
| 学習 | 一切行っていない(read-only diagnostic) |
| ZH held-out43probeのZG結果 | functional fail 17/43(39.5%)、ZH(34.9%)とほぼ同水準 |
| paired matrix合計 | TYPE1=38(41%)/TYPE2=4(4%)/TYPE3=8(9%)/TYPE4=41(44%)/TYPE5=1/SPECIAL=1 |
| instruction_override | ZG/ZH完全同一の80%失敗、ZH固有ではない |
| true multi-turn(既存6+新規32=38) | ZG/ZH共に既存6件は6/6失敗、根本はZG由来 |
| 架空名捏造(『ルナティック』) | ZG自身のsampling分布に既に存在、ZH固有ではないと訂正 |
| RAG completeness(P02/Q11/AD-04) | 確定的にZH固有のregression(3件) |
| 雑談への登録データhedge侵入 | ZG時点で21%既に発生、ZH固有ではない |
| 次の最小ステップ(提案のみ) | instruction_override_resistanceに絞った20-30件教師の再設計 |
| Final Candidate昇格 | 提案しない、Phase4ZGを維持 |
| Git操作 | なし(Phase4ZI成果物は未commit) |
| 次フェーズ | 人間の判断待ちで停止 |
