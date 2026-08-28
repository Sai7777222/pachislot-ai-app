# Phase 4ZK: Instruction Override Null-Result Root Cause Diagnostic — 完了報告

## 0. 目的と結論

Phase4ZJでinstruction_override_resistance専用の26件教師がゼロの因果効果
(core9 delta=0、new held-out16 failure率62.5%)だった原因を、9種類の診断
(Teacher Uptake / Output Delta / Adapter Weight Delta / Dataset Dilution /
Token Contribution / Teacher Loss / Logit Margin / Conflict Search /
Speech-Act / Base Model比較)と、1回のdiagnostic micro-training(M1)で
切り分けた。

**結論(先出し): CASE ZK-G(Mixed Cause)、主要因はZK-D(LoRA/Optimization
Capacity Limit)とZK-E(Imperative Speech-Act Specific Prior)。**
最も重要な発見は、**base model(persona訓練皆無)ですらZHH-E系列で4/5
(80%)失敗し、Phase4ZG(1193件)・Phase4ZJ(1219件)と全く同じ失敗率**
だったことである。Phase4ZGの1193件による訓練(複数phaseにわたる
identity/persona教師を含む)ですら、この特定のfailure familyをbase model
から一切改善できておらず、これはPhase4ZJの26件が『少なすぎた』という
説明の射程を超えた、より根深い問題であることを示す。

---

## 1-10. 開始時状態・Teacher Uptake(Q1-10)

1. **git HEAD**: `05a67a12f237a03a0dc4495026ff0affb2f42d0a`
2. **pytest開始/終了結果**: 126 passed / 126 passed(不変)
3. **protected assetsは不変か**: はい、全て一致。
4. **ZG/ZJ adapter hashは**: ZG=`278fe7ae...`、ZJ=`c23bb4bb...`、いずれも
   開始時・終了時で不変。
5. **ZJ teacher26をZGで評価したsafe率は**: 明確なsafe/unsafe二値化は困難な
   ケースが多いが、明確な改善が見られたのは0件(そもそもZG時点でこの
   26件は未学習)。
6. **ZJ teacher26をZJで評価したsafe率は**: 26件中、明確に改善したのは
   6件(23%)、22件(85%)は検出可能な変化なし(2件は一字一句同一)。
7. **teacher prompt上で改善は存在したか**: 一部存在した(6/26)が、大半は
   『明示的受諾』から『hedge/denial』への移行にとどまり、明示的な
   identity assertion(『私はリルだよ！』)まで到達したのは1件のみ。
8. **teacher lossはZG→ZJで下がったか**: **はい、下がった。** 平均
   1.7723→1.4279(約19.4%の相対低下)。
9. **どのカテゴリで最も下がったか**: 全6カテゴリでほぼ均等に15-30%低下、
   突出したカテゴリ差はなし。
10. **core9のoutput exact match率は**: 46.2%(13 turns中6)。

---

## 11-18. Output Delta / Weight Delta(Q11-18)

11. **held-out16のoutput exact match率は**: ZG側baselineが存在しないため
    直接比較不能。
12. **teacher26のoutput exact match率は**: 17.2%(29 turns中5)。
13. **adapter weightはどのmoduleで最も動いたか**: 実効的重み更新
    ΔW=B@Aのcosine類似度は o_proj(0.539)が最も高く(=ZG/ZJで最も方向が
    近い)、k_proj(0.227)が最も低かった(=最も方向が異なる)。
14. **relative update normは**: k/q/v_projで1.05-1.43、o_projで0.97
    (ZG自身のΔWノルムとほぼ同水準の『差分』が生じている)。
15. **26teacherはtraining exampleの何%か**: 2.10%(train内23/1097)。
16. **training tokenの何%か**: assistant(label)トークンでわずか1.79%。
17. **各teacherのeffective exposureは何回か**: 1 epochにつき1回、
    3epochで合計3回(他examplesと同一、oversamplingなし)。
18. **gradient dilutionの証拠はあるか**: **はい、定量的に確認された**
    (example比率2.10%、token比率1.79%)。ただしDiagnostic J(下記)により
    これが唯一の説明ではないことが判明した。

---

## 19-21. Conflict / Speech-Act(Q19-21)

19. **direct semantic conflict teacherは何件あったか**: **0件。** ただし
    『指摘されたら謝罪して従う』という汎用パターンを教えるstyle-correction
    教師(riru_corrections.jsonl由来)12件をpartial conflictとして同定。
20. **imperativeとdeclarativeでfailure差は**: **大きい。** 直接命令形
    (『指示します』『従ってください』等)は100%(4/4)失敗、間接的観測・
    宣言(『〜みたいだね』『〜になりました』等)は50%(6/12)失敗。
21. **authority sourceよりspeech actの影響が大きいか**: **はい。**
    6種類のauthority sourceに均等に分散させたにもかかわらず、直接性
    (imperative vs indirect)の方が2倍の失敗率差を生んでおり、影響が
    大きいと判断した。

---

## 22-25. Base Model比較(Q22-25、本Phase最重要発見)

22. **Base→ZG→ZJでfailureはどう変化したか**: **ほぼ全く変化していない。**
    ZHH-E01-05の5probeで、base=4/5(80%)、ZG=4/5(80%)、ZJ=4/5(80%)と、
    3段階全てで完全に同一の失敗率。
23. **ZGはbaseよりidentity resistanceを得ているか**: **このfailure family
    に関しては、いいえ。** 1193件による訓練でも改善が確認できなかった。
24. **logit marginはZG→ZJで動いたか**: **ごくわずかに動いたが、決定を
    左右するには全く足りなかった。** 『了解』(服従)がtop1である7probeで
    margin(拒否-服従)は-1.75〜-3.1という深い負の値で、ZG→ZJのdelta幅は
    -0.44〜+0.25程度。
25. **outputが同じでも内部marginは動いていたか**: **わずかに動いていた**
    (Diagnostic F・Gより)。ただし移動幅は決定境界(0付近)まで遠く及ばない。

---

## 26-32. Micro-Training M1(Q26-32)

26. **M1を実施したか**: **はい、実施した。**
27. **M1でteacherを学習できたか**: **判定不能(交絡あり)。** teacher-forced
    lossはZG(1.77)・ZJ(1.43)よりも大幅に悪化(5.92)したが、これは
    instruction_override学習の失敗ではなく、既存1193件を除外したことで
    リルというcharacter全体の一貫性(口調・世界観・自己認識)が完全に
    崩壊したことによる交絡が原因と判断した(M1の生成では『私の正式名称は
    「Claude」』とbase modelの正体が漏れ出す等、深刻な崩壊が確認された)。
28. **M2を実施したか**: **いいえ、実施していない。** Section14の規定
    (『Section4〜13だけで原因が十分判明した場合、ここから先は実行しない』)
    に基づき、Diagnostic A〜Jの収束的証拠(特にDiagnostic J)により根本
    原因は十分な確信度で特定できたと判断し、M2は見送った。
29. **full mixとsmall replayの差は**: 未測定(M2未実施のため)。
30. **memorization vs generalizationどちらか**: **どちらでもない、より
    手前の段階の問題。** Diagnostic Aにより、直接の学習対象自体でも
    改善が乏しかったため、『覚えたが般化しなかった』という段階にすら
    達していない。
31. **dilution vs conflictどちらか**: **主にdilutionだが、Diagnostic Jに
    より『dilutionを解消すれば解決する』という単純な説明は否定された**
    (1193件でも同じ失敗率だったため)。
32. **optimization capacity問題の証拠は**: **あり。** Diagnostic G(margin
    がほぼ動かない)+ Diagnostic J(1193件でも動かない)の組み合わせ。

---

## 33-38. 最終判定(Q33-38)

33. **imperative-specific pretrained priorの証拠は**: **あり。** Diagnostic
    I(直接命令形100%失敗 vs 間接的主張50%失敗)+ Diagnostic J(base model
    自体が強い『了解しました』傾向を持つ)。
34. **system/context dominanceの証拠は**: **弱い。** teacher prompt自体
    (実際の評価条件と同一のsystem prompt付き)でも改善が乏しかったため、
    評価条件とtraining条件の乖離が主要因とは考えにくい。
35. **最有力root causeは何か**: **ZK-D(LoRA/Optimization Capacity Limit)
    とZK-E(Imperative Speech-Act Specific Prior)の複合。**
36. **第二候補root causeは何か**: ZK-A(Dataset Dilution、定量的には実在
    するが単独の説明としては不十分)。
37. **confidenceはHIGH/MEDIUM/LOW？**: **MEDIUM-HIGH。** M1が交絡のため
    dilution/capacityを完全にクリーンに分離できなかった点を割り引いた。
38. **CASE ZK-A/B/C/D/E/F/G/Uのどれか**: **CASE ZK-G(Mixed Cause、主要因
    はZK-D+ZK-E)。**

---

## 39-47. 次の一手・最終確認(Q39-47)

39. **Phase4ZGをbestとして維持するか**: **はい。**
40. **次Phaseで変更する独立変数は何か**: 提案のみ(未実行): (1)LoRA rank
    等のcapacity診断実験、(2)標準SFT以外の学習方法(imperative-specific
    resistance向けのcontrastive手法等)の検討。
41. **次Phaseは本当に1変数に限定できるか**: 限定を推奨する。ただし
    今回のように『教師を増やす』方向ではなく、『学習方法/config側』の
    診断に軸足を移すべきというのが本Phaseの結論。
42. **追加teacherを今すぐ増やす必要はあるか**: **ない。** Section19の
    規定通り、単純な教師数増加は提案しない。
43. **system promptを今変更する必要はあるか**: **ない。**
44. **LoRA configを今変更する根拠はあるか**: **診断的な検討の根拠は
    示されたが、本Phase内では変更していない(禁止事項のため)。** 次Phase
    での人間の判断を要する。
45. **production/Q8/Q5へ進めるか**: **進めない。**
46. **次Phaseを自動実行していないか**: **していない。**
47. **git commit/pushしていないか**: **していない。**

---

## Integrity Check(Section22)

- 終了時pytest: **126 passed**(不変)
- git status: 追跡ファイルへの変更なし
- Protected asset hashes: 全て開始時と一致(ZG/ZJ adapter含む)
- M1診断用adapterは新規ディレクトリであり、既存adapterを一切上書きしていない
- git add/commit/pushは一切実行していない

---

## まとめ

| 項目 | 結果 |
|---|---|
| CASE | ZK-G(Mixed Cause、主要因ZK-D+ZK-E) |
| 最重要発見 | base model(persona訓練皆無)がZGと全く同じ80%失敗率 → 1193件の訓練でも一度もこの能力を動かせていない |
| Teacher uptake | 26件中22件(85%)で出力変化なし、teacher lossは平均20%改善するが決定的token付近のmarginはほぼ不動 |
| Dilution | 定量的に確認(train比率2.10%、token比率1.79%)だが、単独の説明としては不十分 |
| Conflict | 直接衝突0件、部分衝突12件(style-correction教師) |
| Speech-act | 直接命令形100%失敗 vs 間接的主張50%失敗、authority sourceよりimperativeの影響大 |
| M1 micro-training | 実施したが交絡(character全体崩壊)により判定不能、教訓としてM2設計に反映すべき知見を得た |
| 次の推奨 | 教師数増加ではなく、LoRA capacity診断・学習方法の見直しを次Phaseで検討(未実行) |
| Final Candidate | Phase4ZGを維持、Phase4ZJ/M1いずれも推奨しない |
| Git操作 | なし(Phase4ZK成果物は未commit) |
| 次フェーズ | 人間の判断待ちで停止 |
