# Phase4ZT Policy C 設計

## 基本アーキテクチャ

```
User Input
    |
    v
Conservative Dispatch (Phase4ZR、無変更)
    |
    +-- SMALL_TALK / OOD_FACTUAL / PACHISLOT_CONVERSATIONAL / PACHISLOT_FACTUAL (confident)
    |        -> 既存のmode別policy(Phase4ZP)をそのまま使用。本フェーズで一切変更しない。
    |
    +-- UNKNOWN
             |
             v
        Policy C (本フェーズの対象)
```

**最重要不変条件(Section2/Mandatory Invariant)**: `UNKNOWN` かつ `strict_RAG_generation` かつ `context_absent` という組み合わせは0件でなければならない。Phase4ZRのUNKNOWN経路はこの組み合わせを許してしまっていた(context自体を一切注入していなかった)ため、Phase4ZSでfabricationを引き起こした。Policy Cはこの穴を塞ぐことが唯一の目的である。

## Policy C候補

### C1 — Retrieval + Context + RAG Answer
全てのUNKNOWNについて、既存retriever(read-only)でcontextを取得し、必ず既存の strict RAG system prompt(`config/prompts/system.jinja2`、無変更)へcontext付きで送る。

### C2 — Clarification First
全てのUNKNOWNについて、retrieval/RAG生成を一切行わず、Phase4ZRで作成済みのclarificationプロンプト(`phase4zr_unknown_ux_prompt_b.txt`)へcontextなしで送る。

### C3 — Conservative Mixed Policy
UNKNOWNについてまずretrieval(read-only)を実行するが、**retrieval score/hitそのものをmode分類器として使わない**(Section3準拠)。代わりに、**クエリ文字列と検索結果(タイトル+本文)の間の字句的(lexical)重なり**を計算する: 日本語の一般的な助詞・助動詞・機能語(は/が/を/に/で/と/の/か/も/よ/ね/だ/です/ます等、閉じた品詞的stopword、機種固有名詞辞書ではない)を除いた残りの部分文字列(2文字以上)が、検索結果のテキストに literal に出現するかを確認する。出現すれば「pachislot文脈である可能性が高い」と判断しC1と同じRAG経路へ、出現しなければC2と同じclarification経路へ送る。

これは:
- retrieval scoreのしきい値ではない(Section3で明示的に禁止)
- 機種固有名詞の新規辞書ではない(除外するのは一般的な助詞のみで、機種名・AT名等は一切列挙しない)
- 既存のconservative dispatchの判定ルール自体は一切変更しない(dispatchはPolicy Cの前段で完結しており、Policy Cはdispatchが出したUNKNOWNという結果を受け取るだけ)

### C3設計上の限界(正直な事前開示)
Conservative dispatch(Phase4ZR)の設計上、`GENERAL_PACHISLOT_TERMS`や`FACTUAL_METRIC_KEYWORDS`に一致する語を含むqueryは、そもそもUNKNOWNではなくPACHISLOT_FACTUAL/CONVERSATIONALへ直接dispatchされる。したがって、UNKNOWN集合の中には、dispatch自身が使う語彙による「pachislot信号」は原理的に存在しない。C3の字句重なりチェックは、dispatchが見ていない**検索結果側の語彙**との重なりを見るという点で、dispatchとは独立した新しい情報源(retrieval結果)を使っており、意味のある追加信号になり得るが、事前に「全てclarification行きになる可能性」も認識した上で実施する。
