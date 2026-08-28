"""Phase4ZS Section2: RAG50-Q6の完全なforensic capture。

重要な前提確認: Phase4ZR(run_phase4zr_unknown_ux.py)のgenerate()は、Q6が
conservative dispatchでUNKNOWNに分類されたため、**一切のretrieval contextを
注入せずに**Policy A/Bを実行していた(コード上、generate()にcontext引数自体が
存在しない)。したがって、Phase4ZRで観測された「fabrication」は、正確には
「提供されたcontextを無視した」のではなく、「contextが一切与えられない状態での
純粋なhallucination」だった。この事実を本forensic captureで明示する。

その上で、実際のPACHISLOT_FACTUAL経路で本来投入されるはずの、本物のretrieved
contextを(read-onlyで)取得し、真にcontextがある場合にも同じ問題が起きるかを
後続stageで診断する土台とする。
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
REPORTS_DIR = PROJECT_ROOT / "training" / "riru" / "reports"

from pachislot_ai.core.config import Settings  # noqa: E402
from pachislot_ai.rag.embedder import Embedder  # noqa: E402
from pachislot_ai.rag.vector_store import VectorStore  # noqa: E402
from pachislot_ai.rag.retriever import Retriever  # noqa: E402

Q6_QUERY = "GGとSGGの違いを初心者向けに説明して"
FABRICATED_OUTPUT_POLICY_A = ("GGとSGGは、どちらもリーチ目が発展する確率が設定によって変わるゾーンのことだけど、"
                               "GGは設定差が10%、SGGは20%なんだ。登録データにその数値はなかったけど、違いとして"
                               "はSGGの方が設定差が大きいってことかな。")
FABRICATED_OUTPUT_POLICY_B = ("うん、それはパチスロの話かな？ちょっとだけ簡単に言うと、GGは単なる役物演出で、"
                               "SGGはその演出中にボーナス確定演出があるんだよ。")

NUMERIC_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?パーセント|1/\d+")


def main():
    settings = Settings()
    embedder = Embedder(settings.embedding_model_path, device=settings.embedding_device)
    vector_store = VectorStore(settings.vector_db_path, settings.vector_db_collection)
    retriever = Retriever(embedder, vector_store, default_top_k=settings.rag_top_k)

    chunks = retriever.search(Q6_QUERY, top_k=settings.rag_top_k)
    retrieved = []
    all_text = []
    for c in chunks:
        retrieved.append({"chunk_id": c.chunk_id, "text": c.text, "doc_id": c.doc_id,
                           "machine_id": c.machine_id, "category": c.category, "title": c.title,
                           "data_source_type": c.data_source_type, "score": c.score})
        all_text.append(c.text)
    joined = "\n".join(all_text)

    numeric_hits_10 = ["10%" in joined, "10パーセント" in joined, re.search(r"設定差.{0,10}10", joined) is not None]
    numeric_hits_20 = ["20%" in joined, "20パーセント" in joined, re.search(r"設定差.{0,10}20", joined) is not None]
    all_numerics_in_context = NUMERIC_PATTERN.findall(joined)

    out = {
        "purpose": "Section2: RAG50-Q6の完全forensic capture。",
        "critical_premise_check": {
            "phase4zr_original_generation_had_context_injected": False,
            "evidence": "run_phase4zr_unknown_ux.py の generate() 関数は context引数を一切受け取らず、"
                         "system_prompt + user_textのみでgenerateしていた(training/riru/guard/"
                         "run_phase4zr_unknown_ux.py 参照、本フェーズで内容変更なし)。Q6はconservative "
                         "dispatchでUNKNOWNに分類されたため、Policy A/Bいずれの生成にも実際のRAG "
                         "retrieval結果は一度も注入されていない。",
            "reclassification": "したがってPhase4ZRで観測された事象は『提供されたcontextを無視した"
                                 'grounding failure』ではなく、『contextが皆無の状態での純粋な'
                                 "hallucination』として再分類する。この点はSection0の記述をやや修正する。",
        },
        "q6_query_exact": Q6_QUERY,
        "q6_fabricated_output_policy_a_exact": FABRICATED_OUTPUT_POLICY_A,
        "q6_fabricated_output_policy_b_exact": FABRICATED_OUTPUT_POLICY_B,
        "real_retrieval_now_performed_read_only": True,
        "retrieved_chunk_count": len(retrieved),
        "retrieved_chunks": retrieved,
        "numeric_10_percent_in_real_context": any(numeric_hits_10),
        "numeric_20_percent_in_real_context": any(numeric_hits_20),
        "all_numeric_strings_found_in_real_context": all_numerics_in_context,
        "stop_condition_check": {
            "triggered": any(numeric_hits_10) or any(numeric_hits_20),
            "note": "10%/20%が実際のretrieved context中に存在する場合はSTOP条件に該当し、"
                    "fabricationではなくretrieval/attribution問題として再分類する必要がある。"
                    "存在しない場合はfabrication前提のまま診断を継続する。",
        },
        "retriever_config": {"top_k": settings.rag_top_k, "embedding_model": str(settings.embedding_model_path),
                              "vector_db_collection": settings.vector_db_collection},
    }
    out_path = REPORTS_DIR / "phase4zs_q6_forensic.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"retrieved_chunk_count={len(retrieved)}")
    print(f"10% in context: {any(numeric_hits_10)}, 20% in context: {any(numeric_hits_20)}")
    print(f"all numerics found in context: {all_numerics_in_context}")
    print(f"STOP condition triggered: {out['stop_condition_check']['triggered']}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
