import os, sys, json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "eval" / "results"
OUTPUT_FILE  = RESULTS_DIR / "human_eval_output.txt"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

from retrieve.orchestrator import answer_query

questions = [
    ("q01", "Show me PO-2026-P0058"),
    ("q02", "Get me invoice INV-2025-0001"),
    ("q03", "Look up customer cust_0005"),
    ("q04", "Find email msg_0001"),
    ("q05", "Show all purchase orders for Gerard Walsh"),
    ("q06", "Any customers complaining about water dripping?"),
    ("q10", "Need someone urgently, job is dangerous"),
    ("q11", "What has Gerard Walsh had done before?"),
    ("q16", "How much does boiler installation usually cost?"),
    ("q_mat", "What materials are needed for invoice INV-2025-0001?"),
    ("q_quote", "Get me a quote for a boiler installation"),
]

lines = []
lines.append("DACARag — Human Evaluation Output")
lines.append("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
lines.append("=" * 60)
lines.append("")
lines.append("Instructions:")
lines.append("  Read each ANSWER and SOURCES below.")
lines.append("  Score each metric 0.0-1.0 in the scorecard widget.")
lines.append("  Faithfulness  : are all claims backed by the sources shown?")
lines.append("  Ans Relevancy : does the answer address the question asked?")
lines.append("  Ctx Precision : were the retrieved sources actually useful?")
lines.append("  Ctx Recall    : did sources contain everything needed?")
lines.append("")

for qid, query in questions:
    print("Running " + qid + ": " + query[:55] + "...")
    try:
        result  = answer_query(query)
        answer  = result.get("answer", "")
        sources = result.get("sources", [])
        routing = result.get("routing", {})
        latency = result.get("latency_ms", "?")

        lines.append("=" * 60)
        lines.append("[" + qid + "] " + query)
        lines.append("=" * 60)
        lines.append("")
        lines.append("ROUTING : " + str(routing.get("intent","?")) + " via " + str(routing.get("paths",[])))
        lines.append("LATENCY : " + str(latency) + "ms")
        lines.append("")
        lines.append("ANSWER:")
        lines.append(answer)
        lines.append("")
        lines.append("SOURCES (" + str(len(sources)) + " retrieved):")
        for i, s in enumerate(sources, 1):
            lines.append("  [" + str(i) + "] " + str(s))
        lines.append("")
        lines.append("YOUR SCORES (fill in after reading above):")
        lines.append("  Faithfulness   : ___")
        lines.append("  Ans Relevancy  : ___")
        lines.append("  Ctx Precision  : ___")
        lines.append("  Ctx Recall     : ___")
        lines.append("")

    except Exception as e:
        lines.append("=" * 60)
        lines.append("[" + qid + "] " + query)
        lines.append("ERROR: " + str(e))
        lines.append("")

lines.append("=" * 60)
lines.append("SUMMARY TABLE (copy into report)")
lines.append("=" * 60)
lines.append("")
lines.append("Q ID     | Question                                    | Faith | AnsRel | CtxPre | CtxRec")
lines.append("---------|---------------------------------------------|-------|--------|--------|-------")
for qid, query in questions:
    lines.append(qid.ljust(8) + " | " + query[:43].ljust(43) + " |  ___  |  ___   |  ___   |  ___")
lines.append("         | AVERAGE                                     |  ___  |  ___   |  ___   |  ___")
lines.append("")
lines.append("Targets: Faithfulness > 0.90 | Ans Relevancy > 0.85 | Ctx Precision > 0.80 | Ctx Recall > 0.85")

output = "\n".join(lines)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(output)

print("")
print("Done. Output saved to:")
print(str(OUTPUT_FILE))
print("")
print("Open that file, read each answer and sources, then fill in your scores.")
