#!/usr/bin/env python3
"""Per-sample revisit package (doc §30). Usage: make_sample_report.py <run_dir>"""
import json, os, sys, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

run_dir = sys.argv[1]
gen = json.loads(open(os.path.join(run_dir, "outputs", "generations.jsonl")).read().splitlines()[0])
summ = json.load(open(os.path.join(run_dir, "analysis", "metrics_run_summary.json")))
sl = pd.read_parquet(os.path.join(run_dir, "analysis", "metrics_sample_layer.parquet"))
sel = pd.read_parquet(os.path.join(run_dir, "traces", "selected_kv.parquet"))
sample_meta = json.loads(open(os.path.join(run_dir, "prompts", "sample.json")).read())

sid = gen["sample_id"]
rep = os.path.join(run_dir, "analysis", "sample_reports", sid)
os.makedirs(rep, exist_ok=True)

json.dump(sample_meta, open(os.path.join(rep, "metadata.json"), "w"), indent=2)
open(os.path.join(rep, "generation.txt"), "w").write(gen["generated_text"])
json.dump(gen["generated_token_ids"], open(os.path.join(rep, "generation_tokens.json"), "w"))
json.dump({"is_correct": gen["is_correct"], "reference": gen["reference_answer"],
           "prediction": gen["benchmark_prediction"]}, open(os.path.join(rep, "benchmark_result.json"), "w"), indent=2)
sel.sample(min(5000, len(sel))).to_parquet(os.path.join(rep, "selected_kv_subset.parquet"), index=False)

# access raster (middle CSA layer)
layers = sorted(sel.layer_id.unique())
lid = layers[len(layers) // 2]
g = sel[sel.layer_id == lid]
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(g.decode_step, g.compressed_kv_index, s=2, alpha=0.3)
# mark needle compressed index if known
ratio = int(g.compression_ratio.iloc[0]) if len(g) else 4
ax.set(title=f"{sid} L{lid} access raster", xlabel="decode step", ylabel="compressed idx")
fig.tight_layout(); fig.savefig(os.path.join(rep, "access_raster.png"), dpi=110); plt.close(fig)

# retention plot (mean)
LAGS = [1, 2, 4, 8, 16, 32, 64]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(LAGS, [summ["overall_retention"][f"lag_{l}"] for l in LAGS], "k-o")
ax.set(title=f"{sid} retention", xlabel="lag", ylabel="retained", xscale="log")
fig.tight_layout(); fig.savefig(os.path.join(rep, "retention_plot.png"), dpi=110); plt.close(fig)

# notes
hi = sl.loc[sl.adjacent_overlap_mean.idxmax()]
lo = sl.loc[sl.adjacent_overlap_mean.idxmin()]
notes = f"""# Sample report: {sid}

- **Correct:** {gen['is_correct']}  (reference={gen['reference_answer']}, prediction={gen['benchmark_prediction'][:60]!r})
- **Context length (target):** {summ['context_length']} tokens; decode steps: {summ['n_decode_steps']}
- **Needle value:** {sample_meta.get('needle_value')} (HF token position ~{sample_meta.get('token_position_answer_hf')})
- **Highest-locality layer:** L{int(hi.layer_id)} adjacent_overlap={hi.adjacent_overlap_mean:.3f}
- **Lowest-locality layer:** L{int(lo.layer_id)} adjacent_overlap={lo.adjacent_overlap_mean:.3f}
- **Overall adjacent overlap:** {summ['overall_adjacent_overlap_mean']:.3f}; locality lift vs random: {summ['overall_locality_lift_mean']:.2f}x
- **Recency-baseline overlap:** {summ['overall_recency_overlap_mean']:.3f} (separates semantic locality from pure recency)

## Anomalies / notes
- Q2 quantized, CPU reference path; logical (not physical) KV reuse.
"""
open(os.path.join(rep, "notes.md"), "w").write(notes)
print(f"sample report -> {rep}")
