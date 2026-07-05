#!/usr/bin/env python3
"""Per-layer MoE routing structure: expert-usage concentration + a static-vs-dynamic
decomposition of the adjacent-step overlap.

Usage: analyze_moe_concentration.py <out_plot_dir> <run_dir> [<run_dir> ...]
Reads  run_dir/traces/selected_experts.parquet and run_dir/analysis/moe_metrics_sample_layer.parquet.
Writes run_dir/analysis/moe_concentration.json (per-layer table + summary), and one plot
       <out_plot_dir>/moe_06_per_layer_concentration.png for the run with the most decode steps.

Idea: "adjacent overlap" (temporal) conflates two things. A layer can look temporally local simply
because it has FAVORITE experts (a concentrated marginal usage distribution), even if consecutive tokens
were independent. We separate:
  static preference  = expected adjacent overlap if tokens were independent draws from the layer's own
                       marginal usage distribution  = (sum_i p_i^2)/n_used, where p_i = usage rate of
                       expert i (fraction of tokens whose top-k includes i).
  dynamic corr (x)   = observed adjacent overlap / static preference  (the extra step-to-step stickiness).
Uniform baseline is n_used/n_expert. Concentration is summarized by the top-expert usage rate, the number
of experts covering 50% of routing, top-k coverage, and normalized entropy.
"""
import json, os, sys, math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN, GREY = "#1f6feb", "#c55a11", "#1a7f37", "#647083"


def perlayer(run_dir):
    sel = pd.read_parquet(os.path.join(run_dir, "traces", "selected_experts.parquet"),
                          columns=["decode_step", "layer_id", "is_hash_layer", "expert_id", "n_expert", "n_used"])
    n_expert = int(sel.n_expert.dropna().iloc[0]) if sel.n_expert.notna().any() else 256
    k = int(sel.n_used.dropna().iloc[0]) if sel.n_used.notna().any() else 6
    mo = pd.read_parquet(os.path.join(run_dir, "analysis", "moe_metrics_sample_layer.parquet"))[
        ["layer_id", "adjacent_overlap_mean"]]
    obs = {int(r.layer_id): float(r.adjacent_overlap_mean) for _, r in mo.iterrows()}
    rows = []
    for (lid, ishash), g in sel.groupby(["layer_id", "is_hash_layer"]):
        steps = g.decode_step.nunique()
        counts = g.expert_id.value_counts()
        c = counts.values.astype(float)
        total = c.sum()
        p = counts / steps          # usage rate per expert
        q = c / total               # share of selection slots
        cs = np.sort(c)[::-1]
        cum = np.cumsum(cs) / total
        nfor = lambda f: int(np.searchsorted(cum, f) + 1)
        rows.append(dict(
            layer=int(lid), is_hash=bool(ishash), n_steps=int(steps),
            distinct=int((c > 0).sum()), top_expert_rate=float(p.max()),
            n_experts_50pct=nfor(.5), top6_cov=float(cs[:6].sum() / total),
            top32_cov=float(cs[:32].sum() / total),
            entropy_norm=float(-(q * np.log2(q)).sum() / math.log2(n_expert)),
            static_overlap=float((p.values ** 2).sum() / k),
            observed_overlap=obs.get(int(lid), float("nan")),
        ))
    d = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
    d["dynamic_mult"] = d["observed_overlap"] / d["static_overlap"]
    return d, n_expert, k


def summarize(d, n_expert, k):
    uni = k / n_expert
    ln = d[~d.is_hash]
    hs = d[d.is_hash]
    n = len(ln)
    thirds = {"shallow": ln.iloc[:n // 3], "middle": ln.iloc[n // 3:2 * n // 3], "deep": ln.iloc[2 * n // 3:]}
    grp = {name: {
        "layers": [int(s.layer.min()), int(s.layer.max())],
        "top_expert_rate": float(s.top_expert_rate.mean()),
        "observed_overlap": float(s.observed_overlap.mean()),
        "static_overlap": float(s.static_overlap.mean()),
        "dynamic_mult": float(s.dynamic_mult.mean()),
    } for name, s in thirds.items()}
    def agg(s):
        return {"n_layers": int(len(s)), "distinct": float(s.distinct.mean()),
                "top_expert_rate": float(s.top_expert_rate.mean()),
                "n_experts_50pct": float(s.n_experts_50pct.mean()),
                "top6_cov": float(s.top6_cov.mean()), "entropy_norm": float(s.entropy_norm.mean()),
                "static_overlap": float(s.static_overlap.mean()),
                "observed_overlap": float(s.observed_overlap.mean()),
                "dynamic_mult": float(s.dynamic_mult.mean())}
    return {
        "n_expert": n_expert, "n_used": k, "uniform_overlap": uni,
        "learned": agg(ln), "hash": agg(hs), "depth_thirds": grp,
        "corr_static_observed": float(ln.static_overlap.corr(ln.observed_overlap)),
        "corr_dynamic_toprate": float(ln.dynamic_mult.corr(ln.top_expert_rate)),
        "peakiest_layers": [[int(r.layer), round(float(r.top_expert_rate), 2)]
                            for _, r in ln.nlargest(4, "top_expert_rate").iterrows()],
        "flattest_layers": [[int(r.layer), round(float(r.top_expert_rate), 2)]
                            for _, r in ln.nsmallest(4, "top_expert_rate").iterrows()],
    }


def plot(d, n_expert, k, out_png):
    L = d.layer.values
    ishash = d.is_hash.values
    uni = k / n_expert
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].bar(L, d.top_expert_rate, color=[ORANGE if h else BLUE for h in ishash])
    ax[0].axhline(uni, ls="--", color=GREY, lw=1)
    ax[0].set(ylabel="most-used expert's\nper-token rate",
              title=f"Per-layer MoE routing concentration ({int(d.n_steps.max())} decode steps)")
    ax[0].text(0.5, uni + .05, "hash layers", color=ORANGE, fontsize=9)
    ax[0].text(len(L) * 0.45, 0.9, "learned layers", color=BLUE, fontsize=9)
    ax[0].set_ylim(0, 1); ax[0].grid(alpha=.3, axis="y")
    static = d.static_overlap.values
    dyn = np.clip(d.observed_overlap.values - static, 0, None)
    ax[1].bar(L, static, color=GREEN, label="static preference (marginal-only overlap)")
    ax[1].bar(L, dyn, bottom=static, color=BLUE, label="dynamic step-to-step correlation")
    ax[1].axhline(uni, ls="--", color=GREY, lw=1, label=f"uniform-random floor ({k}/{n_expert})")
    ax[1].set(xlabel="layer index (hash then learned)", ylabel="adjacent-step\nexpert overlap",
              title="What drives each layer's overlap: static expert preference vs temporal correlation")
    ax[1].legend(fontsize=8, loc="upper left"); ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    out_dir = sys.argv[1]
    run_dirs = sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    best = None
    for rd in run_dirs:
        d, n_expert, k = perlayer(rd)
        summ = summarize(d, n_expert, k)
        out = {"run_id": os.path.basename(rd), **summ,
               "per_layer": d.round(4).to_dict(orient="records")}
        json.dump(out, open(os.path.join(rd, "analysis", "moe_concentration.json"), "w"), indent=2)
        ln = summ["learned"]
        print(f"{os.path.basename(rd)}: learned top_rate={ln['top_expert_rate']:.2f} "
              f"static={ln['static_overlap']:.3f} observed={ln['observed_overlap']:.3f} "
              f"dyn={ln['dynamic_mult']:.2f}x  corr(static,obs)={summ['corr_static_observed']:.2f} "
              f"corr(dyn,conc)={summ['corr_dynamic_toprate']:.2f}")
        if best is None or int(d.n_steps.max()) > int(best[0].n_steps.max()):
            best = (d, n_expert, k)
    if best:
        d, n_expert, k = best
        plot(d, n_expert, k, os.path.join(out_dir, "moe_06_per_layer_concentration.png"))
        print(f"wrote {out_dir}/moe_06_per_layer_concentration.png (from run with {int(d.n_steps.max())} steps)")


if __name__ == "__main__":
    main()
