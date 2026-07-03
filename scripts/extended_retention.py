#!/usr/bin/env python3
"""Long-horizon retention / working-set for a long-decode run (exploits many steps).
Usage: extended_retention.py <run_dir>
"""
import json, os, sys
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

run_dir = sys.argv[1]
LAGS = [1,2,4,8,16,32,64,128,256,512,1024,2048]
WINS = [1,2,4,8,16,32,64,128,256,512,1024]
sel = pd.read_parquet(os.path.join(run_dir,"traces","selected_kv.parquet"),
                      columns=["layer_id","decode_step","compressed_kv_index"])
# per (layer, step) -> set
sets = {}
for (l,s), g in sel.groupby(["layer_id","decode_step"]):
    sets[(l,s)] = set(g.compressed_kv_index.tolist())
layers = sorted({l for (l,_) in sets})
steps = sorted({s for (_,s) in sets})
smin, smax = steps[0], steps[-1]

def overlap(a,b):
    return len(a&b)/len(a) if a else float("nan")

retention = {}
for lag in LAGS:
    vals=[]
    for l in layers:
        for s in steps:
            if (l,s) in sets and (l,s-lag) in sets:
                vals.append(overlap(sets[(l,s)], sets[(l,s-lag)]))
    retention[lag] = sum(vals)/len(vals) if vals else float("nan")

ws = {}
for w in WINS:
    ratios=[]
    for l in layers:
        for i,s in enumerate(steps):
            window=[sets[(l,steps[j])] for j in range(max(0,i-w+1), i+1) if (l,steps[j]) in sets]
            if window:
                u=set().union(*window)
                # top_k ~ max set size
                k=max(len(x) for x in window)
                ratios.append(len(u)/(w*k))
    ws[w]=sum(ratios)/len(ratios) if ratios else float("nan")

out={"n_steps":len(steps),"n_layers":len(layers),"retention":retention,"working_set_ratio":ws}
json.dump(out, open(os.path.join(run_dir,"analysis","extended_retention.json"),"w"), indent=2)

fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].plot(LAGS,[retention[l] for l in LAGS],"o-",color="#2f5496")
ax[0].set(xscale="log",xlabel="decode lag (steps)",ylabel="retained fraction",
          title=f"Long-horizon retention ({len(steps)} steps, 11% keep)")
ax[0].grid(alpha=.3)
ax[1].plot(WINS,[ws[w] for w in WINS],"s-",color="#c55a11")
ax[1].set(xscale="log",xlabel="decode window (steps)",ylabel="working set / (w·top_k)",
          title="Working-set growth")
ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(run_dir,"analysis","extended_retention.png"),dpi=120)
print("retention:", {l:round(retention[l],3) for l in LAGS})
print("working_set:", {w:round(ws[w],3) for w in WINS})
