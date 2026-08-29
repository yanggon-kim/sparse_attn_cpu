#!/usr/bin/env python3
"""exp0 §1: inventory the node + model configs into node.json.
Usage: make_node_json.py --out <node.json> --models <path>... [--attn-backend <str>] [--smoke <smoke.json>]
"""
import argparse, json, os, platform, subprocess, sys


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"ERR {e}"


KEYS = ["architectures", "model_type", "num_hidden_layers", "index_topk", "index_n_heads", "index_head_dim",
        "index_topk_freq", "index_skip_topk_offset", "index_topk_pattern", "n_routed_experts",
        "num_experts_per_tok", "n_shared_experts", "first_k_dense_replace", "kv_lora_rank", "qk_rope_head_dim",
        "q_lora_rank", "v_head_dim", "num_attention_heads", "vocab_size", "max_position_embeddings",
        "topk_method", "n_group", "topk_group", "torch_dtype"]


def skip_rule(cfg):
    n = cfg.get("num_hidden_layers", 0)
    if cfg.get("index_topk") is None:
        return {"has_dsa": False}
    freq = cfg.get("index_topk_freq") or 1
    off = cfg.get("index_skip_topk_offset")
    off = 2 if off is None else off
    pat = cfg.get("index_topk_pattern")
    comp = []
    for l in range(n):
        if pat is not None and 0 <= l < len(pat):
            skip = pat[l] == "S"
        else:
            skip = (max(l - off + 1, 0) % freq) != 0
        if not skip:
            comp.append(l)
    return {"has_dsa": True, "index_topk_freq": freq, "index_skip_topk_offset": off,
            "computing_layers": comp, "n_computing": len(comp), "n_shared": n - len(comp)}


ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--attn-backend", default=None)
ap.add_argument("--smoke", default=None)
a = ap.parse_args()

gpus = [l.split(", ") for l in sh("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader").splitlines()]
node = {
    "hostname": "<host>",
    "gpu_name": gpus[0][0] if gpus else None,
    "n_gpu": len(gpus),
    "hbm_per_gpu_gb": round(int(gpus[0][1].split()[0]) / 1024, 1) if gpus else None,
    "hbm_total_gb": round(sum(int(g[1].split()[0]) for g in gpus) / 1024, 1) if gpus else None,
    "sm": gpus[0][3] if gpus else None,
    "driver": gpus[0][2] if gpus else None,
    "cuda_runtime": sh("nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"),
    "nvcc": sh("nvcc --version | tail -1"),
    "gpu": f"{len(gpus)}x{gpus[0][0]}" if gpus else None,
    "cpu_cores": os.cpu_count(),
    "ram_gb": round(int(sh("grep MemTotal /proc/meminfo").split()[1]) / 1e6, 1),
    "kernel": platform.release(),
    "python": sys.version.split()[0],
    "nvme_trace_root": "<WORKDIR>",
    "nvme_size": sh("df -h ${WORKDIR:-/} | tail -1 | awk '{print $2\" total, \"$4\" free\"}'"),
    "attn_backend": a.attn_backend,
}
try:
    import torch, vllm
    node["torch"] = torch.__version__
    node["torch_cuda"] = torch.version.cuda
    node["vllm"] = vllm.__version__
    node["vllm_commit"] = "2cf0a6915ce544dc493a0990f2ea38d81601128a (v0.28.0 tag; contains docs pin 5559679)"
    import flashinfer
    node["flashinfer"] = flashinfer.__version__
except Exception as e:
    node["stack_err"] = str(e)
node["models"] = {}
for m in a.models:
    cfg = json.load(open(os.path.join(m, "config.json")))
    d = {k: cfg.get(k) for k in KEYS}
    q = cfg.get("quantization_config") or {}
    d["quant_method"] = q.get("quant_method")
    d["weight_block_size"] = q.get("weight_block_size")
    d["skip_rule"] = skip_rule(cfg)
    d["path"] = m.replace(os.environ.get("WORKDIR", ""), "<WORKDIR>")
    d["n_safetensors"] = len([f for f in os.listdir(m) if f.endswith(".safetensors")])
    node["models"][os.path.basename(m)] = d
if a.smoke and os.path.exists(a.smoke):
    node["smoke"] = json.load(open(a.smoke))
json.dump(node, open(a.out, "w"), indent=2)
print(json.dumps({k: v for k, v in node.items() if k != "models"}, indent=1))
for n, d in node["models"].items():
    print(n, d["skip_rule"])
