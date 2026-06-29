#!/bin/bash
# Environment check for the DeepSeek-V4 KV locality experiment (doc Part XI).
echo "== CPU =="; lscpu | grep -E "Model name|^CPU\(s\)|Socket|NUMA node\(s\)|Thread"
echo "== RAM =="; free -h | head -2
echo "== Disk (/home) =="; df -h /home | tail -1
echo "== Python =="; python3 --version
for p in numpy pandas pyarrow matplotlib transformers tokenizers tabulate scipy; do
  python3 -c "import $p,sys;print('  $p',getattr(__import__('$p'),'__version__','?'))" 2>/dev/null || echo "  $p MISSING"
done
echo "== ds4 binary =="; ls -la <WORKDIR>/ds4/ds4 2>/dev/null
echo "== model =="; ls -la <WORKDIR>/models/*.gguf 2>/dev/null | awk '{print $5,$9}'
