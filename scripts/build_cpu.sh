#!/bin/bash
# Build the instrumented ds4 CPU engine.
set -e
cd <WORKDIR>/ds4
make cpu -j"$(nproc)"
echo "built: ds4 ds4-server ds4-bench ds4-eval ds4-agent"
