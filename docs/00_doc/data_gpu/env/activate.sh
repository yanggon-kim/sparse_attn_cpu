# source this before any campaign command
export WORKDIR=<WORKDIR>
export CUDA_HOME=<WORKDIR>/env/cuda
export PATH=$WORKDIR/shim:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_LOGGING_LEVEL=INFO
source $WORKDIR/env/vllm/bin/activate
export NLTK_DATA=<WORKDIR>/nltk_data
export PYTHONPATH=<HOME>/00_sparse_attn/01_github/sparse_attn_cpu/scripts/gpu:${PYTHONPATH:-}
