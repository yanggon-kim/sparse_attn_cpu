"""Worker-side selection-trace hook for vLLM 0.28.0 (DeepSeek-V3.2 / GLM-5 DSA models).
Import `selhook.worker_ext` inside every TP worker via LLM(worker_extension_cls="selhook.worker_ext.SelHookExt").
Active only when the SEL_TRACE env var (output directory) is set.
"""
