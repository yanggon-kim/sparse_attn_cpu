### Run configuration summary

| run_id                 |   context_target |   context_actual_tok | quant              |   top_k |   csa_layers |   decode_steps |   prefill_tok_s | wall_clock   |   peak_rss_gb | correct   |
|:-----------------------|-----------------:|---------------------:|:-------------------|--------:|-------------:|---------------:|----------------:|:-------------|--------------:|:----------|
| niah_single_2_16384_q2 |            16384 |                16264 | IQ2XXS-w2Q2K-AProj |     512 |           21 |            117 |           1.112 | 4:08:36      |          92.5 | True      |
| niah_single_2_4096_q2  |             4096 |                 3969 | IQ2XXS-w2Q2K-AProj |     512 |           21 |            128 |           1.353 | 54:09.15     |          81.6 | True      |
| niah_single_2_8192_q2  |             8192 |                 7485 | IQ2XXS-w2Q2K-AProj |     512 |           21 |            128 |           1.243 | 1:45:40      |          85.3 | True      |
