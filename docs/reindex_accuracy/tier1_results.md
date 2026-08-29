# exp3 tier 1 — re-index accuracy (perplexity primary, RULER sanity)

Teacher-forced perplexity of a scored continuation after the KV prefix was physically re-indexed (impl A: 64-token block permutation with block-table update; impl B: per-token row permutation, block table untouched). `clean2` = identical rerun = run-to-run noise floor; `ctrl_identity` = hook active with identity permutation; `ctrl_numeric` = same item processed alongside a filler request. Verdict `equivalent_to_noise_floor`: |mean ΔPPL| within the clean2 CI bound and per-token |Δlogprob| p90 within 3× the floor's.

| model | benchmark | prefix | mode | impl | n | PPL | ΔPPL vs clean [95 % CI] | token |Δlogprob| mean / p90 | equivalent |
|---|---|---:|---|---|---:|---:|---|---|---|
| DeepSeek-V3.2 | wikitext2 | 2048 | clean | - | 50 | 3.0006 | +0.00000 [+0.00000, +0.00000] | 0.0000 / 0.0000 | — |
| DeepSeek-V3.2 | wikitext2 | 2048 | clean2 | - | 50 | 3.0024 | +0.00180 [-0.00002, +0.00385] | 0.0219 / 0.0587 | — |
| DeepSeek-V3.2 | wikitext2 | 2048 | ctrl_identity | B | 50 | 3.0023 | +0.00167 [-0.00021, +0.00380] | 0.0216 / 0.0578 | True |
| DeepSeek-V3.2 | wikitext2 | 2048 | ctrl_numeric | - | 50 | 3.0021 | +0.00149 [-0.00348, +0.00590] | 0.0450 / 0.1342 | True |
| DeepSeek-V3.2 | wikitext2 | 2048 | perm_once_A | A | 50 | 3.0027 | +0.00208 [-0.00018, +0.00438] | 0.0230 / 0.0646 | True |
| DeepSeek-V3.2 | wikitext2 | 2048 | perm_once_B | B | 50 | 3.0033 | +0.00272 [-0.00112, +0.00671] | 0.0671 / 0.1947 | True |
| DeepSeek-V3.2 | longbook_ppl | 32768 | clean | - | 10 | 1.0939 | +0.00000 [+0.00000, +0.00000] | 0.0000 / 0.0000 | — |
| DeepSeek-V3.2 | longbook_ppl | 32768 | clean2 | - | 10 | 1.0931 | -0.00081 [-0.00354, +0.00135] | 0.0258 / 0.0127 | — |
| DeepSeek-V3.2 | longbook_ppl | 32768 | clean3 | - | 10 | 1.0940 | +0.00004 [-0.00236, +0.00242] | 0.0267 / 0.0121 | True |
| DeepSeek-V3.2 | longbook_ppl | 32768 | clean4 | - | 10 | 1.0959 | +0.00196 [-0.00149, +0.00655] | 0.0272 / 0.0124 | True |
| DeepSeek-V3.2 | longbook_ppl | 32768 | ctrl_identity | B | 10 | 1.0942 | +0.00023 [-0.00151, +0.00202] | 0.0275 / 0.0128 | True |
| DeepSeek-V3.2 | longbook_ppl | 32768 | ctrl_numeric | - | 10 | 1.0923 | -0.00161 [-0.00335, +0.00038] | 0.0265 / 0.0123 | True |
| DeepSeek-V3.2 | longbook_ppl | 32768 | perm_once_A | A | 10 | 1.0925 | -0.00146 [-0.00601, +0.00190] | 0.0269 / 0.0127 | True |
| DeepSeek-V3.2 | longbook_ppl | 32768 | perm_once_A@8 | A | 10 | 1.0932 | -0.00075 [-0.00334, +0.00186] | 0.0269 / 0.0124 | True |
| DeepSeek-V3.2 | longbook_ppl | 32768 | perm_once_B | B | 10 | 1.1001 | +0.00620 [-0.00040, +0.01781] | 0.0277 / 0.0140 | False |
| DeepSeek-V3.2 | longbook_ppl | 32768 | perm_once_B@8 | B | 10 | 1.0931 | -0.00089 [-0.00201, +0.00018] | 0.0273 / 0.0120 | True |
| DeepSeek-V3.2 | longbook_ppl | 32768 | perm_once_B@9 | B | 10 | 1.0941 | +0.00012 [-0.00136, +0.00163] | 0.0267 / 0.0129 | True |
| DeepSeek-V3.2 | longbook_ppl | 65536 | clean | - | 10 | 1.1357 | +0.00000 [+0.00000, +0.00000] | 0.0000 / 0.0000 | — |
| DeepSeek-V3.2 | longbook_ppl | 65536 | clean2 | - | 10 | 1.1372 | +0.00147 [-0.00160, +0.00510] | 0.0371 / 0.0251 | — |
| DeepSeek-V3.2 | longbook_ppl | 65536 | perm_once_A | A | 10 | 1.1360 | +0.00030 [-0.00315, +0.00464] | 0.0366 / 0.0270 | True |
| DeepSeek-V3.2 | longbook_ppl | 65536 | perm_once_B | B | 10 | 1.1371 | +0.00138 [-0.00286, +0.00654] | 0.0368 / 0.0263 | True |
| DeepSeek-V3.2 | longbook_ppl | 131072 | clean | - | 10 | 1.0966 | +0.00000 [+0.00000, +0.00000] | 0.0000 / 0.0000 | — |
| DeepSeek-V3.2 | longbook_ppl | 131072 | clean2 | - | 10 | 1.0935 | -0.00312 [-0.00941, +0.00222] | 0.0331 / 0.0218 | — |
| DeepSeek-V3.2 | longbook_ppl | 131072 | perm_once_B | B | 10 | 1.0954 | -0.00121 [-0.00485, +0.00271] | 0.0343 / 0.0220 | True |
| GLM-5.2 | wikitext2 | 2048 | clean | - | 50 | 2.9835 | +0.00000 [+0.00000, +0.00000] | 0.0000 / 0.0000 | — |
| GLM-5.2 | wikitext2 | 2048 | clean2 | - | 50 | 2.9847 | +0.00127 [-0.00622, +0.00877] | 0.1170 / 0.3526 | — |
| GLM-5.2 | wikitext2 | 2048 | ctrl_identity | B | 50 | 2.9867 | +0.00321 [-0.00410, +0.01094] | 0.1174 / 0.3516 | True |
| GLM-5.2 | wikitext2 | 2048 | ctrl_numeric | - | 50 | 2.9896 | +0.00617 [-0.00164, +0.01456] | 0.1163 / 0.3498 | True |
| GLM-5.2 | wikitext2 | 2048 | perm_once_A | A | 50 | 2.9856 | +0.00215 [-0.00565, +0.01077] | 0.1184 / 0.3490 | True |
| GLM-5.2 | wikitext2 | 2048 | perm_once_B | B | 50 | 2.9827 | -0.00078 [-0.00765, +0.00632] | 0.1178 / 0.3512 | True |
| GLM-5.2 | longbook_ppl | 32768 | clean | - | 10 | 1.8828 | +0.00000 [+0.00000, +0.00000] | 0.0000 / 0.0000 | — |
| GLM-5.2 | longbook_ppl | 32768 | clean2 | - | 10 | 1.8770 | -0.00581 [-0.01977, +0.00602] | 0.1443 / 0.4375 | — |
| GLM-5.2 | longbook_ppl | 32768 | ctrl_identity | B | 10 | 1.8770 | -0.00579 [-0.01757, +0.00601] | 0.1418 / 0.4372 | True |
| GLM-5.2 | longbook_ppl | 32768 | ctrl_numeric | - | 10 | 1.8697 | -0.01308 [-0.03081, +0.00365] | 0.1404 / 0.4287 | True |
| GLM-5.2 | longbook_ppl | 32768 | perm_once_A | A | 10 | 1.8755 | -0.00732 [-0.01947, +0.00432] | 0.1386 / 0.4137 | True |
| GLM-5.2 | longbook_ppl | 32768 | perm_once_B | B | 10 | 1.8755 | -0.00729 [-0.02092, +0.00743] | 0.1436 / 0.4310 | True |
| GLM-5.2 | longbook_ppl | 65536 | clean | - | 10 | 2.2146 | +0.00000 [+0.00000, +0.00000] | 0.0000 / 0.0000 | — |
| GLM-5.2 | longbook_ppl | 65536 | clean2 | - | 10 | 2.2276 | +0.01296 [-0.00763, +0.03614] | 0.1720 / 0.5398 | — |
| GLM-5.2 | longbook_ppl | 65536 | perm_once_A | A | 10 | 2.2252 | +0.01065 [-0.00844, +0.03049] | 0.1728 / 0.5552 | True |
| GLM-5.2 | longbook_ppl | 65536 | perm_once_B | B | 10 | 2.2253 | +0.01065 [-0.00747, +0.02826] | 0.1732 / 0.5645 | True |
| GLM-5.2 | longbook_ppl | 131072 | clean | - | 10 | 2.5960 | +0.00000 [+0.00000, +0.00000] | 0.0000 / 0.0000 | — |
| GLM-5.2 | longbook_ppl | 131072 | clean2 | - | 10 | 2.5908 | -0.00521 [-0.03246, +0.02342] | 0.2020 / 0.6640 | — |
| GLM-5.2 | longbook_ppl | 131072 | perm_once_B | B | 10 | 2.5955 | -0.00054 [-0.02170, +0.01425] | 0.1996 / 0.6364 | True |
| GLM-5 | wikitext2 | 2048 | clean | - | 50 | 2.8630 | +0.00000 [+0.00000, +0.00000] | 0.0000 / 0.0000 | — |
| GLM-5 | wikitext2 | 2048 | clean2 | - | 50 | 2.8588 | -0.00424 [-0.01088, +0.00195] | 0.1102 / 0.3418 | — |
| GLM-5 | wikitext2 | 2048 | ctrl_identity | B | 50 | 2.8636 | +0.00059 [-0.00653, +0.00758] | 0.1116 / 0.3437 | True |
| GLM-5 | wikitext2 | 2048 | ctrl_numeric | - | 50 | 2.8621 | -0.00088 [-0.00697, +0.00495] | 0.1110 / 0.3418 | True |
| GLM-5 | wikitext2 | 2048 | perm_once_A | A | 50 | 2.8641 | +0.00115 [-0.00614, +0.00795] | 0.1092 / 0.3406 | True |
| GLM-5 | wikitext2 | 2048 | perm_once_B | B | 50 | 2.8607 | -0.00228 [-0.00988, +0.00519] | 0.1111 / 0.3450 | True |
| GLM-5 | longbook_ppl | 32768 | clean | - | 10 | 1.6099 | +0.00000 [+0.00000, +0.00000] | 0.0000 / 0.0000 | — |
| GLM-5 | longbook_ppl | 32768 | clean2 | - | 10 | 1.6018 | -0.00809 [-0.02342, +0.00219] | 0.0941 / 0.2614 | — |
| GLM-5 | longbook_ppl | 32768 | ctrl_identity | B | 10 | 1.5991 | -0.01082 [-0.02708, +0.00198] | 0.0970 / 0.2699 | True |
| GLM-5 | longbook_ppl | 32768 | ctrl_numeric | - | 10 | 1.6040 | -0.00586 [-0.01434, +0.00125] | 0.0935 / 0.2598 | True |
| GLM-5 | longbook_ppl | 32768 | perm_once_A | A | 10 | 1.6053 | -0.00462 [-0.02105, +0.00569] | 0.0918 / 0.2626 | True |
| GLM-5 | longbook_ppl | 32768 | perm_once_B | B | 10 | 1.6050 | -0.00486 [-0.01751, +0.00507] | 0.0962 / 0.2535 | True |
| GLM-5 | longbook_ppl | 65536 | clean | - | 10 | 1.8880 | +0.00000 [+0.00000, +0.00000] | 0.0000 / 0.0000 | — |
| GLM-5 | longbook_ppl | 65536 | clean2 | - | 10 | 1.8950 | +0.00696 [-0.00268, +0.01929] | 0.1164 / 0.3476 | — |
| GLM-5 | longbook_ppl | 65536 | perm_once_A | A | 10 | 1.8912 | +0.00319 [-0.01397, +0.02021] | 0.1157 / 0.3447 | True |
| GLM-5 | longbook_ppl | 65536 | perm_once_B | B | 10 | 1.8921 | +0.00409 [-0.00524, +0.01538] | 0.1151 / 0.3410 | True |
| GLM-5 | longbook_ppl | 131072 | clean | - | 10 | 2.0919 | +0.00000 [+0.00000, +0.00000] | 0.0000 / 0.0000 | — |
| GLM-5 | longbook_ppl | 131072 | clean2 | - | 10 | 2.0879 | -0.00396 [-0.02366, +0.01018] | 0.1267 / 0.3857 | — |
| GLM-5 | longbook_ppl | 131072 | perm_once_B | B | 10 | 2.0913 | -0.00056 [-0.01050, +0.00668] | 0.1239 / 0.3938 | True |

| model | benchmark | context | mode | n | accuracy | Δ vs clean | identical token streams | median first divergence |
|---|---|---:|---|---:|---:|---:|---:|---:|
| DeepSeek-V3.2 | ruler_niah2+qa1 | 32768 | clean | 10 | 0.80 | +0.00 | 10/10 | 9999.0 |
| DeepSeek-V3.2 | ruler_niah2+qa1 | 131072 | clean | 10 | 0.80 | +0.00 | 10/10 | 9999.0 |
| DeepSeek-V3.2 | ruler_niah2+qa1 | 32768 | perm_once_B | 10 | 0.80 | +0.00 | 8/10 | 9999.0 |
| DeepSeek-V3.2 | ruler_niah2+qa1 | 131072 | perm_once_B | 10 | 0.80 | +0.00 | 8/10 | 9999.0 |
| DeepSeek-V3.2 | ruler_niah2+qa1 | 32768 | perm_periodic_B | 10 | 0.80 | +0.00 | 8/10 | 9999.0 |
| DeepSeek-V3.2 | ruler_niah2+qa1 | 131072 | perm_periodic_B | 10 | 0.80 | +0.00 | 8/10 | 9999.0 |
| GLM-5.2 | ruler_niah2+qa1 | 32768 | clean | 10 | 0.90 | +0.00 | 10/10 | 9999.0 |
| GLM-5.2 | ruler_niah2+qa1 | 131072 | clean | 10 | 0.90 | +0.00 | 10/10 | 9999.0 |
| GLM-5.2 | ruler_niah2+qa1 | 32768 | perm_once_B | 10 | 0.90 | +0.00 | 10/10 | 9999.0 |
| GLM-5.2 | ruler_niah2+qa1 | 131072 | perm_once_B | 10 | 0.90 | +0.00 | 8/10 | 9999.0 |
| GLM-5.2 | ruler_niah2+qa1 | 32768 | perm_periodic_B | 10 | 0.90 | +0.00 | 10/10 | 9999.0 |
| GLM-5.2 | ruler_niah2+qa1 | 131072 | perm_periodic_B | 10 | 0.90 | +0.00 | 8/10 | 9999.0 |
| GLM-5 | ruler_niah2+qa1 | 32768 | clean | 10 | 1.00 | +0.00 | 10/10 | 9999.0 |
| GLM-5 | ruler_niah2+qa1 | 131072 | clean | 10 | 1.00 | +0.00 | 10/10 | 9999.0 |
| GLM-5 | ruler_niah2+qa1 | 32768 | perm_once_B | 10 | 1.00 | +0.00 | 10/10 | 9999.0 |
| GLM-5 | ruler_niah2+qa1 | 131072 | perm_once_B | 10 | 1.00 | +0.00 | 8/10 | 9999.0 |
| GLM-5 | ruler_niah2+qa1 | 32768 | perm_periodic_B | 10 | 1.00 | +0.00 | 10/10 | 9999.0 |
| GLM-5 | ruler_niah2+qa1 | 131072 | perm_periodic_B | 10 | 1.00 | +0.00 | 8/10 | 9999.0 |

![ΔPPL](reindex_ppl_delta.png)
