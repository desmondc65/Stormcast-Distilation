# MeanFlow spectral-loss (log-PSD) ablation

MeanFlow WITH the radial log-PSD term (`meanflow_lspec_*`) vs. WITHOUT it (`meanflow_no_lspec_*`), cleaned 192x96 + log1p, 2022 validation year. One row per sampler NFE (1, 2). Everything except the spectral training term is held equal.

| method | Time/Seq.(s) | CRPS↓ | CSI-M@3km↑ | HSS-M@3km↑ | FAR-M@3km↓ | FSS-P16@3km↑ | CSI-M@15km↑ | HSS-M@15km↑ | FAR-M@15km↓ | FSS-P16@15km↑ | CSI-M@27km↑ | HSS-M@27km↑ | FAR-M@27km↓ | FSS-P16@27km↑ | CSI-M@45km↑ | HSS-M@45km↑ | FAR-M@45km↓ | FSS-P16@45km↑ | RMSE_u10↓ | RMSE_v10↓ | RMSE_t2m↓ | RMSE_qpepre↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| meanflow_lspec_nfe1 | 0.909 | 0.6710 | 0.2092 | 0.2621 | 0.4525 | 0.0018 | 0.2443 | 0.2861 | 0.2985 | 0.0017 | 0.2663 | 0.2987 | 0.2184 | 0.0014 | 0.2938 | 0.3081 | 0.1553 | 0.0011 | 1.3767 | 1.6915 | 0.8087 | 0.9834 |
| meanflow_lspec_nfe2 | 1.126 | 0.6093 | 0.2156 | 0.2709 | 0.4974 | 0.0295 | 0.2690 | 0.3154 | 0.3738 | 0.0264 | 0.2950 | 0.3315 | 0.3142 | 0.0228 | 0.3267 | 0.3452 | 0.2631 | 0.0184 | 1.2986 | 1.4887 | 0.8472 | 1.0054 |
| meanflow_no_lspec_nfe1 | 0.913 | 0.5501 | 0.2127 | 0.2655 | 0.4620 | 0.0105 | 0.2689 | 0.3131 | 0.2918 | 0.0099 | 0.3040 | 0.3403 | 0.2038 | 0.0082 | 0.3443 | 0.3654 | 0.1349 | 0.0066 | 1.3314 | 1.4844 | 0.7498 | 0.9655 |
| meanflow_no_lspec_nfe2 | 1.100 | 0.5207 | 0.2141 | 0.2704 | 0.4192 | 0.0279 | 0.2656 | 0.3128 | 0.2563 | 0.0242 | 0.2954 | 0.3332 | 0.1841 | 0.0210 | 0.3307 | 0.3522 | 0.1247 | 0.0171 | 1.2651 | 1.4286 | 0.7315 | 0.9637 |
