# Inference timing — one A6000

GPU `NVIDIA RTX A6000` | ensemble 50 | rollout 24 h | 1 init time(s) | warmup 1

`sec_per_ensemble_forecast` = wall-clock for one 50-member 24 h ensemble forecast (the headline cost).

| method | nfe_per_step | ensemble | n_sequences | n_steps | n_rollouts | total_s | sec_per_ensemble_forecast | sec_per_member_rollout | sec_per_forecast_hour | sec_per_nfe | member_rollout_std_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stormcast_edm | 36 | 50 | 1 | 24 | 50 | 1898.031 | 1898.031 | 37.9606 | 1.5817 | 0.04394 | 2.6839 |
| flowcast_nfe10 | 10 | 50 | 1 | 24 | 50 | 645.2 | 645.2 | 12.904 | 0.5377 | 0.05377 | 1.3075 |
| flowcast_nfe15 | 15 | 50 | 1 | 24 | 50 | 826.438 | 826.438 | 16.5288 | 0.6887 | 0.04591 | 1.1171 |
| flowcast_nfe20 | 20 | 50 | 1 | 24 | 50 | 1086.291 | 1086.291 | 21.7258 | 0.9052 | 0.04526 | 1.8401 |
| meanflow_nfe1 | 1 | 50 | 1 | 24 | 50 | 191.925 | 191.925 | 3.8385 | 0.1599 | 0.15994 | 0.2369 |
| meanflow_nfe2 | 2 | 50 | 1 | 24 | 50 | 239.098 | 239.098 | 4.782 | 0.1992 | 0.09962 | 0.3644 |
