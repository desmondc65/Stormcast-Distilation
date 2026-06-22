# Neighborhood (kernel) CSI on qpepre

CSI with spatial tolerance: both event masks are dilated by a k×k square (~2 km/px, so k=4 ≈ 8 km). A forecast event within ~k px of an observed one scores as a hit rather than a false-alarm + miss. **k=1 = exact-pixel** (matches the scoreboard). Scored on the ensemble mean.

## ≥ 0.1 mm/h
| method | k=1 (~2km) | k=2 (~4km) | k=4 (~8km) | k=8 (~16km) | k=16 (~32km) |
|---|---|---|---|---|---|
| diffusion | 0.245 | 0.262 | 0.289 | 0.338 | 0.434 |

## ≥ 1.0 mm/h
| method | k=1 (~2km) | k=2 (~4km) | k=4 (~8km) | k=8 (~16km) | k=16 (~32km) |
|---|---|---|---|---|---|
| diffusion | 0.102 | 0.120 | 0.148 | 0.195 | 0.284 |

## ≥ 5.0 mm/h
| method | k=1 (~2km) | k=2 (~4km) | k=4 (~8km) | k=8 (~16km) | k=16 (~32km) |
|---|---|---|---|---|---|
| diffusion | 0.009 | 0.010 | 0.013 | 0.018 | 0.027 |

## ≥ 10.0 mm/h
| method | k=1 (~2km) | k=2 (~4km) | k=4 (~8km) | k=8 (~16km) | k=16 (~32km) |
|---|---|---|---|---|---|
| diffusion | 0.005 | 0.006 | 0.008 | 0.010 | 0.015 |

## ≥ 16.0 mm/h
| method | k=1 (~2km) | k=2 (~4km) | k=4 (~8km) | k=8 (~16km) | k=16 (~32km) |
|---|---|---|---|---|---|
| diffusion | 0.003 | 0.004 | 0.005 | 0.008 | 0.012 |

