# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train Gated-Spectral MeanFlow (GS-MeanFlow) on the StormCast residual.

GS-MeanFlow keeps the MeanFlow average-velocity residual head (1-2 NFE) and
adds two pillars motivated by the Taiwan RWRF data (see README.md):

  1. group-decoupled adaptive weighting -- the smooth channels (t2m, u10, v10)
     and the precip channel (qpepre) are re-weighted independently, resolving
     the cross-channel objective conflict the qpw ablation exposes;
  2. an occurrence (hurdle) gate -- a tiny deterministic head predicts qpepre
     wet/dry and forces confidently-dry pixels to exact zero at inference,
     fixing the "always slightly wet" failure a pure continuous flow cannot.

Lives in stormcast/gated-spectral_meanflow/ but reuses the shared config tree
and utils/datasets packages; launch it from the stormcast/ directory.
"""

import os
import sys
import glob

# Make the shared stormcast packages (utils, datasets, config) and this
# module's siblings importable regardless of where torchrun is launched from.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_STORMCAST_ROOT = os.path.dirname(_MODULE_DIR)
for _p in (_STORMCAST_ROOT, _MODULE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_CONFIG_DIR = os.path.join(_STORMCAST_ROOT, "config")

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from physicsnemo.distributed import DistributedManager

from trainer_gsmeanflow import gsmeanflow_training_loop


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="gsmeanflow")
def main(cfg: DictConfig) -> None:
    """GS-MeanFlow entry point."""

    DistributedManager.initialize()
    dist = DistributedManager()

    if dist.rank == 0:
        print("GS-MeanFlow configuration:")
        print(OmegaConf.to_yaml(cfg))

    if cfg.training.seed < 0:
        seed = torch.randint(1 << 31, size=[], device=torch.device("cuda"))
        torch.distributed.broadcast(seed, src=0)
        cfg.training.seed = int(seed)

    wandb_resume = False
    os.makedirs(cfg.training.rundir, exist_ok=True)
    training_states = glob.glob(
        os.path.join(cfg.training.rundir, "checkpoints_gsmeanflow/checkpoint*.pt")
    )
    if training_states:
        wandb_resume = True

    if dist.rank == 0 and cfg.training.log_to_wandb:
        entity, project = "wandb_entity", "wandb_project"
        wandb.init(
            dir=cfg.training.rundir,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            name=os.path.basename(cfg.training.rundir),
            project=project,
            entity=entity,
            resume=wandb_resume,
            mode=cfg.training.wandb_mode,
        )

    gsmeanflow_training_loop(cfg)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
