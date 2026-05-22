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

"""Train the anchored Stochastic-Interpolant Bridge on the StormCast residual.

Direction 1 of the FlowCast improvement plan
(see ../flowcast_improvement.md). The frozen regression network F_theta
provides mu_{t+1}, which is used as the *prior endpoint* of a stochastic
interpolant (Albergo & Vanden-Eijnden 2023): the trained vector field
v_theta integrates from mu directly to M_{t+1}, instead of from white noise
to the residual r_{t+1}.
"""

import os
import glob

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from physicsnemo.distributed import DistributedManager

from utils.trainer_bridge import bridge_training_loop


@hydra.main(version_base=None, config_path="config", config_name="bridge")
def main(cfg: DictConfig) -> None:
    """Bridge-matching entry point."""

    DistributedManager.initialize()
    dist = DistributedManager()

    if dist.rank == 0:
        print("Bridge configuration:")
        print(OmegaConf.to_yaml(cfg))

    if cfg.training.seed < 0:
        seed = torch.randint(1 << 31, size=[], device=torch.device("cuda"))
        torch.distributed.broadcast(seed, src=0)
        cfg.training.seed = int(seed)

    wandb_resume = False
    os.makedirs(cfg.training.rundir, exist_ok=True)
    training_states = glob.glob(
        os.path.join(cfg.training.rundir, "checkpoints_bridge/checkpoint*.pt")
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

    bridge_training_loop(cfg)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
