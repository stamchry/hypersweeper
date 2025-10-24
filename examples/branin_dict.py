"""
Branin
^^^^^^
"""

import hydra
import numpy as np
from omegaconf import DictConfig

__copyright__ = "Copyright 2022, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


@hydra.main(config_path="configs", config_name="branin_rs", version_base="1.1")
def branin(cfg: DictConfig):
    x0 = cfg.x0
    x1 = cfg.x1
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    ret = a * (x1 - b * x0**2 + c * x0 - r) ** 2 + s * (1 - t) * np.cos(x0) + s

    # --- New Cost Function ---
    # This cost function is designed to have features near the Branin minima
    # to create a more interesting trade-off landscape.

    # Valley (cheap area) near (pi, 2.275)
    cost1 = -1.0 * np.exp(-0.5 * ((x0 - np.pi) ** 2 + (x1 - 2.275) ** 2))

    # Moderate peak near (9.42, 2.475)
    cost2 = 0.7 * np.exp(-0.5 * ((x0 - 9.42) ** 2 + (x1 - 2.475) ** 2))

    # High peak (expensive area) near (-pi, 12.275)
    cost3 = 1.0 * np.exp(-0.5 * ((x0 + np.pi) ** 2 + (x1 - 12.275) ** 2))

    # Combine and normalize to a range of roughly [0.1, 1.1]
    cost = 0.5 * (cost1 + cost2 + cost3) + 0.6

    return {"performance": ret, "cost": cost}


if __name__ == "__main__":
    branin()
