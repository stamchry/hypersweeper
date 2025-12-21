"""HyperSMAC implementation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from hydra.utils import get_class
from hydra_plugins.hypersweeper.utils import Info, convert_to_configuration
from omegaconf import OmegaConf, ListConfig
from smac.scenario import Scenario
from smac.facade import HyperparameterOptimizationFacade
from smac.intensifier.hyperband import Hyperband
from smac.runhistory.dataclasses import TrialInfo, TrialValue


def read_additional_configs(initial_design_fn, configspace):
    """Read configurations from csv-logfile.

    Parameters
    ----------
    initial_design_fn : str
        The path to the log file.
    search_space : DictConfig
        The search space which will be converted to a ConfigSpace.ConfigurationSpace, by default None.
        The search space can be loaded via `search_space = OmegaConf.load(search_space_fn)`.

    Returns.
    --------
    list[Configuration]
        The configurations from the log file.
    """
    initial_design = pd.read_csv(initial_design_fn)
    return initial_design.apply(convert_to_configuration, args=(configspace,), axis=1).to_list()


OmegaConf.register_new_resolver("get_class", get_class, replace=True)
OmegaConf.register_new_resolver("read_additional_configs", read_additional_configs, replace=True)


class HyperSMACAdapter:
    """Adapt SMAC ask/tell interface to HyperSweeper ask/tell interface."""

    def __init__(self, smac):
        """Initialize the adapter."""
        self.smac = smac

    def ask(self):
        """Ask for the next configuration."""
        smac_info = self.smac.ask()
        info = Info(config=smac_info.config, budget=smac_info.budget, load_path=None, seed=smac_info.seed)
        terminate = False
        optimizer_termination = False

        return info, terminate, optimizer_termination

    def tell(self, info, value):
        """Tell the result of the configuration."""
        smac_info = TrialInfo(info.config, seed=info.seed, budget=info.budget)
        # For cost-aware optimization, the resource cost is expected in additional_info
        additional_info = {}
        if value.cost is not None:
            additional_info["resource_cost"] = value.cost
        smac_value = TrialValue(time=value.cost, cost=value.performance, additional_info=additional_info)
        self.smac.tell(smac_info, smac_value)

    def finish_run(self, output_path):
        """Do nothing for SMAC."""


def make_smac(configspace, smac_args):
    """Make a SMAC instance for optimization."""

    def dummy_func(arg, seed, budget):  # noqa:ARG001
        return 0.0

    if "output_directory" in smac_args["scenario"]:
        smac_args["scenario"]["output_directory"] = Path(smac_args["scenario"]["output_directory"])
    scenario = Scenario(configspace, **smac_args.pop("scenario"))
    smac_kwargs = {}

    if "callbacks" not in smac_args:
        smac_kwargs["callbacks"] = []
    elif "callbacks" in smac_args and isinstance(smac_args["callbacks"], dict):
        smac_kwargs["callbacks"] = list(smac_args["callbacks"].values())
    elif "callbacks" in smac_args and isinstance(smac_args["callbacks"], (list, ListConfig)):
        smac_kwargs["callbacks"] = smac_args["callbacks"]

    if "acquisition_function" in smac_args and "acquisition_maximizer" in smac_args:
        smac_kwargs["acquisition_maximizer"] = smac_args["acquisition_maximizer"](
            configspace=configspace,
            acquisition_function=smac_args["acquisition_function"],
        )
        if hasattr(smac_args["acquisition_maximizer"], "selector") and hasattr(
            smac_args["acquisition_maximizer"].selector, "expl2callback"
        ):
            smac_kwargs["callbacks"].append(smac_args["acquisition_maximizer"].selector.expl2callback)

    if "config_selector" in smac_args:
        smac_kwargs["config_selector"] = smac_args["config_selector"](scenario=scenario)

    if "initial_design" in smac_args:
        initial_design_obj = smac_args["initial_design"]
        warmstart_file = None

        # Check if it is a partial (instantiated by Hydra) and look in keywords
        if hasattr(initial_design_obj, "keywords") and "warmstart_file" in initial_design_obj.keywords:
            warmstart_file = initial_design_obj.keywords["warmstart_file"]
        # Fallback: Check if it is a dict/DictConfig (raw config)
        elif hasattr(initial_design_obj, "__getitem__") and "warmstart_file" in initial_design_obj:
            warmstart_file = initial_design_obj["warmstart_file"]

        if warmstart_file:
            config_list = read_additional_configs(
                initial_design_fn=warmstart_file,
                configspace=configspace,
            )
            initial_design = HyperparameterOptimizationFacade.get_initial_design(
                scenario=scenario,
                n_configs=0,
                additional_configs=config_list,
            )
        else:
            initial_design = smac_args["initial_design"](scenario=scenario)
        smac_kwargs["initial_design"] = initial_design

    if "intensifier" in smac_args:
        smac_kwargs["intensifier"] = smac_args["intensifier"](scenario)

    if "random_design" in smac_args:
        smac_kwargs["random_design"] = smac_args["random_design"]()

    smac = smac_args["smac_facade"](scenario, dummy_func, **smac_kwargs)
    return HyperSMACAdapter(smac)


if __name__ == "__main__":
    read_additional_configs()
