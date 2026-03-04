from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def init_clearml_task(cfg: DictConfig) -> Optional[Any]:
    """Initialize ClearML Task for experiment tracking.

    ClearML auto-captures TensorBoard logs, hyperparameters, console output,
    and installed packages without additional code.

    :param cfg: Hydra DictConfig containing the full config tree.
    :return: ClearML Task instance or None if disabled.
    """
    clearml_cfg = cfg.get("clearml")
    if not clearml_cfg or not clearml_cfg.get("enable", False):
        log.info("ClearML is disabled.")
        return None

    try:
        from clearml import Task
    except ImportError:
        log.warning("ClearML is not installed. Skipping ClearML initialization.")
        return None

    task = Task.init(
        project_name=clearml_cfg.get("project_name", "Default"),
        task_name=clearml_cfg.get("task_name", "training"),
        tags=list(clearml_cfg.get("tags", [])),
        auto_connect_frameworks=OmegaConf.to_container(
            clearml_cfg.get("auto_connect_frameworks", {}), resolve=True
        ),
    )

    hydra_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    task.connect(hydra_cfg, name="hydra_config")

    log.info(
        f"ClearML Task initialized: "
        f"project='{clearml_cfg.project_name}', "
        f"task='{task.name}', "
        f"id='{task.id}'"
    )

    return task
