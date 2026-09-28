import losses as L


import torch
import torch.nn as nn
from torch.optim import lr_scheduler


class OptimizerSchedulerInst:
    """
    Instantiates optimizer and LR scheduler from a parsed YAML config dict.

    Expected config structure:
        training_parameters:
          lr: 3e-3
          <SchedulerName>:
            <param>: <value>
            ...
    """

    SUPPORTED_SCHEDULERS = {"ReduceLROnPlateau", "ExponentialLR", "MultiplicativeLR"}

    def __init__(self, config: dict, model: nn.Module):
        params = config.get("training_parameters", config)

        self.base_lr = float(params.get("lr", 1e-3))
        self.optimizer = self._build_optimizer(model)
        self.scheduler, self.scheduler_name = self._build_scheduler(params)

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.base_lr)

    def _build_scheduler(self, params: dict):
        found = self.SUPPORTED_SCHEDULERS & params.keys()

        if not found:
            return None, None
        if len(found) > 1:
            raise ValueError(f"Multiple schedulers specified: {found}. Use only one.")

        name = found.pop()
        sched_params = {k: float(v) for k, v in params[name].items()}

        match name:
            case "ReduceLROnPlateau":
                scheduler = lr_scheduler.ReduceLROnPlateau(
                    optimizer=self.optimizer,
                    factor=sched_params.get("factor", 0.1),
                    patience=int(sched_params.get("patience", 10)),
                    cooldown=int(sched_params.get("cooldown", 0)),
                    min_lr=sched_params.get("min_lr", 0.0),
                )
                # Override lr if scheduler-level lr is specified
                if "lr" in sched_params:
                    for g in self.optimizer.param_groups:
                        g["lr"] = sched_params["lr"]

            case "ExponentialLR":
                scheduler = lr_scheduler.ExponentialLR(
                    optimizer=self.optimizer,
                    gamma=sched_params["gamma"],
                )

            case "MultiplicativeLR":
                factor = sched_params["factor"]
                epoch_cycle = int(sched_params["epoch_cycle"])
                scheduler = lr_scheduler.MultiplicativeLR(
                    optimizer=self.optimizer,
                    lr_lambda=lambda epoch: (
                        factor if (epoch + 1) % epoch_cycle == 0 else 1.0
                    ),
                )

            case _:
                raise ValueError(f"Unsupported scheduler: {name}")

        return scheduler, name

    def step(self, metric=None):
        """Call at the end of each epoch. Pass metric for ReduceLROnPlateau."""
        if self.scheduler is None:
            return
        if self.scheduler_name == "ReduceLROnPlateau":
            if metric is None:
                raise ValueError(
                    "ReduceLROnPlateau requires a metric value for .step()"
                )
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def __repr__(self):
        return (
            f"TrainingConfig(\n"
            f"  base_lr={self.base_lr},\n"
            f"  optimizer={type(self.optimizer).__name__},\n"
            f"  scheduler={self.scheduler_name}\n"
            f")"
        )
