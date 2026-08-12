from lightning import LightningModule
from lightning.pytorch.callbacks import BaseFinetuning
from torch.optim.optimizer import Optimizer

import re
import yaml

def _is_parent_module_unfrozen(module_name, potential_parent_modules):
    for potential_parent_module_name in potential_parent_modules.keys():
        if module_name.startswith(potential_parent_module_name):
            return True

    return False

class GradualUnfreezing(BaseFinetuning):
    def __init__(self, unfreeze_schedule_path: str, initial_denom_lr: float = 1.0):
        """
        Args:
            unfreeze_schedule_path: YAML mapping epoch -> list of module-name regexes.
                Names are matched against `pl_module.named_modules()`, so with this
                framework's naming they start with `backbone.` or `head.`.
            initial_denom_lr: newly unfrozen parameters enter the optimizer at
                `lr / initial_denom_lr`. Lightning's own default is 10.0, which silently
                trained the RiNALMo backbone at 1e-6 instead of the requested 1e-5 for a
                whole MRL run. Default 1.0 here means "use the configured learning rate".
        """
        super().__init__()

        self.initial_denom_lr = initial_denom_lr

        # Load unfreezing/fine-tuning schedule
        with open(unfreeze_schedule_path, "r") as f:
            self.unfreeze_schedule = yaml.safe_load(f)

        # "Merge" regexes for each epoch
        for epoch in self.unfreeze_schedule:
            self.unfreeze_schedule[epoch] = re.compile('|'.join(self.unfreeze_schedule[epoch]))

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        phase_zero = self.unfreeze_schedule[0]

        trained_from_the_start = {
            module_name
            for module_name, _ in pl_module.named_modules()
            if module_name and bool(phase_zero.match(module_name))
        }

        models_to_freeze = []
        for module_name, module in pl_module.named_modules():
            # Ignore root module (module_name = '')
            if not module_name:
                continue

            if module_name in trained_from_the_start:
                continue

            # An *ancestor* of a module that is trained from the start must not be frozen
            # wholesale -- `self.freeze` walks the whole subtree, so freezing 'backbone'
            # would freeze 'backbone.transformer' with it even though the schedule asked
            # for that subtree to train. Without this guard a schedule whose phase 0
            # unfreezes a subtree (rather than the whole backbone) silently trains the head
            # and nothing else.
            if any(name.startswith(module_name + ".") for name in trained_from_the_start):
                continue

            # Collect all modules that are not tuned in the first epoch
            models_to_freeze.append(module)

        # Freeze collected modules
        self.freeze(models_to_freeze)

    def finetune_function(self, pl_module: LightningModule, current_epoch: int, optimizer: Optimizer) -> None:
        if current_epoch in self.unfreeze_schedule and current_epoch != 0:
            modules_to_unfreeze = {}

            # Collect next phase modules
            for module_name, module in pl_module.named_modules():
                if bool(self.unfreeze_schedule[current_epoch].match(module_name)) and not _is_parent_module_unfrozen(module_name, modules_to_unfreeze):
                    modules_to_unfreeze[module_name] = module

            # Unfreeze collected modules
            self.unfreeze_and_add_param_group(
                modules=modules_to_unfreeze.values(),
                optimizer=optimizer,
                initial_denom_lr=self.initial_denom_lr,
            )
