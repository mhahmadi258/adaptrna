"""Secondary structure prediction (bpRNA / ArchiveII).

The interesting bit is `self.threshold`: a plain Python float, tuned on the validation set
every N epochs, that predictions depend on and that `state_dict` knows nothing about. The
source hand-stashed it into `checkpoint['state_dict']['threshold']` and its `load_state_dict`
override then did an unguarded `state_dict["threshold"]`, which `KeyError`s on any dict
lacking the key. Here it travels through the typed non-tensor channel instead --
`adapter_extra_payload()` / `load_adapter_extra()` -- and `state_dict` stays tensors-only.
"""

from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from rinalmo.data.alphabet import Alphabet
from rinalmo.data.downstream.secondary_structure.datamodule import SecondaryStructureDataModule
from rinalmo.model.downstream import SecStructPredictionHead
from rinalmo.utils.sec_struct import prob_mat_to_sec_struct, save_to_ct, ss_f1, ss_precision, ss_recall

from rinalmo_hub.module import BaseDownstreamModule
from rinalmo_hub.registry import register_task

THRESHOLD_TUNE_METRIC = "f1"
THRESHOLD_CANDIDATES = [i / 100 for i in range(1, 30)]
DEFAULT_THRESHOLD = 0.5


@register_task("sec_struct")
class SecStructModule(BaseDownstreamModule):
    TASK_NAME = "sec_struct"
    ADAPTER_EXTRA_PREFIXES = ()

    # The head is O(L^2) in memory and the datamodule feeds one variable-length structure at
    # a time, so batching would need padding the head cannot see through.
    DEFAULT_PREDICT_BATCH_SIZE = 1

    PRIMARY_METRIC = "test/f1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tune_threshold_every_n_epoch = int(self.task_config.get("tune_threshold_every_n_epoch", 1))
        self.save_test_ct_files = bool(self.task_config.get("save_test_ct_files", False))

        self.threshold = float(self.task_config.get("threshold", DEFAULT_THRESHOLD))
        self._eval_outputs = self._empty_eval_outputs()

    @staticmethod
    def _empty_eval_outputs():
        return defaultdict(lambda: defaultdict(list))

    def build_head(self, embed_dim, num_resnet_blocks: int = 2, conv_dim: int = 64,
                   kernel_size: int = 3, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected head config keys for sec_struct: {sorted(kwargs)}")

        return SecStructPredictionHead(
            embed_dim, num_blocks=num_resnet_blocks, conv_dim=conv_dim, kernel_size=kernel_size
        )

    def batch_tokens(self, batch):
        # Batches are (ss_id, seq, tokens, sec_struct_true).
        return batch[2]

    def extract_features(self, representation, tokens):
        # Drop CLS and EOS; the structure matrix is over nucleotides only.
        return representation[..., 1:-1, :]

    def forward(self, tokens):
        return super().forward(tokens).squeeze(-1)

    def compute_loss(self, outputs, batch):
        *_, sec_struct_true = batch
        seq_len = sec_struct_true.shape[1]

        upper_tri = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=outputs.device), diagonal=1
        )

        return nn.functional.binary_cross_entropy_with_logits(
            outputs[..., upper_tri], sec_struct_true[..., upper_tri].to(outputs.dtype)
        )

    # ------------------------------------------------------------------ metrics

    def reset_metrics(self, stage):
        if stage in ("val", "test"):
            self._eval_outputs = self._empty_eval_outputs()

    def _is_tuning_epoch(self) -> bool:
        if self.trainer is None:
            return True

        return (self.trainer.current_epoch + 1) % self.tune_threshold_every_n_epoch == 0

    def update_metrics(self, outputs, batch, stage):
        if stage == "train":
            return

        if stage == "val" and not self._is_tuning_epoch():
            return

        thresholds = THRESHOLD_CANDIDATES if stage == "val" else [self.threshold]
        ss_ids, seqs, _, sec_struct_true = batch

        probs = torch.sigmoid(outputs.float()).cpu().numpy()
        targets = sec_struct_true.cpu().numpy()

        for i in range(probs.shape[0]):
            for threshold in thresholds:
                predicted = prob_mat_to_sec_struct(probs=probs[i], seq=seqs[i], threshold=threshold)

                self._eval_outputs["precision"][threshold].append(ss_precision(targets[i], predicted))
                self._eval_outputs["recall"][threshold].append(ss_recall(targets[i], predicted))
                self._eval_outputs["f1"][threshold].append(ss_f1(targets[i], predicted))

            if stage == "test" and self.save_test_ct_files:
                f1 = self._eval_outputs["f1"][self.threshold][-1]
                output_dir = Path(self.trainer.default_root_dir)
                save_to_ct(output_dir / f"{ss_ids[i]}_pred_f1={f1}.ct", sec_struct=predicted, seq=seqs[i])

    def compute_metrics(self, stage):
        if stage == "train" or not self._eval_outputs:
            return {}

        if stage == "val":
            if not self._is_tuning_epoch():
                return {}

            # Pick the threshold with the best mean validation F1, then report at it.
            best_threshold, best_value = DEFAULT_THRESHOLD, -1.0
            for threshold in THRESHOLD_CANDIDATES:
                scores = self._eval_outputs[THRESHOLD_TUNE_METRIC][threshold]
                if not scores:
                    continue

                value = sum(scores) / len(scores)
                if value > best_value:
                    best_value, best_threshold = value, threshold

            if best_value < 0.0:
                return {}

            self.threshold = best_threshold

            return {
                f"val/{THRESHOLD_TUNE_METRIC}": best_value,
                "val/threshold": float(self.threshold),
            }

        metrics = {}
        for key in self._eval_outputs:
            scores = self._eval_outputs[key][self.threshold]
            if scores:
                metrics[f"test/{key}"] = sum(scores) / len(scores)

        return metrics

    def postprocess_predictions(self, outputs, tokens, sequences):
        """Binary base-pairing matrices, one `numpy` array per sequence."""
        probs = torch.sigmoid(outputs.float()).cpu().numpy()

        return [
            prob_mat_to_sec_struct(probs=probs[i], seq=sequences[i], threshold=self.threshold)
            for i in range(probs.shape[0])
        ]

    # ------------------------------------------------------------------ non-tensor state

    def adapter_extra_payload(self):
        return {"threshold": float(self.threshold)}

    def load_adapter_extra(self, extra):
        # Tolerates an adapter written before the task had a threshold; the source's
        # unguarded `state_dict["threshold"]` did not.
        self.threshold = float(extra.get("threshold", DEFAULT_THRESHOLD))

    @staticmethod
    def build_datamodule(cfg):
        data = cfg["data"]

        return SecondaryStructureDataModule(
            data_root=data["root"],
            alphabet=Alphabet(),
            num_workers=data.get("num_workers", 0),
            pin_memory=data.get("pin_memory", False),
            min_seq_len=data.get("min_seq_len", 0),
            max_seq_len=data.get("max_seq_len", 999_999_999),
            dataset=data.get("dataset", "bpRNA"),
            skip_data_preparation=not data.get("prepare", False),
        )
