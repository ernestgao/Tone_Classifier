from __future__ import annotations

import torch
from torch import nn
from transformers import Trainer


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = torch.tensor(
                self.class_weights,
                dtype=logits.dtype,
                device=logits.device,
            )
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
