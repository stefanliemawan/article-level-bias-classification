import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from transformers import Trainer


class StandardTrainer(Trainer):
    def calculate_class_weights(self):
        train_labels = self.train_dataset["label"]
        self.class_weights = np.asarray(
            compute_class_weight(
                class_weight="balanced", classes=np.unique(train_labels), y=train_labels
            )
        ).astype(np.float32)

        print(f"class_weights: {self.class_weights}")

    def compute_loss(self, model, inputs, return_outputs=False):
        print(inputs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights)
        ).to("mps")
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
