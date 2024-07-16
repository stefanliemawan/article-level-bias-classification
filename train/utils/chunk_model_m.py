import platform

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import AutoModel, get_linear_schedule_with_warmup


class ChunkModelM(nn.Module):
    def __init__(
        self,
        tf_model_name,
        num_tf_layers,
        hidden_dim,
        metadata_hidden_dim,
        num_classes,
        train_labels,
        dropout_prob=0.2,
    ):
        super(ChunkModelM, self).__init__()
        if platform.system() == "Darwin":
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.tf_model_name = tf_model_name
        self.dropout_prob = dropout_prob

        self.init_layers(num_tf_layers, hidden_dim, metadata_hidden_dim, num_classes)
        self.calculate_class_weights(train_labels)
        self.init_loss_optimiser()

    def init_layers(self, num_tf_layers, hidden_dim, metadata_hidden_dim, num_classes):
        self.tf_model = AutoModel.from_pretrained(self.tf_model_name)
        self.tf_model = self.tf_model.to(self.device)

        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.tf_model.config.hidden_size,  # 768 for magpie
                    nhead=8,
                    dim_feedforward=hidden_dim,
                )
                for _ in range(num_tf_layers)
            ]
        )

        self.dropout = nn.Dropout(self.dropout_prob)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim + 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(hidden_dim + 128, num_classes),
        )

    def init_loss_optimiser(self):
        self.loss_function = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights).to(self.device)
        )
        learning_rate = 1e-5
        self.optimiser = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        print(f"AdamW, learning_rate: {learning_rate}")

    def calculate_class_weights(self, train_labels):
        self.class_weights = np.asarray(
            compute_class_weight(
                class_weight="balanced", classes=np.unique(train_labels), y=train_labels
            )
        ).astype(np.float32)

        print(f"class_weights: {self.class_weights}")

    def handle_chunks(self, input_ids, attention_mask):
        num_of_chunks = [len(chunk) for chunk in input_ids]

        input_ids_combined = []
        for id in input_ids:
            input_ids_combined.extend(id)

        input_ids_combined_tensors = torch.stack(
            [torch.tensor(x).to(self.device) for x in input_ids_combined]
        )

        attention_mask_combined = []
        for mask in attention_mask:
            attention_mask_combined.extend(mask)

        attention_mask_combined_tensors = torch.stack(
            [torch.tensor(x).to(self.device) for x in attention_mask_combined]
        )

        return (
            input_ids_combined_tensors,
            attention_mask_combined_tensors,
            num_of_chunks,
        )

    # not working, since 1 chunk is 1 mini-batch, need to make it so that metadata is added to one batch: several chunks
    def forward(self, input_ids, attention_mask, metadata):
        tf_model_output = self.tf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        transformer_output = tf_model_output.last_hidden_state

        for layer in self.transformer_layers:
            transformer_output = layer(transformer_output)

        transformer_output = self.dropout(transformer_output)

        expanded_attention_mask = (
            attention_mask.unsqueeze(-1).expand(transformer_output.size()).float()
        )
        masked_output = transformer_output * expanded_attention_mask
        sum_masked_output = torch.sum(masked_output, dim=1)
        mean_pooled_output = sum_masked_output / torch.clamp(
            expanded_attention_mask.sum(dim=1), min=1e-9
        )

        metadata = metadata.to(mean_pooled_output.dtype)
        expanded_metadata = (
            metadata.unsqueeze(1)
            .expand(-1, mean_pooled_output.shape[0] // len(metadata) + 1)
            .reshape(-1)[: mean_pooled_output.shape[0]]
        )
        expanded_metadata = expanded_metadata.unsqueeze(1)

        linear_metadata = nn.Sequential(nn.Linear(1, 128), nn.ReLU()).to(self.device)(
            expanded_metadata
        )

        combined_output = torch.cat((mean_pooled_output, linear_metadata), dim=1)

        mlp_output = self.mlp(combined_output)

        return mlp_output

    def batchify(self, inputs, metadata, batch_size=8):  # better way to do this?
        input_ids = [f["input_ids"] for f in inputs]
        attention_mask = [f["attention_mask"] for f in inputs]
        labels = torch.tensor([f["labels"] for f in inputs]).to(self.device)
        metadata = torch.tensor(metadata).to(self.device)

        dataloader = []
        for i in range(0, len(input_ids), batch_size):
            batch_input_ids = input_ids[i : i + batch_size]
            batch_attention_mask = attention_mask[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]
            batch_metadata = metadata[i : i + batch_size]

            dataloader.append(
                [batch_input_ids, batch_attention_mask, batch_labels, batch_metadata]
            )

        return dataloader

    def train_loop(self, train_dataloader):
        loss = 0
        for (
            batch_input_ids,
            batch_attention_mask,
            batch_labels,
            batch_metadata,
        ) in train_dataloader:
            (
                input_ids_combined_tensors,
                attention_mask_combined_tensors,
                num_of_chunks,
            ) = self.handle_chunks(batch_input_ids, batch_attention_mask)

            self.optimiser.zero_grad()

            logits = self.forward(
                input_ids_combined_tensors,
                attention_mask_combined_tensors,
                batch_metadata,
            )

            logits_split = logits.split(num_of_chunks)

            pooled_logits = torch.cat(
                [torch.mean(x, axis=0, keepdim=True) for x in logits_split]
            )

            batch_loss = self.loss_function(pooled_logits, batch_labels)

            batch_loss.backward()
            self.optimiser.step()
            if self.scheduler:
                self.scheduler.step()

            loss += batch_loss.detach().item()

        loss = loss / (len(train_dataloader))
        return loss

    def validation_loop(self, valid_dataloader):
        loss = 0
        all_pooled_logits = torch.tensor([]).to(self.device)
        with torch.no_grad():
            for (
                batch_input_ids,
                batch_attention_mask,
                batch_labels,
                batch_metadata,
            ) in valid_dataloader:
                (
                    input_ids_combined_tensors,
                    attention_mask_combined_tensors,
                    num_of_chunks,
                ) = self.handle_chunks(batch_input_ids, batch_attention_mask)

                logits = self.forward(
                    input_ids_combined_tensors,
                    attention_mask_combined_tensors,
                    batch_metadata,
                )

                logits_split = logits.split(num_of_chunks)

                pooled_logits = torch.cat(
                    [torch.mean(x, axis=0, keepdim=True) for x in logits_split]
                )

                batch_loss = self.loss_function(pooled_logits, batch_labels)

                loss += batch_loss.detach().item()

                all_pooled_logits = torch.cat((all_pooled_logits, pooled_logits), dim=0)

        loss = loss / (len(valid_dataloader))

        labels = []
        for _, _, batch_labels, _ in valid_dataloader:
            labels.extend(batch_labels.tolist())

        metrics = {"loss": loss, **self.compute_metrics(all_pooled_logits, labels)}

        return metrics

    def fit(self, train_dataloader, valid_dataloader, epochs=3):
        print("Training and validating model")

        num_training_steps = len(train_dataloader) * epochs

        num_warmup_steps = int(0.1 * num_training_steps)
        print(f"warmup_steps: {num_warmup_steps}")

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimiser,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        for epoch in range(epochs):
            print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

            train_loss = self.train_loop(train_dataloader)

            validation_metrics = self.validation_loop(valid_dataloader)

            print(f"Training loss: {train_loss}")
            print(f"Validation metrics: {validation_metrics}")
            print()

    def compute_metrics(self, logits, labels):
        preds = logits.cpu().numpy().argmax(-1)

        report = classification_report(labels, preds, zero_division=1)
        print(report)

        precision = precision_score(labels, preds, average="weighted", zero_division=1)
        recall = recall_score(labels, preds, average="weighted", zero_division=1)
        f1 = f1_score(labels, preds, average="weighted", zero_division=1)

        return {"precision": precision, "recall": recall, "f1": f1}

    def predict(self, inputs, metadata):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = torch.tensor(inputs["labels"]).to(self.device)
        metadata = torch.tensor(metadata).to(self.device)

        with torch.no_grad():
            (
                input_ids_combined_tensors,
                attention_mask_combined_tensors,
                num_of_chunks,
            ) = self.handle_chunks(input_ids, attention_mask)

            logits = self.forward(
                input_ids_combined_tensors, attention_mask_combined_tensors, metadata
            )

            logits_split = logits.split(num_of_chunks)

            pooled_logits = torch.cat(
                [torch.mean(x, axis=0, keepdim=True) for x in logits_split]
            )

            loss = self.loss_function(pooled_logits, labels)
            loss = loss.detach().item()

        labels = labels.tolist()
        metrics = {"loss": loss, **self.compute_metrics(pooled_logits, labels)}

        print(metrics)

        return metrics
