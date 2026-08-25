import numpy as np
import torch
import torch.nn as nn

from encoder_pipeline.evaluation.metrics import classification_metrics


class LinearProbe:
    """Trains a single linear layer on frozen embeddings to predict labels
    -- full-batch Adam + cross-entropy"""

    def __init__(self, epochs: int = 1000, lr: float = 3e-4) -> None:
        self.epochs = epochs
        self.lr = lr

    def evaluate(self, embeddings: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[dict[str, float], dict[str, list[float]]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensors = {
            split: (torch.tensor(x, dtype=torch.float32, device=device), torch.tensor(y, dtype=torch.long, device=device))
            for split, (x, y) in embeddings.items()
        }
        x_train, y_train = tensors["train"]

        model = nn.Linear(x_train.shape[1], int(y_train.max().item()) + 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        loss_curves: dict[str, list[float]] = {f"{split}_loss": [] for split in tensors}
        for _ in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            train_loss = criterion(model(x_train), y_train)
            train_loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                for split, (x, y) in tensors.items():
                    split_loss = train_loss.item() if split == "train" else criterion(model(x), y).item()
                    loss_curves[f"{split}_loss"].append(split_loss)

        model.eval()
        metrics: dict[str, float] = {}
        with torch.no_grad():
            for split, (x, y) in tensors.items():
                logits = model(x)
                y_true, y_pred = y.cpu().numpy(), logits.argmax(dim=1).cpu().numpy()
                y_score = torch.softmax(logits, dim=1).cpu().numpy()
                metrics[f"{split}_accuracy"] = float((y_pred == y_true).mean())
                for name, value in classification_metrics(y_true, y_pred, y_score).items():
                    metrics[f"{split}_{name}"] = value
        return metrics, loss_curves
