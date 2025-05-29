from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn
from sklearn.metrics import roc_auc_score

from src.utils import mixup_data

console = Console()


class Trainer:
    """Handles training & validation loops."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        dataloaders: dict[str, torch.utils.data.DataLoader],
        device: torch.device,
        logger: Any,
        config: Any,
    ) -> None:
        self.model = model.to(device)
        self.opt = optimizer
        self.crit = criterion
        self.dls = dataloaders
        self.device = device
        self.logger = logger
        self.config = config

    def mixup_criterion(
        self, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float
    ) -> torch.Tensor:
        """Mixup loss calculation."""
        return lam * self.crit(pred, y_a) + (1 - lam) * self.crit(pred, y_b)

    def train_epoch(self, epoch: int) -> tuple[float, float, float]:
        self.model.train()
        train_loader = self.dls["train"]
        total_batches = len(train_loader)
        total_loss = 0.0
        correct = 0
        total = 0

        # For AUC calculation
        all_preds: list[np.ndarray] = []
        all_labels: list[int] = []

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[green]Epoch {epoch} Train", total=total_batches)

            for batch_idx, (specs, labels) in enumerate(train_loader):
                specs, labels = specs.to(self.device), labels.to(self.device)

                # Apply mixup if enabled
                if self.config.mixup_alpha > 0 and self.model.training:
                    specs, targets_a, targets_b, lam = mixup_data(
                        specs, labels, self.config.mixup_alpha
                    )

                    self.opt.zero_grad()
                    preds = self.model(specs)
                    loss = self.mixup_criterion(preds, targets_a, targets_b, lam)

                    # Approximate accuracy for mixup
                    _, predicted = preds.max(1)
                    total += labels.size(0)
                    correct += (
                        lam * predicted.eq(targets_a).sum().item()
                        + (1 - lam) * predicted.eq(targets_b).sum().item()
                    )

                    # For AUC, use original labels (not mixed)
                    probs = F.softmax(preds, dim=1)
                    all_preds.extend(probs.detach().cpu().numpy())
                    all_labels.extend(labels.cpu().numpy().tolist())
                else:
                    self.opt.zero_grad()
                    preds = self.model(specs)
                    loss = self.crit(preds, labels)

                    # Standard accuracy
                    _, predicted = preds.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    # For AUC calculation
                    probs = F.softmax(preds, dim=1)
                    all_preds.extend(probs.detach().cpu().numpy())
                    all_labels.extend(labels.cpu().numpy().tolist())

                loss.backward()
                self.opt.step()

                total_loss += loss.item()

                # Update progress bar
                current_acc = 100.0 * correct / total if total > 0 else 0
                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"[green]Epoch {epoch} Train — "
                        f"loss: {loss.item():.4f}, acc: {current_acc:.1f}%"
                    ),
                )

        # Calculate metrics
        avg_loss = total_loss / total_batches
        avg_acc = 100.0 * correct / total if total > 0 else 0

        # Calculate AUC ROC
        try:
            all_preds_array = np.array(all_preds)
            all_labels_array = np.array(all_labels)

            if len(np.unique(all_labels_array)) > 1:  # Need at least 2 classes for AUC
                if self.config.num_classes == 2:
                    # Binary classification
                    auc = roc_auc_score(all_labels_array, all_preds_array[:, 1])
                else:
                    # Multi-class classification
                    auc = roc_auc_score(
                        all_labels_array, all_preds_array, multi_class="ovr", average="macro"
                    )
            else:
                auc = 0.0
        except Exception as e:
            console.log(f"[yellow]Warning: Could not calculate AUC for training: {e}")
            auc = 0.0

        self.logger.log_metric("train_loss", avg_loss, step=epoch)
        self.logger.log_metric("train_acc", avg_acc, step=epoch)
        self.logger.log_metric("train_auc", auc, step=epoch)

        console.log(
            f"[bold green]Epoch {epoch} Train ⏩ "
            f"avg_loss: {avg_loss:.4f}, acc: {avg_acc:.1f}%, auc: {auc:.4f}"
        )
        return avg_loss, avg_acc, auc

    def validate(self, epoch: int) -> tuple[Optional[float], Optional[float], Optional[float]]:
        self.model.eval()
        val_loader = self.dls.get("val", None)
        if val_loader is None:
            return None, None, None

        total_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(val_loader)

        # For AUC calculation
        all_preds: list[np.ndarray] = []
        all_labels: list[int] = []

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[cyan]Epoch {epoch} Val", total=total_batches)

            with torch.no_grad():
                for specs, labels in val_loader:
                    specs, labels = specs.to(self.device), labels.to(self.device)
                    preds = self.model(specs)
                    loss = self.crit(preds, labels)

                    total_loss += loss.item()

                    # Calculate accuracy
                    _, predicted = preds.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    # For AUC calculation
                    probs = F.softmax(preds, dim=1)
                    all_preds.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy().tolist())

                    current_acc = 100.0 * correct / total if total > 0 else 0
                    progress.update(
                        task,
                        advance=1,
                        description=(
                            f"[cyan]Epoch {epoch} Val — "
                            f"loss: {loss.item():.4f}, acc: {current_acc:.1f}%"
                        ),
                    )

        # Calculate metrics
        avg_loss = total_loss / total_batches
        avg_acc = 100.0 * correct / total if total > 0 else 0

        # Calculate AUC ROC
        try:
            all_preds_array = np.array(all_preds)  # Shape: (n_samples, 206)
            all_labels_array = np.array(all_labels)  # Shape: (n_samples,)

            unique_labels = np.unique(all_labels_array)
            if len(unique_labels) > 1:
                # Extract predictions only for classes that appear in validation
                relevant_preds = all_preds_array[
                    :, unique_labels
                ]  # Shape: (n_samples, n_present_classes)

                # Renormalize the extracted predictions to sum to 1.0 if we need to reduce the prediction space
                relevant_preds = relevant_preds / relevant_preds.sum(axis=1, keepdims=True)

                # Remap labels to match the reduced prediction space
                label_mapping = {
                    old_label: new_idx for new_idx, old_label in enumerate(unique_labels)
                }
                remapped_labels = np.array([label_mapping[label] for label in all_labels_array])

                auc = roc_auc_score(
                    remapped_labels, relevant_preds, multi_class="ovr", average="macro"
                )
            else:
                auc = 0.0
        except Exception as e:
            console.log(f"[yellow]Warning: Could not calculate AUC for validation: {e}")
            auc = 0.0

        self.logger.log_metric("val_loss", avg_loss, step=epoch)
        self.logger.log_metric("val_acc", avg_acc, step=epoch)
        self.logger.log_metric("val_auc", auc, step=epoch)

        console.log(
            f"[bold cyan]Epoch {epoch} Val ⏩ "
            f"avg_loss: {avg_loss:.4f}, acc: {avg_acc:.1f}%, auc: {auc:.4f}"
        )
        return avg_loss, avg_acc, auc

    def fit(self, epochs: int) -> None:
        """Complete training loop."""
        best_val_auc = 0.0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc, train_auc = self.train_epoch(epoch)
            val_loss, val_acc, val_auc = self.validate(epoch)

            # Save best model (you can choose metric: loss or AUC, AUC required by the kaggle competition)
            if val_loss is not None and val_auc is not None and val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(self.model.state_dict(), f"{self.config.output_dir}/best_model.pth")
                console.log(f"[bold green]New best model saved! Val AUC: {val_auc:.4f}")
