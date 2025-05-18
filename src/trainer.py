import torch
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console

console = Console()

class Trainer:
    """Handles training & validation loops, with Rich progress bars."""
    def __init__(self, model, optimizer, criterion, dataloaders, device, logger):
        self.model = model.to(device)
        self.opt   = optimizer
        self.crit  = criterion
        self.dls   = dataloaders
        self.device = device
        self.logger = logger  # MLflowLogger

    def train_epoch(self, epoch: int):
        self.model.train()
        train_loader = self.dls['train']
        total_batches = len(train_loader)

        # Setup a rich.Progress instance
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"[green]Epoch {epoch} Train", total=total_batches)

            for specs, labels in train_loader:
                specs, labels = specs.to(self.device), labels.to(self.device)
                preds = self.model(specs)
                loss  = self.crit(preds, labels)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # log and update bar
                self.logger.log_metric("train_loss", loss.item(), step=epoch)
                progress.update(
                    task,
                    advance=1,
                    description=f"[green]Epoch {epoch} Train — loss: {loss.item():.4f}"
                )

    def validate(self, epoch: int):
        self.model.eval()
        val_loader = self.dls.get('val', None)
        if val_loader is None:
            return None

        total_loss = 0.0
        total_batches = len(val_loader)

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Epoch {epoch} Val", total=total_batches)

            with torch.no_grad():
                for specs, labels in val_loader:
                    specs, labels = specs.to(self.device), labels.to(self.device)
                    preds = self.model(specs)
                    loss  = self.crit(preds, labels)
                    total_loss += loss.item()

                    progress.update(
                        task,
                        advance=1,
                        description=f"[cyan]Epoch {epoch} Val — loss: {loss.item():.4f}"
                    )

        avg_loss = total_loss / total_batches
        self.logger.log_metric("val_loss", avg_loss, step=epoch)
        console.log(f"[bold cyan]Epoch {epoch} Val ⏩ avg_loss: {avg_loss:.4f}")
        return avg_loss