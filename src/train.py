import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
import mlflow

from src.config import Config
from src.dataset import BirdClefDataset, collate_fn
from src.utils import get_mel_log_transform
from src.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train BirdCLEF model")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to config YAML"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config(args.config)

    # 1) Load metadata and map labels via taxonomy
    df = pd.read_csv(cfg.train_csv)
    tax_df = pd.read_csv(cfg.taxonomy_csv)
    code2idx = {code: idx for idx, code in enumerate(tax_df["primary_label"])}
    df["label_idx"] = df["primary_label"].map(code2idx)

    # 2) Split train/val
    if cfg.val_fraction > 0:
        train_df, val_df = train_test_split(
            df,
            test_size=cfg.val_fraction,
            stratify=df["label_idx"],
            random_state=42,
        )
    else:
        train_df, val_df = df, None

    # 3) Prepare transforms
    mel_transform = get_mel_log_transform(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0
    )

    # 4) Build DataLoaders
    train_ds = BirdClefDataset(train_df, cfg.train_audio_dir, config=cfg, transform=mel_transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    if val_df is not None:
        val_ds = BirdClefDataset(val_df, cfg.train_audio_dir, config=cfg, transform=mel_transform)
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        dataloaders = {"train": train_loader, "val": val_loader}
    else:
        dataloaders = {"train": train_loader}

    # 5) Instantiate model, loss, optimizer
    model = timm.create_model(
        cfg.backbone,
        pretrained=True,
        num_classes=cfg.num_classes,
        in_chans=1
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    # 6) Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 7) Run and log with MLflow
    mlflow.set_experiment("birdclef25_train")
    with mlflow.start_run():
        mlflow.log_params({
            "backbone": cfg.backbone,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "epochs": cfg.epochs,
            "val_fraction": cfg.val_fraction,
        })

        trainer = Trainer(model, optimizer, criterion, dataloaders, device, mlflow)
        best_val_loss = float("inf")

        for epoch in range(1, cfg.epochs + 1):
            trainer.train_epoch(epoch)
            if "val" in dataloaders:
                val_loss = trainer.validate(epoch)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    os.makedirs(cfg.output_dir, exist_ok=True)
                    best_path = os.path.join(cfg.output_dir, "best_model.pth")
                    torch.save(model.state_dict(), best_path)
                    mlflow.log_artifact(best_path, artifact_path="models")

        # final checkpoint
        os.makedirs(cfg.output_dir, exist_ok=True)
        final_path = os.path.join(cfg.output_dir, "final_model.pth")
        torch.save(model.state_dict(), final_path)
        mlflow.log_artifact(final_path, artifact_path="models")

if __name__ == "__main__":
    main()