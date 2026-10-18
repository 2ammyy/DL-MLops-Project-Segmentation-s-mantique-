import argparse
import random
import time
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import get_datasets
from src.losses import get_loss
from src.metrics import dice_score, iou_score
from src.models.factory import create_model


def parse_args():
    p = argparse.ArgumentParser(description="Entrainement PolypCopilot")
    p.add_argument("--data-dir", default="data/raw/Kvasir-SEG")
    p.add_argument("--model", default="unet")
    p.add_argument("--encoder", default=None)
    p.add_argument("--encoder-weights", default="imagenet")
    p.add_argument("--loss", default="bce_dice", choices=["dice", "bce_dice", "bce"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--encoder-lr-factor", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--checkpoint-dir", default="models/checkpoints")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(model, lr, encoder_lr_factor, weight_decay):
    encoder_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            other_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": lr * encoder_lr_factor},
            {"params": other_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    total_loss, n_samples = 0.0, 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss, dice_sum, iou_sum, n_samples = 0.0, 0.0, 0.0, 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)
        dice_sum += dice_score(logits.float(), masks) * images.size(0)
        iou_sum += iou_score(logits.float(), masks) * images.size(0)
        n_samples += images.size(0)
    n = max(n_samples, 1)
    return {"loss": total_loss / n, "dice": dice_sum / n, "iou": iou_sum / n}


def save_checkpoint(path, model, optimizer, epoch, best_dice, args, metrics):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "model": args.model,
            "encoder": args.encoder,
            "encoder_weights": args.encoder_weights,
            "loss": args.loss,
            "image_size": 256,
            "seed": args.seed,
            "metrics": metrics,
        },
        path,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    use_amp = args.amp and device.type == "cuda"
    print(f"device={device} | amp={use_amp}")

    data_dir = Path(args.data_dir)
    train_ds, val_ds, test_ds = get_datasets(data_dir / "images", data_dir / "masks", seed=args.seed)
    if args.max_train_samples:
        train_ds = Subset(train_ds, range(min(args.max_train_samples, len(train_ds))))
    if args.max_val_samples:
        val_ds = Subset(val_ds, range(min(args.max_val_samples, len(val_ds))))
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    model = create_model(
        name=args.model,
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights if args.encoder_weights != "none" else None,
    ).to(device)
    criterion = get_loss(args.loss)
    optimizer = build_optimizer(model, args.lr, args.encoder_lr_factor, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    mlflow.set_experiment("PolypCopilot")
    run_name = f"{args.model}_{args.encoder or 'default'}_{datetime.now():%Y%m%d-%H%M%S}"
    best_dice, best_epoch, epochs_no_improve = 0.0, -1, 0

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: v for k, v in vars(args).items()})
        mlflow.set_tag("device", str(device))
        print(f"mlflow run_id={run.info.run_id}")

        for epoch in range(args.epochs):
            start = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
            val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
            scheduler.step(val_metrics["dice"])
            epoch_time = time.time() - start
            current_lr = optimizer.param_groups[-1]["lr"]

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_dice": val_metrics["dice"],
                    "val_iou": val_metrics["iou"],
                    "lr": current_lr,
                    "epoch_time_s": epoch_time,
                },
                step=epoch,
            )
            print(
                f"epoch {epoch + 1:02d}/{args.epochs} | "
                f"train_loss {train_loss:.4f} | val_dice {val_metrics['dice']:.4f} | "
                f"val_iou {val_metrics['iou']:.4f} | lr {current_lr:.2e} | {epoch_time:.1f}s"
            )

            save_checkpoint(last_path, model, optimizer, epoch, best_dice, args, val_metrics)
            if val_metrics["dice"] > best_dice:
                best_dice = val_metrics["dice"]
                best_epoch = epoch
                epochs_no_improve = 0
                save_checkpoint(best_path, model, optimizer, epoch, best_dice, args, val_metrics)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"early stopping epoch {epoch + 1} (patience={args.patience})")
                    break

        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(model, test_loader, criterion, device, use_amp)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_artifact(str(best_path))
        mlflow.set_tag("best_epoch", best_epoch)

        print(f"\nMEILLEUR checkpoint: {best_path} (epoch {best_epoch + 1}, val_dice {best_dice:.4f})")
        print(f"TEST: dice={test_metrics['dice']:.4f} iou={test_metrics['iou']:.4f} loss={test_metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
