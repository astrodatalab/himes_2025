# Import packages
import argparse
import os
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import mlflow
import mlflow.pytorch
import pynvml as nvidia_smi
from galaxy_loader import GalaxyDataset
from modules import MMAE_ViT, redshift_hsc_loss

"""
GPU Memory Utilities
"""
def set_gpu_memory_limit(memory_limit_gb):
    memory_limit_bytes = memory_limit_gb * 1024 ** 3
    memory_fraction = memory_limit_bytes / get_total_gpu_memory()
    torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
    torch.cuda.empty_cache()

def get_total_gpu_memory():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    return info.total

def print_gpu_utilization():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024**2} MB.")

"""
Helper: Setup optimizer and scheduler
"""
def setup_optimizer_and_scheduler(model, args):
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'pos_embed' in n or 'cls_token' in n], 
         'lr': args.lr * 0.1, 'weight_decay': 0},
        {'params': [p for n, p in model.named_parameters() if not ('pos_embed' in n or 'cls_token' in n)], 
         'lr': args.lr, 'weight_decay': args.weight_decay}
    ]
 
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = None
    warmup_scheduler = None #introduce warmup schedular, optional

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.epochs // 4, T_mult=2, eta_min=args.lr * 0.01
        )

        # Add warmup scheduler
        warmup_epochs = getattr(args, 'warmup_epochs', 10)  # Default to 10 if not specified
        args.warmup_epochs = warmup_epochs
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
    return optimizer, scheduler, warmup_scheduler

"""
Helper: Load model from MLflow if exists
"""
def load_model_from_mlflow(run_name, device):
    try:
        model_uri = f"models:/{run_name}/latest"
        print(f"Loading model from MLflow: {model_uri}")
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
        print("Model loaded from MLflow successfully.")
        return model
    except Exception as e:
        print(f"Could not load model from MLflow: {e}")
        return None

"""
Validation loop
"""
def validate(model, val_loader, args, zero_spec: bool):
    model.eval()

    # Initialize the individual losses and total loss
    val_losses = {'total': 0, 'img': 0, 'spec': 0, 'z': 0}

    with torch.no_grad():
        for img, spec, z_spec, tid in val_loader:
                    img, spec, z_spec = img.to(args.device, non_blocking=True), spec.to(args.device, non_blocking=True), z_spec.to(args.device, non_blocking=True)
    
                    if spec.dim() == 2:
                        spec = spec.unsqueeze(1)

                    # Downsample the spectrum
                    spec = F.avg_pool1d(spec, kernel_size=30, stride=30)  # downsample to (B, 1, 259)

                    if zero_spec:
                        spec = torch.zeros_like(spec)
    
                    # Forward pass with higher mask ratios for better representation learning
                    outputs = model(img, spec, img_mask_ratio=args.img_mask_ratio, spec_mask_ratio=args.spec_mask_ratio)

                    img_recon = outputs['img_recon']
                    spec_recon = outputs['spec_recon']
                    fused_repr = outputs['fused_representation']
                    img_mask = outputs['img_mask']
                    spec_mask = outputs['spec_mask']
                    redshift_pred = outputs['redshift_pred']
            
                    # IMAGE LOSS
                    # Create spatial mask from patch mask
                    B, num_patches = img_mask.shape
                    patches_per_side = model.img_size // model.patch_size
    
                    # Reshape mask to spatial format and expand
                    img_mask_spatial = img_mask.view(B, patches_per_side, patches_per_side)
                    img_mask_expanded = img_mask_spatial.repeat_interleave(model.patch_size, dim=1).repeat_interleave(model.patch_size, dim=2)
                    img_mask_expanded = img_mask_expanded.unsqueeze(1).expand(-1, 5, -1, -1)  # Expand to 5 channels
    
                    img_loss = compute_masked_loss(img_recon, img, img_mask_expanded, loss_fn=F.mse_loss)
    
                    # SPECTRUM LOSS
                    spec_gt = spec.squeeze(1)  # (B, 7783)
    
                    # Truncate to match patching
                    truncated_len = model.spec_num_patches * model.spectrum_patch_size
                    spec_gt_trunc = spec_gt[:, :truncated_len]
                    spec_recon_trunc = spec_recon[:, :truncated_len]
    
                    # Expand spectrum mask to match truncated spectrum length
                    spec_mask_expanded = spec_mask.repeat_interleave(model.spectrum_patch_size, dim=1)
    
                    spec_loss = compute_masked_loss(spec_recon_trunc, spec_gt_trunc, spec_mask_expanded, loss_fn=F.mse_loss) # NEW
    
                    # REDSHIFT LOSS
                    z_loss = redshift_hsc_loss(z_spec, redshift_pred.squeeze())
    
                    # TOTAL LOSS
                    lambda_img = 0.1
                    lambda_spec = 0
                    lambda_z = 1.0
                    loss = lambda_img * img_loss + lambda_spec * spec_loss + lambda_z * z_loss
    
                    # Update metrics
                    val_losses['total'] += loss.item()
                    val_losses['img'] += img_loss.item()
                    val_losses['spec'] += spec_loss.item()
                    val_losses['z'] += z_loss.item()

        for k in val_losses:
            val_losses[k] /= len(val_loader)

    return val_losses

"""
Training loop with MLflow integration and validation
"""
def compute_masked_loss(pred, target, mask, loss_fn=F.huber_loss):
    """Compute loss only on masked regions"""
    if mask.sum() == 0:  # No masked tokens
        return torch.tensor(0.0, device=pred.device)

    masked_pred = pred[mask.bool()]
    masked_target = target[mask.bool()]
    if isinstance(loss_fn, nn.HuberLoss):
        return loss_fn(masked_pred, masked_target, delta = 10)

    return loss_fn(masked_pred, masked_target)

# Training function
def train_mmae(args, model, train_loader, val_loader, mlflow_uri="http://localhost:8080"):
    # Use MLflow or comment this out
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(args.run_name)

    # Replace path with where your models and logs will be saved
    save_dir = os.path.join("/data2/models/", args.run_name)
    log_dir = os.path.join("/data2/logs/", args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize MLflow run
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            'dataset': args.dataset_path,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'epochs': args.epochs,
            'scheduler': args.scheduler,
            'run_name': args.run_name,
            'image_size': 64,
            'model_description': args.model_description,
            'img_mask_ratio': args.img_mask_ratio,
            'spec_mask_ratio': args.spec_mask_ratio
        })

        loaded_model = load_model_from_mlflow(args.run_name, args.device)
        if loaded_model is not None:
            model.load_state_dict(loaded_model.state_dict())
            print("Resuming training from MLflow checkpoint.")
        else:
            print("Starting training from scratch.")

        optimizer, scheduler, warmup_scheduler = setup_optimizer_and_scheduler(model, args)

        best_loss = float('inf')
        writer = SummaryWriter(os.path.join(log_dir, "runs"))

        # Training metrics
        running_losses = {'total': [], 'img': [], 'spec': []}

        for epoch in range(args.epochs):
            model.train()
            epoch_losses = {'total': 0, 'img': 0, 'spec': 0, 'z': 0}
            total_loss = 0.0
            start_time = time.time()

            for img, spec, z_spec, tid in train_loader:
                img, spec, z_spec = img.to(args.device, non_blocking=True), spec.to(args.device, non_blocking=True), z_spec.to(args.device, non_blocking=True)

                if spec.dim() == 2:
                    spec = spec.unsqueeze(1)

                # Downsample spectrum
                spec = F.avg_pool1d(spec, kernel_size=30, stride=30)  # downsample to (B, 1, 259)

                # Decide whether to drop spectra (optionally, can un-comment to drop images)
                drop_spec = (torch.rand(1).item() < args.spec_drop_prob)
                #drop_img  = (torch.rand(1).item() < args.img_drop_prob)
                
                if drop_spec:
                    spec = torch.zeros_like(spec)
                #if drop_img:
                #    imgs = torch.zeros_like(imgs)

                # Forward pass with higher mask ratios for better representation learning
                outputs = model(img, spec, img_mask_ratio=args.img_mask_ratio, spec_mask_ratio=args.spec_mask_ratio)

                img_recon = outputs['img_recon']
                spec_recon = outputs['spec_recon']
                fused_repr = outputs['fused_representation']
                img_mask = outputs['img_mask']
                spec_mask = outputs['spec_mask']
                redshift_pred = outputs['redshift_pred']
                
                # IMAGE LOSS
                # Create spatial mask from patch mask
                B, num_patches = img_mask.shape
                patches_per_side = model.img_size // model.patch_size

                # Reshape mask to spatial format and expand
                img_mask_spatial = img_mask.view(B, patches_per_side, patches_per_side)
                img_mask_expanded = img_mask_spatial.repeat_interleave(model.patch_size, dim=1).repeat_interleave(model.patch_size, dim=2)
                img_mask_expanded = img_mask_expanded.unsqueeze(1).expand(-1, 5, -1, -1)  # Expand to 5 channels

                img_loss = compute_masked_loss(img_recon, img, img_mask_expanded, loss_fn=F.mse_loss)

                # SPECTRUM LOSS
                spec_gt = spec.squeeze(1)  # (B, 7783)

                # Truncate to match patching
                truncated_len = model.spec_num_patches * model.spectrum_patch_size
                spec_gt_trunc = spec_gt[:, :truncated_len]
                spec_recon_trunc = spec_recon[:, :truncated_len]

                # Expand spectrum mask to match truncated spectrum length
                spec_mask_expanded = spec_mask.repeat_interleave(model.spectrum_patch_size, dim=1)

                if not drop_spec:
                    spec_loss = compute_masked_loss(spec_recon_trunc, spec_gt_trunc, spec_mask_expanded, loss_fn=F.mse_loss)
                else:
                    spec_loss = torch.tensor(0.0, device=img.device)

                # REDSHIFT LOSS
                z_loss = redshift_hsc_loss(z_spec, redshift_pred.squeeze())

                # TOTAL LOSS
                lambda_img = args.lambda_img
                lambda_spec = 0.0 if drop_spec else args.lambda_spec
                lambda_z = args.lambda_z
                loss = lambda_img * img_loss + lambda_spec * spec_loss + lambda_z * z_loss

                # Gradient clipping and optimization
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Update metrics
                epoch_losses['total'] += loss.item()
                epoch_losses['img'] += img_loss.item()
                epoch_losses['spec'] += spec_loss.item()
                epoch_losses['z'] += z_loss.item()
                
                total_loss += loss.item()

            if scheduler is not None: #check for warmup scheduler
                if warmup_scheduler is not None and epoch < args.warmup_epochs:
                    warmup_scheduler.step()
                else:
                    scheduler.step()

            # Calculate total train loss and val losses
            avg_train_loss = total_loss / len(train_loader)
            running_losses['total'].append(avg_train_loss)

            # Two validations: one that has spectra, one that does not
            # Currently, spectra weighting in validation is 0
            val_with_spec = validate(model, val_loader, args, zero_spec=False)
            val_no_spec = validate(model, val_loader, args, zero_spec=True)

            # Setting the validation loss that decides the best model as the one without spectra
            val_main = val_no_spec

            # Calculate individual training losses
            avg_train_z_loss = epoch_losses['z'] / len(train_loader)
            avg_train_img_loss = epoch_losses['img'] / len(train_loader)
            avg_train_spec_loss = epoch_losses['spec'] / len(train_loader)

            # Log total loss
            mlflow.log_metric("train_total_loss", avg_train_loss, step=epoch)

            # Log individual training losses
            mlflow.log_metric("train_z_loss", avg_train_z_loss, step=epoch)
            mlflow.log_metric("train_img_loss", avg_train_img_loss, step=epoch)
            mlflow.log_metric("train_spec_loss", avg_train_spec_loss, step=epoch)

            # Log validation loss
            mlflow.log_metric("val_total_with_spec", val_with_spec['total'], step=epoch)
            mlflow.log_metric("val_z_with_spec",     val_with_spec['z'],     step=epoch)
            mlflow.log_metric("val_spec_with_spec",  val_with_spec['spec'],  step=epoch)
            mlflow.log_metric("val_img_with_spec",   val_with_spec['img'],   step=epoch)

            mlflow.log_metric("val_total_no_spec", val_no_spec['total'], step=epoch)
            mlflow.log_metric("val_z_no_spec",     val_no_spec['z'],     step=epoch)
            mlflow.log_metric("val_spec_no_spec",  val_no_spec['spec'],  step=epoch)
            mlflow.log_metric("val_img_no_spec",   val_no_spec['img'],   step=epoch)

            # Log total loss
            writer.add_scalar('Loss/Train_Total', avg_train_loss, epoch)
            writer.add_scalar('Loss/Val_Total_WithSpec', val_with_spec['total'], epoch)
            writer.add_scalar('Loss/Val_Total_NoSpec',   val_no_spec['total'],   epoch)

            # Track timing and monitor progress
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{args.epochs} - Train loss: {avg_train_loss:.4f}, Val loss: {val_main['total']:.4f}, Time: {elapsed:.1f}s")

            # Determining the new best model
            if val_main['total'] < best_loss:
                best_loss = val_main['total']
                mlflow.pytorch.log_model(
                    model,
                    artifact_path="best_model",
                    registered_model_name=args.run_name
                )
                print(f"Saved new best model to MLflow with val loss: {best_loss:.4f}")

            # Log checkpoints in MLflow
            if (epoch + 1) % 10 == 0:
                mlflow.pytorch.log_model(
                    model,
                    artifact_path=f"checkpoint_epoch_{epoch+1}"
                )
                print(f"Saved checkpoint at epoch {epoch+1} to MLflow")

            # Early stopping check (optional)
            if len(running_losses['total']) >= 50:
                recent_losses = running_losses['total'][-50:]
                if max(recent_losses) - min(recent_losses) < 1e-6:
                    print("Early stopping: Loss has converged")
                    break

        writer.close()

    print(f"Training complete. Best val loss: {best_loss:.4f}")
    return model

"""
Launch script with arg parsing
"""
def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='mmae')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--scheduler', action='store_true')
    parser.add_argument('--img_mask_ratio', type=float, default=0.75)
    parser.add_argument('--spec_mask_ratio', type=float, default=0.75)
    parser.add_argument('--spec_drop_prob', type=float, default=0.5)  
    parser.add_argument('--lambda_img', type=float, default=0.1)
    parser.add_argument('--lambda_spec', type=float, default=0.01)     
    parser.add_argument('--lambda_z', type=float, default=1.0)
    parser.add_argument('--dataset_path', type=str, default='/path/to/data')
    parser.add_argument('--model_description', type=str, default="Insert your model description here.")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Optionally, set GPU limits
    set_gpu_memory_limit(30)
    print_gpu_utilization()

    # Create the dataset
    dataset = GalaxyDataset(h5_path=args.dataset_path, transform=None, normalize=True)
    total_size = len(dataset)

    # Define split sizes
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size  # Remaining 15%
    
    # Use a fixed seed for reproducibility
    generator = torch.Generator().manual_seed(0)
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    # Initialize the model
    model = MMAE_ViT(
        img_size=64,
        patch_size=8,
        spectrum_len=259, #7783,
        spectrum_patch_size=8, #128,
        embed_dim=256,
        transformer_depth=4, #6,
        nhead=8,
        cross_attention_layers=4,
        dropout=0.1
    ).to(args.device)

    train_mmae(args, model, train_loader, val_loader)

if __name__ == "__main__":
    launch()