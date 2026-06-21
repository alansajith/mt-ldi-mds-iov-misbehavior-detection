"""
MT-LDI-MDS: Multi-Teacher Distillation Training
================================================
Main training script implementing the multi-teacher knowledge distillation.

LOSS FUNCTION (Key Contribution):
total_loss = ce_loss + λ1*kd_loss_A + λ2*kd_loss_B + λ3*kd_loss_C

Where:
- ce_loss = CrossEntropyLoss(student_output, true_labels)
- kd_loss_A = MSELoss(student_intermediate_features, teacher_A_embeddings)
- kd_loss_B = MSELoss(student_intermediate_features, teacher_B_embeddings)
- kd_loss_C = MSELoss(student_intermediate_features, teacher_C_embeddings)
- λ1, λ2, λ3 are learnable weights from WeightedAggregator (or fixed 1/3 each)

Three experiments:
1. Single teacher baseline (only kd_loss_A, λ2=λ3=0)
2. Multi-teacher fixed weights (mode="fixed")
3. Multi-teacher learned weights (mode="learnable") ← Main contribution
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from student.bilstm_student import BiLSTMStudent, create_student
from aggregator.weighted_aggregator import WeightedAggregator, create_aggregator


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    data_path: str = "/kaggle/working/data/veremi_processed.csv"
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
    
    # Model
    student_hidden_dim: int = 256
    aggregator_mode: str = "learnable"  # "fixed" or "learnable"
    
    # Training
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    
    # Loss weights (for single teacher baseline)
    use_single_teacher: bool = False
    single_teacher: str = "A"  # A, B, or C
    
    # Checkpointing
    checkpoint_dir: str = "/kaggle/working/training"
    save_every: int = 10
    resume_from: Optional[str] = None
    
    # Logging
    log_every: int = 1
    plot_curves: bool = True
    
    # Device
    device: str = "cuda"


# ============================================================================
# Dataset Preparation
# ============================================================================

class VeReMiDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for VeReMi data."""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        teacher_embeddings: Dict[str, np.ndarray] = None,
        seq_len: int = 1
    ):
        """
        Args:
            features: (N, input_dim) or (N, seq_len, input_dim)
            labels: (N,) class labels
            teacher_embeddings: Dict with keys 'A', 'B', 'C' mapping to (N, teacher_dim)
            seq_len: Sequence length for temporal modeling
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.seq_len = seq_len
        
        # Teacher embeddings for KD
        self.teacher_embeddings = {}
        if teacher_embeddings is not None:
            for k, v in teacher_embeddings.items():
                self.teacher_embeddings[k] = torch.FloatTensor(v)
        
        print(f"Dataset created: {len(self.features)} samples, "
              f"feature shape: {self.features.shape}, "
              f"teacher embeddings: {list(self.teacher_embeddings.keys())}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        item = {
            'features': self.features[idx],
            'labels': self.labels[idx]
        }
        for k, v in self.teacher_embeddings.items():
            item[f'teacher_{k}_emb'] = v[idx]
        return item


def load_and_prepare_data(
    data_path: str,
    teacher_emb_paths: Dict[str, str],
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    seq_len: int = 1
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Load processed VeReMi data and teacher embeddings, create train/val/test loaders.
    
    Args:
        data_path: Path to processed CSV (with all samples)
        teacher_emb_paths: Dict mapping 'A', 'B', 'C' to embedding .npy paths
        test_size: Test split ratio
        val_size: Validation split ratio (from remaining)
        random_state: Random seed
        seq_len: Sequence length
        
    Returns:
        (train_loader, val_loader, test_loader, input_dim, num_classes, scaler)
    """
    print(f"\nLoading data from {data_path}...")
    
    # Load main dataset
    df = pd.read_csv(data_path)
    print(f"Full dataset: {len(df)} samples")
    
    # Identify feature columns
    exclude_cols = ['label', 'sender', 'sendTime', 'attackerType', 'attackID']
    feature_cols = [c for c in df.columns if c not in exclude_cols and 
                    pd.api.types.is_numeric_dtype(df[c])]
    
    features = df[feature_cols].values.astype(np.float32)
    labels = df['label'].values.astype(np.int64)
    
    # Load teacher embeddings
    teacher_embeddings = {}
    for teacher_id, emb_path in teacher_emb_paths.items():
        if os.path.exists(emb_path):
            emb = np.load(emb_path)
            teacher_embeddings[teacher_id] = emb.astype(np.float32)
            print(f"  Loaded Teacher {teacher_id} embeddings: {emb.shape}")
        else:
            print(f"  WARNING: Teacher {teacher_id} embeddings not found at {emb_path}")
    
    # Split: first test, then val from remaining
    # Train = 70%, Val = 15%, Test = 15%
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    # Split teacher embeddings accordingly
    def split_embeddings(emb_dict, train_idx, val_idx, test_idx):
        split_dict = {}
        for k, v in emb_dict.items():
            split_dict[k] = {
                'train': v[train_idx],
                'val': v[val_idx],
                'test': v[test_idx]
            }
        return split_dict
    
    # Get split indices
    temp_idx = np.arange(len(X_temp))
    train_idx, val_idx = train_test_split(
        temp_idx, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    test_idx = np.arange(len(X_test)) + len(X_temp)  # Not used directly, but for reference
    
    # Actually we need to track original indices
    all_indices = np.arange(len(features))
    train_indices, test_indices = train_test_split(
        all_indices, test_size=test_size, random_state=random_state, stratify=labels
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=val_ratio, random_state=random_state, stratify=labels[train_indices]
    )
    
    teacher_splits = split_embeddings(teacher_embeddings, train_indices, val_indices, test_indices)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape for sequence modeling if needed
    if seq_len > 1:
        # For now, treat each sample as independent (seq_len=1)
        # Can be extended to create sequences from temporal data
        pass
    
    input_dim = X_train_scaled.shape[1]
    num_classes = len(np.unique(labels))
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    print(f"  Input dim: {input_dim}")
    print(f"  Num classes: {num_classes}")
    print(f"  Class distribution (train): {np.bincount(y_train)}")
    
    # Create datasets
    train_dataset = VeReMiDataset(X_train_scaled, y_train, 
                                   {k: v['train'] for k, v in teacher_splits.items()})
    val_dataset = VeReMiDataset(X_val_scaled, y_val,
                                 {k: v['val'] for k, v in teacher_splits.items()})
    test_dataset = VeReMiDataset(X_test_scaled, y_test,
                                  {k: v['test'] for k, v in teacher_splits.items()})
    
    return train_dataset, val_dataset, test_dataset, input_dim, num_classes, scaler


def create_data_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 64,
    num_workers: int = 2
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """Create PyTorch DataLoaders."""
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader


# ============================================================================
# Training Loop
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    aggregator: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainingConfig,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    aggregator.train()
    
    total_loss = 0.0
    total_ce = 0.0
    total_kd_A = 0.0
    total_kd_B = 0.0
    total_kd_C = 0.0
    correct = 0
    total = 0
    
    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch in pbar:
        features = batch['features'].to(device)  # (batch, input_dim) or (batch, seq, input_dim)
        labels = batch['labels'].to(device)
        
        # Teacher embeddings
        teacher_A_emb = batch.get('teacher_A_emb', None)
        teacher_B_emb = batch.get('teacher_B_emb', None)
        teacher_C_emb = batch.get('teacher_C_emb', None)
        
        if teacher_A_emb is not None:
            teacher_A_emb = teacher_A_emb.to(device)
        if teacher_B_emb is not None:
            teacher_B_emb = teacher_B_emb.to(device)
        if teacher_C_emb is not None:
            teacher_C_emb = teacher_C_emb.to(device)
        
        optimizer.zero_grad()
        
        # Student forward
        logits, student_features = model(features)  # student_features: (batch, 128)
        
        # CE Loss
        ce_loss = ce_criterion(logits, labels)
        
        # KD Losses
        kd_loss_A = torch.tensor(0.0, device=device)
        kd_loss_B = torch.tensor(0.0, device=device)
        kd_loss_C = torch.tensor(0.0, device=device)
        
        if config.use_single_teacher:
            # Single teacher baseline
            if config.single_teacher == "A" and teacher_A_emb is not None:
                kd_loss_A = mse_criterion(student_features, teacher_A_emb)
            elif config.single_teacher == "B" and teacher_B_emb is not None:
                kd_loss_B = mse_criterion(student_features, teacher_B_emb)
            elif config.single_teacher == "C" and teacher_C_emb is not None:
                kd_loss_C = mse_criterion(student_features, teacher_C_emb)
        else:
            # Multi-teacher: use aggregator
            if teacher_A_emb is not None and teacher_B_emb is not None and teacher_C_emb is not None:
                # Aggregator returns adapted features and weights
                aggregated_features, weights = aggregator(teacher_A_emb, teacher_B_emb, teacher_C_emb)
                
                # KD loss: MSE between student features and aggregated teacher features
                # Note: student_features is (batch, 128), aggregated_features is (batch, student_hidden_dim=256)
                # We need to project student_features to match, or use a different approach
                # Actually, the paper uses student_intermediate_features (128) vs teacher embeddings
                # Let's project aggregated to 128 for comparison
                # Or better: project student_features to 256 to match aggregated
                # For simplicity, let's add a projection layer or just use MSE on available dims
                
                # Project student features to aggregator output dim for KD
                # We'll use a simple linear projection (can be part of model)
                if not hasattr(model, 'kd_projection'):
                    model.kd_projection = nn.Linear(student_features.shape[1], aggregated_features.shape[1]).to(device)
                
                student_projected = model.kd_projection(student_features)
                kd_loss = mse_criterion(student_projected, aggregated_features)
                
                # Decompose for logging (approximate)
                kd_loss_A = mse_criterion(student_projected, teacher_A_emb)
                kd_loss_B = mse_criterion(student_projected, teacher_B_emb)
                kd_loss_C = mse_criterion(student_projected, teacher_C_emb)
                
                # Use weighted sum
                if config.aggregator_mode == "learnable":
                    w = F.softmax(aggregator.lambda_params, dim=0)
                    kd_loss = w[0]*kd_loss_A + w[1]*kd_loss_B + w[2]*kd_loss_C
                else:
                    kd_loss = (kd_loss_A + kd_loss_B + kd_loss_C) / 3
        
        # Total loss
        loss = ce_loss + kd_loss_A + kd_loss_B + kd_loss_C
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(aggregator.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_kd_A += kd_loss_A.item() if isinstance(kd_loss_A, torch.Tensor) else kd_loss_A
        total_kd_B += kd_loss_B.item() if isinstance(kd_loss_B, torch.Tensor) else kd_loss_B
        total_kd_C += kd_loss_C.item() if isinstance(kd_loss_C, torch.Tensor) else kd_loss_C
        
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{ce_loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'ce_loss': total_ce / n_batches,
        'kd_loss_A': total_kd_A / n_batches,
        'kd_loss_B': total_kd_B / n_batches,
        'kd_loss_C': total_kd_C / n_batches,
        'accuracy': 100.0 * correct / total
    }


def validate(
    model: nn.Module,
    aggregator: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: TrainingConfig
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    aggregator.eval()
    
    total_loss = 0.0
    total_ce = 0.0
    total_kd_A = 0.0
    total_kd_B = 0.0
    total_kd_C = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[Val]", leave=False):
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            teacher_A_emb = batch.get('teacher_A_emb', None)
            teacher_B_emb = batch.get('teacher_B_emb', None)
            teacher_C_emb = batch.get('teacher_C_emb', None)
            
            if teacher_A_emb is not None:
                teacher_A_emb = teacher_A_emb.to(device)
            if teacher_B_emb is not None:
                teacher_B_emb = teacher_B_emb.to(device)
            if teacher_C_emb is not None:
                teacher_C_emb = teacher_C_emb.to(device)
            
            logits, student_features = model(features)
            
            ce_loss = ce_criterion(logits, labels)
            
            kd_loss_A = torch.tensor(0.0, device=device)
            kd_loss_B = torch.tensor(0.0, device=device)
            kd_loss_C = torch.tensor(0.0, device=device)
            
            if config.use_single_teacher:
                if config.single_teacher == "A" and teacher_A_emb is not None:
                    kd_loss_A = mse_criterion(student_features, teacher_A_emb)
                elif config.single_teacher == "B" and teacher_B_emb is not None:
                    kd_loss_B = mse_criterion(student_features, teacher_B_emb)
                elif config.single_teacher == "C" and teacher_C_emb is not None:
                    kd_loss_C = mse_criterion(student_features, teacher_C_emb)
            else:
                if teacher_A_emb is not None and teacher_B_emb is not None and teacher_C_emb is not None:
                    aggregated_features, weights = aggregator(teacher_A_emb, teacher_B_emb, teacher_C_emb)
                    
                    if hasattr(model, 'kd_projection'):
                        student_projected = model.kd_projection(student_features)
                        kd_loss_A = mse_criterion(student_projected, teacher_A_emb)
                        kd_loss_B = mse_criterion(student_projected, teacher_B_emb)
                        kd_loss_C = mse_criterion(student_projected, teacher_C_emb)
                        
                        if config.aggregator_mode == "learnable":
                            w = F.softmax(aggregator.lambda_params, dim=0)
                            kd_loss = w[0]*kd_loss_A + w[1]*kd_loss_B + w[2]*kd_loss_C
                        else:
                            kd_loss = (kd_loss_A + kd_loss_B + kd_loss_C) / 3
            
            loss = ce_loss + kd_loss_A + kd_loss_B + kd_loss_C
            
            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_kd_A += kd_loss_A.item() if isinstance(kd_loss_A, torch.Tensor) else kd_loss_A
            total_kd_B += kd_loss_B.item() if isinstance(kd_loss_B, torch.Tensor) else kd_loss_B
            total_kd_C += kd_loss_C.item() if isinstance(kd_loss_C, torch.Tensor) else kd_loss_C
            
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    n_batches = len(val_loader)
    from sklearn.metrics import f1_score
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    return {
        'loss': total_loss / n_batches,
        'ce_loss': total_ce / n_batches,
        'kd_loss_A': total_kd_A / n_batches,
        'kd_loss_B': total_kd_B / n_batches,
        'kd_loss_C': total_kd_C / n_batches,
        'accuracy': 100.0 * correct / total,
        'f1_macro': f1_macro
    }


# ============================================================================
# Main Training Function
# ============================================================================

def train_multiteacher(config: TrainingConfig):
    """Main training function for multi-teacher distillation."""
    
    print("="*80)
    print("MT-LDI-MDS: Multi-Teacher Distillation Training")
    print("="*80)
    print(f"Config: {config}")
    
    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Teacher embedding paths
    teacher_emb_paths = {
        'A': '/kaggle/working/teachers/embeddings_A.npy',
        'B': '/kaggle/working/teachers/embeddings_B.npy',
        'C': '/kaggle/working/teachers/embeddings_C.npy'
    }
    
    # Load data
    train_dataset, val_dataset, test_dataset, input_dim, num_classes, scaler = load_and_prepare_data(
        config.data_path, teacher_emb_paths,
        test_size=config.test_size,
        val_size=config.val_size,
        random_state=config.random_state
    )
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.batch_size
    )
    
    # Create student model
    student = create_student(
        input_dim=input_dim,
        num_classes=num_classes,
        device=device
    )
    
    # Create aggregator
    # Teacher embedding dim from Qwen3-8B (4096)
    teacher_hidden_dim = 4096
    aggregator = create_aggregator(
        teacher_hidden_dim=teacher_hidden_dim,
        student_hidden_dim=config.student_hidden_dim,
        mode=config.aggregator_mode
    ).to(device)
    
    # Optimizer (include both student and aggregator params)
    params = list(student.parameters()) + list(aggregator.parameters())
    if hasattr(student, 'kd_projection'):
        params += list(student.kd_projection.parameters())
    
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    
    # LR Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_ce': [], 'train_kd_A': [], 'train_kd_B': [], 'train_kd_C': [],
        'val_loss': [], 'val_ce': [], 'val_kd_A': [], 'val_kd_B': [], 'val_kd_C': [],
        'val_acc': [], 'val_f1': [], 'lr': [], 'lambda_weights': []
    }
    
    if config.resume_from and os.path.exists(config.resume_from):
        print(f"Resuming from {config.resume_from}")
        checkpoint = torch.load(config.resume_from, map_location=device)
        student.load_state_dict(checkpoint['student_state_dict'])
        aggregator.load_state_dict(checkpoint['aggregator_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        history = checkpoint.get('history', history)
        print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Batch size: {config.batch_size}, LR: {config.lr}")
    print(f"Aggregator mode: {config.aggregator_mode}")
    if config.use_single_teacher:
        print(f"Single teacher mode: {config.single_teacher}")
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            student, aggregator, train_loader, optimizer, device, config, epoch
        )
        
        # Validate
        val_metrics = validate(student, aggregator, val_loader, device, config)
        
        # Scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Get lambda weights
        lambda_weights = aggregator.get_weights()
        
        # Log
        history['train_loss'].append(train_metrics['loss'])
        history['train_ce'].append(train_metrics['ce_loss'])
        history['train_kd_A'].append(train_metrics['kd_loss_A'])
        history['train_kd_B'].append(train_metrics['kd_loss_B'])
        history['train_kd_C'].append(train_metrics['kd_loss_C'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_ce'].append(val_metrics['ce_loss'])
        history['val_kd_A'].append(val_metrics['kd_loss_A'])
        history['val_kd_B'].append(val_metrics['kd_loss_B'])
        history['val_kd_C'].append(val_metrics['kd_loss_C'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_macro'])
        history['lr'].append(current_lr)
        history['lambda_weights'].append(lambda_weights)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{config.epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | CE: {train_metrics['ce_loss']:.4f} | "
              f"KD_A: {train_metrics['kd_loss_A']:.4f} | KD_B: {train_metrics['kd_loss_B']:.4f} | KD_C: {train_metrics['kd_loss_C']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | CE: {val_metrics['ce_loss']:.4f} | "
              f"KD_A: {val_metrics['kd_loss_A']:.4f} | KD_B: {val_metrics['kd_loss_B']:.4f} | KD_C: {val_metrics['kd_loss_C']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.2f}% | F1: {val_metrics['f1_macro']:.4f}")
        print(f"  LR: {current_lr:.6f} | λ: A={lambda_weights['teacher_A']:.4f} B={lambda_weights['teacher_B']:.4f} C={lambda_weights['teacher_C']:.4f}")
        
        # Print lambda weights every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  >>> Lambda weights at epoch {epoch+1}: {lambda_weights}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'aggregator_state_dict': aggregator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'config': asdict(config),
                'history': history,
                'input_dim': input_dim,
                'num_classes': num_classes,
                'scaler': scaler
            }, best_path)
            print(f"  >>> New best model saved! Val Acc: {best_val_acc:.2f}%")
        
        # Save periodic checkpoint
        if (epoch + 1) % config.save_every == 0:
            ckpt_path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'aggregator_state_dict': aggregator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'config': asdict(config),
                'history': history,
                'input_dim': input_dim,
                'num_classes': num_classes
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")
        
        # Save training log
        log_path = os.path.join(config.checkpoint_dir, 'training_log.csv')
        log_df = pd.DataFrame({
            'epoch': range(1, len(history['train_loss']) + 1),
            'train_loss': history['train_loss'],
            'train_ce': history['train_ce'],
            'train_kd_A': history['train_kd_A'],
            'train_kd_B': history['train_kd_B'],
            'train_kd_C': history['train_kd_C'],
            'val_loss': history['val_loss'],
            'val_ce': history['val_ce'],
            'val_kd_A': history['val_kd_A'],
            'val_kd_B': history['val_kd_B'],
            'val_kd_C': history['val_kd_C'],
            'val_acc': history['val_acc'],
            'val_f1': history['val_f1'],
            'lr': history['lr'],
            'lambda_A': [w['teacher_A'] for w in history['lambda_weights']],
            'lambda_B': [w['teacher_B'] for w in history['lambda_weights']],
            'lambda_C': [w['teacher_C'] for w in history['lambda_weights']],
        })
        log_df.to_csv(log_path, index=False)
        
        # GPU memory
        if device.type == 'cuda':
            mem = torch.cuda.memory_allocated() / 1e9
            print(f"  GPU Memory: {mem:.2f} GB")
    
    # Plot training curves
    if config.plot_curves:
        plot_training_curves(history, config.checkpoint_dir)
    
    print(f"\n{'='*80}")
    print(f"Training complete! Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {os.path.join(config.checkpoint_dir, 'best_model.pt')}")
    print(f"Training log saved to: {os.path.join(config.checkpoint_dir, 'training_log.csv')}")
    print(f"{'='*80}")
    
    return student, aggregator, history, test_loader


def plot_training_curves(history: Dict, save_dir: str):
    """Plot and save training curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # CE Loss
    axes[0, 1].plot(epochs, history['train_ce'], 'b-', label='Train CE')
    axes[0, 1].plot(epochs, history['val_ce'], 'r-', label='Val CE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('CE Loss')
    axes[0, 1].set_title('Cross-Entropy Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # KD Losses
    axes[0, 2].plot(epochs, history['train_kd_A'], 'b-', label='KD_A')
    axes[0, 2].plot(epochs, history['train_kd_B'], 'g-', label='KD_B')
    axes[0, 2].plot(epochs, history['train_kd_C'], 'r-', label='KD_C')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('KD Loss')
    axes[0, 2].set_title('Knowledge Distillation Losses')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 0].plot(epochs, history['val_acc'], 'g-', label='Val Acc')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Validation Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1 Macro
    axes[1, 1].plot(epochs, history['val_f1'], 'm-', label='Val F1 Macro')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Validation F1 Macro')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Lambda weights
    lambda_A = [w['teacher_A'] for w in history['lambda_weights']]
    lambda_B = [w['teacher_B'] for w in history['lambda_weights']]
    lambda_C = [w['teacher_C'] for w in history['lambda_weights']]
    axes[1, 2].plot(epochs, lambda_A, 'b-', label='λ_A (DoS/Sybil)')
    axes[1, 2].plot(epochs, lambda_B, 'g-', label='λ_B (Position)')
    axes[1, 2].plot(epochs, lambda_C, 'r-', label='λ_C (Speed/Replay)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Weight')
    axes[1, 2].set_title('Learned Aggregation Weights (λ)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")


# ============================================================================
# Experiment Runners
# ============================================================================

def run_experiment(
    experiment_name: str,
    config: TrainingConfig,
    data_path: str
) -> Dict:
    """Run a single experiment configuration."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*80}")
    
    # Update config for this experiment
    exp_config = TrainingConfig(**asdict(config))
    exp_config.checkpoint_dir = os.path.join(config.checkpoint_dir, experiment_name.lower().replace(' ', '_'))
    
    os.makedirs(exp_config.checkpoint_dir, exist_ok=True)
    
    # Run training
    student, aggregator, history, test_loader = train_multiteacher(exp_config)
    
    # Return final metrics
    return {
        'experiment': experiment_name,
        'best_val_acc': max(history['val_acc']),
        'best_val_f1': max(history['val_f1']),
        'final_lambda': history['lambda_weights'][-1] if history['lambda_weights'] else {},
        'history': history
    }


def main():
    parser = argparse.ArgumentParser(description='MT-LDI-MDS Multi-Teacher Distillation Training')
    parser.add_argument('--data', type=str, default='/kaggle/working/data/veremi_processed.csv',
                        help='Path to processed VeReMi CSV')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--mode', type=str, default='learnable', choices=['fixed', 'learnable'],
                        help='Aggregator mode')
    parser.add_argument('--single-teacher', action='store_true',
                        help='Run single teacher baseline')
    parser.add_argument('--teacher', type=str, default='A', choices=['A', 'B', 'C'],
                        help='Single teacher to use (if --single-teacher)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--run-all-experiments', action='store_true',
                        help='Run all three comparison experiments')
    parser.add_argument('--checkpoint-dir', type=str, default='/kaggle/working/training',
                        help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Base config
    base_config = TrainingConfig(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        aggregator_mode=args.mode,
        use_single_teacher=args.single_teacher,
        single_teacher=args.teacher,
        resume_from=args.resume,
        checkpoint_dir=args.checkpoint_dir
    )
    
    if args.run_all_experiments:
        # Run three comparison experiments
        print("\nRunning all three comparison experiments...")
        
        results = []
        
        # Experiment 1: Single teacher baseline (Teacher A)
        exp1_config = TrainingConfig(**asdict(base_config))
        exp1_config.use_single_teacher = True
        exp1_config.single_teacher = 'A'
        exp1_config.checkpoint_dir = os.path.join(base_config.checkpoint_dir, 'exp1_single_teacher_A')
        r1 = run_experiment("Single Teacher Baseline (A)", exp1_config, args.data)
        results.append(r1)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Experiment 2: Multi-teacher fixed weights
        exp2_config = TrainingConfig(**asdict(base_config))
        exp2_config.use_single_teacher = False
        exp2_config.aggregator_mode = 'fixed'
        exp2_config.checkpoint_dir = os.path.join(base_config.checkpoint_dir, 'exp2_multi_fixed')
        r2 = run_experiment("Multi-Teacher Fixed Weights", exp2_config, args.data)
        results.append(r2)
        
        torch.cuda.empty_cache()
        
        # Experiment 3: Multi-teacher learned weights (MAIN)
        exp3_config = TrainingConfig(**asdict(base_config))
        exp3_config.use_single_teacher = False
        exp3_config.aggregator_mode = 'learnable'
        exp3_config.checkpoint_dir = os.path.join(base_config.checkpoint_dir, 'exp3_multi_learnable')
        r3 = run_experiment("Multi-Teacher Learned Weights (MAIN)", exp3_config, args.data)
        results.append(r3)
        
        # Print summary table
        print("\n" + "="*80)
        print("EXPERIMENT COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Experiment':<40} {'Best Val Acc':>12} {'Best Val F1':>12} {'λ_A':>8} {'λ_B':>8} {'λ_C':>8}")
        print("-"*80)
        for r in results:
            lam = r['final_lambda']
            print(f"{r['experiment']:<40} {r['best_val_acc']:>11.2f}% {r['best_val_f1']:>11.4f} "
                  f"{lam.get('teacher_A', 0):>8.4f} {lam.get('teacher_B', 0):>8.4f} {lam.get('teacher_C', 0):>8.4f}")
        
        # Save comparison results
        comparison_path = os.path.join(base_config.checkpoint_dir, 'experiment_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nComparison results saved to: {comparison_path}")
        
    else:
        # Single run
        train_multiteacher(base_config)


if __name__ == '__main__':
    main()