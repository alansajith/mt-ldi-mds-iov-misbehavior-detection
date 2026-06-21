"""
MT-LDI-MDS: Evaluation Module
==============================
This module evaluates the trained student model on the held-out test set.
It runs three comparison experiments and generates comprehensive reports:
- Classification report (precision, recall, F1 per class + macro average)
- Confusion matrix heatmap
- Inference time measurement
- Results saved to JSON

Three Experiments:
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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from student.bilstm_student import BiLSTMStudent, create_student
from aggregator.weighted_aggregator import WeightedAggregator, create_aggregator


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration."""
    # Model checkpoint
    checkpoint_path: str = "/kaggle/working/training/best_model.pt"
    
    # Data
    data_path: str = "/kaggle/working/data/veremi_processed.csv"
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
    
    # Teacher embeddings (for KD loss computation if needed)
    teacher_emb_paths: Dict[str, str] = None
    
    # Output
    output_dir: str = "/kaggle/working/evaluation"
    save_heatmap: bool = True
    save_results: bool = True
    
    # Device
    device: str = "cuda"
    
    def __post_init__(self):
        if self.teacher_emb_paths is None:
            self.teacher_emb_paths = {
                'A': '/kaggle/working/teachers/embeddings_A.npy',
                'B': '/kaggle/working/teachers/embeddings_B.npy',
                'C': '/kaggle/working/teachers/embeddings_C.npy'
            }


# ============================================================================
# Attack Label Names
# ============================================================================

ATTACK_NAMES = {
    0: 'benign',
    1: 'DoS',
    2: 'Sybil',
    3: 'fixed_position',
    4: 'random_position',
    5: 'eventual_stop',
    6: 'fixed_speed',
    7: 'random_speed',
    8: 'data_replay'
}

CLASS_NAMES = [ATTACK_NAMES[i] for i in range(9)]


# ============================================================================
# Data Loading
# ============================================================================

def load_test_data(
    data_path: str,
    teacher_emb_paths: Dict[str, str],
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[torch.utils.data.DataLoader, np.ndarray, np.ndarray, StandardScaler, int, int]:
    """
    Load test data and teacher embeddings.
    
    Returns:
        test_loader, X_test_scaled, y_test, scaler, input_dim, num_classes
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Feature columns
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
    
    # Split indices (same as training)
    all_indices = np.arange(len(features))
    train_indices, test_indices = train_test_split(
        all_indices, test_size=test_size, random_state=random_state, stratify=labels
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=val_size/(1-test_size), random_state=random_state, 
        stratify=labels[train_indices]
    )
    
    # Test set
    X_test = features[test_indices]
    y_test = labels[test_indices]
    
    # Teacher embeddings for test
    teacher_test = {}
    for k, v in teacher_embeddings.items():
        teacher_test[k] = v[test_indices]
    
    # Scale (fit on train, transform test)
    # We need train data to fit scaler
    X_train = features[train_indices]
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    input_dim = X_test_scaled.shape[1]
    num_classes = len(np.unique(labels))
    
    print(f"Test set: {len(X_test)} samples")
    print(f"Input dim: {input_dim}, Num classes: {num_classes}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    # Create dataset and loader
    test_dataset = VeReMiTestDataset(X_test_scaled, y_test, teacher_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    return test_loader, X_test_scaled, y_test, scaler, input_dim, num_classes


def train_test_split(
    indices: np.ndarray,
    test_size: float,
    random_state: int,
    stratify: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper for sklearn's train_test_split to avoid import at top level."""
    from sklearn.model_selection import train_test_split as sk_split
    return sk_split(indices, test_size=test_size, random_state=random_state, stratify=stratify)


class VeReMiTestDataset(torch.utils.data.Dataset):
    """Test dataset with teacher embeddings."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 teacher_embeddings: Dict[str, np.ndarray] = None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.teacher_embeddings = {}
        if teacher_embeddings:
            for k, v in teacher_embeddings.items():
                self.teacher_embeddings[k] = torch.FloatTensor(v)
    
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


# ============================================================================
# Model Loading
# ============================================================================

def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device
) -> Tuple[BiLSTMStudent, WeightedAggregator, Dict, int, int]:
    """
    Load student model and aggregator from checkpoint.
    
    Returns:
        student, aggregator, config, input_dim, num_classes
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    config = checkpoint.get('config', {})
    input_dim = checkpoint.get('input_dim', 20)
    num_classes = checkpoint.get('num_classes', 9)
    
    # Create student model
    student = create_student(
        input_dim=input_dim,
        num_classes=num_classes,
        device=device
    )
    student.load_state_dict(checkpoint['student_state_dict'])
    student.eval()
    
    # Create aggregator
    teacher_hidden_dim = 4096  # Qwen3-8B hidden dim
    student_hidden_dim = config.get('student_hidden_dim', 256)
    aggregator_mode = config.get('aggregator_mode', 'learnable')
    
    aggregator = create_aggregator(
        teacher_hidden_dim=teacher_hidden_dim,
        student_hidden_dim=student_hidden_dim,
        mode=aggregator_mode
    ).to(device)
    aggregator.load_state_dict(checkpoint['aggregator_state_dict'])
    aggregator.eval()
    
    print(f"Model loaded successfully:")
    print(f"  Input dim: {input_dim}")
    print(f"  Num classes: {num_classes}")
    print(f"  Aggregator mode: {aggregator_mode}")
    print(f"  Lambda weights: {aggregator.get_weights()}")
    
    return student, aggregator, config, input_dim, num_classes


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(
    student: BiLSTMStudent,
    aggregator: WeightedAggregator,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Dict,
    class_names: List[str] = None
) -> Dict[str, Any]:
    """
    Evaluate model on test set.
    
    Returns:
        Dictionary with metrics, predictions, and timing info.
    """
    student.eval()
    aggregator.eval()
    
    all_preds = []
    all_labels = []
    all_logits = []
    all_student_features = []
    all_teacher_A = []
    all_teacher_B = []
    all_teacher_C = []
    all_aggregated = []
    
    # For timing
    inference_times = []
    
    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    
    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
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
            
            # Time inference
            start_time = time.perf_counter()
            
            logits, student_features = student(features)
            
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) / features.size(0) * 1000)  # ms per sample
            
            # Aggregator forward (for analysis)
            aggregated_features = None
            if teacher_A_emb is not None and teacher_B_emb is not None and teacher_C_emb is not None:
                aggregated_features, weights = aggregator(teacher_A_emb, teacher_B_emb, teacher_C_emb)
                all_aggregated.append(aggregated_features.cpu().numpy())
                all_teacher_A.append(teacher_A_emb.cpu().numpy())
                all_teacher_B.append(teacher_B_emb.cpu().numpy())
                all_teacher_C.append(teacher_C_emb.cpu().numpy())
            
            all_logits.append(logits.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_student_features.append(student_features.cpu().numpy())
            
            # Compute losses for logging
            ce_loss = ce_criterion(logits, labels)
            total_ce += ce_loss.item() * labels.size(0)
            
            if aggregated_features is not None:
                if hasattr(student, 'kd_projection'):
                    student_proj = student.kd_projection(student_features)
                else:
                    # Create projection on the fly
                    proj = nn.Linear(student_features.shape[1], aggregated_features.shape[1]).to(device)
                    nn.init.xavier_uniform_(proj.weight)
                    nn.init.zeros_(proj.bias)
                    student_proj = proj(student_features)
                kd_loss = mse_criterion(student_proj, aggregated_features)
                total_kd += kd_loss.item() * labels.size(0)
            
            total_loss += (ce_loss.item() + (kd_loss.item() if aggregated_features is not None else 0)) * labels.size(0)
    
    # Concatenate all
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_logits = np.concatenate(all_logits)
    all_student_features = np.concatenate(all_student_features)
    
    n_samples = len(all_labels)
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Per-class metrics
    report = classification_report(
        all_labels, all_preds, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Average inference time
    avg_inference_ms = np.mean(inference_times)
    std_inference_ms = np.std(inference_times)
    
    # Average losses
    avg_loss = total_loss / n_samples
    avg_ce = total_ce / n_samples
    avg_kd = total_kd / n_samples if total_kd > 0 else 0
    
    # Lambda weights
    lambda_weights = aggregator.get_weights()
    
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'avg_loss': avg_loss,
        'avg_ce_loss': avg_ce,
        'avg_kd_loss': avg_kd,
        'avg_inference_ms': avg_inference_ms,
        'std_inference_ms': std_inference_ms,
        'lambda_weights': lambda_weights,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'num_samples': n_samples
    }
    
    return results


def print_classification_report(report: Dict, class_names: List[str]):
    """Print formatted classification report."""
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-"*60)
    
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"{class_name:<20} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                  f"{metrics['f1-score']:>10.4f} {int(metrics['support']):>10}")
    
    print("-"*60)
    # Macro avg
    macro = report['macro avg']
    print(f"{'macro avg':<20} {macro['precision']:>10.4f} {macro['recall']:>10.4f} "
          f"{macro['f1-score']:>10.4f} {int(macro['support']):>10}")
    # Weighted avg
    weighted = report['weighted avg']
    print(f"{'weighted avg':<20} {weighted['precision']:>10.4f} {weighted['recall']:>10.4f} "
          f"{weighted['f1-score']:>10.4f} {int(weighted['support']):>10}")
    print("="*80)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str):
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    
    # Normalize for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def save_results(results: Dict, output_path: str):
    """Save evaluation results to JSON."""
    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results_serializable = convert(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"Results saved to: {output_path}")


# ============================================================================
# Experiment Runners
# ============================================================================

def run_single_experiment(
    experiment_name: str,
    checkpoint_path: str,
    config: EvalConfig,
    device: torch.device
) -> Dict:
    """Run evaluation for a single experiment configuration."""
    print(f"\n{'='*80}")
    print(f"EVALUATING: {experiment_name}")
    print(f"{'='*80}")
    
    # Load model
    student, aggregator, train_config, input_dim, num_classes = load_model_from_checkpoint(
        checkpoint_path, device
    )
    
    # Load test data
    test_loader, X_test, y_test, scaler, _, _ = load_test_data(
        config.data_path,
        config.teacher_emb_paths,
        test_size=config.test_size,
        val_size=config.val_size,
        random_state=config.random_state
    )
    
    # Evaluate
    results = evaluate_model(
        student, aggregator, test_loader, device, train_config, CLASS_NAMES
    )
    
    # Print results
    print(f"\nResults for {experiment_name}:")
    print(f"  Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  F1 Macro: {results['f1_macro']:.4f}")
    print(f"  F1 Weighted: {results['f1_weighted']:.4f}")
    print(f"  Precision Macro: {results['precision_macro']:.4f}")
    print(f"  Recall Macro: {results['recall_macro']:.4f}")
    print(f"  Avg Inference Time: {results['avg_inference_ms']:.2f} ± {results['std_inference_ms']:.2f} ms/sample")
    print(f"  Lambda Weights: {results['lambda_weights']}")
    
    print_classification_report(results['classification_report'], CLASS_NAMES)
    
    # Plot confusion matrix
    if config.save_heatmap:
        cm_path = os.path.join(config.output_dir, f"confusion_matrix_{experiment_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png")
        plot_confusion_matrix(
            np.array(results['confusion_matrix']),
            CLASS_NAMES,
            cm_path
        )
    
    # Save results
    if config.save_results:
        results_path = os.path.join(config.output_dir, f"results_{experiment_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.json")
        save_results(results, results_path)
    
    return results


def run_all_experiments(config: EvalConfig):
    """Run all three comparison experiments."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Experiment configurations
    experiments = [
        {
            'name': 'Single_Teacher_Baseline_A',
            'checkpoint': '/kaggle/working/training/exp1_single_teacher_A/best_model.pt',
            'description': 'Single teacher baseline (Teacher A only)'
        },
        {
            'name': 'Multi_Teacher_Fixed_Weights',
            'checkpoint': '/kaggle/working/training/exp2_multi_fixed/best_model.pt',
            'description': 'Multi-teacher with fixed weights (1/3 each)'
        },
        {
            'name': 'Multi_Teacher_Learned_Weights',
            'checkpoint': '/kaggle/working/training/exp3_multi_learnable/best_model.pt',
            'description': 'Multi-teacher with learned weights (MAIN CONTRIBUTION)'
        }
    ]
    
    all_results = []
    
    for exp in experiments:
        if not os.path.exists(exp['checkpoint']):
            print(f"\nWARNING: Checkpoint not found: {exp['checkpoint']}")
            print(f"Skipping {exp['name']}")
            continue
        
        try:
            results = run_single_experiment(
                exp['name'],
                exp['checkpoint'],
                config,
                device
            )
            results['experiment'] = exp['name']
            results['description'] = exp['description']
            all_results.append(results)
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"ERROR evaluating {exp['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary table
    print("\n" + "="*100)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("="*100)
    print(f"{'Experiment':<35} {'Acc':>8} {'F1-Macro':>10} {'F1-Wt':>10} {'Prec-M':>10} {'Rec-M':>10} {'Infer(ms)':>10} {'λ_A':>8} {'λ_B':>8} {'λ_C':>8}")
    print("-"*100)
    
    for r in all_results:
        lam = r['lambda_weights']
        print(f"{r['experiment']:<35} {r['accuracy']*100:>7.2f}% {r['f1_macro']:>10.4f} {r['f1_weighted']:>10.4f} "
              f"{r['precision_macro']:>10.4f} {r['recall_macro']:>10.4f} {r['avg_inference_ms']:>10.2f} "
              f"{lam.get('teacher_A', 0):>8.4f} {lam.get('teacher_B', 0):>8.4f} {lam.get('teacher_C', 0):>8.4f}")
    
    print("="*100)
    
    # Save combined results
    if config.save_results:
        combined_path = os.path.join(config.output_dir, 'all_experiments_comparison.json')
        save_results({'experiments': all_results}, combined_path)
        print(f"\nCombined results saved to: {combined_path}")
    
    return all_results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MT-LDI-MDS Evaluation')
    parser.add_argument('--checkpoint', type=str, default='/kaggle/working/training/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='/kaggle/working/data/veremi_processed.csv',
                        help='Path to processed VeReMi CSV')
    parser.add_argument('--output-dir', type=str, default='/kaggle/working/evaluation',
                        help='Output directory for results')
    parser.add_argument('--run-all', action='store_true',
                        help='Run all three comparison experiments')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--no-heatmap', action='store_true',
                        help='Disable confusion matrix heatmap')
    parser.add_argument('--test-size', type=float, default=0.15, help='Test split size')
    parser.add_argument('--val-size', type=float, default=0.15, help='Val split size')
    
    args = parser.parse_args()
    
    config = EvalConfig(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        save_heatmap=not args.no_heatmap,
        device=args.device
    )
    
    if args.run_all:
        run_all_experiments(config)
    else:
        # Single evaluation
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        run_single_experiment(
            "Single_Evaluation",
            config.checkpoint_path,
            config,
            device
        )


if __name__ == '__main__':
    main()