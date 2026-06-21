"""
MT-LDI-MDS: Data Splitting Module
==================================
This module handles loading the VeReMi extension dataset, splitting it into three 
specialized subsets for multi-teacher training, applying supervised contrastive 
learning (SupConLoss) for embedding clustering, outlier removal via Z-score,
and finding nearest neighbor pairs.

Dataset: VeReMi Extension (BSM logs for IoV misbehavior detection)
Attack Labels:
  0 = benign
  1 = DoS
  2 = Sybil
  3 = fixed_position
  4 = random_position
  5 = eventual_stop
  6 = fixed_speed
  7 = random_speed
  8 = data_replay

Teacher Subsets:
  D_A (Teacher A): labels [1, 2] + benign [0] → DoS and Sybil attacks
  D_B (Teacher B): labels [3, 4, 5] + benign [0] → Position spoofing attacks
  D_C (Teacher C): labels [6, 7, 8] + benign [0] → Speed and replay attacks

Output: /kaggle/working/data/split_A.csv, split_B.csv, split_C.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import cdist
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SupConLoss Implementation (Supervised Contrastive Loss)
# ============================================================================

class SupConLoss:
    """
    Supervised Contrastive Loss from scratch in PyTorch.
    
    Formula:
    L_sup = sum_i [ -1/|P(i)| * sum_{p in P(i)} log( exp(z_i . z_p / tau) / sum_{a in A(i)} exp(z_i . z_a / tau) ) ]
    
    Where:
    - z_i: normalized embedding of anchor sample i
    - P(i): set of positive samples (same class as i)
    - A(i): set of all samples except i (anchor)
    - tau: temperature parameter (default 0.07)
    
    This implementation uses NumPy for compatibility with the data preprocessing pipeline.
    """
    
    def __init__(self, temperature: float = 0.07):
        self.temperature = temperature
    
    def compute(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute SupCon loss for a batch of embeddings.
        
        Args:
            embeddings: (N, D) array of embeddings
            labels: (N,) array of class labels
            
        Returns:
            Scalar loss value
        """
        # Normalize embeddings to unit sphere
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        embeddings = embeddings / norms
        
        # Compute similarity matrix (cosine similarity since normalized)
        sim_matrix = np.dot(embeddings, embeddings.T) / self.temperature  # (N, N)
        
        N = embeddings.shape[0]
        loss_sum = 0.0
        valid_anchors = 0
        
        for i in range(N):
            # Find positive samples (same class, excluding self)
            pos_mask = (labels == labels[i]) & (np.arange(N) != i)
            pos_indices = np.where(pos_mask)[0]
            
            if len(pos_indices) == 0:
                continue  # No positive pairs for this anchor
            
            # All other samples as negatives (including other classes and self excluded)
            # A(i) = all samples except i
            anchor_sim = sim_matrix[i]  # (N,)
            
            # Compute denominator: sum over all a in A(i) of exp(sim)
            # Exclude self (i) from denominator
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            denom = np.sum(np.exp(anchor_sim[mask]))
            
            # Compute loss for each positive
            anchor_loss = 0.0
            for p in pos_indices:
                numer = np.exp(anchor_sim[p])
                anchor_loss += -np.log(numer / (denom + 1e-8))
            
            # Average over positive pairs
            anchor_loss /= len(pos_indices)
            loss_sum += anchor_loss
            valid_anchors += 1
        
        if valid_anchors == 0:
            return 0.0
        
        return loss_sum / valid_anchors
    
    def apply_scl_clustering(self, embeddings: np.ndarray, labels: np.ndarray, 
                             n_iterations: int = 10, lr: float = 0.01) -> np.ndarray:
        """
        Apply SCL to refine embeddings by gradient descent on SupCon loss.
        This clusters attack embeddings tighter within each class.
        
        Args:
            embeddings: (N, D) initial embeddings
            labels: (N,) class labels
            n_iterations: number of gradient steps
            lr: learning rate
            
        Returns:
            Refined embeddings (N, D)
        """
        # Convert to float32 for numerical stability
        emb = embeddings.astype(np.float32).copy()
        
        for iteration in range(n_iterations):
            # Normalize
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1e-8
            emb_norm = emb / norms
            
            # Compute gradients
            grad = np.zeros_like(emb)
            N = emb.shape[0]
            
            for i in range(N):
                pos_mask = (labels == labels[i]) & (np.arange(N) != i)
                pos_indices = np.where(pos_mask)[0]
                
                if len(pos_indices) == 0:
                    continue
                
                sim_matrix = np.dot(emb_norm, emb_norm.T) / self.temperature
                anchor_sim = sim_matrix[i]
                
                mask = np.ones(N, dtype=bool)
                mask[i] = False
                denom = np.sum(np.exp(anchor_sim[mask]))
                
                for p in pos_indices:
                    numer = np.exp(anchor_sim[p])
                    prob = numer / (denom + 1e-8)
                    # Gradient w.r.t anchor embedding
                    grad[i] += (1 - prob) * (emb_norm[p] - prob * emb_norm[i]) / self.temperature
                
                grad[i] /= len(pos_indices)
            
            # Update embeddings
            emb -= lr * grad
        
        return emb


# ============================================================================
# Outlier Removal via Z-score
# ============================================================================

def remove_outliers_zscore(df: pd.DataFrame, feature_cols: List[str], 
                           threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers using Z-score method.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        threshold: Z-score threshold (default 3.0)
        
    Returns:
        DataFrame with outliers removed
    """
    z_scores = np.abs(zscore(df[feature_cols].values, axis=0))
    # Keep rows where ALL features have z-score < threshold
    keep_mask = np.all(z_scores < threshold, axis=1)
    outliers_removed = np.sum(~keep_mask)
    if outliers_removed > 0:
        print(f"  Removed {outliers_removed} outliers (Z-score > {threshold})")
    return df[keep_mask].reset_index(drop=True)


# ============================================================================
# Nearest Neighbor Pair Finding
# ============================================================================

def find_topk_pairs(df: pd.DataFrame, feature_cols: List[str], 
                    label_col: str, k: int = 5) -> Dict[int, List[Tuple[int, int, float]]]:
    """
    Find top-k closest sample pairs per class using Euclidean distance.
    
    Args:
        df: DataFrame with features and labels
        feature_cols: List of feature column names
        label_col: Label column name
        k: Number of closest pairs per class
        
    Returns:
        Dictionary mapping class label to list of (idx1, idx2, distance) tuples
    """
    features = df[feature_cols].values
    labels = df[label_col].values
    unique_labels = np.unique(labels)
    
    pairs_dict = {}
    
    for label in unique_labels:
        class_indices = np.where(labels == label)[0]
        if len(class_indices) < 2:
            pairs_dict[int(label)] = []
            continue
        
        class_features = features[class_indices]
        
        # Compute pairwise distances
        dist_matrix = cdist(class_features, class_features, metric='euclidean')
        
        # Get upper triangle (excluding diagonal)
        pairs = []
        for i in range(len(class_indices)):
            for j in range(i + 1, len(class_indices)):
                pairs.append((class_indices[i], class_indices[j], dist_matrix[i, j]))
        
        # Sort by distance and take top-k
        pairs.sort(key=lambda x: x[2])
        pairs_dict[int(label)] = pairs[:k]
        
        if pairs:
            avg_dist = np.mean([p[2] for p in pairs[:k]])
            print(f"  Class {label}: top-{k} avg distance = {avg_dist:.4f}")
    
    return pairs_dict


# ============================================================================
# Dataset Loading and Splitting
# ============================================================================

def load_veremi_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load VeReMi extension dataset from CSV.
    
    Args:
        csv_path: Path to veremi.csv
        
    Returns:
        DataFrame with VeReMi data
        
    Raises:
        FileNotFoundError: If dataset not found with clear instructions
    """
    if not os.path.exists(csv_path):
        error_msg = (
            f"\n{'='*80}\n"
            f"ERROR: VeReMi dataset not found at {csv_path}\n"
            f"{'='*80}\n"
            f"Please upload the VeReMi Extension Dataset to Kaggle:\n\n"
            f"Option 1 - Kaggle Dataset Search:\n"
            f"  1. In Kaggle notebook, click 'Add Data' → 'Search datasets'\n"
            f"  2. Search for 'VeReMi extension dataset' or 'VeReMi-Dataset'\n"
            f"  3. Select and add to your notebook\n"
            f"  4. The path will be /kaggle/input/veremi-dataset/veremi.csv\n\n"
            f"Option 2 - Manual Upload:\n"
            f"  1. Download from: https://github.com/josephkamel/VeReMi-Dataset\n"
            f"  2. Upload veremi.csv to Kaggle via 'Add Data' → 'New Dataset'\n"
            f"  3. Add to notebook\n\n"
            f"Expected CSV columns: sender, sendTime, sendPos, speed, acce, yaw, label, ...\n"
            f"{'='*80}"
        )
        raise FileNotFoundError(error_msg)
    
    print(f"Loading VeReMi dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare feature columns for the dataset.
    Excludes non-numeric and label columns.
    
    Args:
        df: Raw VeReMi DataFrame
        
    Returns:
        (processed_df, feature_column_names)
    """
    # Identify numeric feature columns (exclude label and metadata)
    exclude_cols = ['label', 'sender', 'sendTime', 'attackerType', 'attackID']
    feature_cols = [c for c in df.columns if c not in exclude_cols and 
                    pd.api.types.is_numeric_dtype(df[c])]
    
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    
    # Handle any NaN values
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    
    return df, feature_cols


def split_dataset(df: pd.DataFrame, feature_cols: List[str], 
                  label_col: str = 'label') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into three teacher subsets based on attack labels.
    
    D_A (Teacher A): labels [1, 2] + benign [0] → DoS and Sybil
    D_B (Teacher B): labels [3, 4, 5] + benign [0] → Position spoofing
    D_C (Teacher C): labels [6, 7, 8] + benign [0] → Speed and replay
    
    Args:
        df: Full VeReMi DataFrame
        feature_cols: Feature column names
        label_col: Label column name
        
    Returns:
        (df_A, df_B, df_C) - Three DataFrames for each teacher
    """
    # Define attack label groups
    teacher_A_labels = [0, 1, 2]      # Benign + DoS + Sybil
    teacher_B_labels = [0, 3, 4, 5]   # Benign + Position spoofing
    teacher_C_labels = [0, 6, 7, 8]   # Benign + Speed/Replay
    
    df_A = df[df[label_col].isin(teacher_A_labels)].copy().reset_index(drop=True)
    df_B = df[df[label_col].isin(teacher_B_labels)].copy().reset_index(drop=True)
    df_C = df[df[label_col].isin(teacher_C_labels)].copy().reset_index(drop=True)
    
    print(f"\nSplit sizes:")
    print(f"  Teacher A (DoS, Sybil): {len(df_A)} samples")
    print(f"  Teacher B (Position):   {len(df_B)} samples")
    print(f"  Teacher C (Speed/Replay): {len(df_C)} samples")
    
    return df_A, df_B, df_C


def print_class_distribution(df: pd.DataFrame, name: str, label_col: str = 'label'):
    """Print class distribution for a dataset split."""
    dist = df[label_col].value_counts().sort_index()
    print(f"\n{name} Class Distribution:")
    for label, count in dist.items():
        attack_names = {
            0: 'benign', 1: 'DoS', 2: 'Sybil', 3: 'fixed_position',
            4: 'random_position', 5: 'eventual_stop', 6: 'fixed_speed',
            7: 'random_speed', 8: 'data_replay'
        }
        name_str = attack_names.get(label, f'class_{label}')
        print(f"  {label} ({name_str}): {count}")


def process_split(df: pd.DataFrame, feature_cols: List[str], label_col: str,
                  split_name: str, apply_scl: bool = True, 
                  scl_iterations: int = 10, zscore_threshold: float = 3.0,
                  topk: int = 5) -> pd.DataFrame:
    """
    Process a single dataset split: outlier removal, SCL, nearest neighbors.
    
    Args:
        df: DataFrame for this split
        feature_cols: Feature column names
        label_col: Label column name
        split_name: Name for logging
        apply_scl: Whether to apply SupCon loss clustering
        scl_iterations: SCL gradient steps
        zscore_threshold: Z-score outlier threshold
        topk: Number of nearest neighbor pairs
        
    Returns:
        Processed DataFrame
    """
    print(f"\n{'='*60}")
    print(f"Processing {split_name}...")
    print(f"{'='*60}")
    
    # 1. Remove outliers using Z-score
    print(f"\n1. Removing outliers (Z-score > {zscore_threshold})...")
    df_clean = remove_outliers_zscore(df, feature_cols, zscore_threshold)
    
    # 2. Apply SupCon Loss for embedding clustering (if enabled)
    if apply_scl and len(df_clean) > 1:
        print(f"\n2. Applying SupCon Loss clustering ({scl_iterations} iterations)...")
        features = df_clean[feature_cols].values.astype(np.float32)
        labels = df_clean[label_col].values
        
        supcon = SupConLoss(temperature=0.07)
        initial_loss = supcon.compute(features, labels)
        print(f"   Initial SupCon loss: {initial_loss:.4f}")
        
        # Apply SCL refinement
        refined_features = supcon.apply_scl_clustering(
            features, labels, n_iterations=scl_iterations, lr=0.01
        )
        
        final_loss = supcon.compute(refined_features, labels)
        print(f"   Final SupCon loss: {final_loss:.4f}")
        
        # Update features in DataFrame (for downstream use)
        for i, col in enumerate(feature_cols):
            df_clean[col] = refined_features[:, i]
    else:
        print("\n2. Skipping SupCon Loss (insufficient samples or disabled)")
    
    # 3. Find top-k nearest neighbor pairs per class
    print(f"\n3. Finding top-{topk} nearest neighbor pairs per class...")
    pairs = find_topk_pairs(df_clean, feature_cols, label_col, topk)
    
    # Store pair info as metadata (could be used for data augmentation)
    df_clean.attrs['neighbor_pairs'] = pairs
    
    return df_clean


def save_splits(df_A: pd.DataFrame, df_B: pd.DataFrame, df_C: pd.DataFrame,
                output_dir: str = '/kaggle/working/data'):
    """Save the three dataset splits to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    path_A = os.path.join(output_dir, 'split_A.csv')
    path_B = os.path.join(output_dir, 'split_B.csv')
    path_C = os.path.join(output_dir, 'split_C.csv')
    
    df_A.to_csv(path_A, index=False)
    df_B.to_csv(path_B, index=False)
    df_C.to_csv(path_C, index=False)
    
    print(f"\nSaved splits to:")
    print(f"  {path_A} ({len(df_A)} samples)")
    print(f"  {path_B} ({len(df_B)} samples)")
    print(f"  {path_C} ({len(df_C)} samples)")


def main():
    parser = argparse.ArgumentParser(description='Split VeReMi dataset for MT-LDI-MDS')
    parser.add_argument('--input', type=str, default='/kaggle/input/veremi-dataset/veremi.csv',
                        help='Path to VeReMi CSV file')
    parser.add_argument('--output-dir', type=str, default='/kaggle/working/data',
                        help='Output directory for splits')
    parser.add_argument('--no-scl', action='store_true',
                        help='Disable SupCon loss clustering')
    parser.add_argument('--scl-iterations', type=int, default=10,
                        help='Number of SCL gradient iterations')
    parser.add_argument('--zscore-threshold', type=float, default=3.0,
                        help='Z-score outlier threshold')
    parser.add_argument('--topk', type=int, default=5,
                        help='Top-k nearest neighbors per class')
    args = parser.parse_args()
    
    print("="*80)
    print("MT-LDI-MDS: VeReMi Dataset Splitting")
    print("="*80)
    
    # Load dataset
    try:
        df = load_veremi_dataset(args.input)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)
    
    # Prepare features
    df, feature_cols = prepare_features(df)
    
    # Split into three teacher subsets
    df_A, df_B, df_C = split_dataset(df, feature_cols)
    
    # Process each split
    df_A = process_split(df_A, feature_cols, 'label', 'Teacher A (DoS, Sybil)',
                         apply_scl=not args.no_scl, scl_iterations=args.scl_iterations,
                         zscore_threshold=args.zscore_threshold, topk=args.topk)
    df_B = process_split(df_B, feature_cols, 'label', 'Teacher B (Position)',
                         apply_scl=not args.no_scl, scl_iterations=args.scl_iterations,
                         zscore_threshold=args.zscore_threshold, topk=args.topk)
    df_C = process_split(df_C, feature_cols, 'label', 'Teacher C (Speed/Replay)',
                         apply_scl=not args.no_scl, scl_iterations=args.scl_iterations,
                         zscore_threshold=args.zscore_threshold, topk=args.topk)
    
    # Print class distributions
    print_class_distribution(df_A, 'Teacher A')
    print_class_distribution(df_B, 'Teacher B')
    print_class_distribution(df_C, 'Teacher C')
    
    # Save splits
    save_splits(df_A, df_B, df_C, args.output_dir)
    
    print("\n" + "="*80)
    print("Dataset splitting complete!")
    print("="*80)


if __name__ == '__main__':
    main()