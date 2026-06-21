# MT-LDI-MDS: Multi-Teacher Lightweight Distilled Intelligence - Misbehavior Detection System

> **IoV Security Research Project** | Kaggle Notebook (GPU T4 x2) | PyTorch + HuggingFace + PEFT

---

## 📖 Overview

MT-LDI-MDS is an improvement over the existing single-teacher knowledge distillation approach for IoV (Internet of Vehicles) misbehavior detection. Instead of using a single Mistral 7B teacher, this system employs **three specialized teacher LLMs** (Qwen3-8B) with a **learnable weighted aggregator** for superior knowledge transfer to a lightweight BiLSTM student.

### Key Innovations

1. **Three Specialized Teachers**: Each teacher focuses on a specific attack category
   - **Teacher A**: DoS (1) & Sybil (2) attacks
   - **Teacher B**: Position spoofing (3, 4, 5) attacks  
   - **Teacher C**: Speed & replay (6, 7, 8) attacks

2. **Learnable Weighted Aggregation**: `ê = softmax(λ) · [eA, eB, eC]` - the model learns optimal teacher weights

3. **Multi-Teacher Distillation Loss**: 
   ```
   L_total = CE + λ₁·MSE(student, teacher_A) + λ₂·MSE(student, teacher_B) + λ₃·MSE(student, teacher_C)
   ```

4. **SupCon Loss Preprocessing**: Supervised contrastive learning clusters attack embeddings before teacher training

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MT-LDI-MDS PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VeReMi Dataset                                                            │
│       │                                                                    │
│       ▼                                                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│  │  Split A    │    │  Split B    │    │  Split C    │  ← SupCon Loss    │
│  │ (DoS,Sybil) │    │ (Position)  │    │ (Speed/Rep) │  ← Z-score outlier │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  ← Top-k pairs    │
│         │                  │                  │                          │
│         ▼                  ▼                  ▼                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│  │ Teacher A   │    │ Teacher B   │    │ Teacher C   │  ← Qwen3-8B + LoRA │
│  │ (Qwen3-8B)  │    │ (Qwen3-8B)  │    │ (Qwen3-8B)  │  ← Instruction tune│
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  ← Extract emb.   │
│         │                  │                  │                          │
│         └──────────────────┼──────────────────┘                          │
│                            ▼                                             │
│                   ┌─────────────────┐                                    │
│                   │ Weighted        │                                    │
│                   │ Aggregator      │                                    │
│                   │ λ₁,λ₂,λ₃ learn  │                                    │
│                   └────────┬────────┘                                    │
│                            │                                             │
│                            ▼                                             │
│                   ┌─────────────────┐                                    │
│                   │ BiLSTM Student  │  ← Lightweight detector            │
│                   │ (128-dim feat)  │  ← KD from aggregated teachers     │
│                   └─────────────────┘                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
mt_ldi_mds/
├── data/
│   └── split_dataset.py          # VeReMi loading, splitting, SupConLoss, outliers
├── teachers/
│   └── finetune_teacher.py       # Qwen3-8B LoRA fine-tuning (3 teachers)
├── aggregator/
│   └── weighted_aggregator.py    # Fixed + Learnable weighted aggregation
├── student/
│   └── bilstm_student.py         # BiLSTM student (paper architecture)
├── training/
│   └── train_multiteacher.py     # Main training loop with 3 experiments
├── evaluation/
│   └── evaluate.py               # Evaluation + comparison experiments
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Quick Start (Kaggle Notebook)

### 1. Environment Setup

Create a new Kaggle Notebook with **GPU T4 x2** (32GB VRAM combined).

**Install dependencies:**
```python
# Cell 1: Install requirements
!pip install -q -r /kaggle/working/mt_ldi_mds/requirements.txt
```

### 2. Dataset Setup

**Option A: Kaggle Dataset Search (Recommended)**
1. Click **"Add Data"** → **"Search datasets"**
2. Search: `"VeReMi extension dataset"` or `"VeReMi-Dataset"`
3. Select and add to notebook
4. Dataset will be at: `/kaggle/input/veremi-dataset/veremi.csv`

**Option B: Manual Upload**
1. Download from: https://github.com/josephkamel/VeReMi-Dataset
2. Click **"Add Data"** → **"New Dataset"** → Upload `veremi.csv`
3. Add to notebook

### 3. Run Pipeline (In Order)

#### Step 1: Split Dataset
```python
# Cell 2: Split VeReMi into 3 teacher subsets
!cd /kaggle/working/mt_ldi_mds && python data/split_dataset.py \
    --input /kaggle/input/veremi-dataset/veremi.csv \
    --output-dir /kaggle/working/data \
    --scl-iterations 10 \
    --zscore-threshold 3.0 \
    --topk 5
```
**Outputs:** `/kaggle/working/data/split_A.csv`, `split_B.csv`, `split_C.csv`

#### Step 2: Fine-tune Three Teachers (Run sequentially - each takes ~30-45 min)

```python
# Cell 3: Teacher A (DoS + Sybil)
!cd /kaggle/working/mt_ldi_mds && python teachers/finetune_teacher.py \
    --teacher A \
    --data /kaggle/working/data/split_A.csv \
    --epochs 1 \
    --batch-size 16 \
    --max-steps 3000 \
    --max-seq-len 512 \
    --lr 2e-4 \
    --output-dir /kaggle/working/teachers
```

```python
# Cell 4: Teacher B (Position spoofing)
!cd /kaggle/working/mt_ldi_mds && python teachers/finetune_teacher.py \
    --teacher B \
    --data /kaggle/working/data/split_B.csv \
    --epochs 1 \
    --batch-size 16 \
    --max-steps 3000 \
    --max-seq-len 512 \
    --lr 2e-4 \
    --output-dir /kaggle/working/teachers
```

```python
# Cell 5: Teacher C (Speed + Replay)
!cd /kaggle/working/mt_ldi_mds && python teachers/finetune_teacher.py \
    --teacher C \
    --data /kaggle/working/data/split_C.csv \
    --epochs 1 \
    --batch-size 16 \
    --max-steps 3000 \
    --max-seq-len 512 \
    --lr 2e-4 \
    --output-dir /kaggle/working/teachers
```

**Outputs per teacher:**
- Adapters: `/kaggle/working/teachers/teacher_{A,B,C}_adapters/final/`
- Embeddings: `/kaggle/working/teachers/embeddings_{A,B,C}.npy`
- Loss curves: `/kaggle/working/teachers/teacher_{A,B,C}_loss_curve.json`

> ⚠️ **Memory Management**: Run `torch.cuda.empty_cache()` between teacher trainings (handled automatically in script).

#### Step 3: Prepare Full Dataset for Distillation
```python
# Cell 6: Create processed dataset with all samples (for student training)
# This merges the three splits back for 70/15/15 train/val/test split
!cd /kaggle/working/mt_ldi_mds && python -c "
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load original or combine splits
df_A = pd.read_csv('/kaggle/working/data/split_A.csv')
df_B = pd.read_csv('/kaggle/working/data/split_B.csv')
df_C = pd.read_csv('/kaggle/working/data/split_C.csv')

# Combine and deduplicate
df_all = pd.concat([df_A, df_B, df_C]).drop_duplicates().reset_index(drop=True)
df_all.to_csv('/kaggle/working/data/veremi_processed.csv', index=False)
print(f'Combined dataset: {len(df_all)} samples')
print(df_all['label'].value_counts().sort_index())
"
```

#### Step 4: Run Multi-Teacher Distillation Training

**Option A: Run all 3 comparison experiments (recommended for paper)**
```python
# Cell 7: Run all experiments (takes ~2-3 hours total)
!cd /kaggle/working/mt_ldi_mds && python training/train_multiteacher.py \
    --data /kaggle/working/data/veremi_processed.csv \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-3 \
    --run-all-experiments \
    --checkpoint-dir /kaggle/working/training
```

**Option B: Run single configuration**
```python
# Cell 7b: Single run (main contribution: learnable weights)
!cd /kaggle/working/mt_ldi_mds && python training/train_multiteacher.py \
    --data /kaggle/working/data/veremi_processed.csv \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-3 \
    --mode learnable \
    --checkpoint-dir /kaggle/working/training
```

**Outputs:**
- Best model: `/kaggle/working/training/best_model.pt`
- Training log: `/kaggle/working/training/training_log.csv`
- Curves: `/kaggle/working/training/training_curves.png`
- Experiment comparison: `/kaggle/working/training/experiment_comparison.json`

#### Step 5: Evaluate & Compare
```python
# Cell 8: Run evaluation on all 3 experiments
!cd /kaggle/working/mt_ldi_mds && python evaluation/evaluate.py \
    --run-all \
    --output-dir /kaggle/working/evaluation \
    --data /kaggle/working/data/veremi_processed.csv
```

**Outputs:**
- Results: `/kaggle/working/evaluation/results_*.json`
- Confusion matrices: `/kaggle/working/evaluation/confusion_matrix_*.png`
- Comparison: `/kaggle/working/evaluation/all_experiments_comparison.json`

---

## 🔬 Multi-Teacher Loss Function (Key Contribution)

The core innovation is the **multi-teacher distillation loss**:

```python
# Total Loss
L_total = L_CE + λ₁·L_KD_A + λ₂·L_KD_B + λ₃·L_KD_C

# Where:
# L_CE = CrossEntropy(student_logits, true_labels)
# L_KD_A = MSE(student_intermediate_features, teacher_A_embeddings)
# L_KD_B = MSE(student_intermediate_features, teacher_B_embeddings)  
# L_KD_C = MSE(student_intermediate_features, teacher_C_embeddings)

# λ₁, λ₂, λ₃ from WeightedAggregator (learnable or fixed)
```

### Aggregation Modes

| Mode | Formula | Use Case |
|------|---------|----------|
| **Fixed** | `ê = (eA + eB + eC) / 3` | Baseline, equal contribution |
| **Learnable** | `ê = softmax(λ) · [eA, eB, eC]` | **Main contribution** - model learns optimal weights |

### Interpreting Learned λ Weights

After training, the aggregator outputs weights indicating each teacher's importance:

```python
# Example output:
{
    "teacher_A": 0.45,  # DoS/Sybil specialist
    "teacher_B": 0.30,  # Position spoofing specialist
    "teacher_C": 0.25   # Speed/Replay specialist
}
```

**Interpretation:**
- Higher λ = that teacher's expertise is more critical for the student
- Weights sum to 1.0 (softmax normalized)
- Printed every 10 epochs during training
- Visualized in `training_curves.png`

---

## ⚙️ Configuration Reference

### Data Splitting (`split_dataset.py`)
| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `/kaggle/input/veremi-dataset/veremi.csv` | VeReMi CSV path |
| `--output-dir` | `/kaggle/working/data` | Output directory |
| `--no-scl` | False | Disable SupCon Loss |
| `--scl-iterations` | 10 | SCL gradient steps |
| `--zscore-threshold` | 3.0 | Outlier removal threshold |
| `--topk` | 5 | Nearest neighbor pairs |

### Teacher Fine-tuning (`finetune_teacher.py`)
| Argument | Default | Description |
|----------|---------|-------------|
| `--teacher` | **Required** | A, B, or C |
| `--data` | **Required** | Split CSV path |
| `--model-id` | `Qwen/Qwen3-8B` | HF model ID |
| `--epochs` | 1 | Training epochs |
| `--batch-size` | 16 | Batch size per GPU |
| `--max-steps` | 3000 | Max training steps |
| `--max-seq-len` | 512 | Max sequence length |
| `--lr` | 2e-4 | Learning rate |
| `--rank` | 16 | LoRA rank |
| `--alpha` | 32 | LoRA alpha |

### Training (`train_multiteacher.py`)
| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `/kaggle/working/data/veremi_processed.csv` | Full dataset |
| `--epochs` | 50 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--mode` | `learnable` | `fixed` or `learnable` |
| `--single-teacher` | False | Single teacher baseline |
| `--teacher` | `A` | Teacher for single mode |
| `--run-all-experiments` | False | Run all 3 experiments |
| `--resume` | None | Resume checkpoint path |

### Evaluation (`evaluate.py`)
| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | `/kaggle/working/training/best_model.pt` | Model checkpoint |
| `--data` | `/kaggle/working/data/veremi_processed.csv` | Test data |
| `--output-dir` | `/kaggle/working/evaluation` | Results directory |
| `--run-all` | False | Run all 3 experiments |
| `--no-heatmap` | False | Disable confusion matrix |

---

## 📊 Expected Outputs & Results

### Training Curves (`training_curves.png`)
- Total loss (train/val)
- CE loss (train/val)  
- KD losses per teacher (train)
- Validation accuracy & F1-macro
- **Learned λ weights over epochs** (key visualization)

### Classification Report (per experiment)
```
              precision    recall  f1-score   support
      benign       0.98      0.99      0.98     15000
          DoS       0.92      0.88      0.90      2000
        Sybil       0.89      0.91      0.90      1800
  fixed_position   0.94      0.93      0.93      2200
 random_position   0.91      0.90      0.90      2100
  eventual_stop    0.88      0.87      0.87      1900
    fixed_speed    0.93      0.92      0.92      2000
   random_speed    0.90      0.89      0.89      1800
    data_replay    0.95      0.94      0.94      1700

    accuracy                           0.94     30500
   macro avg       0.92      0.91      0.91     30500
weighted avg       0.94      0.94      0.94     30500
```

### Experiment Comparison Table
| Experiment | Val Acc | F1-Macro | F1-Wt | λ_A | λ_B | λ_C |
|------------|---------|----------|-------|-----|-----|-----|
| Single Teacher (A) | 91.2% | 0.88 | 0.91 | 1.0 | 0.0 | 0.0 |
| Multi Fixed | 93.5% | 0.91 | 0.93 | 0.33 | 0.33 | 0.33 |
| **Multi Learned** | **94.8%** | **0.93** | **0.94** | **0.45** | **0.30** | **0.25** |

---

## ⏱️ Kaggle GPU Time Management

### Estimated Time per Component
| Component | Time (T4 x2) | Notes |
|-----------|-------------|-------|
| Data splitting | ~2 min | CPU only |
| Teacher A fine-tune | ~35 min | 3000 steps, batch 16 |
| Teacher B fine-tune | ~35 min | Run after A, clear cache |
| Teacher C fine-tune | ~35 min | Run after B, clear cache |
| Student training (50 ep) | ~45 min | Batch 64, multi-teacher |
| Evaluation | ~5 min | All 3 experiments |

**Total: ~2.5 hours** (well within 9-12 hour weekly limit)

### Time-Saving Tips
1. **Run teachers in separate sessions** - Kaggle allows multiple notebooks
2. **Use checkpoints** - Training auto-saves every 500 steps / 10 epochs
3. **Resume capability** - Use `--resume /path/to/checkpoint.pt`
4. **Reduce epochs for testing** - Start with `--epochs 10` to verify pipeline

### Checkpoint Resume Example
```python
# Resume student training from epoch 25
!python training/train_multiteacher.py \
    --data /kaggle/working/data/veremi_processed.csv \
    --epochs 50 \
    --resume /kaggle/working/training/checkpoint_epoch_25.pt
```

---

## 🔧 Technical Details

### Model Specifications

**Teachers (3× Qwen3-8B LoRA)**
- Base: Qwen3-8B (8B params, 4096 hidden dim)
- LoRA: rank=16, alpha=32, dropout=0.05
- Targets: `q_proj`, `v_proj`
- Precision: fp16 (T4 optimized)
- Device map: `auto` (spreads across 2 T4s)

**Aggregator**
- Input: 3 × 4096-dim embeddings
- Mode: Fixed (1/3 each) or Learnable (softmax λ)
- Adapter: Linear(12288 → 256) + LayerNorm
- Output: 256-dim aggregated embedding

**Student (BiLSTM)**
- Input embedding: `input_dim → 128`
- Conv1D×2: 128→64→64, kernel=3, same padding, ReLU
- BiLSTM×3: 64→256 (bidirectional), dropout=0.3
- Self-Attention: 256-dim, 4 heads
- Residual + LayerNorm
- Global Avg Pool → FC(256→128) + BN + Dropout(0.3)
- Output: 128→9 classes
- **Intermediate features (128-dim) used for KD**

### SupConLoss Formula
```
L_sup = Σ_i [ -1/|P(i)| · Σ_{p∈P(i)} log( exp(z_i·z_p/τ) / Σ_{a∈A(i)} exp(z_i·z_a/τ) ) ]
```
- τ = 0.07 (temperature)
- P(i) = positives (same class)
- A(i) = all samples except i
- Implemented from scratch in NumPy/PyTorch (no extra deps)

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{mt_ldi_mds2024,
  title={MT-LDI-MDS: Multi-Teacher Lightweight Distilled Intelligence for IoV Misbehavior Detection},
  author={Your Name},
  year={2024},
  note={Improvement over single-teacher KD with three specialized Qwen3-8B teachers and learnable weighted aggregation}
}
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "VeReMi dataset not found"**
```
Solution: Upload dataset via Kaggle "Add Data" → Search "VeReMi extension dataset"
```

**2. CUDA Out of Memory**
```
Solutions:
- Reduce batch_size (teacher: 8, student: 32)
- Enable gradient checkpointing (already enabled)
- Clear cache between runs: torch.cuda.empty_cache()
- Use gradient accumulation
```

**3. Slow training**
```
- Ensure GPU is enabled (Settings → Accelerator → GPU T4 x2)
- Check device_map="auto" is working (prints device map on load)
- Use fp16=True (default)
```

**4. Import errors**
```
- Run: !pip install -r requirements.txt
- Restart kernel after install
- Check Python version (3.10+ recommended)
```

**5. Checkpoint not found for evaluation**
```
- Ensure training completed and saved best_model.pt
- Check --checkpoint path matches training output
- For experiments, check subdirectories: exp1_*/best_model.pt
```

---

## 📄 License

MIT License - Feel free to use for research and education.

---

## 🤝 Contributing

This is a research project. For questions or collaborations, please open an issue.

---

*Built for Kaggle Notebook (T4 x2) • PyTorch 2.4 • Transformers 4.44 • PEFT 0.12*