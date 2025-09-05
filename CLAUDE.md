# CLAUDE.md - LLM Training Pipeline

Transformer-based language model training pipeline using CodeLlama tokenizer and FineWeb dataset (277.4GB, 100B tokens from HuggingFace).

## Path Definitions
- `$DATA`: `/home/andrea/Desktop/data`
- `$PROJECT`: `/home/andrea/Desktop/LargeLanguageModel`

## Directory Structure

```
LargeLanguageModel/
├── src/
│   ├── dataset_preparation/
│   │   ├── tokenizer.py           # CodeLlama tokenizer wrapper
│   │   ├── fineweb_download.py    # Parallel FineWeb downloader
│   │   ├── fineweb_prep.py        # JSONL→tokenized sequences
│   │   └── lima_tokenizer.py      # LIMA dataset prep
│   ├── training/
│   │   ├── pre_training.py        # Pre-training loop
│   │   ├── post_training.py       # Supervised fine-tuning
│   │   └── dataloader.py          # PyTorch DataLoader
│   ├── inference/
│   │   ├── Inference.py           # Text generation interface
│   │   └── benchmark.py           # LAMBADA evaluation
│   └── utils/
│       ├── model.py               # Transformer architecture
│       ├── logging_utils.py       # Dual logging system
│       ├── metrics.py             # Metrics tracking
│       ├── scheduler.py           # LR scheduling
│       ├── rope.py                # Rotary embeddings
│       └── activations.py         # Activation functions
├── tokenizers/codellama_tokenizer/  # 32016 vocab
├── checkpoints/                # Hierarchical: run_*/checkpoint_*.pt, sft_lima_*/
└── logs/                       # tensorboard/, raw_metrics/

$DATA/
├── raw/fineweb/               # JSONL chunks from download
├── post-training/             # LIMA tokenized data
└── tokenized_datasets/        # fineweb_{test,full}_dataset/
```

## Pipeline Components

### 1. Tokenizer (src/dataset_preparation/tokenizer.py)
**CodeLlama-7b-hf**: 32,016 vocab, special tokens: `<s>`(1), `</s>`(2), `<unk>`(0)  
**Features**: Unicode support, batch encoding, Rust backend

### 2. FineWeb Pipeline

#### 2a. Download (src/dataset_preparation/fineweb_download.py)
**Architecture**: N parallel workers + 1 writer, HF native sharding, zero inter-worker overhead  
**Scale**: Test=2 chunks/200k docs | Full=100B tokens/148M docs/277GB  
**Defaults**: 8 workers, 100k docs/chunk → `$DATA/raw/fineweb/`

#### 2b. Processing (src/dataset_preparation/fineweb_prep.py)  
**Features**: Continuous packing (no gaps), BOS/EOS preservation, 2048-token windows  
**Batching**: 700k docs/file (7 chunks) → `$DATA/tokenized_datasets/`

### 3. DataLoader (src/training/dataloader.py)
**Memory**: 32-chunk division for efficient memory use, train on GPU (zero-copy), val on CPU (pinned)  
**Split**: 90/10 train/val (seed=42), auto num_workers optimization

### 4. Model Architecture (src/utils/model.py)

| Component | Implementation | Details |
|-----------|---------------|---------|
| Attention | MultiHeadAttention | Standard multi-head attention, Flash Attention |
| Position | RoPE | Parameter-free rotary embeddings |
| Activation | SwiGLU | 3-matrix FFN with Swish gating |
| Norm | RMSNorm/LayerNorm | Pre-norm architecture |
| Optimization | Tied embeddings, bf16 | TorchInductor support |

**Generation**: Temperature/top-k/top-p sampling, batch support, EOS handling

### 5. Pre-Training (src/training/pre_training.py)
**Features**: Gradient accumulation, cosine+warmup scheduler, dual logging (TB+JSON)  
**Memory**: GPU-resident train data, async val transfers, windowed smoothing  
**Checkpointing**: Auto-resume, best model tracking, 10% epoch validation

### 6. Utils Components

| Module | Purpose | Key Features |
|--------|---------|--------------|
| rope.py | Rotary embeddings | Configurable theta |
| activations.py | Modern activations | SwiGLU/GeGLU/ReGLU variants |
| logging_utils.py | Dual logging | TensorBoard + JSON simultaneous |
| metrics.py | Metrics tracking | Smoothing, throughput, early stopping |
| scheduler.py | LR scheduling | Cosine annealing with warmup |

### 7. Inference (src/inference/Inference.py)
**Modes**: Single model | Multi-model comparison  
**Features**: Hierarchical checkpoint selection, Unicode support, ANSI colors  
**Parameters**: temp=0.8, top_k=50, top_p=0.95, max_length=200

### 8. Benchmark (src/inference/benchmark.py)
**LAMBADA**: Context-dependent word prediction, strict accuracy  
**Configs**: Quick=50 | Standard=100 | Comprehensive=200 samples  
**Output**: `benchmark_results/` with JSON+text summaries

### 9. Post-Training

#### LIMA Preparation (src/dataset_preparation/lima_tokenizer.py)
**Dataset**: 1,030 curated examples (97% single-turn), 542 kept (<2048 tokens)  
**Format**: `<s>User: {q}\nAssistant: {a}</s>` → `$DATA/post-training/`

#### Fine-Tuning (src/training/post_training.py)
**Training**: LR=5e-5, batch=8, ~60 steps/epoch, GPU-resident  
**Output**: `checkpoints/sft_lima_*/` with best/latest/epoch checkpoints

## Usage Commands

### Quick Reference Table

| Task | Test Mode | Full Mode | Custom Options |
|------|-----------|-----------|----------------|
| **Tokenizer** | `python3 -m src.dataset_preparation.tokenizer` | - | `--encode "text"` `--decode "1,2,3"` |
| **Download** | `python3 -m src.dataset_preparation.fineweb_download --test` | `python3 -m src.dataset_preparation.fineweb_download` | `--workers N --chunk-size M` |
| **Process** | `python3 -m src.dataset_preparation.fineweb_prep --test` | `python3 -m src.dataset_preparation.fineweb_prep` | `--window W --batch-size B` |
| **Pre-Train** | `python3 -m src.training.pre_training --test` | `python3 -m src.training.pre_training` | `--resume checkpoint.pt --name exp` |
| **Inference** | `python3 -m src.inference.Inference` → Select mode | - | Interactive prompts |
| **Benchmark** | Via `python3 -m src.inference.Inference` option 3 | - | quick/standard/comprehensive |
| **LIMA Prep** | `python3 -m src.dataset_preparation.lima_tokenizer` | - | `--max-length L --batch-size B` |
| **SFT** | `python3 -m src.training.post_training --checkpoint path` | - | `--epochs E --lr R --batch-size B` |

### Model Creation Example
```python
from src.utils.model import create_model
model = create_model("config.json")  # ~25M params default

# Training
outputs = model(input_ids, labels=input_ids)

# Generation  
generated = model.generate(prompt_ids, max_length=100, temperature=0.8)
```

### DataLoader Example
```python
from src.training.dataloader import create_simple_train_val_dataloaders

train_loader, val_loader, info = create_simple_train_val_dataloaders(
    batch_size=64, val_split=0.1, seed=42
)
# Train on GPU (zero-copy), Val on CPU (pinned memory)
```

### Monitoring
```bash
tensorboard --logdir logs/tensorboard
ls benchmark_results/*.txt
```

## Configuration

`config.json`: Centralized configuration for all components

### Key Parameters
- Model: hidden_size, num_layers, num_heads, vocab_size=32016
- Training: batch_size, learning_rate, warmup_steps, grad_accumulation
- Data: window_size=2048, val_split=0.1

## Development Standards

**Philosophy**: Clarity > cleverness | Modular design | Efficiency at scale | Fail fast  
**Implementation**: Descriptive errors | Progress feedback | Memory-efficient | Clean separation

## Important Instructions

- Do what has been asked; nothing more, nothing less
- NEVER create files unless absolutely necessary
- ALWAYS prefer editing existing files
- NEVER proactively create documentation unless requested

---
*Updated as components are implemented*