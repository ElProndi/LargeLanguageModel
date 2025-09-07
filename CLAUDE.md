# CLAUDE.md - LLM Training Pipeline

Transformer-based language model training pipeline using LLaMA-2 tokenizer and FineWeb dataset (277.4GB, 100B tokens from HuggingFace).

## Path Definitions
- `$DATA`: `./data` (within project root)
- `$PROJECT`: `.` (project root)

## Directory Structure

```
LargeLanguageModel/
├── src/
│   ├── dataset_preparation/
│   │   ├── tokenizer.py           # LLaMA-2 tokenizer wrapper
│   │   ├── fineweb_download.py    # Parallel FineWeb downloader
│   │   ├── fineweb_prep.py        # JSONL→tokenized sequences
│   │   ├── fineweb_recovery.py    # Recovery for missing parquet files
│   │   └── lima_tokenizer.py      # LIMA dataset prep
│   ├── training/
│   │   ├── pre_training.py        # Pre-training loop
│   │   ├── post_training.py       # Supervised fine-tuning
│   │   └── dataloader.py          # PyTorch DataLoader
│   ├── inference/
│   │   ├── Inference.py           # Text generation interface
│   │   ├── benchmark.py           # LAMBADA evaluation
│   │   ├── hellaswag_benchmark.py # HellaSwag commonsense reasoning
│   │   └── perplexity_benchmark.py # Perplexity on held-out data
│   └── utils/
│       ├── model.py               # Transformer architecture
│       ├── logging_utils.py       # Dual logging system
│       ├── metrics.py             # Metrics tracking
│       ├── scheduler.py           # LR scheduling
│       ├── rope.py                # Rotary embeddings
│       └── activations.py         # Activation functions
├── tokenizers/llama2_tokenizer/  # 32000 vocab
├── checkpoints/                # Hierarchical: run_*/checkpoint_*.pt, sft_lima_*/
└── logs/                       # tensorboard/, raw_metrics/

data/                           # (within project root)
├── raw/fineweb/               # JSONL chunks from download
├── post-training/             # LIMA tokenized data
└── tokenized_datasets/        # fineweb_{test,full}_dataset/
```

## Pipeline Components

### 1. Tokenizer (src/dataset_preparation/tokenizer.py)
**LLaMA-2-7b-hf**: 32,000 vocab, special tokens: `<s>`(1), `</s>`(2), `<unk>`(0)  
**Features**: Unicode support, batch encoding, Rust backend

### 2. FineWeb Pipeline

#### 2a. Download (src/dataset_preparation/fineweb_download.py)
**Architecture**: N parallel workers + 1 writer, HF native sharding, zero inter-worker overhead  
**Scale**: Test=2 chunks/200k docs | Full=100B tokens/148M docs/277GB  
**Defaults**: 2 workers, 100k docs/chunk → `data/raw/fineweb/`

#### 2b. Recovery (src/dataset_preparation/fineweb_recovery.py)
**Purpose**: Download missing parquet files directly from HuggingFace  
**Features**: Preserves existing 274GB downloads, checks for missing files  
**Process**: Verify missing → Download directly → Validate completeness

#### 2c. Processing (src/dataset_preparation/fineweb_prep.py)  
**Features**: Continuous packing (no gaps), BOS/EOS preservation, 1024-token windows  
**Batching**: 700k docs/file (7 chunks) → `data/tokenized_datasets/`

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

### 8. Benchmark Suite

#### LAMBADA (src/inference/benchmark.py)
**Task**: Context-dependent word prediction, strict accuracy  
**Configs**: Quick=50 | Standard=100 | Comprehensive=200 samples  
**Output**: `benchmark_results/` with JSON+text summaries

#### HellaSwag (src/inference/hellaswag_benchmark.py)
**Task**: Commonsense reasoning - predicting plausible scenario continuations  
**Dataset**: Downloaded from HuggingFace, cached in `benchmark_cache/`  
**Metrics**: Accuracy on 4-way multiple choice (human ~95%, random 25%)  
**Features**: Sequence scoring via log-probability, batch evaluation

#### Perplexity (src/inference/perplexity_benchmark.py)
**Task**: Evaluate model quality on held-out FineWeb data (`tokens_0.npy`)  
**Metrics**: Perplexity (lower is better), median/std statistics  
**Features**: Sliding window for long sequences, memory-mapped data loading  
**Interpretation**: <20 excellent, 20-50 good, 50-100 moderate, >100 poor

## Usage Commands

### Quick Reference Table

| Task | Test Mode | Full Mode | Custom Options |
|------|-----------|-----------|----------------|
| **Tokenizer** | `python3 -m src.dataset_preparation.tokenizer` | - | `--encode "text"` `--decode "1,2,3"` |
| **Download** | `python3 -m src.dataset_preparation.fineweb_download --test` | `python3 -m src.dataset_preparation.fineweb_download` | `--workers N --chunk-size M` |
| **Recovery** | `python3 -m src.dataset_preparation.fineweb_recovery` | - | Automatic detection of missing files |
| **Process** | `python3 -m src.dataset_preparation.fineweb_prep --test` | `python3 -m src.dataset_preparation.fineweb_prep` | `--window W --batch-size B` |
| **Pre-Train** | `python3 -m src.training.pre_training --test` | `python3 -m src.training.pre_training` | `--resume checkpoint.pt --name exp` |
| **Inference** | `python3 -m src.inference.Inference` → Select mode | - | Interactive prompts |
| **LAMBADA** | Via `python3 -m src.inference.Inference` option 3 | - | quick/standard/comprehensive |
| **HellaSwag** | `python3 -m src.inference.hellaswag_benchmark --checkpoint path` | - | `--max-samples N --split val` |
| **Perplexity** | `python3 -m src.inference.perplexity_benchmark --checkpoint path` | - | `--max-sequences N --batch-size B` |
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
- Model: hidden_size, num_layers, num_heads, vocab_size=32000
- Training: batch_size, learning_rate, warmup_steps, grad_accumulation
- Data: window_size=1024, val_split=0.1

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