# CLAUDE.md - LLM Training Pipeline

This file provides guidance to Claude Code when working with the LLM training pipeline implementation.

## Project Overview

A transformer-based language model training pipeline leveraging the pre-trained CodeLlama tokenizer for superior text and code representation. The pipeline supports two high-quality data sources:
- **Wikipedia**: Structured encyclopedic knowledge (50GB, ~15B tokens)
- **FineWeb**: Diverse, high-quality web content from HuggingFace (27.6GB, 10B tokens)

Both datasets use the same CodeLlama tokenizer and training infrastructure, allowing seamless switching via configuration.

## Directory Structure

```
LargeLanguageModel/
├── CLAUDE.md              # This file - project guidance
├── main.py                # Pipeline router and orchestrator
├── config.json            # Centralized configuration for entire pipeline
├── cleaner.py             # Wikipedia data extraction and cleaning
├── tokenizer.py           # CodeLlama tokenizer wrapper from HuggingFace
├── dataset_prep.py        # Wikipedia dataset tokenization with sliding windows
├── fineweb_prep.py        # FineWeb dataset streaming and tokenization
├── dataloader.py          # PyTorch DataLoader for efficient training (supports both datasets)
├── model.py               # Transformer language model architecture
├── train.py               # Main training loop with mixed precision
├── Inference.py           # Interactive text generation from checkpoints
├── lima_tokenizer.py      # LIMA dataset preparation for instruction tuning
├── post_training.py       # Supervised fine-tuning (SFT) script
├── utils/                 # Training utilities
│   ├── __init__.py        # Package initialization
│   ├── logging_utils.py   # Dual logging (TensorBoard + JSON)
│   ├── metrics.py         # Metrics tracking and smoothing
│   ├── scheduler.py       # Learning rate scheduling
│   ├── rope.py            # Rotary Position Embeddings (RoPE)
│   └── activations.py     # Modern activation functions (SwiGLU, GeGLU)
├── tokenizers/            # Saved tokenizer models
│   └── codellama_tokenizer/  # CodeLlama tokenizer (32016 vocab)
├── checkpoints/           # Model checkpoints (hierarchical structure)
│   ├── run_2024_12_15_14_30/  # Training run directory (timestamp-based)
│   │   ├── checkpoint_latest.pt    # Most recent checkpoint
│   │   ├── checkpoint_best.pt      # Best validation loss
│   │   └── checkpoint_step_*.pt    # Step-based checkpoints
│   ├── sft_lima_*/        # SFT checkpoints from instruction tuning
│   │   ├── checkpoint_best.pt      # Best validation loss
│   │   ├── checkpoint_latest.pt    # Most recent checkpoint
│   │   ├── checkpoint_epoch_*.pt   # Per-epoch checkpoints
│   │   └── sft_config.json        # Fine-tuning configuration
│   └── [legacy flat structure]     # Old format: checkpoints directly in base dir
└── logs/                  # Training logs
    ├── tensorboard/       # TensorBoard event files
    └── raw_metrics/       # Raw JSON metrics for analysis
```

## Data Organization

```
Desktop/
├── data/                    # Centralized training data
│   ├── raw/                # Raw Wikipedia JSONL files
│   │   └── enwiki_namespace_0/
│   │       └── *.jsonl     # 38 Wikipedia dump files (50GB+)
│   ├── cleaned_articles/   # Processed text (Unicode preserved)
│   │   └── cleaned_*.txt   # One article per line format
│   ├── post-training/      # Instruction tuning datasets
│   │   ├── lima_train.jsonl         # Original LIMA dataset
│   │   ├── tokenized_examples.npy   # Tokenized LIMA sequences
│   │   ├── metadata.json            # Dataset statistics
│   │   └── example_conversations.json # Sample formatted conversations
│   └── tokenized_datasets/ # Tokenized sequences for training
│       ├── codellama_test_dataset/  # Wikipedia test mode output
│       │   ├── tokens_*.npy         # NumPy arrays of token sequences
│       │   └── metadata.json        # Dataset statistics
│       ├── codellama_full_dataset/  # Wikipedia full dataset
│       │   ├── tokens_*.npy         # 38 files of tokenized sequences
│       │   └── metadata.json        # Dataset statistics
│       ├── fineweb_test_dataset/    # FineWeb test mode output (10k docs)
│       │   ├── tokens_*.npy         # NumPy arrays of token sequences
│       │   └── metadata.json        # Dataset statistics
│       └── fineweb_full_dataset/    # FineWeb full dataset
│           ├── tokens_*.npy         # Up to 38 files of tokenized sequences
│           └── metadata.json        # Dataset statistics
└── LargeLanguageModel/     # Active development directory
```

## Pipeline Architecture

### Pipeline Orchestration (main.py)
**Purpose**: Unified router and orchestrator for all pipeline modules, providing both interactive and CLI interfaces

**Key Features**:
- Interactive menu mode for user-friendly pipeline navigation
- Direct CLI mode with argparse for automation and scripting
- Module registry with centralized configuration
- Subprocess execution for clean memory isolation between stages
- Intelligent error handling distinguishing user interrupts from failures
- Argument forwarding to individual modules
- Session continuity with "run another module" prompts

**Input**: User selection via interactive menu or CLI arguments
**Output**: Executes selected pipeline modules with specified parameters

**Processing Characteristics**:
- Two execution modes: interactive (default) and CLI (with arguments)
- Maps module names to script files: cleanup→cleaner.py, tokenize→tokenizer.py, etc.
- Return code interpretation: 0 (success), 130 (Ctrl-C), other (errors)
- Subprocess isolation prevents memory leaks between pipeline stages
- Supports module chaining in interactive mode
- Preserves module independence while providing unified access

### Phase 1: Data Cleaning (cleaner.py)
**Purpose**: Extract and clean Wikipedia articles from raw JSONL dumps

**Key Features**:
- Processes 38 Wikipedia JSONL files sequentially
- Extracts article text from nested JSON structure
- Cleans text preserving Unicode characters for richer representation
- Removes metadata, references, and formatting artifacts
- Implements efficient streaming processing with minimal memory footprint

**Input**: Raw Wikipedia dumps at `/home/andrea/Desktop/data/raw/enwiki_namespace_0/*.jsonl`
**Output**: Cleaned text at `/home/andrea/Desktop/data/cleaned_articles/cleaned_*.txt`

**Processing Characteristics**:
- Multi Processor, each process 1 file
- Real-time statistics: articles processed, processing speed, file progress
- Output format: One complete article per line

### Phase 2: Tokenization (tokenizer.py)
**Purpose**: Use pre-trained CodeLlama tokenizer for superior text and code representation

**Key Features**:
- Pre-trained CodeLlama tokenizer from HuggingFace (codellama/CodeLlama-7b-hf)
- Fixed vocabulary size of 32,016 tokens (2x larger than custom)
- Special tokens: <s> (BOS, ID=1), </s> (EOS, ID=2), <unk> (UNK, ID=0)
- Full Unicode support for international text and symbols
- Optimized for both natural language and code
- Fast batch encoding/decoding with Rust implementation

**Input**: Cleaned articles from `/home/andrea/Desktop/data/cleaned_articles/`
**Output**: Saved tokenizer models in `tokenizers/` directory

**Processing Characteristics**:
- Downloads pre-trained model from HuggingFace on first use
- Perfect reconstruction of Unicode text
- Compression ratio: ~2.76x on typical English text
- Better performance on code and technical content
- Automatic caching of downloaded tokenizer

### Phase 3a: Wikipedia Dataset Preparation (dataset_prep.py)
**Purpose**: Tokenize cleaned Wikipedia articles into training sequences with sliding windows

**Key Features**:
- Loads CodeLlama tokenizer (always full 32,016 vocab)
- Processes cleaned files sequentially with memory efficiency
- Creates sliding windows with 50% overlap for continuity
- Batch tokenization using HuggingFace transformers
- Saves sequences as NumPy uint16 arrays for efficient loading

**Input**: Cleaned articles from `/home/andrea/Desktop/data/cleaned_articles/`
**Output**: Tokenized sequences in `/home/andrea/Desktop/data/tokenized_datasets/codellama_*_dataset/`

**Processing Characteristics**:
- Test mode: First 10,000 articles only for quick iteration
- Configurable window size (default 512 tokens)
- Batch processing (default 10,000 articles per batch)
- Real-time progress tracking with speed statistics
- Metadata generation with compression metrics
- Outputs to codellama_test_dataset or codellama_full_dataset

### Phase 3b: FineWeb Dataset Preparation (fineweb_prep.py)
**Purpose**: Stream and tokenize FineWeb-10BT dataset from HuggingFace for training

**Key Features**:
- Streams FineWeb-10BT dataset (10B GPT-2 tokens) without full download
- Quality filtering using language scores (threshold: 0.65)
- Filters short documents (< 100 tokens)
- Uses same CodeLlama tokenizer for compatibility
- Creates sliding windows matching Wikipedia format
- Saves as NumPy arrays compatible with existing infrastructure

**Input**: Streamed from HuggingFace dataset `HuggingFaceFW/fineweb` (sample-10BT)
**Output**: Tokenized sequences in `/home/andrea/Desktop/data/tokenized_datasets/fineweb_*_dataset/`

**Processing Characteristics**:
- Test mode: First 10,000 documents for quick iteration
- Streaming mode: No need to download full 27.6GB dataset
- Batch processing (default 10,000 documents per batch)
- Creates up to 38 output files to match Wikipedia structure
- Real-time progress with quality filtering statistics
- Compatible sliding window generation (512 tokens, 50% overlap)

### Phase 4: Data Loading (dataloader.py)
**Purpose**: High-performance PyTorch DataLoader for training with both Wikipedia and FineWeb datasets

**Key Features**:
- Supports both Wikipedia and FineWeb datasets via `dataset_source` parameter
- Memory-efficient fourth-by-fourth loading to save VRAM (~20GB per fourth)
- Train data on GPU for zero-copy access, validation on CPU with pinned memory
- Implements standard PyTorch Dataset interface
- Automatic concatenation of tokenized files within each fourth

**Input**: Tokenized sequences from either:
- Wikipedia: `/home/andrea/Desktop/data/tokenized_datasets/codellama_*_dataset/`
- FineWeb: `/home/andrea/Desktop/data/tokenized_datasets/fineweb_*_dataset/`

**Output**: PyTorch DataLoader ready for training

**Processing Characteristics**:
- Dataset selection based on config.json `dataset.source` field
- Fourth-based loading: splits dataset into 4 parts, loads one at a time
- Fails fast if dataset is incomplete (missing metadata.json)
- Direct tensor indexing with no I/O during training
- Automatic optimization when data is on GPU (num_workers=0)
- Consistent train/val split across runs (seed=42)

### Phase 5: Model Architecture (model.py)
**Purpose**: Modern transformer-based language model with state-of-the-art optimizations

**Key Features**:
- **FastMultiHeadAttention** with Grouped Query Attention (GQA) for reduced memory usage
- **Rotary Position Embeddings (RoPE)** for better length extrapolation (parameter-free)
- **SwiGLU activation** function for improved gradient flow
- **RMSNorm** option alongside LayerNorm for faster training
- **Flash Attention** enforcement for optimal performance on modern GPUs
- **Pre-norm architecture** for better training stability
- Tied input/output embeddings to reduce parameters
- Vocabulary size of 32,016 tokens (CodeLlama)

**Architecture Components**:
- **Grouped Query Attention (GQA)**: Reduces KV heads while maintaining Q heads (e.g., 16 Q heads, 4 KV heads = 4:1 ratio)
- **RoPE**: Encodes positions through rotation matrices, no learned parameters
- **SwiGLU FFN**: Three projection matrices (gate, up, down) with Swish gating
- **Memory Optimizations**: bfloat16 autocast, TorchInductor compilation support

**Generation Capabilities**:
- Temperature-based sampling
- Top-k and Top-p (nucleus) sampling
- Batch generation with proper EOS handling
- Support for padding and attention masks
- bfloat16 inference for speed

### Phase 6: Model Training (train.py)
**Purpose**: Complete training pipeline with GPU optimization and monitoring

**Key Features**:
- Gradient accumulation for effective batch size scaling
- Cosine learning rate scheduling with linear warmup
- Dual logging system (TensorBoard + raw JSON)
- Automatic checkpointing with interruption recovery
- Memory-efficient train/val split with GPU/CPU optimization

**Architecture**:
- GPU-resident training data for zero-copy access
- CPU-resident validation data with pinned memory
- Non-blocking async transfers for validation
- Windowed metrics smoothing for stable monitoring
- Early stopping and best model tracking

**Training Characteristics**:
- Requires CUDA GPU (fails fast if not available)
- Validation every 10% of epoch
- Automatic checkpoint management
- Graceful interruption handling with state preservation

### Phase 7: Advanced Model Components (utils/rope.py, utils/activations.py)
**Purpose**: Modern transformer building blocks for state-of-the-art performance

**Components**:
1. **Rotary Position Embeddings** (rope.py):
   - Parameter-free position encoding through rotation matrices
   - Better length extrapolation than learned embeddings
   - TorchInductor-compatible real-valued operations
   - Support for Grouped Query Attention (different Q/KV head counts)
   - Configurable theta and scaling factors

2. **Modern Activation Functions** (activations.py):
   - **SwiGLU**: Swish-gated linear unit used in LLaMA/PaLM
   - **GeGLU**: GELU-gated variant
   - **ReGLU**: ReLU-gated variant
   - Three-matrix architecture (gate, up, down projections)
   - Automatic intermediate size calculation for optimal GPU utilization

### Phase 8: Utilities (utils/)
**Purpose**: Modular components for training infrastructure

**Components**:
1. **DualLogger** (logging_utils.py):
   - Simultaneous TensorBoard and JSON logging
   - Experiment tracking with timestamps
   - Buffered writes for efficiency
   - Configuration and hyperparameter logging

2. **MetricsTracker** (metrics.py):
   - Sliding window smoothing for noisy metrics
   - Throughput calculation (tokens/second)
   - Memory usage tracking
   - Best metrics persistence
   - Early stopping support

3. **Scheduler** (scheduler.py):
   - Cosine annealing with warmup
   - Configurable minimum learning rate
   - Resume-friendly state management
   - Integration with PyTorch optimizers

### Phase 9: Model Inference (Inference.py)
**Purpose**: Interactive text generation using trained model checkpoints with single and multi-model comparison modes

**Key Features**:
- **Two Inference Modes**:
  - Single model inference for standard text generation
  - Multi-model comparison for A/B testing different checkpoints
- Hierarchical checkpoint organization by training runs
- Interactive two-step selection: training run → checkpoint
- Custom prompt input with full Unicode support
- Configurable generation parameters (temperature, top-k, top-p)
- Color-coded terminal interface with ANSI codes
- Real-time generation statistics and performance metrics
- Continuous generation loop without model reloading
- No ASCII cleaning required - handles all text types

**Multi-Model Comparison Features**:
- Load multiple checkpoints simultaneously
- Side-by-side generation from the same prompt
- Performance comparison statistics (speed, tokens/sec)
- Memory warnings for GPU constraints
- Automatic identification of fastest/most productive models

**Input**: Trained checkpoints from `checkpoints/` directory
**Output**: Generated text with statistics (single or comparative)

**Processing Characteristics**:
- Automatic GPU/CPU detection and optimization
- Handles compiled model state dictionaries (removes "_orig_mod." prefix)
- Uses CodeLlama tokenizer exclusively
- Memory-efficient with GPU cache management
- Batch generation support with proper EOS handling
- Backward compatibility with flat checkpoint structure
- Full Unicode text generation support

**Checkpoint Organization**:
- New: Hierarchical structure with subdirectories per training run
- Legacy: Flat structure with all checkpoints in base directory
- Automatic detection and handling of both formats

**Generation Parameters**:
- **Max Length**: Maximum tokens to generate (default: 200)
- **Temperature**: Controls randomness (default: 0.8)
- **Top-k**: Limits vocabulary to top k tokens (default: 50, 0 to disable)
- **Top-p**: Nucleus sampling threshold (default: 0.95, 1.0 to disable)

### Phase 10: Model Benchmarking (benchmark.py)
**Purpose**: Comprehensive evaluation framework for language models using standard NLP metrics

**Key Features**:
- **Four Evaluation Metrics**:
  - Perplexity measurement on WikiText-103 dataset
  - Text completion quality via BLEU and ROUGE scores
  - Next token prediction accuracy (Top-1, Top-5, Top-10)
  - Commonsense reasoning through multiple-choice scenarios
- **Dataset Management**:
  - Automatic downloading and caching of benchmark datasets
  - WikiText-103 for perplexity evaluation
  - LAMBADA for context-dependent word prediction
  - HellaSwag-style tasks for commonsense reasoning
  - Fallback samples for offline evaluation
- **Multi-Model Comparison**:
  - Side-by-side evaluation of multiple checkpoints
  - Automatic best-score highlighting
  - Performance ranking across all metrics
- **Result Persistence**:
  - JSON output for programmatic analysis
  - Human-readable text summaries
  - Timestamped result files for tracking progress

**Input**: Trained model checkpoints from `checkpoints/` directory
**Output**: Comprehensive evaluation metrics saved to `benchmark_results/`

**Processing Characteristics**:
- Memory-efficient evaluation with disabled gradients
- Teacher-forcing for perplexity calculation
- Autoregressive generation for completion tasks
- Configurable sample sizes (quick/standard/comprehensive)
- Real-time progress tracking with tqdm
- Automatic GPU/CPU optimization
- Batch processing for efficient evaluation

**Benchmark Configurations**:
- **Quick**: 50 WikiText, 20 LAMBADA, 10 HellaSwag samples (~5 minutes)
- **Standard**: 100 WikiText, 50 LAMBADA, 20 HellaSwag samples (~10 minutes)
- **Comprehensive**: 200 WikiText, 100 LAMBADA, 50 HellaSwag samples (~20 minutes)

**Metrics Explained**:
- **Perplexity**: Lower is better; measures prediction uncertainty
- **BLEU**: 0-100 score; measures n-gram precision in generated text
- **ROUGE**: 0-100 score; evaluates recall of important phrases
- **Top-k Accuracy**: Percentage where correct token is in top-k predictions
- **Commonsense**: Percentage of correctly chosen completions

### Phase 11: Post-Training - Instruction Tuning (lima_tokenizer.py, post_training.py)
**Purpose**: Transform pretrained language models into instruction-following chatbots through supervised fine-tuning (SFT) on high-quality conversation data

**Key Features**:
- **LIMA Dataset Integration**:
  - High-quality instruction dataset (1,030 carefully curated examples)
  - Diverse sources: StackExchange, WikiHow, manual authoring
  - 97% single-turn, 3% multi-turn conversations
  - Quality over quantity approach for effective fine-tuning
- **Chat Template Formatting**:
  - Role-based conversation structure with "User:" and "Assistant:" markers
  - Proper BOS/EOS token placement for each conversation turn
  - Multi-turn support with clear conversation boundaries
- **Efficient SFT Training**:
  - Lower learning rates (5e-5) to prevent catastrophic forgetting
  - Full dataset loaded to GPU memory (only ~0.5MB tokenized)
  - Architecture verification from pretrained checkpoints
  - Separate checkpoint directory for fine-tuned models

**Components**:
1. **lima_tokenizer.py**: Prepares LIMA dataset for training
   - Downloads and processes LIMA from HuggingFace
   - Formats conversations with role markers and special tokens
   - Filters examples by length (max 512 tokens)
   - Saves tokenized sequences as NumPy arrays

2. **post_training.py**: Supervised fine-tuning script
   - Loads pretrained checkpoints with architecture verification
   - Implements instruction-specific training loop
   - Tracks validation loss and generates samples
   - Saves SFT checkpoints with full configuration

**Input**: 
- lima_tokenizer.py: Raw LIMA dataset from HuggingFace
- post_training.py: Pretrained model checkpoint + tokenized LIMA data

**Output**: 
- lima_tokenizer.py: `data/post-training/tokenized_examples.npy`
- post_training.py: `checkpoints/sft_lima_*/checkpoint_best.pt`

**Processing Characteristics**:
- Tokenization: 542/1030 examples kept (512 token limit)
- Training: ~60 steps per epoch with batch size 8
- Memory: Entire dataset fits in GPU memory
- Speed: Full epoch in ~1 minute on modern GPU
- Validation: Every 20% of epoch with sample generation

**Chat Format Example**:
```
<s>User: How do I implement a binary search tree in Python?
Assistant: Here's a complete implementation of a binary search tree in Python:

[Implementation details...]</s>
<s>User: Can you add a method to find the height?
Assistant: [Height method implementation...]</s>
```

### Upcoming Phases (To Be Implemented)

**Phase 12: Inference Server**
- REST API for text generation
- Streaming generation support
- Model quantization for deployment

## Usage

### Using the Pipeline Router (main.py)
```bash
# Interactive mode - user-friendly menu navigation
python3 main.py

# Direct CLI execution - for automation and scripting
python3 main.py cleanup --test              # Run Wikipedia cleanup in test mode
python3 main.py tokenize --encode "text"    # Encode text with tokenizer
python3 main.py prepare --window 1024       # Prepare Wikipedia dataset with custom window
python3 main.py fineweb --test              # Prepare FineWeb dataset in test mode
python3 main.py train --test --name exp1    # Train model with experiment name
python3 main.py inference                   # Launch interactive text generation

# Interactive workflow:
# 1) Run python3 main.py
# 2) Select module from menu (1-7)
# 3) Enter optional arguments when prompted
# 4) Module executes in subprocess
# 5) Choose to run another module or quit

# Module mapping:
# 1 → Raw Cleanup (cleaner.py) - Wikipedia only
# 2 → Tokenization (tokenizer.py)
# 3 → Wikipedia Prep (dataset_prep.py)
# 4 → FineWeb Prep (fineweb_prep.py)
# 5 → Training (train.py)
# 6 → Inference (Inference.py)
# 7 → Quit
```

### Running the Data Cleaner
```bash
# Test mode, test 50000 articles per process.
python3 cleaner.py --test

# Processes all 38 Wikipedia files with 4 workers
python3 cleaner.py

# Output: /home/andrea/Desktop/data/cleaned_articles/
```

### Using the CodeLlama Tokenizer
```bash
# Download and save CodeLlama tokenizer
python3 tokenizer.py

# Encode text with CodeLlama tokenizer
python3 tokenizer.py --encode "Hello world"

# Encode code with CodeLlama tokenizer
python3 tokenizer.py --encode "def hello(): return 'world'"

# Decode token IDs
python3 tokenizer.py --decode "123,456,789"

# Load saved tokenizer
python3 tokenizer.py --load tokenizers/codellama_tokenizer

# Output: tokenizers/codellama_tokenizer/
```


### FineWeb Prep (Parallel Streaming)
```bash
# Test mode (single-threaded by default)
python3 fineweb_prep.py --test

# Full mode with parallel streaming (e.g., 4 workers)
python3 fineweb_prep.py --workers 4

# Via router with args forwarding
python3 main.py fineweb -- --workers 4

# Notes:
# - The --workers flag enables parallel streaming by sharding the dataset
#   across worker threads, increasing docs/sec for the phase that shows:
#   "Documents processed: x docs [.., y docs/s]".
# - For low I/O environments or testing, omit --workers to use prefetch mode.
```

### Preparing Wikipedia Dataset
```bash
# Test mode: tokenize first 10,000 articles with CodeLlama
python3 dataset_prep.py --test

# Full dataset: tokenize all 38 cleaned files
python3 dataset_prep.py

# Custom window size (default 512)
python3 dataset_prep.py --window 1024

# Custom batch size (default 10000)
python3 dataset_prep.py --batch 5000

# Output: /home/andrea/Desktop/data/tokenized_datasets/codellama_*_dataset/
```

### Preparing FineWeb Dataset
```bash
# Test mode: process first 10,000 documents
python3 fineweb_prep.py --test

# Full dataset: stream and process entire FineWeb-10BT
python3 fineweb_prep.py

# Custom parameters
python3 fineweb_prep.py --window 1024 --batch-size 5000 --max-files 20

# Output: /home/andrea/Desktop/data/tokenized_datasets/fineweb_*_dataset/

# Note: FineWeb streams from HuggingFace, no download needed
# Quality filtering applied automatically (language_score > 0.65)
```

### Switching Between Datasets for Training
```bash
# Edit config.json to switch datasets:
# For Wikipedia:
"dataset": {
  "source": "wikipedia",
  ...
}

# For FineWeb:
"dataset": {
  "source": "fineweb",
  ...
}

# Then train normally - the dataloader will automatically use the selected dataset
python3 train.py
```

### Creating and Testing the Model
```python
# Create model from configuration
from model import create_model

model = create_model("config.json")
# Output: Model with ~25M parameters

# Test forward pass
import torch
input_ids = torch.randint(0, 32016, (batch_size, seq_len))
outputs = model(input_ids, labels=input_ids)
print(f"Loss: {outputs['loss'].item()}")

# Generate text
generated = model.generate(
    prompt_ids,
    max_length=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)
```

### Using the DataLoader
```python
# In your training script
from dataloader import create_simple_train_val_dataloaders

# Create memory-efficient train/val dataloaders
train_loader, val_loader, info = create_simple_train_val_dataloaders(
    batch_size=64,
    val_split=0.1,      # 10% validation split
    shuffle_train=True,
    test_mode=False,    # Set True for quick testing
    verbose=True,
    seed=42            # Consistent split across runs
)

# Training loop
for epoch in range(num_epochs):
    # Training data is on GPU for zero-copy access
    for batch in train_loader:
        # batch shape: (batch_size, window_size)
        outputs = model(batch)  # Already on GPU
        loss = criterion(outputs)
        # ... training step
    
    # Validation data is on CPU with pinned memory
    for batch in val_loader:
        batch = batch.to('cuda', non_blocking=True)
        # ... validation step

# Access dataset info
print(f"Train sequences: {info['train_size']:,}")
print(f"Val sequences: {info['val_size']:,}")
print(f"Train device: {info['train_device']}")
print(f"Val device: {info['val_device']}")
```

### Training the Model
```bash
# Test mode: single epoch on small dataset
python3 train.py --test

# Full training: complete dataset with all epochs
python3 train.py

# Resume from checkpoint
python3 train.py --resume checkpoints/checkpoint_latest.pt

# Custom experiment name for logging
python3 train.py --name experiment_lr_1e3

# Monitor training with TensorBoard
tensorboard --logdir logs/tensorboard

# Training outputs:
# - Checkpoints in checkpoints/
# - TensorBoard logs in logs/tensorboard/
# - Raw metrics in logs/raw_metrics/
```

### Running Benchmarks
```bash
# Run benchmark evaluation through Inference.py
python3 Inference.py
# Select option 3: Benchmark models
# Choose configuration: quick/standard/comprehensive
# Results saved to benchmark_results/

# Or use benchmark module directly
python3 -c "
from benchmark import ModelBenchmark, format_benchmark_results
from model import create_model
from tokenizer import WikipediaTokenizer
import torch

# Load models
models_dict = {
    'model_latest': (model1, tokenizer1),
    'model_best': (model2, tokenizer2)
}

# Run benchmarks
benchmark = ModelBenchmark()
results = benchmark.run_full_benchmark(models_dict)
print(format_benchmark_results(results))
"

# View saved results
ls -la benchmark_results/
cat benchmark_results/benchmark_*.txt
```

### Post-Training: Instruction Tuning

#### Preparing LIMA Dataset
```bash
# Download and tokenize LIMA dataset
python3 lima_tokenizer.py

# Custom parameters
python3 lima_tokenizer.py --max-length 1024 --batch-size 50

# Output location: /home/andrea/Desktop/data/post-training/
# Files created:
# - tokenized_examples.npy (tokenized sequences)
# - metadata.json (dataset statistics)
# - example_conversations.json (sample formatted conversations)
```

#### Running Supervised Fine-Tuning
```bash
# Basic SFT with pretrained checkpoint
python3 post_training.py --checkpoint checkpoints/110M_FINE_COMPLETE/110MPre-Trained.pt

# Custom training parameters
python3 post_training.py \
    --checkpoint checkpoints/110M_FINE_COMPLETE/110MPre-Trained.pt \
    --epochs 2 \
    --lr 3e-5 \
    --batch-size 4 \
    --name my_sft_experiment

# Quick test with smaller batch
python3 post_training.py \
    --checkpoint checkpoints/110M_FINE_COMPLETE/110MPre-Trained.pt \
    --batch-size 2 \
    --epochs 1

# Output: checkpoints/sft_lima_[timestamp]/
# - checkpoint_best.pt (best validation loss)
# - checkpoint_latest.pt (most recent)
# - checkpoint_epoch_*.pt (per-epoch saves)
# - sft_config.json (training configuration)
```

### Running Inference
```bash
# Run interactive inference with trained model
python3 Inference.py

# The script will:
# 1. Select inference mode (single model or multi-model comparison)
# 2. List available training runs (or use base directory if flat structure)
# 3. Select specific checkpoint(s) within the chosen run
# 4. Load selected model(s) and tokenizer
# 5. Enter custom prompts
# 6. Configure or use default generation parameters
# 7. Generate and display text with statistics

# Single Model Workflow:
# - Select mode: 1 (Single model inference)
# - Select training run from list (most recent highlighted)
# - Select checkpoint (best/latest highlighted in colors)
# - Enter custom prompt: "The future of AI is"
# - Choose: 1 (Use default parameters) or 2 (Customize)
# - View generated text and statistics
# - Continue with new prompts or quit

# Multi-Model Comparison Workflow:
# - Select mode: 2 (Multi-model comparison)
# - Select training run from list
# - Select multiple checkpoints: enter comma-separated numbers (e.g., 1,3,5)
#   or 'all' to load all checkpoints
# - System shows memory usage and warnings if needed
# - Enter custom prompt for all models
# - Configure generation parameters (applied to all models)
# - View side-by-side generation results
# - See performance summary (fastest, most tokens, highest speed)
# - Continue with new prompts or quit

# Notes:
# - GPU memory is automatically managed with cache clearing
# - Supports backward compatibility with old flat checkpoint structure
# - Color coding: GREEN for best checkpoint, BLUE for latest
# - Memory warnings appear when loading multiple large models
# - Works with both pretrained and SFT checkpoints (e.g., checkpoints/sft_lima_*/checkpoint_best.pt)
```

## Configuration System

The `config.json` file centralizes all configuration for the pipeline.

### Dataset Configuration
```json
"dataset": {
  "source": "wikipedia",  // or "fineweb" - selects which dataset to use
  "wikipedia_test_dir": "/path/to/wikipedia/test",
  "wikipedia_full_dir": "/path/to/wikipedia/full",
  "fineweb_test_dir": "/path/to/fineweb/test",
  "fineweb_full_dir": "/path/to/fineweb/full"
}
```

- **source**: Controls which dataset the training pipeline uses
  - `"wikipedia"`: Use Wikipedia dataset (default, structured encyclopedic content)
  - `"fineweb"`: Use FineWeb-10BT dataset (diverse web content)
- Dataset paths are automatically selected based on the source field
- Both datasets use identical tokenization and training infrastructure

## Development Guidelines

### Code Philosophy
- **Clarity over cleverness**: Readable, maintainable code
- **Modular design**: Single responsibility per module
- **Efficiency at scale**: Handle 50GB+ datasets gracefully
- **Fail fast**: Clear error messages, no silent failures

### Implementation Standards
- Implement proper error handling with descriptive messages
- Provide progress feedback for long-running operations
- Memory-efficient processing for large datasets
- Clean separation of concerns between modules
- Keep main classes focused on their core functionality
- Extract complex operations to dedicated modules

## Error Handling

The pipeline implements strict error handling:
- Immediate failure on critical errors
- Clear error messages with context
- No automatic fallbacks that hide problems
- Progress state preserved on interruption

---

*This document will be updated as new components are implemented.*
