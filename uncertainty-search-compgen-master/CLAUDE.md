# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project focused on "Uncertainty search for compositional generalization" in natural language processing. The project implements methods for training language models with uncertainty-guided search techniques, specifically targeting compositional generalization tasks like semantic parsing.

## Key Components

### Core Architecture
- **Two-stage training process**: 
  - Stage 1: Base model fine-tuning 
  - Stage 2: ATS (Adaptive Temperature Selection) head training
- **ATS Head**: A 2-layer Transformer + 2-layer MLP architecture with GELU activation and LayerNorm for adaptive temperature prediction
- **Uncertainty-guided search**: Uses entropy-based uncertainty metrics to guide beam search during inference

### Main Modules
- `uncertainty_search_compgen/train_lm.py`: Core training logic with dual-stage training
- `uncertainty_search_compgen/inference.py`: Inference with uncertainty-guided search
- `uncertainty_search_compgen/data.py`: Data loading and preprocessing for various semantic parsing datasets
- `uncertainty_search_compgen/dataset.py`: PyTorch dataset implementations
- `uncertainty_search_compgen/load_hf_lm.py`: HuggingFace model and tokenizer loading
- `uncertainty_search_compgen/divergence_metrics.py`: Uncertainty and divergence measurement utilities

### Dataset Support
The project supports multiple semantic parsing datasets:
- SMCalFlow (simplified and full versions)
- ATIS semantic parsing
- GeoQuery
- Text2SQL tasks
- Various NLI and text classification tasks

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Alternative: Install as package
pip install -e .

# Install from root directory (use absolute paths in train_lm.py)
cd uncertainty-search-compgen-master
pip install -e .
```

### Python Path Configuration
The codebase requires specific Python path configuration:
- Main training scripts expect to run from the `uncertainty-search-compgen-master` directory
- Absolute paths are hardcoded in `train_lm.py:5` and `train_lm.py:11` for project root
- Package imports use relative imports with `__package__ = "uncertainty_search_compgen"`

### Training Commands
```bash
# Train on full data with ATS approach
python train_on_full_data.py

# Test ATS improvements
python test_ats_improvements.py

# Quick ATS testing
python quick_test_ats.py

# Aggressive ATS testing
python test_aggressive_ats.py
```

### Analysis and Evaluation
```bash
# Analyze existing experimental results
python analyze_existing_data.py

# Generate correlation reports
python generate_correlation_report.py

# Temperature scaling diagnostics
python diagnose_ats_vs_entropy.py

# Test different temperature scaling approaches
python test_temperature_scaling_effect.py

# Test improved ATS methods
python test_improved_ats_methods.py
```

### Jupyter Notebooks
```bash
# Interactive inference and analysis
jupyter notebook notebook/inference.ipynb

# Or run inference directly
python notebook/inference.py
```

### SLURM Integration
For GPU cluster usage:
```bash
# Submit GPU job (launches Jupyter notebook)
sbatch slurm/submit_interactive_gpu.sh

# Submit CPU job  
sbatch slurm/submit_interactive_cpu.sh

# Check job status
squeue -u $USER

# View logs
tail -f logs/output_<job_id>.log
tail -f logs/error_<job_id>.log
```

### Data Setup
```bash
# Download required datasets (from README)
wget https://sspilsbury-data.s3.eu-north-1.amazonaws.com/text.zip
unzip text.zip
```

## Key Configuration

### Model Configuration
- Default base model: `Salesforce/codet5p-220m`
- Tokenizer: CodeT5+ tokenizer with right padding
- Cache directory: `/tmp/hf` for HuggingFace models
- ATS head architecture: 2-layer Transformer + 2-layer MLP with GELU and LayerNorm
- GPU memory optimization: Supports H200 GPUs with 141GB memory allocation

### Training Parameters
- Learning rates: Base model 0.1x, ATS head 1.0x
- Temperature regularization: MSE alignment with weight 1.5
- Loss function: Combination of cross-entropy and uniform loss with alpha parameter
- Batch processing: Supports variable sequence lengths with padding

### Data Preprocessing
- Input sequences: Max length 256 tokens
- Target sequences: Max length 448 tokens  
- Special tokens: Uses `[eos]` for sequence termination
- Filtering: Removes examples with incomplete PersonName or year matches

## Inference and Evaluation

### Uncertainty Metrics
- Entropy-based uncertainty measurement
- Teacher-student model divergence (KL divergence)
- Temperature-loss correlation analysis

### Search Strategy
- Beam search with uncertainty-guided selection
- Adaptive temperature scaling based on ATS head predictions
- Configurable beam width and selection thresholds

## Experiment Tracking

### Lightning Logs
- Training logs stored in `logs/`, `logs_full_data_stable/`
- TensorBoard integration for experiment monitoring
- Model checkpoints saved with descriptive names

### Results Storage
- Evaluation results: `evaluation_results*.json`
- Correlation analysis: `*_correlation_results.json`
- Figures and plots: `figures/` directory

## Development Notes

### Code Style
- Uses PyTorch Lightning for training orchestration
- Implements custom loss functions with uncertainty regularization
- Heavy use of transformer architectures from HuggingFace
- GPU-optimized with CUDA support throughout
- Absolute path dependencies in training scripts (requires specific working directory)

### Testing Strategy
- Multiple test scripts for different ATS configurations
- Quantitative evaluation with correlation metrics
- Visualization tools for analysis and debugging
- Text2SQL tests available in `text/semparse/text2sql-data/systems/sequence-to-sequence/seq2seq/test/`

### File Structure Notes
- Main package: `uncertainty_search_compgen/`
- Training logs: `logs/`, `logs_full_data_stable/` with TensorBoard events
- Results and analysis: `figures/`, evaluation JSON files
- Model checkpoints: Stored with descriptive names indicating layer and stage
- SLURM integration: Scripts in `slurm/` directory for cluster deployment

## Standard Workflow
1. First think through the problem, read the codebase for relevant files, and write a plan to tasks/todo. md.
2. The plan should have a list of todo items that you can
check off as you complete them
3. Before you begin working, check in with me and I will
verify the plan.
4. Then, begin working on the todo items, marking them as
complete as you go.
5. Please every step of the way just give me a high level
explanation of what changes you made
6. Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
7. Finally, add a review section to the todo.md file with a summary of the changes you made and any other relevant