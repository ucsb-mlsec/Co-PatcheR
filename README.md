# 🔧 Co-PatcheR: Collaborative Software Patching with Component(s)-specific Small Reasoning Models

<p align="center">
    <a href="https://arxiv.org/abs/2505.18955"><img src="https://img.shields.io/badge/arXiv-2505.18955-b31b1b.svg?style=for-the-badge">
    <a href="https://opensource.org/license/mit/"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge">
    <a href="https://huggingface.co/collections/UCSB-SURFI/co-patcher-6834dec588b92b3fd011eecc"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Co--PatcheR-%23ff8811.svg?style=for-the-badge">
</p>

<p align="center">
    🔍&nbsp;<a href="#overview">Overview</a>
    | 🤖&nbsp;<a href="#models">Models</a>
    | 🛠️&nbsp;<a href="#installation">Installation</a>
    | 🚀&nbsp;<a href="#quick-start">Quick Start</a>
    | 📝&nbsp;<a href="#citation">Citation</a>
</p>

## News

- 🚀 **[May 2025]** Co-PatcheR models and code are now available on Hugging Face! 
- 📄 **[May 2025]** Co-PatcheR paper is available on arXiv!

## Overview

### 🔧 Co-PatcheR: A Collaborative Software Patching System

Co-PatcheR introduces an approach to automated software patching through **collaborative small reasoning models**. Instead of using one large model for all tasks, we employ specialized 14B models for different components of the patching pipeline.

**Key Innovations:**
- **🎯 Component-Specific Models**: Dedicated models for localization, generation, and validation
- **📊 SOTA Performance**: 46% resolved rate on SWE-bench-Verified

### 🏗️ Architecture Overview

Co-PatcheR usage consists of three specialized components:

1. **🔍 Fault Localization**: Identifies problematic code locations
2. **⚡ Patch Generation**: Generates multiple patch candidates
3. **🛡️ Patch Validation**: Validation through test case generation and execution

## Models

| Component | Model | HF Checkpoint | Size |
|-----------|-------|---------------|------|
| Localization & Generation | Co-PatcheR-Loc-Gen | 🤗 [UCSB-SURFI/Co-PatcheR-Loc-Gen-14B](https://huggingface.co/UCSB-SURFI/Co-PatcheR-Loc-Gen-14B) | 14B |
| Validation (w/ assertions) | Co-PatcheR-Val-Assert | 🤗 [UCSB-SURFI/Co-PatcheR-Val-assert-14B](https://huggingface.co/UCSB-SURFI/Co-PatcheR-Val-assert-14B) | 14B |
| Validation (w/o assertions) | Co-PatcheR-Val-NoAssert | 🤗 [UCSB-SURFI/Co-PatcheR-Val-no-assert-14B](https://huggingface.co/UCSB-SURFI/Co-PatcheR-Val-no-assert-14B) | 14B |

> _Note_: Both validation models (`Co-PatcheR-Val-assert-14B` and `Co-PatcheR-Val-no-assert-14B`) enhance the validation step, but for a simplified setup you can use just `Co-PatcheR-Val-no-assert-14B` with still good performance.

**Performance on SWE-bench-Verified:**
- **46% resolved rate** with 3×14B models
- **Least training resources** among specialized patching models
- **Smallest model ensemble** achieving SOTA performance

## Installation

### 🐳 Docker Setup (Recommended)

1. **Pull the Docker image:**
```bash
docker pull 3rdn4/patchpilot_verified:v1
```

2. **Run the container:**
```bash
docker run -it 3rdn4/patchpilot_verified:v1
```

### 📦 From Source

1. **Clone the repository:**
```bash
git clone git@github.com:ucsb-mlsec/Co-PatcheR.git
cd Co-PatcheR
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment:**
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Quick Start

### 📥 Preprocessing Data (Optional but Recommended)

For SWE-Bench Verified, we need to checkout repositories and process files. To save time, you can download the preprocessed data:

```bash
# Download preprocessed repository structure
wget https://github.com/ucsb-mlsec/Co-PatcheR/releases/download/v1.0.0/verified_repo_structure.txt

# Export the location 
export PROJECT_STRUCTURE={path_to_downloaded_file}
```

### ⚙️ Model Deployment (Recommended)

We recommend deploying the Co-PatcheR models locally using vLLM:

```bash
export CUDA_VISIBLE_DEVICES=0,1
vllm serve UCSB-SURFI/Co-PatcheR-Loc-Gen-14B --tensor-parallel-size 2 --port 2952
```

### 🎯 1. Fault Localization

#### Step 1: Initial Localization
```bash
python patchpilot/fl/localize.py \
    --file_level \
    --related_level \
    --fine_grain_line_level \
    --output_folder results/localization \
    --backend opensource \
    --model UCSB-SURFI/Co-PatcheR-Loc-Gen-14B \
    --top_n 5 \
    --compress \
    --context_window=20 \
    --temperature 0.7 \
    --match_partial_paths \
    --num_samples 4 \
    --num_threads 32 \
    --task_list_file swe_verify_tasks.txt \
    --benchmark verified \
    --port 2952
```

#### Step 2: Merge Localization Results
```bash
python patchpilot/fl/localize.py \
    --merge \
    --output_folder results/localization/merged \
    --start_file results/localization/loc_outputs.jsonl \
    --num_samples 4
```

### ⚡ 2. Patch Generation  

Generate patches based on merged localization results:

```bash
python patchpilot/repair/repair.py \
    --loc_file results/localization/merged/loc_all_merged_outputs.jsonl \
    --output_folder results/repair \
    --benchmark verified \
    --max_samples 20 \
    --batch_size 1 \
    --num_threads 32 \
    --backend opensource \
    --model UCSB-SURFI/Co-PatcheR-Loc-Gen-14B \
    --task_list_file swe_verify_tasks.txt \
    --port 2952
```

### 🛡️ 3. Patch Validation

#### Step 1: Generate Proof-of-Concepts (POCs)
```bash
python patchpilot/reproduce/reproduce.py \
    --reproduce_folder results/reproduce \
    --num_threads 24 \
    --task_list_file swe_verify_tasks.txt \
    --setup_map setup_result/verified_setup_map.json \
    --tasks_map setup_result/verified_tasks_map.json \
    --model UCSB-SURFI/Co-PatcheR-Val-no-assert-14B \
    --backend opensource \
    --benchmark verified \
    --num_samples 5 \
    --port 2952
```

#### Step 2: Verify POCs
```bash
python patchpilot/reproduce/verify.py \
    --verify_folder results/validation \
    --reproduce_folder results/reproduce \
    --patch_folder results/repair \
    --num_threads 32 \
    --task_list_file swe_verify_tasks.txt \
    --setup_map setup_result/verified_setup_map.json \
    --tasks_map setup_result/verified_tasks_map.json \
    --backend opensource \
    --model UCSB-SURFI/Co-PatcheR-Val-no-assert-14B \
    --port 2952
```

### 🔄 4. Rerank Patches

Final step to rerank patches based on validation results:

```bash
python patchpilot/repair/rerank.py \
    --loc_file results/localization/merged/loc_all_merged_outputs.jsonl \
    --output_folder results/repair \
    --benchmark verified \
    --verify_folder results/verify \
    --setup_map setup_result/full_setup_map.json \
    --tasks_map setup_result/full_tasks_map.json \
    --num_threads 32 \
    --task_list_file swe_verify_tasks.txt \
    --sample_mod
```

### 📋 Complete Pipeline Summary

The complete Co-PatcheR pipeline follows this order:
1. **Localization** → Issue localization
2. **Generation** → Generate patch candidates  
3. **Validation** → Generate POCs + dynamic validation
4. **Rerank** → Final ranking of patches


## 📝 Citation

If you find Co-PatcheR useful in your research, please cite our paper:

```bibtex
@article{tang2025copatcher,
  title={Co-PatcheR: Collaborative Software Patching with Component(s)-specific Small Reasoning Models},
  author={Tang, Yuheng and Li, Hongwei and Zhu, Kaijie and Yang, Michael and Ding, Yangruibo and Guo, Wenbo},
  journal={arXiv preprint arXiv:2505.18955},
  year={2025}
}
```

---

<p align="center">
    Made with ❤️ by the UCSB-SURFI Team
</p>