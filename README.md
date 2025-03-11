# SafeMERGE

## Abstract
Fine-tuning large language models (LLMs) on downstream tasks can inadvertently erode their safety alignment, even for benign fine-tuning datasets. We address this challenge by proposing **SafeMERGE** (accepted to ICLR 2025), a post–fine-tuning framework that preserves safety while maintaining task utility. It achieves this by selectively merging fine-tuned and safety-aligned model layers *only* when those deviate from safe behavior, measured by a cosine similarity criterion. We evaluate SafeMERGE against other fine-tuning- and post–fine-tuning-stage approaches for Llama-2-7B-Chat and Qwen-2-7B-Instruct models on GSM8K and PubMedQA tasks while exploring different merging strategies. We find that SafeMERGE consistently reduces harmful outputs compared to other baselines without significantly sacrificing performance, sometimes even enhancing it. The results suggest that our selective, subspace-guided, and per-layer merging method provides an effective safeguard against the inadvertent loss of safety in fine-tuned LLMs while outperforming simpler post–fine-tuning-stage defenses.

---

## This Repo
This repository provides a tool to perform SafeLoRA-based merging of adapters for transformer models. The approach leverages projection matrices computed from the difference between a base (unaligned) model and an aligned (safe) model to determine the appropriate mix between a finetuned adapter and a safety-tuned adapter.

---

## Overview
The key idea is to use a projection matrix to measure how much a finetuned adapter's weights deviate from a safe reference. If the cosine similarity between the projected finetuned weights and the original weights falls below a specified threshold, a partial merge is applied (e.g., using weights `[0.8, 0.2]` for the finetuned and safe adapters, respectively). Otherwise, the finetuned adapter is used without adjustment.

### Why Qwen Models Are Handled Differently
Qwen models have unique characteristics compared to other architectures like LLaMA. Specifically, Qwen models include additional parameters—such as biases—that are not part of the LoRA layers. These extra parameters often have shapes that do not match the expected 2D structure used for LoRA projections (e.g., 1D biases). As a result, when a Qwen model is detected (by checking if `"Qwen"` is present in the model path), the code **skips non-2D parameters** to ensure that only valid 2D LoRA parameters are processed during projection.

---

## Files
- **`utils.py`**  
  Contains helper functions to compute projection matrices and cosine similarity between LoRA weight differences, and defines the `SafeLoRAMerger` class which encapsulates the merging logic.

- **`get_safemerge_model.py`**  
  A command-line script that loads the base model along with finetuned and safety adapters, computes the necessary projections, and then applies the SafeLoRA merging procedure before saving the final merged model.

---

## Requirements

- **Python**  
  Tested with Python 3.11.4

- **PyTorch**  
  Tested with PyTorch 2.4.1 + CUDA 12.1, which can be installed via:
  ```bash
  pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

- **Remaining Requirements**  
  Install the remaining repository requirements via pip:
  ```bash
  pip install -r requirements.txt


## Usage Example
Suppose you have the following models:
- Base Model: "meta-llama/Llama-2-7b-chat-hf"
- Finetuned Adapter: "my_hf_repo/llama_2_7b_chat_hf_gsm8k"
- Safety Adapter: "my_hf_repo/llama_2_7b_chat_hf_gsm8k_safe"
- Unaligned model (unsafe): "meta-llama/Llama-2-7b-hf"
- Aligned model (safe): "meta-llama/Llama-2-7b-chat-hf"

Run the script from the command line as follows:
```bash
    python get_safemerge_model.py \
    --base_model_id meta-llama/Llama-2-7b-chat-hf \
    --finetuned_model_id my_hf_repo/llama_2_7b_chat_hf_gsm8k \
    --safety_model_id my_hf_repo/llama_2_7b_chat_hf_safety_tuned \
    --safelora_unaligned_model_id meta-llama/Llama-2-7b-hf \
    --safelora_aligned_model_id meta-llama/Llama-2-7b-chat-hf \
    --cos_threshold 0.35 \
    --default_merge_ratio 0.2 \
    --weighting "[0.8, 0.2]" \
    --merge_type linear \
    --density 0.5 \
    --output_path ./safemerge_models
```

This command will:
1. Load the base model.
2. Load the finetuned and safety adapters. 
3. Compute the SafeLoRA projection matrices from the specified unaligned and aligned models.
4. Merge the adapters based on the cosine similarity threshold (using partial merging if the similarity is below 0.35).
5. Save the final merged model in the specified output directory.