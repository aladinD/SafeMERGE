<div align="center">

# SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging

[![Paper](https://img.shields.io/badge/paper-arXiv%3A2503.17239-b31b1b)](https://arxiv.org/abs/2503.17239)
[![Conference](https://img.shields.io/badge/ACL-2026-2e7d32)](https://aclanthology.org/2026.findings-acl.1761/)

</div>

<div style="margin-bottom: 1em;">
  <strong>Abstract</strong><br>
  Fine-tuning large language models (LLMs) is a common practice to adapt generalist models to specialized domains. However, recent studies show that fine-tuning can erode safety alignment, causing LLMs to respond to harmful or unethical prompts. Many methods to realign safety have been proposed, but often introduce custom algorithms that are difficult to implement or compromise task utility. In this work, we propose SafeMERGE, a lightweight, post-fine-tuning framework that restores safety while maintaining downstream performance. SafeMERGE selectively merges fine-tuned with safety-aligned model layers only when they deviate from safe behavior, measured by a cosine similarity criterion. Across four LLMs and several tasks, SafeMERGE consistently reduces harmful outputs compared to other defenses, with negligible or even positive impact on utility. Our results demonstrate that selective, layer-wise merging offers a robust safeguard against the inadvertent loss of safety during fine-tuning, establishing SafeMERGE as a simple yet effective post-fine-tuning defense.
</div>

<br>

<p align="center">
  <img src="safeMERGE.png" alt="SafeMERGE" width="70%">
  <br>
  <em>SafeMERGE: merges harmful and safe LoRAs if the layers deviate from safe behavior, measured by a projection-based cosine similarity.</em>
</p>

---

## 🔎 Overview

The key idea is to use a projection matrix to measure how much a finetuned adapter's weights deviate from a safe reference. If the cosine similarity between the projected finetuned weights and the original weights falls below a specified threshold, a partial merge is applied (e.g., using weights `[0.8, 0.2]` for the finetuned and safe adapters, respectively). Otherwise, the finetuned adapter is used without adjustment.

---

## 📁 Files

- **`utils.py`**  
  Contains helper functions to compute projection matrices and cosine similarity between LoRA weight differences, and defines the `SafeLoRAMerger` class which encapsulates the merging logic. Merging is done 1:1 as in PEFT! 

- **`get_safemerge_model.py`**  
  A simple command-line script that computes the SafeMERGE model and saves it to an output directory.

---

## ⚙️ Requirements

Tested with Python 3.11.4 and PyTorch 2.4.1 + CUDA 12.1, which can be installed via:
```bash
  pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

Install the repository requirements via:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Example

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
3. Compute the safety subspace from the specified unaligned and aligned models.
4. Merge the adapters based on the cosine similarity threshold (using partial merging if the similarity is below 0.35).
5. Save the final SafeMERGE model in the specified output directory.

---

### 📝 Implementational Note: Why Qwen Models Are Handled Differently
Qwen models include additional parameters, such as biases, that are not part of the LoRA layers. These extra parameters often have shapes that do not match the expected 2D structure used for LoRA projections (e.g., 1D biases). As a result, when a Qwen model is detected, the code **skips non-2D parameters** to ensure that only valid 2D LoRA parameters are processed during projection.

---

## 📚 Citation

If you find SafeMERGE useful in your research, please cite:

```bibtex
@inproceedings{Djuhera_2026,
  title={SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging},
  url={http://dx.doi.org/10.18653/v1/2026.findings-acl.1761},
  DOI={10.18653/v1/2026.findings-acl.1761},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  publisher={Association for Computational Linguistics},
  author={Djuhera, Aladin and Kadhe, Swanand Ravindra and Ahmed, Farhan and Zawad, Syed and Boche, Holger},
  year={2026},
  pages={35316–35335}
}
```
