import torch
import numpy as np
from transformers import AutoModelForCausalLM

import torch
from dataclasses import replace
from typing import Literal
from peft.utils import (
    _freeze_adapter,
    _get_submodules,
)
from peft.tuners.lora.model import LoraModel
from peft.tuners.lora.layer import LoraLayer
from peft import PeftModel


def compute_safelora_projection_matrices(
    base_model_path: str,
    aligned_model_path: str,
    target_modules: list[str],
    device: str = "cuda"
):
    """
    Given a base (unaligned) model and an aligned (safe) model, compute the
    SafeLoRA projection matrices for each parameter that matches target_modules.
    Returns a list of 2D matrices, each one is V V^T / ||V||, stored in the order
    that we encounter matching parameters. Different implementations for Llama and Qwen models.
    """
    # 1. Load base and aligned models
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, device_map="cpu", load_in_8bit=False, low_cpu_mem_usage=True
    )
    aligned_model = AutoModelForCausalLM.from_pretrained(
        aligned_model_path, device_map="cpu", load_in_8bit=False, low_cpu_mem_usage=True
    )

    proj_matrices = []

    # 2. Iterate over base vs. aligned parameters

    # Qwen models have biases which are not part of LoRA layers and therefore need to be ignored for projection! 
    # Thus, we check if either path has "Qwen" to decide whether to skip non-2D weights.
    is_qwen = ("Qwen" in base_model_path) or ("Qwen" in aligned_model_path)
    if is_qwen:
        print("Detected Qwen model. Will skip non-2D weights (like biases).")
    else:
        print("Detected LLaMA or other model. Will project all matched shape params.")

    # Iterate over parameters from base and aligned models
    for (b_name, b_param), (a_name, a_param) in zip(
        base_model.named_parameters(),
        aligned_model.named_parameters()
    ):
        # Only process parameters that correspond to LoRA layers (e.g., `q_proj.lora_A`) in target modules.
        if any(mod in a_name for mod in target_modules):

            # Must match shape
            if b_param.shape != a_param.shape:
                continue

            if is_qwen:
                # For Qwen: skip anything not 2D (like 1D biases)
                if a_param.dim() != 2:
                    # e.g. skipping q_proj.bias
                    continue

            # Compute the difference `vec` between the aligned and base model weights representing the change due to alignment       
            vec = (a_param - b_param).to(device)
            fro_norm = torch.norm(vec, p="fro")
            if fro_norm < 1e-9:
                # no difference => store zero or identity
                C = torch.zeros(vec.shape[0], vec.shape[0], device=device)
            else:
                C = torch.mm(vec, vec.t()) / fro_norm

            proj_matrices.append(C.cpu().detach())

    return proj_matrices

def compute_safelora_cos(
    P: torch.Tensor,    # projection matrix from base vs aligned difference
    loraA_ft: torch.Tensor,
    loraB_ft: torch.Tensor,
    device: str = "cuda"
) -> float:
    """
    Reproduce the SafeLoRA style measure:
      - Project the finetuned B matrix with P
      - Multiply by A to get the final projected LoRA
      - Compare with original A x B
      - Return cosine similarity
    """
    # Move to device
    P = P.to(device)
    A_ft = loraA_ft.to(device)
    B_ft = loraB_ft.to(device)

    # 1. Project B_ft => W_proj = P * B_ft
    W_proj = torch.mm(P, B_ft)

    # 2. fW_proj = W_proj x A_ft
    #    (We must confirm the shape is correct: typically LoRA_B is [r, out_dim],
    #     LoRA_A is [in_dim, r], so the multiplication order might need transposition,
    #     depending on how the LoRA is shaped.)
    fW_proj = torch.mm(W_proj, A_ft)

    # 3. Original fW = B_ft x A_ft
    fW_ori  = torch.mm(B_ft, A_ft)

    # 4. Flatten and compute cos sim
    # SafeLoRA does cos(sim) of shape (1,-1)
    cos_val = torch.nn.functional.cosine_similarity(
        fW_proj.view(1, -1),
        fW_ori.view(1, -1),
        dim=1
    ).item()

    return float(np.round(cos_val, 5))

class SafeLoRAMerger:
    """
    Mixes the idea of SafeLoRA's layer-wise projection-based
    "alignment measure" with standard PEFT merges ("linear", "dare_linear", etc.).
    """
    def __init__(
        self,
        peft_model: PeftModel,
        base_model_path: str,
        aligned_model_path: str,
        threshold: float = 0.35,
        default_merge_ratio: float = 0.2,
        device: str = "cuda",
    ):
        """
        :param peft_model: The model that already has (e.g.) 2 adapters loaded:
                           - a finetuned adapter
                           - a "safe" adapter
        :param base_model_path: Path to unaligned base model (for SafeLoRA).
        :param aligned_model_path: Path to aligned model (for SafeLoRA).
        :param threshold: Cosine similarity threshold. If below => do partial merge.
        :param default_merge_ratio: The fraction for the "safe" adapter if below threshold
                                    (So final weights might be [0.8, 0.2]).
        :param device: "cuda" or "cpu"
        """
        self.peft_model = peft_model
        self.threshold = threshold
        self.default_merge_ratio = default_merge_ratio
        self.device = device

        # We retrieve which modules are "target_modules" from the first peft_config
        self.peft_config = peft_model.peft_config

        # We assume the first adapter has the main config
        first_key = list(self.peft_config.keys())[0]
        target_modules = list(self.peft_config[first_key].target_modules)

        # 1. Compute the SafeLoRA projection matrices
        self.proj_matrices = compute_safelora_projection_matrices(
            base_model_path=base_model_path,
            aligned_model_path=aligned_model_path,
            target_modules=target_modules,
            device=device
        )

        # We'll keep an index to match each LoRA-labeled parameter
        self.proj_index = 0

    def add_weighted_adapter_safelora(
        self,
        adapters: list[str],
        weights: list[float],
        adapter_name: str,
        combination_type: str = "linear",
        density: float | None = None,
        majority_sign_method: Literal["total", "frequency"] = "total",
    ) -> None:
        """
        Replaces add_weighted_adapter. For each LoRA layer:
        1. Compute SafeLoRA cosine measure.
        2. If cos < threshold => partial merge [1 - default_merge_ratio, default_merge_ratio].
        3. Else => keep finetuned only [1.0, 0.0].
        4. Then call _generalized_task_arithmetic_weighted_adapter with that local weighting.

        Only supports "linear" or "dare_linear" in this implementation!
        """

        # Step A: set up the new adapter's config
        if adapter_name in self.peft_config:
            return

        comb_type, new_rank, new_target_modules = self.peft_model._check_add_weighted_adapter(
            adapters=adapters,
            combination_type=combination_type,
            svd_rank=None,
        )

        # Create new config
        self.peft_model.peft_config[adapter_name] = replace(
            self.peft_model.peft_config[adapters[0]],
            r=new_rank,
            lora_alpha=new_rank,
            target_modules=new_target_modules,
        )
        self.peft_model.inject_adapter(self.peft_model.model, adapter_name)
        _freeze_adapter(self.peft_model.model, adapter_name)

        # We'll track how many got "merged" vs. how many kept finetuned,
        # and also record the keys (names) of the processed layers.
        merged_count = 0
        total_count  = 0
        processed_keys = []
        merged_keys = []

        # Step B: iterate over each LoraLayer
        key_list = [key for key, _ in self.peft_model.model.named_modules()
                    if self.peft_model.prefix not in key]

        for key in key_list:
            _, target, _ = _get_submodules(self.peft_model.model, key)
            if not isinstance(target, LoraLayer):
                continue

            total_count += 1
            processed_keys.append(key)  # record the key of the processed layer

            # The new weighted adapter's LoRA A/B
            if adapter_name in target.lora_A:
                target_lora_A = target.lora_A[adapter_name].weight
                target_lora_B = target.lora_B[adapter_name].weight
            elif adapter_name in target.lora_embedding_A:
                target_lora_A = target.lora_embedding_A[adapter_name]
                target_lora_B = target.lora_embedding_B[adapter_name]
            else:
                continue

            # Zero them out first.
            target_lora_A.data.zero_()
            target_lora_B.data.zero_()

            # We only implement for "linear"/"dare_linear".
            if combination_type not in ["linear", "dare_linear"]:
                continue

            # Expect exactly 2 adapters: [finetuned_name, safe_name]
            if len(adapters) != 2:
                raise ValueError("For threshold-based merges, pass exactly two adapters (finetuned & safe).")

            ft_name, safe_name = adapters

            # We read the "base" ratio from the user (e.g. [0.8, 0.2]) but we
            # can override for each layer if cos < threshold or not:
            ft_weight, safe_weight = weights  # e.g. [0.8, 0.2]

            # Retrieve the LoRA A/B for the finetuned adapter
            if ft_name in target.lora_A:
                loraA_ft = target.lora_A[ft_name].weight
                loraB_ft = target.lora_B[ft_name].weight
            else:
                continue

            # And for the safe adapter
            if safe_name in target.lora_A:
                loraA_safe = target.lora_A[safe_name].weight
                loraB_safe = target.lora_B[safe_name].weight
            else:
                continue

            # (1) Compute SafeLoRA cosine similarity with the next projection matrix
            P = self.proj_matrices[self.proj_index]
            cos_val = compute_safelora_cos(P, loraA_ft, loraB_ft, device=self.device)
            self.proj_index += 1

            # Decide local weights
            if cos_val < self.threshold:
                # They differ => partial merge => we do the user's ratio, e.g. [0.8,0.2]
                local_weights = [ft_weight, safe_weight]
                merged_count += 1
                merged_keys.append(key)  # record the key for a merged layer
            else:
                # They are "aligned enough" => keep finetuned only => [1.0, 0.0]
                local_weights = [1.0, 0.0]

            # (2) Call the standard PEFT merging logic for this layer.
            local_deltas = self.peft_model._generalized_task_arithmetic_weighted_adapter(
                combination_type,
                [ft_name, safe_name],
                local_weights,
                target,
                density,
                majority_sign_method
            )
            # This returns [loraA_merged, loraB_merged]
            newA, newB = local_deltas
            target_lora_A.data.copy_(newA)
            target_lora_B.data.copy_(newB)

        print(f"[SafeLoRAMerger] Processed {total_count} LoRA layers:")
        print(f"[SafeLoRAMerger] Merged {merged_count} layers (cos < {self.threshold}):")
        for k in merged_keys:
            print(f"  - {k}")