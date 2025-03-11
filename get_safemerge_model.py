import argparse
import ast
import os

from transformers import AutoModelForCausalLM
from peft import PeftModel

from utils import SafeLoRAMerger

def get_safemerge(args):
    "Gets the SafeMERGE model."

    # (A) Load base model
    base_model_id = args.base_model_id
    model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")

    # (B) Load finetuned adapter
    finetuned_model_peft_id = args.finetuned_model_id
    finetuned_adapter_name = "finetuned"
    peft_model = PeftModel.from_pretrained(
        model,
        finetuned_model_peft_id,
        adapter_name=finetuned_adapter_name,
    )

    # (C) Load "safe" adapter
    safety_model_id = args.safety_model_id
    safety_adapter_name = "safe"
    peft_model.load_adapter(safety_model_id, adapter_name=safety_adapter_name)

    # (D) Create a SafeLoRAMerger, referencing some base vs aligned model for the projection calc
    #  (These can be any "base" and "aligned" that define the "safe subspace".)
    base_model_path_for_safelora = args.safelora_unaligned_model_id     # unaligned means unsafe, e.g. "meta-llama/Llama-2-7b-hf"
    aligned_model_path_for_safelora = args.safelora_aligned_model_id    # aligned means safe, e.g. "meta-llama/Llama-2-7b-chat-hf"

    merger = SafeLoRAMerger(
        peft_model=peft_model,
        base_model_path=base_model_path_for_safelora,
        aligned_model_path=aligned_model_path_for_safelora,
        threshold=args.cos_threshold,           # cos similarity threshold
        default_merge_ratio=args.default_merge_ratio,  # means if cos < threshold => [0.8, 0.2]
        device="cuda"
    )

    # (E) Add the weighted adapter using the SafeLoRA-based threshold logic
    weighted_adapter_name = "weighted"
    if args.default_merge_ratio is not None:
        print("Using default merge ratio")
    else:
        print("Using custom merge ratio: ", args.weighting)

    merger.add_weighted_adapter_safelora(
        adapters=[finetuned_adapter_name, safety_adapter_name],
        weights=[1.0 - merger.default_merge_ratio, merger.default_merge_ratio] if merger.default_merge_ratio is not None else args.weighting,
        adapter_name=weighted_adapter_name,
        combination_type=args.merge_type, # "linear" or "dare_linear"
        density=args.density
    )

    # (F) Activate the merged adapter, remove the old ones, and save
    peft_model.set_adapter(weighted_adapter_name)
    peft_model.delete_adapter(finetuned_adapter_name)
    peft_model.delete_adapter(safety_adapter_name)

    # (G) Save the model
    dataset_name = args.finetuned_model_id.split("-")[-2]
    safety_model_name = args.safety_model_id.split("-")[-2]
    peft_name = f"safemerge_{dataset_name}_SafetyModel_{safety_model_name}_threshold_{args.cos_threshold}_weighting_{args.weighting[0]}_{args.weighting[1]}_{args.merge_type}_density_{args.density}"
    save_path = os.path.join(args.output_path, peft_name)
    peft_model.save_pretrained(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model_id", type=str, required=True) # the base model for PEFT loading, e.g. "meta-llama/Llama-2-7b-chat-hf"
    parser.add_argument("--finetuned_model_id", type=str, required=True) # your finetuned model, e.g. "my_hf_repo/llama_2_7b_chat_hf_gsm8k"
    parser.add_argument("--safety_model_id", type=str, required=True) # your safety tuned model, e.g. "my_hf_repo/llama_2_7b_chat_hf_safety_tuned"
    parser.add_argument("--safelora_unaligned_model_id", type=str, required=True) # unaligned model, e.g. "meta-llama/Llama-2-7b-hf"
    parser.add_argument("--safelora_aligned_model_id", type=str, required=True) # aligned model, e.g. "meta-llama/Llama-2-7b-chat-hf"
    parser.add_argument("--cos_threshold", type=float, default=0.35, required=False)
    parser.add_argument("--default_merge_ratio", type=float, default=None, required=False)
    parser.add_argument("--weighting", type=lambda x: ast.literal_eval(x) if isinstance(x, str) else x, default=[0.8, 0.2], required=False)
    parser.add_argument("--merge_type", type=str, default="linear", required=False)
    parser.add_argument("--density", type=float, default=None, required=False)
    parser.add_argument("--output_path", type=str, default="./safemerge_models", required=False)

    args = parser.parse_args()

    get_safemerge(args)