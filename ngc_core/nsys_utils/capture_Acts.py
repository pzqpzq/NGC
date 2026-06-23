

import torch
from torch import Tensor
import torch.nn as nn
from typing import Dict, Iterable, Tuple, Set, Callable, List
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------- util funcs ---------------------------------------------------
def _mlp_prefix(pattern: str) -> str:
    if ".mlp." not in pattern: raise ValueError(f"pattern 缺少 '.mlp.': {pattern}")
    return pattern.split(".mlp.")[0] + ".mlp"

def _attn_prefix(pattern: str) -> str:
    if ".self_attn." in pattern: return pattern.split(".self_attn.")[0] + ".self_attn"
    if ".attn." in pattern: return pattern.split(".attn.")[0] + ".attn"
    raise ValueError(f"pattern 缺少 '.self_attn.'或'.attn.': {pattern}")

def gather_mlp_targets(patterns: Iterable[str]) -> Set[str]:
    return {_mlp_prefix(p) for p in patterns}

def gather_attn_targets(patterns: Iterable[str]) -> Set[str]:
    """Given a list of fully‑qualified module names, return the unique attention prefixes."""
    return {_attn_prefix(p) for p in patterns}

# ---------- main capture -------------------------------------------------
_TensorPair = Tuple[torch.Tensor, torch.Tensor]

def capture_llm_Acts(
    model: nn.Module,
    mlp_prefixes: Set[str],
    attn_prefixes: Set[str],
    tokenizer: AutoTokenizer,
    layer_types: List[str],
    prompt: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, _TensorPair]:

    records: Dict[str, _TensorPair] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def make_io_hook(name: str):
        def _hook(mod: nn.Module, inp, out):  # type: ignore[override]
            x_cpu = inp[0].detach().to(dtype).cpu()
            y_cpu = out.detach().to(dtype).cpu()
            records[name] = (x_cpu, y_cpu)
            del x_cpu, y_cpu
        return _hook

    wanted = set(layer_types)

    for module_name, module in model.named_modules():
        # ---- MLP sub‑modules ------------------------------------------------
        if module_name in mlp_prefixes:
            if "mlp.gate_proj" in wanted and hasattr(module, "gate_proj"):
                handles.append(module.gate_proj.register_forward_hook(
                    make_io_hook(f"{module_name}.gate_proj")))
            if "mlp.up_proj" in wanted and hasattr(module, "up_proj"):
                handles.append(module.up_proj.register_forward_hook(
                    make_io_hook(f"{module_name}.up_proj")))
            if "mlp.down_proj" in wanted and hasattr(module, "down_proj"):
                handles.append(module.down_proj.register_forward_hook(
                    make_io_hook(f"{module_name}.down_proj")))

        # ---- Attention sub‑modules -----------------------------------------
        if module_name in attn_prefixes:
            if "self_attn.q_proj" in wanted and hasattr(module, "q_proj"):
                handles.append(module.q_proj.register_forward_hook(
                    make_io_hook(f"{module_name}.q_proj")))
            if "self_attn.k_proj" in wanted and hasattr(module, "k_proj"):
                handles.append(module.k_proj.register_forward_hook(
                    make_io_hook(f"{module_name}.k_proj")))
            if "self_attn.v_proj" in wanted and hasattr(module, "v_proj"):
                handles.append(module.v_proj.register_forward_hook(
                    make_io_hook(f"{module_name}.v_proj")))
            if "self_attn.o_proj" in wanted and hasattr(module, "o_proj"):
                handles.append(module.o_proj.register_forward_hook(
                    make_io_hook(f"{module_name}.o_proj")))

    if not handles: raise ValueError("No sub‑modules matched — check your prefix lists and layer_types.")

    device = next(model.parameters()).device
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    model_dtype = next(model.parameters()).dtype  # keep forward consistent

    # Run the model
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=model_dtype): _ = model(**model_inputs)
    
    # Cleanup
    for h in handles: h.remove()
    torch.cuda.empty_cache()

    return records
