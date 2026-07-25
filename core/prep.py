import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_tokenizer_and_model(model_id: str, device=device, dtype=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model

def load_tokenizer_and_model_peft(model_id: str, checkpoint: str, device=device, dtype=None):
    tokenizer, model = load_tokenizer_and_model(model_id, device, dtype)
    # Load the GFlowNet LoRA adapter and permanently merge it into the base weights
    peft_model = PeftModel.from_pretrained(model, checkpoint)
    model = peft_model.merge_and_unload()
    model.to(device)
    model.eval()
    return tokenizer, model

def get_embedding_matrix(model):
    """Returns the detached embedding matrix (V, D)"""
    emb_layer = model.get_input_embeddings()
    return emb_layer.weight.detach()

def project_to_vocab_by_l2(s: torch.FloatTensor, emb_matrix: torch.FloatTensor):
    """
    Projects continuous embedding(s) to the nearest token using L2 distance.
    s: (..., D) -> natively handles (M, D)
    """
    D = emb_matrix.size(1)
    original_shape = s.shape[:-1] 
    s_flat = s.view(-1, D)         

    diff = s_flat[:, None, :] - emb_matrix[None, :, :]  
    dist2 = torch.sum(diff ** 2, dim=-1)                
    top_idx = dist2.argmin(dim=-1)                      

    return top_idx.view(*original_shape)

def joint_log_prob_from_inputs_embeds(model, inputs_embeds, target_ids):
    """Calculates the negative NLL (Log Joint Probability) of the sequence."""
    outputs = model(inputs_embeds=inputs_embeds, return_dict=True)
    logits = outputs.logits[:, :-1, :].contiguous()
    targets = target_ids[:, 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
    vocab_size = logits.shape[-1]
    return -loss_fct(logits.view(-1, vocab_size), targets.view(-1))