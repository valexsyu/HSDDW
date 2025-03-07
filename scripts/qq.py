import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Function to freeze attention layers
def freeze_all_except_lm_head(model):
    for name, param in model.named_parameters():
        if name != 'lm_head.weight':
            param.requires_grad = False
        else:
            param.requires_grad = True
            print(name)
            
            
target_model_name = '/work/valex1377/LLMSpeculativeSampling/llama_model/llama-68m'
# target_model_name = '/work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-7b-chat'
large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                    torch_dtype=torch.float16,
                                                    device_map="auto",
                                                    trust_remote_code=True)              

freeze_all_except_lm_head(large_model)            