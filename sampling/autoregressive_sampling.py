import torch

from tqdm import tqdm
from sampling.utils import norm_logits, sample






import os
import sys
from transformers import AutoTokenizer
import time

COLORS = [
    "\033[33m",  # 黄色
    "\033[34m",  # 蓝色
    "\033[31m",  # 红色
    "\033[32m",  # 绿色
    "\033[33m",  # 黄色
    "\033[35m",  # 紫色
    "\033[36m",  # 青色
]

WHITE = "\033[37m"  # 白色
RESET = "\033[0m"   # 重置颜色

def clear_and_print(tokens, pos ,color,tokenizer,sleep=0.5):
    # 清屏并移动光标到左上角
    sys.stdout.write("\033[2J\033[H")
    # 分段控制颜色：前10字符白色，之后动态颜色
    prefix = f"{WHITE}{tokenizer.decode(tokens[:pos])}{RESET}"  # 白色部分
    colored_suffix = f"{color}{tokenizer.decode(tokens[pos:])}{RESET}"  # 动态颜色部分
    sys.stdout.write(prefix + colored_suffix + "\n")
    # sys.stdout.flush()
    time.sleep(sleep)
RECORD_STEP=True






@torch.no_grad()
def autoregressive_sampling(x : torch.Tensor, model : torch.nn.Module, max_len : int, random_seed : int = None,
                            temperature : float = 1, top_k : int = 0, top_p : float = 0):
    n = len(x)
    T = len(x) + max_len
    if random_seed:
        torch.manual_seed(random_seed)
    past_key_values = None
    while n < T:
        # outputs = model(x)
        if past_key_values:
            last_ids = x[:, -1]
            if last_ids.dim() == 1:
                last_ids = torch.unsqueeze(last_ids, 0)
            outputs = model(last_ids, past_key_values = past_key_values, use_cache = True)
        else:
            outputs = model(x)
        last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p)
        past_key_values = outputs.past_key_values
        idx_next = sample(last_p)
        x = torch.cat((x, idx_next), dim=1)
        n += 1
    return x


@torch.no_grad()
def autoregressive_sampling_with_eos(x : torch.Tensor, model : torch.nn.Module, max_len : int, random_seed : int = None,
                            temperature : float = 1, top_k : int = 0, top_p : float = 0, eos_token_id : int = None,decoder=None):
    n = x.size(1)
    T = x.size(1) + max_len

    if random_seed:
        torch.manual_seed(random_seed)



    if RECORD_STEP:
        tokenizer = AutoTokenizer.from_pretrained("/work/valex1377/LLMSpeculativeSampling/llama_model/llama-2-70b-chat") 
        breakpoint()
        time.sleep(5)


    past_key_values = None
    seq_len = n
    samll_step=0
    while n < T:
        
        
        if RECORD_STEP:
            if n == seq_len :
                clear_and_print(x[0],n ,COLORS[0],tokenizer,sleep=0) 
                        
        
        # outputs = model(x)
        if past_key_values:
            last_ids = x[:, -1]
            if last_ids.dim() == 1:
                last_ids = torch.unsqueeze(last_ids, 0)
            outputs = model(last_ids, past_key_values = past_key_values, use_cache = True)
        else:
            outputs = model(x)
        last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p)
        past_key_values = outputs.past_key_values
        idx_next = sample(last_p)
        x = torch.cat((x, idx_next), dim=1)
        n += 1
        
        if RECORD_STEP:
            samll_step=samll_step+1
            # if samll_step > 1 :
            #     clear_and_print(x[0],n-2 ,COLORS[0],tokenizer,sleep=0)           
            #     samll_step=0
            clear_and_print(x[0],n ,COLORS[0],tokenizer,sleep=0) 
            
        
        
        
        
        if eos_token_id is not None :
            if eos_token_id in x[0] : ## only support batch = 1
                # find the indices where x is equal to eos_token
                eos_indices = torch.eq(x[0], eos_token_id).nonzero(as_tuple=True)[0]

                if eos_indices.nelement() > 0:
                    # get the index of the first occurrence of eos_token
                    first_eos_index = eos_indices[0]

                    # select the elements in x before the first eos_token
                    x = x[:,:first_eos_index]             
                
                break        
    return x[:,seq_len:]