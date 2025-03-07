# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/mt_bench/question.jsonl

import copy
import json

import torch
from torch.utils.data import Dataset


# PROMPT_TEMPLATE = (
#     f"{{question}}\n---\nYour Answser:\n"
# )

PROMPT_TEMPLATE = (
    f"""
    [INST] <<SYS>> 
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
    Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
    Please ensure that your responses are socially unbiased and positive in nature.\n 
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
    If you don't know the answer to a question, please don't share false information. <</SYS>> \n
    {{question}}
    [/INST]
    """
)


class mt_bench_dataset(Dataset):    
    def __init__(self, dataset_path, tokenizer):
        self.ann = json.load(open(dataset_path))
        self.tokenizer = tokenizer
        

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        prompt =  self.tokenizer.encode(self.tokenizer.bos_token + PROMPT_TEMPLATE.format(question=ann["turns"]), add_special_tokens=False)
        if ann.get("reference", "") == "":
            reference = self.tokenizer.encode("NO Reference", add_special_tokens=False)   
        else:
            reference = self.tokenizer.encode("NO Reference", add_special_tokens=False)     
            # reference = self.tokenizer.encode(ann["reference"] +  self.tokenizer.eos_token, add_special_tokens=False)        
            
    
        example = torch.tensor(prompt, dtype=torch.int64)
        labels = torch.tensor(reference, dtype=torch.int64)
        example_mask = example.ge(0)
        example[~example_mask] = 0           

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }               

