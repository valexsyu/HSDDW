
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertForMaskedLM, AdamW, BertForPreTraining
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from transformers import BertConfig
import wandb
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm 
from multiprocessing import Pool
from joblib import Parallel, delayed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os
import logging
from typing import Any


# Set the number of tokens to mask and max length
n = 5  # This can be adjusted
debug = False
# Training loop
epochs = 2 if debug else 20
batch_size = 15 # 1G=20
use_CLS = False
mask_policy = 'random_start_continue_mask'
model_name='bert-base-multilingual-cased'
dataset_name='Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1'
if debug :
    save_name='debug'
else:
    # Save the model to a directory
    # save_name='n3_e20_magpie_ft'
    # save_name='n_test'
    # save_name='n3_e20_wocls_magpie_ft'       
    # save_name='n3_e20_wocls_rdmask_magpie_ft'
    # save_name='n3_e20_wocls_bertmask_magpie_ft'    
    save_name='n5_e20_wocls_magpie_ft'
    # save_name='n7_e20_wocls_magpie_ft'
model_save_path = os.path.join('checkpoints',save_name)



# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load the configuration for the model_name model
config = BertConfig.from_pretrained(model_name)

# Access the max_position_embeddings parameter
max_len = config.max_position_embeddings
if torch.cuda.device_count() > 1:
    use_DDP=True
else:   
    use_DDP=False

    





def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.barrier()
    dist.get_world_size()
 

def cleanup():
    dist.destroy_process_group()
    



if use_DDP :
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    setup(rank, world_size)
else:
    rank = 0


device = torch.device(f'cuda:{rank}')
if rank == 0:
    if debug :
        wandb.init(mode="disabled")
    else:
        # Initialize a new run
        wandb.init(project="mbert_finetuning", name=save_name, config={
            "learning_rate": 5e-5,
            "epochs": epochs,
            "batch_size": batch_size,
            "max_len": max_len,
            "n_masked_tokens": n,
        })

    # Access configuration parameters
    config = wandb.config
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()    
    print(f"GPU Number:{torch.cuda.device_count()}")



from transformers.utils.doc import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.bert.modeling_bert import BERT_INPUTS_DOCSTRING, BertForPreTrainingOutput, _CONFIG_FOR_DOC
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class BertNoNSLForPreTraining(BertForPreTraining):
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertForPreTrainingOutput]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
                pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BertForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> model = BertForPreTraining.from_pretrained("bert-base-uncased")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None :
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            # next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            # 
            total_loss = masked_lm_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    




class RandomMaskedDataset(Dataset):
    def __init__(self, dataset, tokenizer, n, max_len=128, batch_size=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.n = n
        self.max_len = max_len
        self.batch_size = batch_size
        self.mlm_probability: float = 0.15

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['input'] + " "+ self.dataset[idx]['output']
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        num_tokens = len(token_ids)
        
        # Randomly choose a starting point and length for the input
        if num_tokens < self.n:
            start_idx = 0
            self.n = num_tokens
        else:
            start_idx = random.randint(0, num_tokens-self.n)

        if use_CLS :
            available_len = random.randint(self.n, min(self.max_len - 2, num_tokens-start_idx))
        else:
            available_len = random.randint(self.n, min(self.max_len - 1, num_tokens-start_idx))
        end_idx = start_idx + available_len
        
        selected_token_ids = token_ids[start_idx:end_idx]
            
        selected_token_len = len(selected_token_ids)
        masked_token_ids = selected_token_ids.copy()
        # Labels are the same as input_ids but with the non-masked tokens set to -100 to ignore them during loss computation
        # labels = [-100] * self.max_len
        labels = [-100] * selected_token_len
        
        
        if mask_policy == "random_start_continue_mask" :
            #=======================================================================
            ## random start point for mask
            # Randomly choose a starting point and length for the input
            if num_tokens < self.n:
                start_mask_idx = 0
                self.n = num_tokens
            else:
                start_mask_idx = random.randint(0, selected_token_len-self.n) 
                
            # Mask the last n tokens
            for i in range(start_mask_idx, start_mask_idx + self.n):
                masked_token_ids[i] = self.tokenizer.mask_token_id
                labels[i] = selected_token_ids[i]   
        
        elif mask_policy == "last_contiune_mask":        
            #========================================================================              
            
            # # Mask the last n tokens
            for i in range(selected_token_len - self.n, selected_token_len):
                masked_token_ids[i] = self.tokenizer.mask_token_id
                labels[i] = selected_token_ids[i]
            #==========================================================================
        elif mask_policy == "bert_mask":
            masked_token_ids, labels = self.torch_mask_tokens(torch.tensor(masked_token_ids))
            
        
        
        
        if use_CLS :
            # Add [CLS] and [SEP] tokens
            input_ids = [self.tokenizer.cls_token_id] + masked_token_ids + [self.tokenizer.sep_token_id]
            labels = [-100] + labels +[-100]
        else:
            # Add [CLS] tokens
            input_ids = [self.tokenizer.cls_token_id] + masked_token_ids
            labels = [-100] + labels
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)

        # Pad the input and attention mask to max_len
        padding_len = self.max_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_len
        labels += [-100] * padding_len
        attention_mask += [0] * padding_len


        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels)
    
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # if special_tokens_mask is None:
        #     special_tokens_mask = [
        #         self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        #     ]
        #     special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        # else:
        #     special_tokens_mask = special_tokens_mask.bool()

        # probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs.tolist(), labels.tolist()    

# Load dataset from Hugging Face
dataset = load_dataset(dataset_name)

# Initialize the custom dataset with the last n tokens masked
train_dataset = RandomMaskedDataset(dataset['train'], tokenizer, n, max_len)
if use_DDP :
    train_sampler = DistributedSampler(train_dataset)
    # Create a DataLoader
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, pin_memory=False, prefetch_factor=2, num_workers=4)
else:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)




# # Load pre-trained mBERT model for masked language modeling
# model = BertForMaskedLM.from_pretrained(model_name)

# model = AutoModelForCausalLM.from_pretrained(model_name, 
#                                              trust_remote_code=True)

model = BertNoNSLForPreTraining.from_pretrained(model_name, 
                                             trust_remote_code=True)



# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda",rank)
model = model.to(device)
print(f"Using device: {device}")
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
# Move model to the GPU

if use_DDP :
    model = DDP(model, device_ids=[rank],find_unused_parameters=True)




# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)


# Set model to training mode
model.train()

# Initialize GradScaler for mixed precision training
scaler = GradScaler()

for epoch in range(epochs):
    total_loss = 0
    step = 0

    if use_DDP :
        # Set the epoch for the DistributedSampler to shuffle data
        train_sampler.set_epoch(epoch)
    
    # Initialize tqdm only on rank 0
    if rank == 0:
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    else:
        progress_bar = train_loader  # Dummy progress bar for other ranks

    for batch in progress_bar:    

    # for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
        optimizer.zero_grad()

        # Move batch data to the appropriate device
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        with autocast():
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Take the mean of the loss across all GPUs (DDP handles this automatically)
            # Note: Loss is automatically averaged in DDP, so you don't need to manually reduce it

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Zero the gradients for the next iteration
        optimizer.zero_grad()


        
        # Log the loss to WandB
        if rank == 0:
            wandb.log({"batch_loss": loss.item()})
            # Optionally, accumulate the total loss
            total_loss += loss.item()            
        step += 1
        if step % 100 == 0:
            # Only log from the master process (rank 0)
            if rank == 0:
                logger.info(f"batch_loss:{loss.item()}")
            if debug:
                break
    
    # After training, save the model
    if rank == 0:
        if use_DDP :
            # Save the model
            model_to_save = model.module  # Access the original model
            model_to_save.save_pretrained(model_save_path)

            # Save the optimizer state (optional)
            torch.save(optimizer.state_dict(), os.path.join(model_save_path, 'optimizer.pt'))
        else:
            model.save_pretrained(model_save_path)
            torch.save(optimizer.state_dict(), os.path.join(model_save_path, 'optimizer.pt'))
        # model.save_pretrained(model_save_path)
        # Save the tokenizer to the same directory
        tokenizer.save_pretrained(model_save_path)   
        progress_bar.close()  # Close the progress bar only for rank 0
    if use_DDP :
        dist.barrier()        
        
        
    
    # Log the epoch loss to wandb
    if rank == 0:
        avg_loss = total_loss / len(train_loader)
        wandb.log({"epoch_loss": avg_loss})
        logger.info(f"Epoch: {epoch+1}, Loss: {avg_loss}")    
    
if rank == 0:
    print("Training complete.")   
    if not debug :
        # Log the saved model to wandb
        wandb.save(model_save_path)
        wandb.finish()     
if use_DDP :
    cleanup()






