import torch
from transformers import BertTokenizer, BertForMaskedLM, PreTrainedTokenizerFast
import warnings
import random

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=r".*parameter name that contains `beta`.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*parameter name that contains `gamma`.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*Some weights of the model checkpoint.*")

# model_name='bert-large-cased-whole-word-masking'
model_name='bert-base-multilingual-cased'
# Load pre-trained model tokenizer (vocabulary)
# tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
# Load pre-trained model for masked language modeling
# model = BertForMaskedLM.from_pretrained(model_name)
# model = BertForMaskedLM.from_pretrained('/work/valex1377/LLMSpeculativeSampling/checkpoints/n3_e20_magpie_ft')
# model = BertForMaskedLM.from_pretrained('/work/valex1377/LLMSpeculativeSampling/checkpoints/n3_e20_wocls_magpie_ft')
# model = BertForMaskedLM.from_pretrained('/work/valex1377/LLMSpeculativeSampling/checkpoints/n3_e20_wocls_rdmask_magpie_ft')
# model = BertForMaskedLM.from_pretrained('/work/valex1377/LLMSpeculativeSampling/checkpoints/n3_e20_wocls_bertmask_magpie_ft')
# model = BertForMaskedLM.from_pretrained('/work/valex1377/LLMSpeculativeSampling/checkpoints/n5_e20_wocls_magpie_ft')
model = BertForMaskedLM.from_pretrained('/work/valex1377/LLMSpeculativeSampling/checkpoints/n7_e20_wocls_magpie_ft')

# Text with multiple [MASK] tokens
# question = "[CLS] Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
question = "[CLS] Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions "
# question = "[CLS] Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see "
# question = "[CLS] Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and "
# question = "[CLS] Compose an engaging travel blog post about a recent trip to Hawaii, highlighting "
# question = "[CLS] Compose an engaging travel blog [MASK] [MASK] [MASK] recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
# question = "[CLS] Imagine you are participating in a race with a group of people. If you have just overtaken the second person, what’s your current position? Where is the person you just overtook? [SEP]"
# question = "[CLS] Compose an engaging travel blog post about a recent trip to Hawaii, highlighting [MASK] [MASK] [MASK] [MASK] [MASK]"
# question = "[CLS] Compose an engaging travel blog post [MASK] a recent trip to Hawaii, highlighting "
# question = "[CLS] Compose an engaging travel blog post about a recent trip to Hawaii, highlighting "
# question = "[CLS] Compose an engaging travel blog post [MASK] a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
print("Input Text",question)
max_len=512
for i in range(1,9):
    mask = " [MASK]" * i
    text = question + mask
    # text = question


    tokenized = tokenizer.tokenize(text)


    # Encode the text
    encoded_input = tokenizer(text, return_tensors='pt', add_special_tokens=False)

    # Perform a forward pass to predict the masked tokens
    with torch.no_grad():
        output = model(**encoded_input)

    # Extract the logits for all masked token positions
    mask_token_indices = (encoded_input.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    predicted_tokens = []
    for mask_token_index in mask_token_indices:
        mask_token_logits = output.logits[0, mask_token_index, :]
        predicted_token_id = torch.argmax(mask_token_logits, dim=-1)
        predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id.item())
        predicted_tokens.append(predicted_token)

    # Properly join the subword tokens to form complete words
    predicted_tokens = [token.replace('##', '') if token.startswith('##') else token for token in predicted_tokens]

    # Replace the [MASK] tokens in the original text with the predicted tokens
    for predicted_token in predicted_tokens:
        text = text.replace('[MASK]', predicted_token, 1)

    # print("Original text:", question + mask)
    # print("Original text:", question)
    print("Output token:", predicted_tokens)
    # print("Predicted text:", text)
