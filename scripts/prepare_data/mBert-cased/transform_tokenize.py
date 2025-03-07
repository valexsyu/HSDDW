from transformers import AutoTokenizer, AutoModel, BertTokenizer
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
from datasets import load_dataset
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='input file')
    parser.add_argument('--output', type=str, required=True, help='output file')
    parser.add_argument('--pretrained_model', type=str, required=True, help='pretrained language model')  
    args = parser.parse_args()
    
    
    dataset = load_dataset(args.dataset_name)
    lines=[]
    for i in tqdm(range(len(dataset['train'])),desc='Merage Input and Output'):
        lines.append(dataset['train'][i]['input'] + " "+ dataset['train'][i]['output'])
 
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.pretrained_model)
    tokenized_ids = tokenizer.batch_encode_plus(lines, add_special_tokens=False)['input_ids']

    # Save the tokenized IDs to a JSON file
    with open(args.output, 'w') as f:
        json.dump(tokenized_ids, f)
    
if __name__ == "__main__":
  main()
  
  