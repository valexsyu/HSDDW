root=/work/valex1377/LLMSpeculativeSampling
DATASET_NAME='Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1'
TOKEN_PATH=$root/datasets/mbert_ft
TOKEN_FILE_NAME=Llama-3-Magpie-Pro-1M-v0.1-merge.json
MODEL_NAME=bert-base-multilingual-cased
mkdir $TOKEN_PATH


echo "processing"
python transform_tokenize.py --dataset_name $DATASET_NAME --output $TOKEN_PATH/$TOKEN_FILE_NAME --pretrained_model $MODEL_NAME



wget -O $TOKEN_PATH/vocab.txt https://huggingface.co/google-bert/bert-base-multilingual-cased/raw/main/vocab.txt



# fairseq-preprocess --trainpref $TOKEN_PATH/train  --task masked_lm\
# --destdir ${TOKEN_PATH}/databin --srcdict $TOKEN_PATH/vocab.txt \
# --tgtdict $TOKEN_PATH/vocab.txt --workers 25 \