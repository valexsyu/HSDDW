export OMP_NUM_THREADS=32
# export TORCH_DISTRIBUTED_DEBUG=INFO
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 /work/valex1377/LLMSpeculativeSampling/scripts/mbert/mbert_train.py

# export OMP_NUM_THREADS=16
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 /work/valex1377/LLMSpeculativeSampling/scripts/mbert/mbert_train.py

# export OMP_NUM_THREADS=8
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 /work/valex1377/LLMSpeculativeSampling/scripts/mbert/mbert_train.py

#  CUDA_VISIBLE_DEVICES=0 python /work/valex1377/LLMSpeculativeSampling/scripts/mbert/mbert_train.py