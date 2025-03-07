from .mt_bench_dataset import mt_bench_dataset
from .sp_dataset import mt_bench_dataset, alpaca_dataset,gsm8k_dataset,humaneval_dataset

SPDATASETZOO={
    'mt_bench' : mt_bench_dataset,
    'mt_bench_multi' : mt_bench_dataset,
    'alpaca'   : alpaca_dataset,
    'gsm8k'    : gsm8k_dataset,
    'humaneval': humaneval_dataset,
    
}