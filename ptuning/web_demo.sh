PRE_SEQ_LEN=128

CUDA_VISIBLE_DEVICES=0 python web_demo.py \
    --model_name_or_path ..\model \
    --ptuning_checkpoint ..\checkpoint-3000 \
    --pre_seq_len $PRE_SEQ_LEN

set PRE_SEQ_LEN=128
set CUDA_VISIBLE_DEVICES=0
python web_demo.py --model_name_or_path ..\model --ptuning_checkpoint ..\checkpoint-3000 --pre_seq_len 128
