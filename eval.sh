CUDA_VISIBLE_DEVICES=1 python eval.py \
--quant_model_path ./pre_quantized_models/llama-2-7b-w4a4kv4 \
--ppl_seqlen 1024 \
--eval_ppl \
--spike \
# --eval_tasks winogrande \
# --eval_batch_size 8 \
