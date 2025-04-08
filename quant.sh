CUDA_VISIBLE_DEVICES=1 python main.py \
--model_path meta-llama/Llama-2-7b-hf \
--output_dir ./log/llama-2-7b-w4a4kv4 \
--wbits 4 \
--input_bits 4 \
--input_mode static \
--v_bits 4 \
--k_bits 4 \
--q_bits 4 \
--a_bits 16 \
--kv_group_size 128 \
--kv_mode static \
--mse_init \
--set_prefixed_tokens \
--eval_ppl \
--save_quant_dir ./pre_quantized_models/llama-2-7b-w4a4kv4 \
--calib_dataset wikitext2 \
--pre_rotate \
--qk_online_had \
--down_online_had \
# --eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande \

# CUDA_VISIBLE_DEVICES=2 python main.py \
# --model_path EleutherAI/pythia-6.9b \
# --output_dir ./log/pythia-6.9b-w4a4kv4 \
# --eval_ppl \
# --calib_dataset wikitext2 \
# --save_quant_dir ./pre_quantized_models/pythia-6.9b-w4a4kv4 \
# --wbits 4 \
# --input_bits 4 \
# --input_mode static \
# --v_bits 4 \
# --k_bits 4 \
# --q_bits 4 \
# --a_bits 16 \
# --kv_group_size 128 \
# --kv_mode static \
# --mse_init \
# --set_prefixed_tokens \
# --pre_rotate \
# --qk_online_had \
# --down_online_had \
# --outlier_threshold 3
# # # --eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande \