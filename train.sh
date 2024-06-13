#accelerate launch ./bradley-terry-rm/mistral_rm.py \
#  --model_name mistralai/Mistral-7B-Instruct-v0.3 \
#  --max_length 4096 \
#  --train_set_path llm-jp/hh-rlhf-12k-ja \
#  --output_path ./models/mistral_rm_test \
#  --gradient_accumulation_steps 64

# LoRA
#python ./merge_lora.py --lora_model_path ./models/mistral_rm_test/last_checkpoint --merged_dir ./mistral_rm
#cp ./models/mistral_rm_test/last_checkpoint/tokenizer* ./mistral_rm/
#cp ./models/mistral_rm_test/last_checkpoint/special_tokens_map.json ./mistral_rm/

# full sft
#accelerate launch ./bradley-terry-rm/mistral_rm.py \
#  --model_name tokyotech-llm/Swallow-MS-7b-instruct-v0.1 \
#  --max_length 4096 \
#  --train_set_path llm-jp/hh-rlhf-12k-ja \
#  --output_path ./models/swallow_rm \
#  --gradient_accumulation_steps 64 \
#  --deepspeed ./deepspeed_configs/deepspeed_3.json

## lora sft
#accelerate launch ./bradley-terry-rm/mistral_rm_lora.py \
#  --model_name tokyotech-llm/Swallow-MS-7b-instruct-v0.1 \
#  --max_length 4096 \
#  --train_set_path llm-jp/hh-rlhf-12k-ja \
#  --output_path ./models/swallow_rm_lora \
#  --gradient_accumulation_steps 64 \
#  --deepspeed ./deepspeed_configs/deepspeed_3.json

## full sft
#accelerate launch ./bradley-terry-rm/mixtral_rm.py \
#  --model_name mistralai/Mixtral-8x22B-Instruct-v0.1 \
#  --max_length 4096 \
#  --train_set_path llm-jp/hh-rlhf-12k-ja \
#  --output_path ./models/mixtral_rm \
#  --gradient_accumulation_steps 64 \
#  --deepspeed ./deepspeed_configs/deepspeed_3.json

# lora sft
accelerate launch ./bradley-terry-rm/mixtral_rm_lora.py \
  --model_name mistralai/Mixtral-8x22B-Instruct-v0.1 \
  --max_length 4096 \
  --train_set_path llm-jp/hh-rlhf-12k-ja \
  --output_path ./models/mixtral_rm_lora \
  --gradient_accumulation_steps 64 \
  --deepspeed ./deepspeed_configs/deepspeed_3.json
