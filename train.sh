accelerate launch ./bradley-terry-rm/mistral_rm.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.3 \
  --max_length 4096 \
  --train_set_path llm-jp/hh-rlhf-12k-ja \
  --output_path ./models/mistral_rm_test \
  --gradient_accumulation_steps 64

python ./merge_lora.py --lora_model_path ./models/mistral_rm_test/last_checkpoint --merged_dir ./mistral_rm
cp ./models/mistral_rm_test/last_checkpoint/tokenizer* ./mistral_rm/
cp ./models/mistral_rm_test/last_checkpoint/special_tokens_map.json ./mistral_rm/
