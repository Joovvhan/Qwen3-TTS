# Custom Data Finetuning Pipeline

## Step 1. JSONL 변환

```bash
uv run python custom_data/convert_ysoya21_jsonl.py
```

출력: `custom_data/YSOYA21/YSOYA21_converted.jsonl`

## Step 2. Audio Codes 추가

```bash
uv run python finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl custom_data/YSOYA21/YSOYA21_converted.jsonl \
  --output_jsonl custom_data/YSOYA21/YSOYA21_train.jsonl
```

출력: `custom_data/YSOYA21/YSOYA21_train.jsonl`

## Step 3. Finetuning

```bash
cd finetuning && uv run python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path ../output/YSOYA21 \
  --train_jsonl ../custom_data/YSOYA21/YSOYA21_train.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name YSOYA21
```

출력: `output/YSOYA21/`
