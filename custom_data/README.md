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
  --batch_size 8 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name YSOYA21
```

출력: `output/YSOYA21/`

## 주의사항

### config.json의 speaker_id 대소문자 이슈

모델 추론 코드(`modeling_qwen3_tts.py`)는 speaker 이름을 `speaker.lower()`로 변환하여 lookup합니다.
따라서 체크포인트의 `config.json` 내 `spk_id` 키가 반드시 **소문자**여야 합니다.

`sft_12hz.py`는 저장 시 자동으로 `.lower()` 처리하므로 신규 학습은 문제없습니다.
단, 이 수정 이전에 생성된 체크포인트는 `config.json`을 직접 수정해야 합니다.

```json
// 잘못된 예 (추론 실패)
"spk_id": { "YSOYA21": 3000 }

// 올바른 예
"spk_id": { "ysoya21": 3000 }
```
