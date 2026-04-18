# Custom Data Finetuning Pipeline

새 데이터셋을 추가할 때 이 문서의 규격을 따릅니다.
YSOYA21이 레퍼런스 구현입니다.

---

## 데이터셋 디렉토리 구조

```
custom_data/
  {DATASET}/
    {DATASET}.jsonl          # 원본 메타데이터 (Step 0 입력)
    *.wav                    # 오디오 파일
    {DATASET}_converted.jsonl  # Step 1 출력
    {DATASET}_train.jsonl      # Step 2 출력 (학습에 사용)
  convert_{dataset}.py       # 데이터셋별 변환 스크립트
```

---

## JSONL 포맷 규격

### Step 1 입력 (원본 데이터셋)

데이터셋마다 포맷이 다를 수 있음. YSOYA21의 경우:

```jsonl
{"file": "YSOYA21_0001.wav", "text": "발화 내용"}
```

### Step 1 출력 / Step 2 입력 (`_converted.jsonl`)

변환 스크립트가 반드시 아래 4개 필드를 생성해야 합니다.

| 필드 | 타입 | 설명 |
|------|------|------|
| `audio` | str | 오디오 파일 **절대 경로** |
| `text` | str | 발화 텍스트 |
| `ref_audio` | str | 화자 참조용 오디오 **절대 경로** (중간 길이 1개 선택) |
| `language` | str | 언어 코드 (`"ko"`, `"en"` 등) |

```jsonl
{"audio": "/abs/path/YSOYA21_0001.wav", "text": "발화 내용", "ref_audio": "/abs/path/YSOYA21_0581.wav", "language": "ko"}
```

### Step 2 출력 (`_train.jsonl`)

`prepare_data.py`가 `audio_codes` 필드를 추가합니다. 직접 편집하지 않습니다.

```jsonl
{"audio": "...", "text": "...", "ref_audio": "...", "language": "ko", "audio_codes": [[...]]}
```

---

## 변환 스크립트 작성 규칙

새 데이터셋은 `custom_data/convert_{dataset}.py`를 작성합니다.

- `--input_jsonl`, `--output_jsonl`, `--ref_audio`, `--language` 인자를 지원할 것
- `audio` 필드는 반드시 **절대 경로**로 출력할 것 (`os.path.abspath` 사용)
- 출력 파일은 `encoding="utf-8"`로 저장할 것
- 완료 후 Step 2 명령어를 출력할 것 (YSOYA21 구현 참조)

---

## Step 1. JSONL 변환

데이터셋별 변환 스크립트를 실행합니다.

```bash
# YSOYA21 예시 (프로젝트 루트에서 실행)
uv run python custom_data/convert_ysoya21_jsonl.py
```

출력: `custom_data/{DATASET}/{DATASET}_converted.jsonl`

## Step 2. Audio Codes 추가

```bash
uv run python finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl custom_data/{DATASET}/{DATASET}_converted.jsonl \
  --output_jsonl custom_data/{DATASET}/{DATASET}_train.jsonl
```

출력: `custom_data/{DATASET}/{DATASET}_train.jsonl`

## Step 3. Finetuning

```bash
cd finetuning && uv run python sft_12hz.py ^
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base ^
  --output_model_path ../output/{DATASET} ^
  --train_jsonl ../custom_data/{DATASET}/{DATASET}_train.jsonl ^
  --batch_size 8 ^
  --lr 2e-6 ^
  --num_epochs 10 ^
  --save_epochs 5 ^
  --speaker_name {DATASET}
```

출력: `output/{DATASET}/`

---

## 주의사항

### config.json의 spk_id 대소문자 이슈

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
