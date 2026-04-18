# coding=utf-8
"""
게임 음성 데이터셋 변환 스크립트.

입력 포맷 (metadata.jsonl):
  {"filename": "xxxx.wav", "transcription": "...", ...}

오디오 파일 위치: {dataset_dir}/audio/{filename}

출력 포맷 (_converted.jsonl):
  {"audio": "<abs_path>", "text": "...", "ref_audio": "<abs_path>", "language": "ja"}

game 포맷 감지 기준: 서브디렉토리 안에 metadata.jsonl + audio/ 가 존재하는 경우.
"""
import argparse
import json
import os

import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))


def find_game_dirs(root: str) -> list[str]:
    """metadata.jsonl + audio/ 가 있는 서브디렉토리를 자동으로 찾습니다."""
    result = []
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        if (os.path.exists(os.path.join(entry.path, "metadata.jsonl")) and
                os.path.isdir(os.path.join(entry.path, "audio"))):
            result.append(entry.path)
    return result


def pick_ref_audio(audio_dir: str, filenames: list[str]) -> str:
    """오디오 파일 목록 중 중간 길이의 파일을 ref_audio로 선택합니다."""
    durations = []
    for fname in filenames:
        path = os.path.join(audio_dir, fname)
        if os.path.exists(path):
            info = sf.info(path)
            durations.append((info.duration, path))
    durations.sort(key=lambda x: x[0])
    return durations[len(durations) // 2][1]


def convert(dataset_dir: str, language: str = "ja") -> str:
    dataset_dir = os.path.abspath(dataset_dir)
    audio_dir = os.path.join(dataset_dir, "audio")
    input_jsonl = os.path.join(dataset_dir, "metadata.jsonl")
    dataset_name = os.path.basename(dataset_dir)
    output_jsonl = os.path.join(dataset_dir, f"{dataset_name}_converted.jsonl")

    with open(input_jsonl, encoding="utf-8") as f:
        items = [json.loads(l.strip()) for l in f if l.strip()]

    filenames = [item["filename"] for item in items]
    ref_audio = pick_ref_audio(audio_dir, filenames)

    converted = []
    skipped = 0
    for item in items:
        audio_path = os.path.join(audio_dir, item["filename"])
        if not os.path.exists(audio_path):
            skipped += 1
            continue
        converted.append({
            "audio": audio_path,
            "text": item["transcription"],
            "ref_audio": ref_audio,
            "language": language,
        })

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for entry in converted:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[{dataset_name}] {len(converted)} entries -> {output_jsonl}"
          + (f" (skipped {skipped})" if skipped else ""))
    print(f"  ref_audio: {ref_audio}")
    return output_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="특정 데이터셋 디렉토리. 미지정 시 자동 감지된 모든 game 데이터셋을 처리합니다.")
    parser.add_argument("--language", type=str, default="ja")
    args = parser.parse_args()

    if args.dataset_dir:
        dirs = [args.dataset_dir]
    else:
        dirs = find_game_dirs(_HERE)
        print(f"Found {len(dirs)} game dataset(s): {[os.path.basename(d) for d in dirs]}\n")

    output_jsonls = []
    for d in dirs:
        output_jsonl = convert(d, language=args.language)
        output_jsonls.append(output_jsonl)

    print("\nNext step (prepare_data.py per dataset):")
    for out in output_jsonls:
        train_jsonl = out.replace("_converted.jsonl", "_train.jsonl")
        print(f"  uv run python finetuning/prepare_data.py \\")
        print(f"    --device cuda:0 \\")
        print(f"    --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \\")
        print(f"    --input_jsonl {out} \\")
        print(f"    --output_jsonl {train_jsonl}")
        print()


if __name__ == "__main__":
    main()
