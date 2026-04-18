# coding=utf-8
"""
game 포맷 데이터셋에 대해 prepare_data.py를 순차 실행합니다.
game 포맷 감지 기준: 서브디렉토리 안에 metadata.jsonl + audio/ 가 존재하는 경우.
프로젝트 루트에서 실행: uv run python custom_data/prepare_all_game.py
"""
import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    dirs = find_game_dirs(_HERE)
    print(f"Found {len(dirs)} game dataset(s): {[os.path.basename(d) for d in dirs]}\n")

    for dataset_dir in dirs:
        dataset_name = os.path.basename(dataset_dir)
        input_jsonl = os.path.join(dataset_dir, f"{dataset_name}_converted.jsonl")
        output_jsonl = os.path.join(dataset_dir, f"{dataset_name}_train.jsonl")

        if not os.path.exists(input_jsonl):
            print(f"[{dataset_name}] SKIP — _converted.jsonl not found (convert 먼저 실행 필요)")
            continue

        print(f"[{dataset_name}] Starting prepare_data.py ...")
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_ROOT, "finetuning", "prepare_data.py"),
                "--device", "cuda:0",
                "--tokenizer_model_path", "Qwen/Qwen3-TTS-Tokenizer-12Hz",
                "--input_jsonl", input_jsonl,
                "--output_jsonl", output_jsonl,
                "--batch_size", str(args.batch_size),
            ],
            cwd=_ROOT,
        )

        if result.returncode != 0:
            print(f"[{dataset_name}] FAILED (exit code {result.returncode}). Stopping.")
            sys.exit(result.returncode)

        print(f"[{dataset_name}] Done -> {output_jsonl}\n")

    print("All datasets prepared.")


if __name__ == "__main__":
    main()
