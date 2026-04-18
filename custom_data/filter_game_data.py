# coding=utf-8
"""
game 포맷 데이터셋 (_converted.jsonl) 일괄 필터링.
조건: text >= 8자, audio <= 15초
YSOYA21은 제외합니다.
프로젝트 루트에서 실행: uv run python custom_data/filter_game_data.py
"""
import json
import os

import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))

MIN_CHARS = 8
MIN_AUDIO = 3.0
MAX_AUDIO = 15.0


def find_game_dirs(root: str) -> list[str]:
    result = []
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        if (os.path.exists(os.path.join(entry.path, "metadata.jsonl")) and
                os.path.isdir(os.path.join(entry.path, "audio"))):
            result.append(entry.path)
    return result


def filter_dataset(dataset_dir: str):
    name = os.path.basename(dataset_dir)
    jsonl_path = os.path.join(dataset_dir, f"{name}_converted.jsonl")

    if not os.path.exists(jsonl_path):
        print(f"[{name}] SKIP — _converted.jsonl 없음")
        return

    with open(jsonl_path, encoding="utf-8") as f:
        items = [json.loads(l.strip()) for l in f if l.strip()]

    kept, removed = [], 0
    for item in items:
        chars = len(item.get("text", ""))
        audio_path = item.get("audio", "")
        if not os.path.exists(audio_path):
            removed += 1
            continue
        dur = sf.info(audio_path).duration
        if chars < MIN_CHARS or dur < MIN_AUDIO or dur > MAX_AUDIO:
            removed += 1
            continue
        kept.append(item)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[{name}] {len(items)} -> {len(kept)}  (removed: {removed})")


def main():
    dirs = find_game_dirs(_HERE)
    print(f"Filter: text >= {MIN_CHARS}chars  |  {MIN_AUDIO}s <= audio <= {MAX_AUDIO}s\n")
    for d in dirs:
        filter_dataset(d)
    print("\n완료.")


if __name__ == "__main__":
    main()
