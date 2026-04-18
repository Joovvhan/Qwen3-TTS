# coding=utf-8
"""
custom_data의 _converted.jsonl을 데이터셋별로 대화형으로 분석하고 필터링합니다.
프로젝트 루트에서 실행: uv run python custom_data/curate_data.py [--dataset NAME] [--bins N]
"""
import argparse
import json
import os

import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))


def find_converted_datasets(root: str) -> list[tuple[str, str]]:
    result = []
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        name = entry.name
        jsonl_path = os.path.join(entry.path, f"{name}_converted.jsonl")
        if os.path.exists(jsonl_path):
            result.append((name, jsonl_path))
    return result


def make_edges(hi: float, step: float) -> list[float]:
    """0~10은 1 간격, 이후는 step 간격으로 bin 경계를 생성합니다."""
    edges = list(range(0, 11))  # 0, 1, 2, ..., 10
    cur = 10 + step
    while cur <= hi + step:
        edges.append(cur)
        cur += step
    return [float(e) for e in edges]


def histogram(values: list[float], step: float, unit: str = "", fmt: str = ".0f") -> str:
    if not values:
        return ""
    edges = make_edges(max(values), step)
    counts = [0] * (len(edges) - 1)
    for v in values:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                counts[i] += 1
                break
        else:
            counts[-1] += 1  # max값 처리

    max_count = max(counts) if any(counts) else 1
    bar_width = 30
    lines = []
    for i, count in enumerate(counts):
        if count == 0:
            continue
        left, right = edges[i], edges[i + 1]
        bar = "█" * int(bar_width * count / max_count)
        l_str = str(int(left)) if fmt == "d" else f"{left:{fmt}}"
        r_str = str(int(right)) if fmt == "d" else f"{right:{fmt}}"
        label = f"  {l_str}-{r_str}{unit}"
        lines.append(f"{label:<20} | {bar} {count}")
    return "\n".join(lines)


def load_dataset(jsonl_path: str) -> tuple[list[dict], list[float | None], list[str]]:
    with open(jsonl_path, encoding="utf-8") as f:
        items = [json.loads(l.strip()) for l in f if l.strip()]
    durations, missing = [], []
    for item in items:
        path = item.get("audio", "")
        if not os.path.exists(path):
            missing.append(path)
            durations.append(None)
        else:
            durations.append(sf.info(path).duration)
    return items, durations, missing


def show_distribution(name: str, items: list, durations: list, missing: list, text_step: int, audio_step: float):
    total = len(items)
    char_lens = [len(item.get("text", "")) for item in items]
    valid_durs = [d for d in durations if d is not None]

    print(f"\n{'='*60}")
    print(f"[{name}]  total: {total}  |  missing audio: {len(missing)}")
    print(f"\n  [Text length (chars)]")
    print(f"  min={min(char_lens)}  max={max(char_lens)}  mean={sum(char_lens)/len(char_lens):.1f}")
    print(histogram(char_lens, step=text_step, unit="c", fmt="d"))
    if valid_durs:
        print(f"\n  [Audio duration (sec)]")
        print(f"  min={min(valid_durs):.2f}  max={max(valid_durs):.2f}  mean={sum(valid_durs)/len(valid_durs):.2f}")
        print(histogram(valid_durs, step=audio_step, unit="s", fmt=".0f"))


def compute_filter(
    items: list, durations: list,
    min_chars: int, max_chars: int, min_audio: float, max_audio: float
) -> tuple[list, list]:
    kept, removed = [], []
    for item, dur in zip(items, durations):
        chars = len(item.get("text", ""))
        reasons = []
        if chars < min_chars:
            reasons.append(f"text {chars}c < {min_chars}c")
        if chars > max_chars:
            reasons.append(f"text {chars}c > {max_chars}c")
        if dur is None:
            reasons.append("missing audio")
        elif dur < min_audio:
            reasons.append(f"{dur:.2f}s < {min_audio}s")
        elif dur > max_audio:
            reasons.append(f"{dur:.2f}s > {max_audio}s")
        if reasons:
            removed.append((item, ", ".join(reasons)))
        else:
            kept.append(item)
    return kept, removed


def show_filter_status(items, durations, min_chars, max_chars, min_audio, max_audio):
    kept, removed = compute_filter(items, durations, min_chars, max_chars, min_audio, max_audio)
    print(f"\n  현재 필터: chars [{min_chars}~{max_chars}]  audio [{min_audio}s~{max_audio}s]")
    print(f"  -> keep: {len(kept)}  /  remove: {len(removed)}  /  total: {len(items)}")
    if removed:
        print("  [제거 예시 (최대 3개)]")
        for item, reason in removed[:3]:
            print(f"    {reason} | {item.get('text','')[:40]!r}")
    return kept, removed


def process_dataset(name: str, jsonl_path: str, text_step: int, audio_step: float):
    items, durations, missing = load_dataset(jsonl_path)
    show_distribution(name, items, durations, missing, text_step, audio_step)

    # 기본값
    min_chars, max_chars = 5, 200
    min_audio, max_audio = 0.5, 15.0

    while True:
        kept, removed = show_filter_status(items, durations, min_chars, max_chars, min_audio, max_audio)
        print()
        print("  1. min_chars 설정")
        print("  2. max_chars 설정")
        print("  3. min_audio 설정")
        print("  4. max_audio 설정")
        print("  5. 적용 (파일 저장)")
        print("  6. 넘어가기")

        choice = input("  선택: ").strip()

        if choice == "1":
            val = input(f"  min_chars [{min_chars}]: ").strip()
            if val:
                min_chars = int(val)
        elif choice == "2":
            val = input(f"  max_chars [{max_chars}]: ").strip()
            if val:
                max_chars = int(val)
        elif choice == "3":
            val = input(f"  min_audio [{min_audio}]: ").strip()
            if val:
                min_audio = float(val)
        elif choice == "4":
            val = input(f"  max_audio [{max_audio}]: ").strip()
            if val:
                max_audio = float(val)
        elif choice == "5":
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for item in kept:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  [✓] 저장 완료: {len(items)} -> {len(kept)}")
            break
        elif choice == "6":
            print("  [스킵]")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=str,   default=None, help="특정 데이터셋 이름만 처리")
    parser.add_argument("--text_step",  type=int,   default=20,   help="텍스트 히스토그램 bin 크기 (chars)")
    parser.add_argument("--audio_step", type=float, default=5.0,  help="오디오 히스토그램 bin 크기 (초)")
    args = parser.parse_args()

    datasets = find_converted_datasets(_HERE)
    if not datasets:
        print("_converted.jsonl 파일을 찾을 수 없습니다. convert 스크립트를 먼저 실행하세요.")
        return

    if args.dataset:
        datasets = [(n, p) for n, p in datasets if n == args.dataset]
        if not datasets:
            print(f"Dataset not found: {args.dataset}")
            return

    for name, jsonl_path in datasets:
        process_dataset(name, jsonl_path, args.text_step, args.audio_step)

    print("\n모든 데이터셋 처리 완료.")


if __name__ == "__main__":
    main()
