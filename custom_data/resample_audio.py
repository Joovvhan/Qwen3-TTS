# coding=utf-8
"""
custom_data 하위 모든 데이터셋의 wav 파일을 24kHz로 인플레이스 리샘플링합니다.
- game 포맷 (metadata.jsonl + audio/): audio/ 하위 wav 처리
- flat 포맷 (YSOYA21 등): 데이터셋 디렉토리 직접 wav 처리
프로젝트 루트에서 실행: uv run python custom_data/resample_audio.py [--dataset NAME]
"""
import argparse
import os

import librosa
import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_SR = 24000


def find_audio_dirs(root: str) -> list[tuple[str, str]]:
    """(dataset_name, audio_dir) 목록 반환."""
    result = []
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        audio_subdir = os.path.join(entry.path, "audio")
        if os.path.isdir(audio_subdir):
            result.append((entry.name, audio_subdir))
        else:
            # flat 구조: 디렉토리 내에 wav가 있으면 포함
            wavs = [f for f in os.listdir(entry.path) if f.endswith(".wav")]
            if wavs:
                result.append((entry.name, entry.path))
    return result


def resample_dir(name: str, audio_dir: str):
    wav_files = sorted(f for f in os.listdir(audio_dir) if f.endswith(".wav"))
    if not wav_files:
        print(f"[{name}] wav 없음, 스킵")
        return

    total = len(wav_files)
    skipped = 0
    converted = 0

    print(f"[{name}] {total}개 파일 처리 중...")
    for i, fname in enumerate(wav_files, 1):
        path = os.path.join(audio_dir, fname)
        if sf.info(path).samplerate == TARGET_SR:
            skipped += 1
            continue
        waveform, sr = librosa.load(path, sr=None, mono=False)
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        sf.write(path, waveform.T, TARGET_SR)
        converted += 1
        if i % 100 == 0 or i == total:
            print(f"  {i}/{total}  converted={converted}  skipped(already 24k)={skipped}")

    print(f"[{name}] 완료: converted={converted}, already 24kHz={skipped}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="특정 데이터셋 이름만 처리")
    args = parser.parse_args()

    audio_dirs = find_audio_dirs(_HERE)
    if not audio_dirs:
        print("처리할 데이터셋을 찾을 수 없습니다.")
        return

    if args.dataset:
        audio_dirs = [(n, d) for n, d in audio_dirs if n == args.dataset]
        if not audio_dirs:
            print(f"Dataset not found: {args.dataset}")
            return

    print(f"Target: {TARGET_SR}Hz  |  datasets: {[n for n, _ in audio_dirs]}\n")
    for name, audio_dir in audio_dirs:
        resample_dir(name, audio_dir)

    print("전체 완료.")


if __name__ == "__main__":
    main()
