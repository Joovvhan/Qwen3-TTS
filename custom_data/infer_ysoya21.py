# coding=utf-8
import os
import torch
import soundfile as sf
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

DEVICE = "cuda:0"
CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "output", "YSOYA21", "checkpoint-epoch-14")
SPEAKER = "YSOYA21"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "YSOYA21", "infer_samples")

# fmt: off
TEXTS = [
    # --- 데이터셋 내 문장 (5개) ---
    "한국에서는 개발 책임자의 직함으로 디렉터, 총괄 디렉터, PD, 프로듀서가 혼용돼서 쓰이고 있는데요.",
    "게임 PD가 되어보니 라는 제목으로 PD 직무에 대한 리뷰를 해보려고 합니다.",
    "PD에 대해서 이렇게 생각해 볼 수도 있구나 정도로 봐주시면 좋을 것 같습니다.",
    "나도 나중에는 저렇게 되고 싶다 이런 꿈을 가지고 저도 업계에 들어왔습니다.",
    "머릿속에 그리던 게임이 실체화되고 플레이어분들이 즐겁게 플레이하시는 반응을 볼 때는 많은 보람을 느끼죠.",
    # --- 데이터셋과 무관한 문장 (5개) ---
    "오늘 날씨가 정말 좋네요. 산책하기 딱 좋은 날씨입니다.",
    "저는 커피보다 녹차를 더 좋아합니다.",
    "이번 주말에 친구들과 함께 영화를 보러 갈 예정입니다.",
    "한국의 사계절은 각각 뚜렷한 특색이 있어 매력적입니다.",
    "독서는 마음을 풍요롭게 해주는 좋은 습관입니다.",
]
# fmt: on


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading model from: {CHECKPOINT}")
    tts = Qwen3TTSModel.from_pretrained(
        CHECKPOINT,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=DEVICE,
    )

    for i, text in enumerate(TEXTS):
        tag = "dataset" if i < 5 else "unseen"
        out_path = os.path.join(OUTPUT_DIR, f"{tag}_{i:02d}.wav")
        print(f"[{i+1:02d}/{len(TEXTS)}] {tag}: {text[:40]}...")
        wavs, sr = tts.generate_custom_voice(text=text, speaker=SPEAKER)
        sf.write(out_path, wavs[0], sr)
        print(f"  -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
