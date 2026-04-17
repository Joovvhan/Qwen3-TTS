import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT  = os.path.join(_HERE, "YSOYA21", "YSOYA21.jsonl")
DEFAULT_OUTPUT = os.path.join(_HERE, "YSOYA21", "YSOYA21_converted.jsonl")
DEFAULT_REF_AUDIO = os.path.join(_HERE, "YSOYA21", "YSOYA21_0581.wav")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_jsonl", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Base directory for audio files. Defaults to the directory of input_jsonl.")
    parser.add_argument("--ref_audio", type=str, default=DEFAULT_REF_AUDIO)
    parser.add_argument("--language", type=str, default="ko")
    args = parser.parse_args()

    audio_dir = args.audio_dir or os.path.dirname(os.path.abspath(args.input_jsonl))
    ref_audio = os.path.abspath(args.ref_audio)

    with open(args.input_jsonl, encoding="utf-8") as f:
        lines = [json.loads(l.strip()) for l in f if l.strip()]

    converted = []
    for item in lines:
        audio_path = os.path.join(audio_dir, item["file"])
        converted.append({
            "audio": audio_path,
            "text": item["text"],
            "ref_audio": ref_audio,
            "language": args.language,
        })

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Converted {len(converted)} entries -> {args.output_jsonl}")
    print("\nNext step:")
    print(f"  uv run python finetuning/prepare_data.py \\")
    print(f"    --device cuda:0 \\")
    print(f"    --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \\")
    print(f"    --input_jsonl {args.output_jsonl} \\")
    print(f"    --output_jsonl {os.path.splitext(args.output_jsonl)[0].replace('_converted', '_train')}.jsonl")


if __name__ == "__main__":
    main()
