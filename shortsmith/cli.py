from __future__ import annotations

import argparse
import os
from pathlib import Path

from .pipeline import process_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Create captioned vertical shorts from a YouTube video.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("-n", "--clips", type=int, default=3, choices=range(1, 6), metavar="1..5")
    parser.add_argument("--min-seconds", type=int, default=20)
    parser.add_argument("--max-seconds", type=int, default=59)
    parser.add_argument("--whisper-model", default=os.getenv("SHORTSMITH_WHISPER_MODEL", "base"))
    parser.add_argument("--language", default=None)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()

    def report(message: str) -> None:
        print(message, flush=True)

    result = process_video(
        url=args.url,
        output_root=args.output,
        clips=args.clips,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        whisper_model=args.whisper_model,
        language=args.language,
        use_llm=not args.no_llm,
        progress=report,
    )
    print("\nFinished clips:")
    for clip in result:
        print(f"- {clip['filename']}: {clip['hook']} ({clip['start']:.1f}–{clip['end']:.1f}s)")


if __name__ == "__main__":
    main()
