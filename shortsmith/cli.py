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
    parser.add_argument("--whisper-model", default=os.getenv("SHORTSMITH_WHISPER_MODEL", "base"),
                        choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--language", default=None)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--cookies", default=None, help="Path to a yt-dlp cookies file, for videos YouTube bot-checks")
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    if not 10 <= args.min_seconds <= args.max_seconds <= 60:
        parser.error("Clip durations must satisfy 10 <= --min-seconds <= --max-seconds <= 60.")
    if args.cookies and not Path(args.cookies).is_file():
        parser.error(f"Cookies file not found: {args.cookies}")

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
        cookies=args.cookies,
    )
    print("\nFinished clips:")
    for clip in result:
        print(f"- {clip['filename']}: {clip['hook']} ({clip['start']:.1f}–{clip['end']:.1f}s)")


if __name__ == "__main__":
    main()
