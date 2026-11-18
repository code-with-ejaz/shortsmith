# ShortSmith

Turn a YouTube link into finished, captioned vertical clips. Paste the URL in the local web app; it analyzes the video's transcript, selects moments, and shows **only the finished clips to download**—there is no preview player or intermediate-video download.

## Low-data workflow

1. ShortSmith downloads **audio only** for transcription, rather than downloading a full video for analysis.
2. Whisper transcribes locally. When `ANTHROPIC_API_KEY` is set, Claude ranks transcript-based candidate moments and writes accurate hooks/captions. The transcript text (not the video) is sent to Anthropic. Without a key, local scoring is used.
3. Only selected sections are fetched from YouTube, at up to 480p by default, then rendered locally as 720x1280 vertical MP4s. Section downloads may include some extra data around keyframes; YouTube/format behavior means exact byte savings vary.
4. The web UI provides the final MP4s only. Files are stored locally under `output/`.

This trades analysis-video bandwidth for a smaller audio download and selected clip downloads. The local Whisper model may also need to download once on first use. A smaller model uses less storage and is faster, while potentially reducing transcription accuracy.

## Requirements

- Python 3.9+
- FFmpeg (including `ffprobe`) on `PATH`
- Internet access for YouTube audio/selected sections and the initial Whisper model download

## Install and run

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -e .
cp .env.example .env  # optional; add your Anthropic key to enable Claude analysis
python -m shortsmith.web
```

Open http://127.0.0.1:8000. Paste a YouTube video URL, choose up to five clips, and wait for processing to finish. Or use the command line:

```bash
shortsmith "https://www.youtube.com/watch?v=VIDEO_ID" --clips 3
```

The CLI writes finished files to `output/<job-id>/`. The local web server binds to loopback only by default; do not expose it to the public internet without adding authentication and appropriate deployment protections.

## Options

```text
--clips N                Number of finished clips (1–5, default 3)
--min-seconds N          Minimum clip duration (default 20)
--max-seconds N          Maximum clip duration (default 59)
--whisper-model MODEL    tiny, base, small, medium, large-v3 (default base)
--language CODE          Optional language code, e.g. en, ur, es
--no-llm                 Use local scoring even when an API key is configured
--output DIR             Output directory (default output)
```

Set `ANTHROPIC_API_KEY` in the environment or `.env` to enable transcript-based Claude analysis. API use may incur charges. `SHORTSMITH_MODEL` overrides the model name. The transcript is sent to Anthropic for ranking and hook generation; the video itself is not sent to Anthropic.

## YouTube sign-in checks

YouTube sometimes blocks downloads with "Sign in to confirm you're not a bot", especially from datacenter networks. If that happens, export cookies from a browser where you are signed in to YouTube (see the [yt-dlp cookie docs](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp)), then either set `SHORTSMITH_COOKIES` to the cookies file path in `.env`, or pass `--cookies <path>` to the CLI. Keep the cookies file private; never commit it.

## Notes and responsible use

- This app does not upload content to TikTok or any other platform.
- It does not show source-video previews; the web UI exposes only final clips.
- Use only videos you own or have permission to reuse. Follow YouTube's terms and applicable copyright law.
- YouTube availability, age restrictions, network conditions, FFmpeg codecs, and format changes can affect success. If YouTube blocks a download or changes its formats, update `yt-dlp`.
- Local processing can take time. Whisper is CPU-capable; GPU acceleration depends on a compatible CTranslate2/CUDA installation.

## Tests

```bash
python -m unittest discover -s tests -v
```

## License

MIT
