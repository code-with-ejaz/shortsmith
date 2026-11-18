from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse


def load_dotenv() -> None:
    """Load a simple .env file without requiring another dependency."""
    path = Path(".env")
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def validate_youtube_url(url: str) -> str:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower().rstrip(".")
    allowed = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be", "youtube-nocookie.com", "www.youtube-nocookie.com"}
    if parsed.scheme not in {"http", "https"} or host not in allowed:
        raise ValueError("Please enter a valid YouTube video URL.")
    if host.endswith("youtu.be"):
        video_id = parsed.path.strip("/").split("/")[0]
    else:
        video_id = parse_qs(parsed.query).get("v", [""])[0]
        if not video_id:
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
                video_id = parts[1]
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,20}", video_id or ""):
        raise ValueError("The URL must point to a single YouTube video, not a playlist or channel.")
    return f"https://www.youtube.com/watch?v={video_id}"


def _cookies_file(explicit: str | None = None) -> str | None:
    """Return a yt-dlp cookie file path, if one was configured and exists."""
    candidate = explicit or os.getenv("SHORTSMITH_COOKIES")
    if candidate and Path(candidate).is_file():
        return candidate
    return None


def _friendly_download_error(exc: Exception) -> RuntimeError:
    message = str(exc)
    if "Sign in to confirm" in message or "bot" in message.lower():
        return RuntimeError(
            "YouTube asked for a sign-in check before this download. Export cookies from a browser "
            "where you are signed in to YouTube (see yt-dlp's cookie docs), then set SHORTSMITH_COOKIES "
            "to that cookies file, or pass --cookies, and try again."
        )
    return RuntimeError(message)


def _run(command: list[str], *, progress: Callable[[str], None] | None = None) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Command failed ({Path(command[0]).name}): {detail[-1800:]}")


def _require_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError("Please install FFmpeg (including ffprobe) and make sure it is on PATH. Missing: " + ", ".join(missing))


def _safe_stem(text: str, fallback: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return (stem[:48].strip("-") or fallback)


def download_audio(url: str, work_dir: Path, cookies: str | None = None) -> Path:
    from yt_dlp import YoutubeDL

    work_dir.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(work_dir / "source_audio.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "overwrites": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "5"}],
        "postprocessor_args": {"ffmpeg": ["-ac", "1", "-ar", "16000"]},
    }
    cookiefile = _cookies_file(cookies)
    if cookiefile:
        options["cookiefile"] = cookiefile
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title") or "youtube-video"
    except Exception as exc:
        raise _friendly_download_error(exc) from exc
    wav = work_dir / "source_audio.wav"
    if not wav.exists():
        # Some source codecs may not use the expected postprocessor output name.
        audio_files = [p for p in work_dir.glob("source_audio.*") if p.suffix.lower() not in {".part", ".ytdl", ".temp"}]
        if not audio_files:
            raise RuntimeError("YouTube audio download completed but no audio file was created.")
        _run(["ffmpeg", "-y", "-i", str(audio_files[0]), "-vn", "-ac", "1", "-ar", "16000", str(wav)])
    (work_dir / "title.txt").write_text(title, encoding="utf-8")
    return wav


def transcribe(audio_path: Path, model_name: str, language: str | None = None) -> tuple[list[dict[str, Any]], float]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments_iter, info = model.transcribe(str(audio_path), language=language, word_timestamps=True, vad_filter=True)
    segments: list[dict[str, Any]] = []
    for segment in segments_iter:
        words = []
        for word in (segment.words or []):
            words.append({"start": float(word.start), "end": float(word.end), "text": word.word.strip()})
        segments.append({"start": float(segment.start), "end": float(segment.end), "text": segment.text.strip(), "words": words})
    if not segments:
        raise RuntimeError("No speech was detected in the downloaded audio.")
    return segments, float(info.duration or segments[-1]["end"])


def _score_text(text: str) -> float:
    lowered = text.lower()
    score = 0.0
    hooks = ("why", "how", "secret", "mistake", "never", "nobody", "truth", "stop", "imagine", "actually", "surprising", "finally")
    power = ("amazing", "incredible", "love", "hate", "fail", "success", "wrong", "best", "worst", "danger", "money", "change", "important", "shocking")
    contrast = ("but", "however", "turns out", "instead", "except", "although")
    score += 2.0 * sum(1 for token in hooks if re.search(r"\b" + re.escape(token) + r"\b", lowered))
    score += 1.0 * sum(1 for token in power if re.search(r"\b" + re.escape(token) + r"\b", lowered))
    score += 1.2 * sum(1 for token in contrast if token in lowered)
    score += 1.0 * sum(1 for token in ("?", "!", "$", "%") if token in text)
    score += min(len(text.split()) / 110.0, 1.5)
    return score


def find_candidates(segments: list[dict[str, Any]], duration: float, minimum: int, maximum: int, limit: int = 12) -> list[dict[str, Any]]:
    if not 10 <= minimum <= maximum <= 60:
        raise ValueError("Clip durations must satisfy 10 <= minimum <= maximum <= 60 seconds.")
    if not segments:
        return []
    ends = [i for i, seg in enumerate(segments) if re.search(r"[.!?…][\"'’)]*$", seg["text"])]
    if not ends:
        ends = list(range(len(segments)))
    candidates = []
    for i, start_seg in enumerate(segments):
        start = float(start_seg["start"])
        for j in ends:
            if j < i:
                continue
            end = float(segments[j]["end"])
            length = end - start
            if length < minimum:
                continue
            if length > maximum:
                break
            text = " ".join(s["text"] for s in segments[i:j + 1]).strip()
            if not text:
                continue
            candidates.append({"start": start, "end": end, "text": text, "score": _score_text(text)})
    candidates.sort(key=lambda c: c["score"], reverse=True)
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        midpoint = (candidate["start"] + candidate["end"]) / 2
        if all(abs(midpoint - (item["start"] + item["end"]) / 2) > minimum * 0.55 for item in selected):
            selected.append(candidate)
            if len(selected) >= limit:
                break
    return selected


def analyze_candidates(candidates: list[dict[str, Any]], count: int, use_llm: bool) -> list[dict[str, Any]]:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if use_llm and api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            model = os.getenv("SHORTSMITH_MODEL", "claude-3-5-haiku-latest")
            compact = [{"id": i, "start": round(c["start"], 1), "end": round(c["end"], 1), "transcript": c["text"][:2200]} for i, c in enumerate(candidates)]
            response = client.messages.create(
                model=model,
                max_tokens=1600,
                system="You select compelling, truthful short-form video moments using transcript text only. Never invent claims or imply facts absent from the transcript. Return only valid JSON with a clips array. Each clip must have id (integer), hook (at most 8 words), caption (one concise sentence), reason (one sentence). Choose distinct moments and rank the strongest first.",
                messages=[{"role": "user", "content": "Select up to " + str(count) + " of these transcript moments. Return JSON only. Moments: " + json.dumps(compact, ensure_ascii=False)}],
            )
            text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
            data = json.loads(text[text.find("{"):text.rfind("}") + 1])
            by_id = {i: candidate for i, candidate in enumerate(candidates)}
            results = []
            for item in data.get("clips", []):
                candidate = by_id.get(item.get("id"))
                if candidate is None:
                    continue
                results.append({**candidate, "hook": str(item.get("hook", "A moment worth hearing"))[:100], "caption": str(item.get("caption", ""))[:500], "reason": str(item.get("reason", "Selected for its strong hook and complete thought."))[:500]})
                if len(results) >= count:
                    break
            if results:
                return results
        except Exception:
            # Keep the application useful when the API is unavailable or returns malformed output.
            pass

    results = []
    for candidate in candidates[:count]:
        words = re.findall(r"[\w'-]+", candidate["text"])
        hook = " ".join(words[:7]) or "A moment worth hearing"
        results.append({**candidate, "hook": hook, "caption": candidate["text"][:240], "reason": "Ranked using local transcript hook, contrast, pacing, and emphasis signals."})
    return results


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, int(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, cs = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", " ")


def write_ass(path: Path, clip: dict[str, Any], segments: list[dict[str, Any]], width: int = 720, height: int = 1280) -> None:
    header = f'''[Script Info]\nScriptType: v4.00+\nPlayResX: {width}\nPlayResY: {height}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Caption,Arial,52,&H00FFFFFF,&H0000FFFF,&H00101010,&H80000000,-1,0,1,4,1,2,38,38,150,1\nStyle: Hook,Arial,55,&H00FFFFFF,&H0000FFFF,&H00101010,&H90000000,-1,0,1,4,2,8,36,36,120,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'''
    events = [f"Dialogue: 1,{_ass_time(0)},{_ass_time(min(3.5, clip['end'] - clip['start']))},Hook,,0,0,0,,{{\\an8}}{_ass_escape(clip['hook'])}"]
    words: list[dict[str, Any]] = []
    for segment in segments:
        for word in segment.get("words", []):
            if word["end"] <= clip["start"] or word["start"] >= clip["end"]:
                continue
            words.append({"start": max(word["start"], clip["start"]) - clip["start"], "end": min(word["end"], clip["end"]) - clip["start"], "text": word["text"]})
    for offset in range(0, len(words), 6):
        group = words[offset:offset + 6]
        if not group:
            continue
        start, end = group[0]["start"], group[-1]["end"]
        karaoke = " ".join("{\\k" + str(max(1, int((w['end'] - w['start']) * 100))) + "}" + _ass_escape(w["text"]) for w in group)
        events.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(max(end, start + 0.25))},Caption,,0,0,0,,{karaoke}")
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def _probe_duration(path: Path) -> float | None:
    """Return a media file's duration in seconds, or None when ffprobe cannot read it."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=False,
        )
        return float(result.stdout.strip())
    except (OSError, ValueError):
        return None


def download_section(url: str, clip: dict[str, Any], destination: Path, max_height: int = 480, cookies: str | None = None) -> Path:
    from yt_dlp import YoutubeDL

    destination.parent.mkdir(parents=True, exist_ok=True)
    section = f"*{max(0, clip['start']):.3f}-{clip['end']:.3f}"
    options: dict[str, Any] = {
        "format": f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best",
        "outtmpl": str(destination.with_suffix(".source.%(ext)s")),
        "download_sections": [section],
        "force_keyframes_at_cuts": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "overwrites": True,
        "merge_output_format": "mp4",
    }
    # NOTE: do not add postprocessor_args with -ss/-t here. yt-dlp appends
    # postprocessor ffmpeg args to every ffmpeg invocation, including the
    # merger that runs on the already-cut section file (which starts near
    # time 0), so an absolute -ss would seek past the end of the clip and
    # produce an empty or truncated file. download_sections plus
    # force_keyframes_at_cuts already performs the cut; render_clip trims
    # the result to the exact requested duration.
    cookiefile = _cookies_file(cookies)
    if cookiefile:
        options["cookiefile"] = cookiefile
    try:
        with YoutubeDL(options) as ydl:
            ydl.download([url])
    except Exception as exc:
        raise _friendly_download_error(exc) from exc
    matches = [p for p in destination.parent.glob(destination.stem + ".source.*") if p.suffix.lower() not in {".part", ".ytdl", ".temp"}]
    if not matches:
        raise RuntimeError("Could not find the downloaded clip section.")
    source = max(matches, key=lambda p: p.stat().st_size)
    expected = clip["end"] - clip["start"]
    actual = _probe_duration(source)
    if actual is not None and actual < expected * 0.5:
        raise RuntimeError(f"The downloaded section is only {actual:.1f}s, but {expected:.1f}s was requested. YouTube may have blocked part of the download; try again or use cookies (SHORTSMITH_COOKIES).")
    return source


def render_clip(source: Path, output: Path, subtitle_file: Path, duration: float) -> None:
    _require_tools()
    # The source section starts at its requested time; subtitles are relative to the selected clip.
    sub_path = str(subtitle_file.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    video_filter = f"[0:v]split=2[bg][fg];[bg]scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,boxblur=20:10[bgblur];[fg]scale=720:1280:force_original_aspect_ratio=decrease[foreground];[bgblur][foreground]overlay=(W-w)/2:(H-h)/2,subtitles='{sub_path}'[v]"
    command = ["ffmpeg", "-y", "-i", str(source), "-filter_complex", video_filter, "-map", "[v]", "-map", "0:a?", "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-movflags", "+faststart", str(output)]
    _run(command)


def process_video(url: str, output_root: Path, clips: int = 3, min_seconds: int = 20, max_seconds: int = 59, whisper_model: str = "base", language: str | None = None, use_llm: bool = True, progress: Callable[[str], None] | None = None, job_id: str | None = None, cookies: str | None = None) -> list[dict[str, Any]]:
    _require_tools()
    url = validate_youtube_url(url)
    if not 1 <= clips <= 5:
        raise ValueError("Choose between 1 and 5 clips.")
    job_id = job_id or uuid.uuid4().hex[:12]
    work_dir = output_root / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    def say(text: str) -> None:
        if progress:
            progress(text)

    say("Downloading audio only for transcription…")
    audio = download_audio(url, work_dir, cookies=cookies)
    say("Transcribing locally… (the first run may download a Whisper model)")
    transcript, duration = transcribe(audio, whisper_model, language)
    candidates = find_candidates(transcript, duration, min_seconds, max_seconds)
    if not candidates:
        raise RuntimeError("Could not find a complete speech segment within the requested duration range. Try shorter minimum duration or another video.")
    say("Choosing moments and writing hooks…")
    selected = analyze_candidates(candidates, clips, use_llm)
    say(f"Downloading and rendering {len(selected)} selected clip(s)…")
    results = []
    for i, clip in enumerate(selected, 1):
        stem = f"{i:02d}-{_safe_stem(clip['hook'], f'clip-{i}')}"
        output = work_dir / f"{stem}.mp4"
        subtitles = work_dir / f"{stem}.ass"
        source = download_section(url, clip, work_dir / f"{stem}.mp4", cookies=cookies)
        write_ass(subtitles, clip, transcript)
        render_clip(source, output, subtitles, clip["end"] - clip["start"])
        source.unlink(missing_ok=True)
        results.append({"filename": output.name, "hook": clip["hook"], "caption": clip["caption"], "reason": clip["reason"], "start": clip["start"], "end": clip["end"]})
    report_path = work_dir / "report.json"
    report_path.write_text(json.dumps({"clips": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    # Remove the full audio/transcript intermediates; keep only final downloads, subtitles and report.
    audio.unlink(missing_ok=True)
    for candidate in work_dir.glob("source_audio.*"):
        candidate.unlink(missing_ok=True)
    say("Finished. Your clips are ready to download.")
    return results
