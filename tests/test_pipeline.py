import unittest
from pathlib import Path
from unittest import mock

from shortsmith.pipeline import download_section, find_candidates, validate_youtube_url

try:
    import yt_dlp  # noqa: F401
except ImportError:
    yt_dlp = None


class UrlTests(unittest.TestCase):
    def test_watch_url_normalizes(self):
        self.assertEqual(validate_youtube_url("https://www.youtube.com/watch?v=abcdefghijk"), "https://www.youtube.com/watch?v=abcdefghijk")

    def test_short_url_normalizes(self):
        self.assertEqual(validate_youtube_url("https://youtu.be/abcdefghijk?t=42"), "https://www.youtube.com/watch?v=abcdefghijk")

    def test_rejects_non_youtube_host(self):
        with self.assertRaises(ValueError):
            validate_youtube_url("https://youtube.com.attacker.example/watch?v=abcdefghijk")

    def test_rejects_playlist_without_video_id(self):
        with self.assertRaises(ValueError):
            validate_youtube_url("https://www.youtube.com/playlist?list=PL123456")

    def test_embed_live_and_v_urls_normalize(self):
        for url in (
            "https://www.youtube.com/embed/abcdefghijk",
            "https://www.youtube.com/live/abcdefghijk",
            "https://www.youtube.com/v/abcdefghijk",
            "https://www.youtube-nocookie.com/embed/abcdefghijk",
        ):
            self.assertEqual(validate_youtube_url(url), "https://www.youtube.com/watch?v=abcdefghijk")

    def test_music_youtube_url_normalizes(self):
        self.assertEqual(
            validate_youtube_url("https://music.youtube.com/watch?v=abcdefghijk"),
            "https://www.youtube.com/watch?v=abcdefghijk",
        )


@unittest.skipUnless(yt_dlp is not None, "yt-dlp is not installed")
class DownloadSectionTests(unittest.TestCase):
    def test_no_absolute_seek_postprocessor_args(self):
        # Regression: postprocessor_args with -ss/-t used to be appended to
        # every ffmpeg invocation, including the merger running on the
        # already-cut section file, seeking past the end of the clip.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            fake = work / "01-test.source.mp4"
            fake.write_bytes(b"fake")
            captured = {}

            class FakeYoutubeDL:
                def __init__(self, options):
                    captured.update(options)

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def download(self, urls):
                    pass

            with mock.patch("yt_dlp.YoutubeDL", FakeYoutubeDL), mock.patch(
                "shortsmith.pipeline._probe_duration", return_value=30.0
            ):
                result = download_section(
                    "https://www.youtube.com/watch?v=abcdefghijk",
                    {"start": 120.0, "end": 150.0},
                    work / "01-test.mp4",
                )
            self.assertEqual(result, fake)
            self.assertNotIn("postprocessor_args", captured)
            self.assertEqual(captured["download_sections"], ["*120.000-150.000"])
            self.assertTrue(captured["force_keyframes_at_cuts"])


class CandidateTests(unittest.TestCase):
    def test_finds_sentence_ending_candidates(self):
        segments = [
            {"start": 0.0, "end": 10.0, "text": "Why does this happen?", "words": []},
            {"start": 10.0, "end": 20.0, "text": "Nobody tells you the secret.", "words": []},
            {"start": 20.0, "end": 30.0, "text": "But here is the truth!", "words": []},
        ]
        candidates = find_candidates(segments, 30, 20, 30)
        self.assertTrue(candidates)
        self.assertGreaterEqual(candidates[0]["end"] - candidates[0]["start"], 20)

    def test_rejects_invalid_duration_range(self):
        with self.assertRaises(ValueError):
            find_candidates([], 60, 59, 20)


if __name__ == "__main__":
    unittest.main()
