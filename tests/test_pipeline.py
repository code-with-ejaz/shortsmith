import unittest

from shortsmith.pipeline import find_candidates, validate_youtube_url


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
