import unittest

from env.models import Clip
from graders.grader_easy import grade_highlight
from graders.grader_hard import grade_intent
from graders.grader_medium import grade_structured


class TestGraderEasy(unittest.TestCase):
    def setUp(self):
        self.clips = {
            "c1": Clip(id="c1", duration=10, importance=1.0, emotion="happiness", motion="high", tags=["action"]),
            "c2": Clip(id="c2", duration=10, importance=0.5, emotion="sadness", motion="low", tags=["indoor"]),
        }

    def test_perfect_selection(self):
        score = grade_highlight(["c1"], self.clips, 10)
        self.assertGreaterEqual(score, 0.8)

    def test_empty_timeline(self):
        score = grade_highlight([], self.clips, 10)
        self.assertEqual(score, 0.001)

    def test_score_in_range(self):
        score = grade_highlight(["c1", "c2"], self.clips, 20)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


class TestGraderMedium(unittest.TestCase):
    def test_perfect_order(self):
        self.assertEqual(grade_structured(["c1", "c2"], ["c1", "c2"]), 0.999)

    def test_no_overlap(self):
        self.assertEqual(grade_structured(["c3"], ["c1", "c2"]), 0.001)

    def test_partial_match(self):
        score = grade_structured(["c1", "c3", "c2"], ["c1", "c2"])
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


class TestGraderHard(unittest.TestCase):
    def setUp(self):
        self.clips = {
            "c1": Clip(id="c1", duration=10, importance=1.0, emotion="happiness", motion="high", tags=["action"]),
            "c2": Clip(id="c2", duration=10, importance=0.5, emotion="sadness", motion="low", tags=["indoor"]),
        }

    def test_score_in_range(self):
        score = grade_intent(["c1"], self.clips, "balanced", ["c1", "c2"])
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_returns_min(self):
        score = grade_intent([], self.clips, "balanced", ["c1", "c2"])
        self.assertEqual(score, 0.001)


if __name__ == "__main__":
    unittest.main()
