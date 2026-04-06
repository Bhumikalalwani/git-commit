import unittest
from graders.grader_easy import grade_highlight
from graders.grader_medium import grade_structured
from graders.grader_hard import grade_intent
from env.models import Clip

class TestGraders(unittest.TestCase):
    def setUp(self):
        self.clips = {
            "c1": Clip(id="c1", duration=10, importance=1.0, emotion="happiness", motion="high", tags=["action"]),
            "c2": Clip(id="c2", duration=10, importance=0.5, emotion="sadness", motion="low", tags=["indoor"]),
        }

    def test_grader_easy(self):
        score_perfect = grade_highlight(["c1"], self.clips, 10)
        self.assertGreaterEqual(score_perfect, 0.8)
        
        score_worst = grade_highlight([], self.clips, 10)
        self.assertLessEqual(score_worst, 0.1)

    def test_grader_medium(self):
        score_perfect = grade_structured(["c1", "c2"], ["c1", "c2"])
        self.assertEqual(score_perfect, 1.0)
        
        score_worst = grade_structured(["c3"], ["c1", "c2"])
        self.assertEqual(score_worst, 0.0)

    def test_grader_hard(self):
        score = grade_intent(["c1"], self.clips, "balanced", ["c1", "c2"])
        self.assertTrue(0.0 <= score <= 1.0)

if __name__ == '__main__':
    unittest.main()
