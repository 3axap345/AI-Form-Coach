from __future__ import annotations

from test_support import (
    JOINT_INDEX,
    Config,
    analyze_form,
    synthetic_squat_sample,
    top_detected_issues,
    unittest,
)


class TestFormAnalysis(unittest.TestCase):
    def test_form_analysis_shallow_depth(self):
        sample = synthetic_squat_sample()
        bottom = slice(25, 36)
        sample[bottom, JOINT_INDEX["left_hip"], :2] = (0.30, 0.55)
        sample[bottom, JOINT_INDEX["right_hip"], :2] = (-0.30, 0.55)
        sample[bottom, JOINT_INDEX["left_knee"], :2] = (0.34, 1.1)
        sample[bottom, JOINT_INDEX["right_knee"], :2] = (-0.34, 1.1)
        analysis = analyze_form(sample, Config())
        self.assertTrue(analysis["shallow_depth"]["detected"])
        self.assertIn("Squat depth", analysis["shallow_depth"]["message"])

    def test_form_analysis_excessive_forward_lean(self):
        sample = synthetic_squat_sample()
        bottom = slice(25, 36)
        sample[bottom, JOINT_INDEX["left_shoulder"], 0] += 0.9
        sample[bottom, JOINT_INDEX["right_shoulder"], 0] += 0.9
        analysis = analyze_form(sample, Config())
        self.assertTrue(analysis["excessive_forward_lean"]["detected"])

    def test_form_analysis_knee_valgus(self):
        sample = synthetic_squat_sample()
        bottom = slice(25, 36)
        sample[bottom, JOINT_INDEX["left_knee"], 0] = 0.08
        sample[bottom, JOINT_INDEX["right_knee"], 0] = -0.08
        analysis = analyze_form(sample, Config())
        self.assertTrue(analysis["knee_valgus"]["detected"])

    def test_form_analysis_heel_instability(self):
        sample = synthetic_squat_sample()
        bottom = slice(25, 36)
        sample[bottom, JOINT_INDEX["left_heel"], :2] += (0.25, -0.15)
        sample[bottom, JOINT_INDEX["right_heel"], :2] += (-0.25, -0.15)
        analysis = analyze_form(sample, Config())
        self.assertTrue(analysis["heel_instability"]["detected"])

    def test_top_detected_issues_limits_and_sorts(self):
        analysis = {
            "knee_valgus": {"detected": True, "score": 0.4, "message": "a"},
            "shallow_depth": {"detected": True, "score": 0.9, "message": "b"},
            "heel_instability": {"detected": True, "score": 0.6, "message": "c"},
            "_metrics": {},
        }
        issues = top_detected_issues(analysis, limit=2)
        self.assertEqual([issue["message"] for issue in issues], ["b", "c"])
