import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
    import torch  # noqa: F401 - optional dependency availability check
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, "Optional API dependencies not installed")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dispatchlab.api import app

        cls.client = TestClient(app)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["synthetic"])

    def test_input_validation(self):
        for body in [{"seed": -1}, {"scenario": "unknown"}, {"policy": "oracle"}]:
            self.assertEqual(self.client.post("/simulate", json=body).status_code, 422)

    def test_fresh_simulation_reproducible(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / "advanced/results/training.json").exists():
            self.skipTest("Model outputs not bundled")
        body = dict(seed=987654, policy="planner", scenario="standard")
        a = self.client.post("/simulate", json=body)
        b = self.client.post("/simulate", json=body)
        self.assertEqual(a.status_code, 200)
        self.assertEqual(a.json()["frames"], b.json()["frames"])
        self.assertEqual(len(a.json()["frames"]), 61)
        m = a.json()["metrics"]
        self.assertEqual(
            m["requests"], m["delivered"] + m["rejected"] + m["unfinished"]
        )

    def test_neural_model_serves(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / "advanced/models/ddqn.pt").exists():
            self.skipTest("Trained model not bundled")
        r = self.client.post("/simulate", json={"seed": 999999, "policy": "ddqn"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["frames"][-1]["time"], 60)
