import unittest
from dataclasses import replace
import numpy as np
from dispatchlab.environment import Config, DispatchEnv, TRAVEL
from dispatchlab.agent import LinearQAgent, features, train_episode
from dispatchlab.experiment import run_episode
from dispatchlab.policies import heuristic


class EnvironmentTests(unittest.TestCase):
    def test_reproducibility_and_no_future_exposure(self):
        a, b = DispatchEnv(), DispatchEnv()
        x, _ = a.reset(seed=21)
        y, _ = b.reset(seed=21)
        self.assertEqual(x, y)
        self.assertNotIn("schedule", x)
        self.assertEqual(a._schedule, b._schedule)
        for _ in range(20):
            action = heuristic(a.observation(), a.action_mask())
            oa, ra, *_ = a.step(action)
            ob, rb, *_ = b.step(action)
            self.assertEqual(oa, ob)
            self.assertEqual(ra, rb)

    def test_conservation_and_capacity(self):
        rng = np.random.default_rng(6)
        for seed in range(40):
            e = DispatchEnv()
            e.reset(seed=seed)
            while not e.done:
                e.step(int(rng.choice(np.flatnonzero(e.action_mask()))))
                self.assertLessEqual(len(e.orders), e.config.max_orders)
                self.assertLessEqual(
                    sum(o["status"] for o in e.orders), e.config.capacity
                )
                self.assertEqual(
                    e.metrics["requests"],
                    e.metrics["delivered"] + e.metrics["rejected"] + len(e.orders),
                )

    def test_delivery_timing_and_no_teleportation(self):
        e = DispatchEnv(replace(Config(), arrival_probability=0))
        e.reset(seed=2)
        dest = e.orders[0]["dest"]
        distance = int(TRAVEL[0, dest])
        for tick in range(distance):
            e.step(dest + 1 if tick == 0 else 0)
            self.assertEqual(e.metrics["delivered"], int(tick == distance - 1))
        self.assertEqual(e.location, dest)
        self.assertEqual(e.metrics["travel_ticks"], distance)

    def test_invalid_actions_and_terminal(self):
        e = DispatchEnv()
        e.reset(seed=3)
        with self.assertRaises(ValueError):
            e.step(1)
        with self.assertRaises(ValueError):
            e.step(-1)
        while not e.done:
            e.step(0)
        with self.assertRaises(RuntimeError):
            e.step(0)
        self.assertEqual(e.t, e.config.horizon)

    def test_late_cost_charged_once_each_tick(self):
        e = DispatchEnv(replace(Config(), arrival_probability=0))
        e.reset(seed=0)
        e._schedule = {}
        e.orders = [dict(id=0, dest=3, deadline=1, created=0, status=1)]
        _, r1, *_ = e.step(4)
        _, r2, *_ = e.step(0)
        self.assertAlmostEqual(r1, -0.35)
        self.assertAlmostEqual(r2, 10 - 0.35 - 0.7)

    def test_terminal_unfinished_and_overflow_penalties(self):
        e = DispatchEnv(
            replace(Config(), horizon=3, arrival_cutoff=3, max_orders=1, capacity=1)
        )
        e.reset(seed=0)
        e._schedule = {1: (2, 10), 2: (3, 10)}
        rewards = [e.step(0)[1] for _ in range(3)]
        self.assertEqual(e.metrics["rejected"], 2)
        self.assertAlmostEqual(sum(rewards), -24)
        self.assertEqual(e.metrics["unfinished"], 1)

    def test_action_independent_arrivals(self):
        a, b = DispatchEnv(), DispatchEnv()
        a.reset(seed=19)
        b.reset(seed=19)
        schedule = a._schedule.copy()
        while not a.done:
            a.step(heuristic(a.observation(), a.action_mask()))
            b.step(0)
        self.assertEqual(schedule, b._schedule)
        self.assertEqual(a.metrics["requests"], b.metrics["requests"])

    def test_baselines_obey_action_mask(self):
        for mode in ("nearest", "deadline"):
            for k in (1, 2, 3):
                for seed in range(10):
                    run_episode(Config(), seed, lambda o, m: heuristic(o, m, mode, k))

    def test_semimarkov_discount_and_terminal_update(self):
        e = DispatchEnv()
        obs, info = e.reset(seed=1)
        a = LinearQAgent(alpha=1, gamma=0.5)
        x = features(obs, 0)
        a.weights[:] = 0.1
        before = a.weights.copy()
        a.update(obs, 0, 3, 4, obs, info["action_mask"], True)
        expected = before + (3 - before @ x) * x / (1 + x @ x)
        np.testing.assert_allclose(a.weights, expected)
        before = a.weights.copy()
        bootstrap = max(a.values(obs, info["action_mask"]).values())
        a.update(obs, 0, 3, 4, obs, info["action_mask"], False)
        expected = before + (3 + 0.5**4 * bootstrap - before @ x) * x / (1 + x @ x)
        np.testing.assert_allclose(a.weights, expected)

    def test_training_and_model_roundtrip(self):
        import tempfile
        from pathlib import Path

        a = LinearQAgent(seed=1)
        for seed in range(10):
            train_episode(DispatchEnv(), a, seed, 0.3)
        self.assertTrue(np.isfinite(a.weights).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            a.save(path)
            b = LinearQAgent.load(path)
            np.testing.assert_array_equal(a.weights, b.weights)


if __name__ == "__main__":
    unittest.main()
