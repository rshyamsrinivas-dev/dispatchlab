import unittest
from dataclasses import replace
import numpy as np
from dispatchlab.fleet import FleetEnv, FleetConfig, DIST, encode, OBS_SIZE
from dispatchlab.fleet_policies import fleet_heuristic


class FleetTests(unittest.TestCase):
    def test_random_trajectories_obey_physics(self):
        rng = np.random.default_rng(18)
        for seed in range(20):
            env = FleetEnv()
            obs, mask = env.reset(seed)
            while not env.done:
                obs, _, _, mask = env.step(int(rng.choice(np.flatnonzero(mask))))
                self.assertEqual(encode(obs).shape, (OBS_SIZE,))
                self.assertTrue(np.isfinite(encode(obs)).all())
                self.assertEqual(
                    env.metrics["requests"],
                    env.metrics["delivered"]
                    + env.metrics["rejected"]
                    + len(env.orders),
                )
                self.assertLessEqual(len(env.orders), env.config.max_orders)
                self.assertEqual(
                    sum(v["battery"] for v in env.vehicles),
                    2 * env.config.battery
                    + env.metrics["energy_charged"]
                    - env.metrics["energy_used"],
                )
                for i, v in enumerate(env.vehicles):
                    self.assertGreaterEqual(v["battery"], 0)
                    self.assertLessEqual(v["battery"], env.config.battery)
                    reserve = DIST[v["target"] if v["remaining"] else v["location"], 0]
                    self.assertGreaterEqual(v["battery"], reserve)
                    self.assertLessEqual(
                        sum(o["vehicle"] == i for o in env.orders), env.config.capacity
                    )

    def test_exogenous_streams_independent_of_actions(self):
        a, b = FleetEnv(), FleetEnv()
        a.reset(131)
        b.reset(131)
        while not a.done:
            a.step(fleet_heuristic(a.observation(), a.mask(), "planner"))
            b.step(0)
        self.assertEqual(a._arrivals, b._arrivals)
        np.testing.assert_array_equal(a._delays, b._delays)
        self.assertEqual(a.metrics["requests"], b.metrics["requests"])

    def test_traffic_changes_not_demand(self):
        a, b = FleetEnv(), FleetEnv(replace(FleetConfig(), traffic_probability=0.8))
        a.reset(18)
        b.reset(18)
        self.assertEqual(a._arrivals, b._arrivals)
        self.assertFalse(np.array_equal(a._delays, b._delays))

    def test_energy_mask_and_service_time(self):
        e = FleetEnv(replace(FleetConfig(), demand_rate=0, traffic_probability=0))
        e.reset(1)
        e._arrivals = {}
        e._delays[:] = 0
        e.orders = [dict(id=0, dest=2, created=0, deadline=20, priority=1, vehicle=0)]
        e.vehicles[0]["battery"] = 2
        self.assertFalse(e.mask()[3 * 8])  # needs 2 outward + 2 reserve
        e.vehicles[0]["battery"] = 4
        e.step(3 * 8)
        self.assertEqual(e.metrics["delivered"], 0)
        e.step(0)
        self.assertEqual(e.metrics["delivered"], 1)
        self.assertEqual(e.vehicles[0]["cooldown"], 1)
        self.assertFalse(e.mask()[8])
        e.step(0)
        self.assertTrue(e.mask()[8])

    def test_both_vehicles_advance_together(self):
        e = FleetEnv()
        e.reset(1)
        e._arrivals = {}
        e._delays[:] = 0
        e.orders = [
            dict(id=i, dest=2 + 3 * i, created=0, deadline=20, priority=1, vehicle=i)
            for i in range(2)
        ]
        e.step(3 * 8 + 6)
        self.assertEqual([v["remaining"] for v in e.vehicles], [1, 1])
        e.step(0)
        self.assertEqual(e.metrics["delivered"], 2)

    def test_baseline_validity(self):
        for mode in ["nearest", "deadline", "planner"]:
            for seed in range(5):
                e = FleetEnv()
                o, m = e.reset(seed)
                while not e.done:
                    o, _, _, m = e.step(fleet_heuristic(o, m, mode))

    def test_terminal_and_invalid_action(self):
        e = FleetEnv()
        e.reset(3)
        with self.assertRaises(ValueError):
            e.step(64)
        while not e.done:
            e.step(0)
        with self.assertRaises(RuntimeError):
            e.step(0)
        self.assertEqual(e.metrics["unfinished"], len(e.orders))

    def test_no_private_forecasts_in_observation(self):
        e = FleetEnv()
        o, _ = e.reset(3)
        self.assertNotIn("_arrivals", o)
        self.assertNotIn("_delays", o)
        o["vehicles"][0]["battery"] = 0
        self.assertEqual(e.vehicles[0]["battery"], e.config.battery)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch extra not installed")
class DeepAgentTests(unittest.TestCase):
    def test_masked_action_and_checkpoint(self):
        import tempfile
        from pathlib import Path
        from dispatchlab.deep_agent import DoubleDQN

        e = FleetEnv()
        o, m = e.reset(2)
        a = DoubleDQN(seed=2, training=False)
        self.assertTrue(m[a.act(o, m)])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "model.pt"
            a.save(p)
            b = DoubleDQN.load(p)
            np.testing.assert_array_equal(a.q_values(o), b.q_values(o))

    def test_real_gradient_update(self):
        from dispatchlab.deep_agent import DoubleDQN

        a = DoubleDQN(seed=1)
        e = FleetEnv()
        o, m = e.reset(3)
        x = encode(o)
        for i in range(1024):
            a.buffer.add(x, 0, -1, x, True, m)
        before = a.q_values(o).copy()
        loss = a.optimise()
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(np.array_equal(before, a.q_values(o)))
