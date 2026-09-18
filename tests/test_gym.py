import unittest

try:
    import gymnasium
except ImportError:
    gymnasium = None


@unittest.skipIf(gymnasium is None, "Optional gym extra not installed")
class GymTests(unittest.TestCase):
    def test_api_contract(self):
        from gymnasium.utils.env_checker import check_env
        from dispatchlab.gym_env import GymDispatchEnv

        check_env(GymDispatchEnv(), skip_render_check=True)

    def test_valid_trajectory_matches_core(self):
        from dispatchlab.environment import DispatchEnv
        from dispatchlab.gym_env import GymDispatchEnv
        from dispatchlab.policies import heuristic

        a, b = DispatchEnv(), GymDispatchEnv()
        a.reset(seed=100)
        b.reset(seed=100)
        while not a.done:
            action = heuristic(a.observation(), a.action_mask())
            _, r1, d1, *_ = a.step(action)
            obs, r2, d2, _, _ = b.step(action)
            self.assertEqual((r1, d1), (r2, d2))
            self.assertTrue(b.observation_space.contains(obs))
