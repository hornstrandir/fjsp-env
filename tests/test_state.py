import unittest
import numpy as np
from fjsp_env.envs.fjsp_env import FjspEnv
from utils.config import ENV_CONFIG


class TestState(unittest.TestCase):
    def test_random(self):
        env = FjspEnv(
            env_config=ENV_CONFIG,
        )
        average = 0
        for _ in range(100):
            state = env.reset()
            self.assertEqual(env.current_time_step, 0)
            legal_actions = env.get_legal_actions()
            done = False
            total_reward = 0
            self.assertTrue(
                max(state["real_obs"].flatten()) <= 1.0, "Out of max bound state"
            )
            self.assertTrue(
                min(state["real_obs"].flatten()) >= 0.0, "Out of min bound state"
            )
            self.assertTrue(
                not np.isnan(state["real_obs"]).any(), "NaN inside state rep!"
            )
            self.assertTrue(
                not np.isinf(state["real_obs"]).any(), "Inf inside state rep!"
            )
            machines_available = set()
            for operation in range(len(env.legal_actions[:-1])):
                if env.legal_actions[operation]:
                    machine_needed = env.needed_machine_operation[operation]
                    machines_available.add(machine_needed)
            self.assertEqual(
                len(machines_available),
                env.nb_machine_legal,
                f"machine available {machines_available} and nb machine available {env.nb_machine_legal} are not coherant, needed_machines: {env.needed_machine_operation}",
            )
            actions_taken = list()
            while not done:
                actions = np.random.choice(
                    len(legal_actions), 1, p=(legal_actions / legal_actions.sum())
                )[0]
                actions_taken.append(actions)
                self.assertEqual(
                legal_actions[:-1].sum(),
                env.nb_legal_actions,
                f"""
                legal_actions {legal_actions[:-1].sum()} and nb legal actions {env.nb_legal_actions} are not coherant, 
                legal_actions: {env.legal_actions}
                """,
                )
                state, rewards, done, _ = env.step(actions)
                legal_actions = env.get_legal_actions()
                total_reward += rewards


                self.assertTrue(
                    max(state["real_obs"].flatten()) <= 1.0, "Out of max bound state"
                    f"state: {state['real_obs']}"
                )
                self.assertTrue(
                    min(state["real_obs"].flatten()) >= 0.0, "Out of min bound state"
                )
                self.assertTrue(
                    not np.isnan(state["real_obs"]).any(), "NaN inside state rep!"
                )
                self.assertTrue(
                    not np.isinf(state["real_obs"]).any(), "Inf inside state rep!"
                )
                machines_available = set()
                for operation in range(len(env.legal_actions[:-1])):
                    if env.legal_actions[operation]:
                        machine_needed = env.needed_machine_operation[operation]
                        machines_available.add(machine_needed)
                self.assertEqual(
                    len(machines_available),
                    env.nb_machine_legal,
                    f"""
                    machines available {machines_available} and nb_machines_legal {env.nb_machine_legal} is not coherant, 
                    machine legal: {env.machine_legal}, 
                    legal actions: {env.legal_actions},
                    current activity jobs: {env.todo_activity_jobs},
                    last activity jobs: {env.last_activity_jobs},
                    needed machines operations: {env.needed_machine_operation},
                    """
                )
                assert len(machines_available) == env.nb_machine_legal

            average += env.last_time_step
            self.assertEqual(len(env.next_time_step), 0)
            for job in range(env.jobs):
                self.assertEqual(
                    env.todo_activity_jobs[job], 
                    env.last_activity_jobs[job] + 1,
                    f"""
                    last activity jobs: {env.last_activity_jobs},
                    todo activity jobs: {env.todo_activity_jobs},
                    solution: {env.solution}
                    actions taken: {actions_taken}
                    time until activity finished: {env.time_until_activity_finished_jobs}
                    time until machine free: {env.time_until_available_machine}
                    """)
            self.assertTrue(env.total_energy_costs, )
            self.assertIsInstance(env.total_energy_costs, float,
            f"energy costs: {env.total_energy_costs}")

if __name__ == '__main__':
    unittest.main()