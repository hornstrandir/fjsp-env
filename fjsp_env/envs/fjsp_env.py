import gym
import random
import bisect
import numpy as np
import datetime
from pathlib import Path


class IllegalActionError(Exception):
    pass


# TODO: check if illegal actions do have some heavy computations. If so: use: if legal_action(action): ...
class FjspEnv(gym.Env):
    """ """

    def __init__(self, env_config) -> None:
        super().__init__()
        instance_path = env_config["instance_path"]
        price_data_path = env_config["energy_data_path"]
        self.loose_noop_restrictions = env_config["loose_noop_restrictions"]
        self.count_noop = 0
        self.sum_time_activities = 0  # used to scale observations
        self.jobs = 0
        self.machines = 0
        self.max_alternatives = 0
        self.operations = 0
        self.instance_map = {}
        self.jobs_max_duration = 0
        self.max_time_op = 0
        self.max_time_jobs = 0
        self.last_activity_jobs = None
        # load energy data
        self.prices = np.load(price_data_path)
        self._total_electricity_costs = 0
        self.max_price = np.amax(self.prices)
        self.min_price = np.amin(self.prices)
        self.current_price = None
        self.alpha = env_config["alpha"]
        # initial values for variables used for solving (to reinitialize when reset() is called)
        self.solution = None
        self.solution_machines = None
        self.last_solution = None
        self.last_time_step = float("inf")
        self.current_time_step = float("inf")
        self.next_time_step = list()
        self.next_action = list()
        self.legal_actions = None
        self.time_until_available_machine = None
        self.time_until_activity_finished_jobs = None
        self.todo_activity_jobs = None
        self.total_perform_op_time_jobs = None
        self.needed_machine_operation = None
        self.total_idle_time_jobs = None
        self.idle_time_jobs_last_op = None
        self.state = None
        self.illegal_actions = None
        self.action_illegal_no_op = None
        self.machine_legal = None
        self.legal_jobs = None
        # initial values for variables used for representation
        self.start_timestamp = datetime.datetime.now().timestamp()
        self.sum_op = 0
        with open(instance_path, "r") as instance_file:
            for line_count, line in enumerate(instance_file):
                splitted_line = line.split()
                if len(splitted_line) == 0:
                    break
                if line_count == 0:
                    self.jobs = int(splitted_line[0])
                    self.machines = int(splitted_line[1])
                    self.max_activities_jobs = int(splitted_line[2])
                    self.max_alternatives = int(splitted_line[3])
                    self.operations = self.jobs * self.max_alternatives
                    self.jobs_max_duration = np.zeros(self.jobs, dtype=int)
                    self.last_activity_jobs = np.zeros(self.jobs, dtype=int)
                    self.legal_jobs = np.ones(self.jobs, dtype=int)
                else:
                    idx = 1
                    # start counting jobs at null
                    job_nb = line_count - 1
                    id_activity = 0
                    self.last_activity_jobs[job_nb] = str(int(splitted_line[0]) - 1)
                    while idx < len(splitted_line):
                        number_operations = int(splitted_line[idx])
                        max_time_activity = 0
                        for id_operation in range(1, number_operations + 1):
                            machine, time = int(
                                splitted_line[idx + 2 * id_operation - 1]
                            ) - 1, int(splitted_line[idx + 2 * id_operation])
                            if time > max_time_activity:
                                max_time_activity = time
                            key = self._ids_to_key(
                                job_nb, id_activity, id_operation - 1
                            )
                            self.instance_map[key] = machine, time
                        self.jobs_max_duration[job_nb] += max_time_activity
                        self.max_time_op = max(self.max_time_op, max_time_activity)
                        self.sum_op += max_time_activity
                        id_activity += 1
                        idx += 1 + 2 * number_operations
                        self.sum_time_activities += max_time_activity
        self.max_time_jobs = np.amax(self.jobs_max_duration)
        self.power_consumption_machines = np.array(
            env_config["power_consumption_machines"].get(str(self.machines))
        )
        self.max_power_consumption = np.max(self.power_consumption_machines)
        assert self.max_time_op > 0
        assert self.max_time_jobs > 0
        assert self.jobs > 0
        assert self.machines > 1, "We need at least 2 machines"
        assert self.instance_map is not None
        assert self.min_price >= 0
        self.action_space = gym.spaces.Discrete(self.operations + 1)
        # used for plotting
        self.colors = [
            tuple([random.random() for _ in range(3)]) for _ in range(self.machines)
        ]
        """
        matrix with the following attributes for each job:
            -Legal job
            -Left over time on the current op
            -Current operation %
            -Total left over time
            -When next machine available
            -Time since IDLE: 0 if not available, time otherwise
            -Total IDLE time in the schedule
        """
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": gym.spaces.Box(0, 1, shape=(self.operations + 1,)),
                "real_obs": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self.operations, 9), dtype=float
                ),
            }
        )

    @property
    def total_energy_costs(self):
        total_energy_costs = 0
        for operation in range(self.operations):
            id_job = operation // self.max_alternatives
            for activity in range(self.last_activity_jobs[id_job] + 1):
                op_start_time = self.solution[operation][activity]
                if op_start_time == -1:
                    continue
                key = self._ids_to_key(
                    id_job, activity, operation - id_job * self.max_alternatives
                )
                operation_data = self.instance_map.get(key)
                if operation_data is None:
                    continue
                machine_needed, time_needed = operation_data
                avg_price = np.average(
                    self.prices[op_start_time : op_start_time + time_needed]
                )
                avg_price *= self.max_price - self.min_price
                avg_price += self.min_price
                total_energy_costs += (
                    avg_price
                    * 60
                    / 1000
                    * self.power_consumption_machines[machine_needed]
                    * time_needed
                )
        return total_energy_costs

    @property
    def nb_legal_actions(self):
        return self.legal_actions[:-1].sum()

    @property
    def nb_machine_legal(self):
        return self.machine_legal.sum()

    def _ids_to_key(self, id_job: int, id_activity: int, id_operation: int) -> str:
        return str(id_job) + str(id_activity) + str(id_operation)

    def _get_current_state_representation(self):
        self.state[:, 0] = self.legal_actions[:-1]
        return {
            "real_obs": self.state,
            "action_mask": self.legal_actions,
        }

    def get_legal_actions(self):
        return self.legal_actions

    def reset(self):
        self.current_time_step = 0
        self.next_time_step = list()
        self.next_action = list()
        self.legal_actions = np.ones(self.operations + 1, dtype=bool)
        self.legal_actions[self.operations] = False
        self.machine_legal = np.zeros(self.machines, dtype=bool)
        self.legal_jobs = np.ones(self.jobs, dtype=bool)
        self.solution = np.full(
            (self.operations, self.max_activities_jobs), -1, dtype=int
        )
        self.time_until_available_machine = np.zeros(self.machines, dtype=int)
        self.time_until_activity_finished_jobs = np.zeros(self.jobs, dtype=int)
        self.todo_activity_jobs = np.zeros(self.jobs, dtype=int)
        self.total_perform_op_time_jobs = np.zeros(self.jobs, dtype=int)
        self.needed_machine_operation = np.full(self.operations, -1)
        self.total_idle_time_jobs = np.zeros(self.jobs, dtype=int)
        self.idle_time_jobs_last_op = np.zeros(self.jobs, dtype=int)
        self.illegal_actions = np.zeros((self.machines, self.operations), dtype=bool)
        self.action_illegal_no_op = np.zeros(self.operations, dtype=bool)
        self.count_noop = 0
        self._total_electricity_costs = 0
        for operation in range(self.operations):
            id_job = operation // self.max_alternatives
            id_operation = operation - id_job * self.max_alternatives
            key = self._ids_to_key(id_job, 0, id_operation)
            operation_data = self.instance_map.get(key)
            if operation_data is None:
                continue
            needed_machine = operation_data[0]
            self.needed_machine_operation[operation] = needed_machine
            if not self.machine_legal[needed_machine]:
                self.machine_legal[needed_machine] = True
        for operation, machine in enumerate(self.needed_machine_operation):
            if machine == -1:
                # This is just a placeholder. Thus this action must be illegal
                self.legal_actions[operation] = False
        self.state = np.zeros((self.operations, 9), dtype=float)
        return self._get_current_state_representation()

    def _prioritization_non_final(self):
        if self.nb_machine_legal < 1:
            return
        for machine in range(self.machines):
            if not self.machine_legal[machine]:
                continue
            final_operations = list()
            keys_final_operations = list()
            non_final_operations = list()
            min_non_final = float("inf")
            for operation in range(self.operations):
                id_job = operation // self.max_alternatives
                if not (
                    self.needed_machine_operation[id_job] == machine
                    and self.legal_actions[operation]
                ):
                    continue
                id_activity = self.todo_activity_jobs[id_job]
                id_operation = operation - id_job * self.max_alternatives
                key_this_step = self._ids_to_key(id_job, id_activity, id_operation)
                # TODO: check if it is possible if this can be called on the really last activity not the penultimate activity
                if id_activity == self.last_activity_jobs[id_job]:
                    keys_final_operations.append(key_this_step)
                    final_operations.append(operation)
                else:
                    operation_data = self.instance_map.get(key_this_step)
                    if operation_data is None:
                        continue
                    time_needed_legal = operation_data[1]
                    key_next_step = self._ids_to_key(
                        id_job, id_activity + 1, id_operation
                    )
                    operation_data = self.instance_map.get(key_next_step)
                    if operation_data is None:
                        continue
                    machine_needed_nextstep = operation_data[0]
                    if self.time_until_available_machine[machine_needed_nextstep] == 0:
                        min_non_final = min(min_non_final, time_needed_legal)
                        non_final_operations.append(operation)
            if len(non_final_operations) > 0:
                for idx, key in enumerate(keys_final_operations):
                    operation = final_operations[idx]
                    operation_data = self.instance_map.get(key)
                    if operation_data is None:
                        continue
                    time_needed_legal = operation_data[1]
                    if time_needed_legal > min_non_final:
                        self.legal_actions[operation] = False
            needed_machines = self.needed_machine_operation[self.legal_actions[:-1]]
        for machine in range(self.machines):
            if machine in needed_machines:
                self.machine_legal[machine] = True
            else:
                self.machine_legal[machine] = False

    def _check_no_op(self):
        """
        Restrictions:
        1. Disallow No-Op if: four or more machines with some allocatable job.
        2. Disallow No-Op if: five or more allocatable jobs.
        3. If we make a pause longer than D it would be better to
            allocate a job with duration D.
        4. We do not wait for jobs that will be rejected by non-final.
        """
        self.legal_actions[self.operations] = False
        if not self.loose_noop_restrictions:
            noop_restriction = self.nb_machine_legal <= 3 and self.nb_legal_actions <= 4
        else:
            noop_restriction = True
        if len(self.next_time_step) > 0 and noop_restriction:
            machine_next = set()
            next_time_step = self.next_time_step[0]
            max_horizon = self.current_time_step
            max_horizon_machine = [
                self.current_time_step + self.max_time_op for _ in range(self.machines)
            ]
            # return if there is an legal job that ends before the next time step
            # max horizion is the min duration the machine is occupied.
            for operation in range(self.operations):
                if self.legal_actions[operation]:
                    id_job = operation // self.max_alternatives
                    id_activity = self.todo_activity_jobs[id_job]
                    id_operation = operation - id_job * self.max_alternatives
                    key = self._ids_to_key(id_job, id_activity, id_operation)
                    operation_data = self.instance_map.get(key)
                    if operation_data is None:
                        continue
                    machine_needed, time_needed = operation_data
                    end_job = self.current_time_step + time_needed
                    if end_job < next_time_step:
                        return
                    max_horizon_machine[machine_needed] = min(
                        max_horizon_machine[machine_needed], end_job
                    )
                    max_horizon = max(max_horizon, max_horizon_machine[machine_needed])
            # TODO: what is happening here
            for operation in range(self.operations):
                if self.legal_actions[operation]:
                    continue
                id_job = operation // self.max_alternatives
                id_activity = self.todo_activity_jobs[id_job]
                id_operation = operation - id_job * self.max_alternatives
                if (
                    self.time_until_activity_finished_jobs[id_job] > 0
                    and id_activity < self.last_activity_jobs[id_job]
                ):
                    time_step = id_activity + 1
                    time_needed = (
                        self.current_time_step
                        + self.time_until_activity_finished_jobs[id_job]
                    )
                    while (
                        time_step < self.last_activity_jobs[id_job] - 1
                        and max_horizon > time_needed
                    ):
                        next_key = self._ids_to_key(id_operation, time_step, id_job)
                        operation_data = self.instance_map.get(next_key)
                        if operation_data is None:
                            time_step += 1
                            continue
                        machine_needed, duration_operation = operation_data
                        if (
                            max_horizon_machine[machine_needed] > time_needed
                            and self.machine_legal[machine_needed]
                        ):
                            machine_next.add(machine_needed)
                            # TODO: Why is that?
                            if len(machine_next) == self.nb_machine_legal:
                                self.legal_actions[self.operations] = True
                                return
                        time_needed += duration_operation
                        time_step += 1
                elif (
                    not self.action_illegal_no_op[operation]
                    and id_activity <= self.last_activity_jobs[id_job]
                ):
                    time_step = self.todo_activity_jobs[id_job]
                    key_next_timestep = self._ids_to_key(
                        id_job, time_step, id_operation
                    )
                    operation_data = self.instance_map.get(key_next_timestep)
                    if operation_data is None:
                        continue
                    machine_needed = operation_data[0]
                    time_needed = (
                        self.current_time_step
                        + self.time_until_available_machine[machine_needed]
                    )
                    while (
                        time_step < self.last_activity_jobs[id_job]
                        and max_horizon > time_needed
                    ):
                        key_next_timestep = self._ids_to_key(
                            id_operation, time_step, id_job
                        )
                        operation_data = self.instance_map.get(key_next_timestep)
                        if operation_data is None:
                            time_step += 1
                            continue
                        machine_needed, duration_operation = operation_data
                        if (
                            max_horizon_machine[machine_needed] > time_needed
                            and self.machine_legal[machine_needed]
                        ):
                            machine_next.add(machine_needed)
                            if len(machine_next) == self.nb_machine_legal:
                                self.legal_actions[self.operations] = True
                                return
                        time_needed += duration_operation
                        time_step += 1

    def step(self, action: int):
        time_reward = 0.0
        if action == self.operations:
            self.count_noop += 1
            for operation in range(self.operations):
                if self.legal_actions[operation]:
                    self.legal_actions[operation] = False
                    needed_machine = self.needed_machine_operation[operation]
                    self.machine_legal[needed_machine] = False
                    self.illegal_actions[needed_machine][operation] = True
                    self.action_illegal_no_op[operation] = True
            while self.nb_machine_legal == 0:
                time_reward -= self.increase_time_step()
            scaled_reward = self._reward_scaler(time_reward, energy_penalty=0)
            self._prioritization_non_final()
            self._check_no_op()
            return (
                self._get_current_state_representation(),
                scaled_reward,
                self._is_done(),
                {},
            )
        else:
            id_job = action // self.max_alternatives
            id_activity = self.todo_activity_jobs[id_job]
            id_operation = action - id_job * self.max_alternatives
            key = self._ids_to_key(id_job, id_activity, id_operation)
            current_activity_job = self.todo_activity_jobs[id_job]
            operation_data = self.instance_map.get(key)
            # TODO: delete if tests are ok
            if operation_data is None:
                print(f"illegal action: {action}")
                raise IllegalActionError
            machine_needed, time_needed = operation_data
            energy_penalty = self._calculate_energy_penalty(action, time_needed)
            time_reward += time_needed
            self.time_until_available_machine[machine_needed] = time_needed
            self.time_until_activity_finished_jobs[id_job] = time_needed
            self.state[action][1] = time_needed / self.max_time_op
            to_add_time_step = self.current_time_step + time_needed
            # We can skip the timesteps where nothing happens in the env.
            if to_add_time_step not in self.next_time_step:
                index = bisect.bisect_left(self.next_time_step, to_add_time_step)
                self.next_time_step.insert(index, to_add_time_step)
                self.next_action.insert(index, action)
            self.solution[action][current_activity_job] = self.current_time_step
            self.legal_jobs[id_job] = False
            for operation in range(self.operations):
                if (
                    self.needed_machine_operation[operation] == machine_needed
                    and self.legal_actions[operation]
                ):
                    self.legal_actions[operation] = False
                # All alternatives of the action taken are set to illegal
                elif (
                    self.legal_actions[operation]
                    and (operation // self.max_alternatives) == id_job
                ):
                    self.legal_actions[operation] = False
            self.machine_legal[machine_needed] = False
            # Reduce number of machine legal if the only operation that needed this machine
            # is an alternative of the action that has been choosen.
            needed_machines = self.needed_machine_operation[self.legal_actions[:-1]]
            for machine in range(self.machines):
                if machine in needed_machines:
                    self.machine_legal[machine] = True
                else:
                    self.machine_legal[machine] = False
            # TODO: check this in the original code. what is it for
            for operation in range(self.operations):
                if self.illegal_actions[machine_needed][operation]:
                    self.action_illegal_no_op[operation] = False
                    self.illegal_actions[machine_needed][operation] = False
            # if we can't allocate new job in the current timestep, we pass to the next one
            while self.nb_machine_legal == 0 and len(self.next_time_step) > 0:
                time_reward -= self.increase_time_step()
            self._prioritization_non_final()
            self._check_no_op()
            # we then need to scale the reward
            scaled_reward = self._reward_scaler(time_reward, energy_penalty)
            return (
                self._get_current_state_representation(),
                scaled_reward,
                self._is_done(),
                {},
            )

    def _update_power_observations(self):
        """
        Must be called after increasing the timestep since the calculations
        depend on the next states repr.
        """
        for operation in range(self.operations):
            id_job = operation // self.max_alternatives
            id_activity = self.todo_activity_jobs[id_job]
            id_operation = operation - id_job * self.max_alternatives
            if self.todo_activity_jobs[id_job] <= self.last_activity_jobs[id_job]:
                key = self._ids_to_key(id_job, id_activity, id_operation)
                operation_data = self.instance_map.get(key)
                if operation_data is None:
                    continue
                machine_needed, time_needed = operation_data
                self.state[operation][7] = (
                    self.power_consumption_machines[machine_needed]
                    / self.max_power_consumption
                )
                # since we are only interested in observations of legal actions,
                # we take the avg of the time when the action can be selected.
                avg_price = np.average(
                    self.prices[
                        self.current_time_step : self.current_time_step + time_needed
                    ]
                )
                # MinMax scaling:
                scaled_avg = (avg_price - self.min_price) / (
                    self.max_price - self.min_price
                )
                self.state[operation][8] = scaled_avg
            else:
                self.state[operation][7] = 1.0
                self.state[operation][8] = 1.0

    # TODO: energy penalty can contain total energy costs
    def _calculate_energy_penalty(self, action: int, processing_time: int):
        """
        The penalty is scaled by the max_price.

        penalty = avg(price_vector) * power_consumption_machine / 60 / max_price
        """
        avg_price = np.average(
            self.prices[
                self.current_time_step : self.current_time_step + processing_time
            ]
        )
        # here we can even use negative values.
        # It is good to schedule energy intense ops when there are negative energy prices.
        power_consumption = self.power_consumption_machines[
            self.needed_machine_operation[action]
        ]
        # using ratio avg_price/max_energy_price thus unitless
        return (
            avg_price / self.max_price * power_consumption / self.max_power_consumption
        )

    # TODO: Impact of ignoring energy_reward for noop.
    def _reward_scaler(self, reward: float, energy_penalty: float = 0):
        """
        Calculate the scaled reward consisting of the regular unscaled reward
        and the scaled energy_penalty.
        """
        if energy_penalty == 0:
            return reward / self.max_time_op
        return (1 - self.alpha) * (
            reward / self.max_time_op
        ) - self.alpha * energy_penalty

    def _update_total_energy_costs(self):
        power_consumption = self.power_consumption_machines.copy()
        # Legal machines are idle. Thus they don't consume energy.
        power_consumption[self.machine_legal] = 0
        self._total_energy_costs += np.sum(
            power_consumption * self.ts_energy_prices[self.current_time_step]
        )

    def increase_time_step(self):
        hole_planning = 0
        next_time_step_to_pick = self.next_time_step.pop(0)
        self.next_action.pop(0)
        time_difference = next_time_step_to_pick - self.current_time_step
        self.current_time_step = next_time_step_to_pick
        for id_job in range(self.jobs):
            was_left_time = self.time_until_activity_finished_jobs[id_job]
            if was_left_time > 0:
                performed_op_job = min(time_difference, was_left_time)
                self.time_until_activity_finished_jobs[id_job] = max(
                    0, self.time_until_activity_finished_jobs[id_job] - time_difference
                )
                self.state[id_job : id_job + self.max_alternatives, 1] = (
                    self.time_until_activity_finished_jobs[id_job] / self.max_time_op
                )
                self.total_perform_op_time_jobs[id_job] += performed_op_job
                self.state[id_job : id_job + self.max_alternatives, 3] = (
                    self.total_perform_op_time_jobs[id_job] / self.max_time_jobs
                )
                if self.time_until_activity_finished_jobs[id_job] == 0:
                    self.legal_jobs[id_job] = True
                    self.total_idle_time_jobs[id_job] += time_difference - was_left_time
                    self.state[id_job : id_job + self.max_alternatives, 6] = (
                        self.total_idle_time_jobs[id_job] / self.sum_op
                    )
                    self.idle_time_jobs_last_op[id_job] = (
                        time_difference - was_left_time
                    )
                    self.state[id_job : id_job + self.max_alternatives, 5] = (
                        self.idle_time_jobs_last_op[id_job] / self.sum_op
                    )
                    self.todo_activity_jobs[id_job] += 1
                    self.state[id_job : id_job + self.max_alternatives, 2] = (
                        self.todo_activity_jobs[id_job] / self.max_activities_jobs
                    )
                    for id_operation in range(self.max_alternatives):
                        operation = id_operation + id_job * self.max_alternatives
                        if (
                            self.todo_activity_jobs[id_job]
                            <= self.last_activity_jobs[id_job]
                        ):
                            key = self._ids_to_key(
                                id_job, self.todo_activity_jobs[id_job], id_operation
                            )
                            operation_data = self.instance_map.get(key)
                            if operation_data is None:
                                self.needed_machine_operation[operation] = -1
                                if self.legal_actions[operation]:
                                    self.legal_actions[operation] = False
                            else:
                                needed_machine = operation_data[0]
                                self.needed_machine_operation[
                                    operation
                                ] = needed_machine
                                self.state[operation][4] = (
                                    max(
                                        0,
                                        self.time_until_available_machine[
                                            needed_machine
                                        ]
                                        - time_difference,
                                    )
                                    / self.max_time_op
                                )
                        else:
                            self.needed_machine_operation[operation] = -1
                            # this allow to have 1 is job is over (not 0 because, 0 strongly indicate that the job is a
                            # good candidate)
                            self.state[operation][4] = 1.0
                            if self.legal_actions[operation]:
                                self.legal_actions[operation] = False
            elif self.todo_activity_jobs[id_job] <= self.last_activity_jobs[id_job]:
                self.total_idle_time_jobs[id_job] += time_difference
                self.idle_time_jobs_last_op[id_job] += time_difference
                self.state[id_job : id_job + self.max_alternatives, 5] = (
                    self.idle_time_jobs_last_op[id_job] / self.sum_op
                )
                self.state[id_job : id_job + self.max_alternatives, 6] = (
                    self.total_idle_time_jobs[id_job] / self.sum_op
                )

        needed_machines = self.needed_machine_operation[self.legal_actions[:-1]]
        for machine in range(self.machines):
            if machine in needed_machines:
                self.machine_legal[machine] = True
            else:
                self.machine_legal[machine] = False

        for machine in range(self.machines):
            if self.time_until_available_machine[machine] < time_difference:
                empty = time_difference - self.time_until_available_machine[machine]
                hole_planning += empty
            self.time_until_available_machine[machine] = max(
                0, self.time_until_available_machine[machine] - time_difference
            )
            if self.time_until_available_machine[machine] == 0:
                # Set actions legal that need this machine.
                for operation in range(self.operations):
                    id_job = operation // self.max_alternatives
                    if (
                        self.needed_machine_operation[operation] == machine
                        and not self.legal_actions[operation]
                        # and not self.illegal_actions[machine][operation]
                        and self.legal_jobs[id_job]
                        and self.todo_activity_jobs[id_job]
                        <= self.last_activity_jobs[id_job]
                    ):
                        self.legal_actions[operation] = True
                        if not self.machine_legal[machine]:
                            self.machine_legal[machine] = True
        self._update_power_observations()
        power_consumption = self.power_consumption_machines.copy()
        # Legal machines are idle. Thus they don't consume energy.
        power_consumption[self.machine_legal] = 0
        self._total_electricity_costs += np.sum(
            power_consumption * self.prices[self.current_time_step]
        )
        return hole_planning

    def _is_done(self):
        if self.nb_legal_actions == 0:
            self.last_time_step = self.current_time_step
            self.last_solution = self.solution
            return True
        return False
