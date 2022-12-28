
import gym
import random
import bisect
import numpy as np
import datetime
from pathlib import Path

#TODO: check if illegal actions do have some heavy computations. If so: use: if legal_action(action): ...
class FJSPEnv(gym.Env):
    """
    """

    def __init__(self, env_config=None) -> None:
        super().__init__()
        
        if env_config is None:
            env_config = {
                "instance_path": str(Path(__file__).parent.absolute())
                + "/instances_preprocessed/Mk01.fjs"
            }
        instance_path = env_config["instance_path"]
        self.sum_time_activities = 0 # used to scale observations
        self.jobs = 0
        self.machines = 0
        self.max_alternatives = 0
        self.actions = 0
        self.instance_map = {}

        self.jobs_max_duration = 0    
        self.max_time_op = 0
        self.max_time_jobs = 0
        self.nb_legal_actions = 0
        self.nb_machine_legal = 0
        self.nb_activities_per_job = None
        self.nb_operations_per_activity = None
        self.last_activity_jobs = None
        # initial values for variables used for solving (to reinitialize when reset() is called)
        self.solution = None
        self.last_solution = None
        self.last_time_step = float("inf")
        self.current_time_step = float("inf")
        self.next_time_step = list()
        self.next_action = list()
        self.legal_actions = None
        self.time_until_available_machine = None
        self.time_until_activity_finished = None # changed from op to activity
        self.todo_activity_jobs = None
        self.total_perform_op_time_jobs = None
        self.needed_machine_operation = None
        self.total_idle_time_jobs = None
        self.idle_time_jobs_last_op = None
        self.state = None
        self.illegal_actions = None
        self.action_illegal_no_op = None
        self.machine_legal = None
        # initial values for variables used for representation
        self.start_timestamp = datetime.datetime.now().timestamp()
        self.sum_op = 0


        with open(instance_path, 'r') as instance_file:
            for line_count, line in enumerate(instance_file):
                splitted_line = line.split()
                if line_count == 0:
                    self.jobs = int(splitted_line[0])
                    self.machines = int(splitted_line[1])
                    self.max_activities_jobs = int(splitted_line[2]) # TODO: do we need it
                    self.max_alternatives = int(splitted_line[3])
                    self.actions = self.jobs * self.max_alternatives
                    self.jobs_max_duration = np.zeros(self.jobs, dtype=int)
                    self.last_activity_jobs = np.zeros(self.jobs, dtype=int)
                    self.nb_operations_per_activity = np.zeros((self.jobs, self.machines), dtype=int)
                else:
                    idx = 1
                    # start counting jobs at null
                    job_nb = line_count - 1
                    id_activity = 0
                    while idx < len(splitted_line):
                        # TODO: Improvement: would be better to set number of activities just once
                        self.last_activity_jobs[job_nb] = id_activity + 1 
                        number_operations = int(splitted_line[idx])
                        self.nb_operations_per_activity[job_nb][id_activity] = number_operations
                        max_time_activity = 0
                        for id_operation in range(1, number_operations+1):
                            machine, time = int(splitted_line[idx + 2 * id_operation - 1]), int(splitted_line[idx + 2 * id_operation])
                            if time > max_time_activity:
                                max_time_activity = time
                            key = self._ids_to_key(job_nb, id_activity, id_operation)
                            self.instance_map[key] = machine, time
                        self.jobs_max_duration[job_nb] += max_time_activity
                        self.max_time_op = max(self.max_time_op, max_time_activity)
                        self.sum_op += max_time_activity
                        id_activity += 1
                        idx += 1 + 2 * number_operations
                        self.sum_time_activities += max_time_activity
        self.max_time_jobs = np.amax(self.jobs_max_duration)
        assert self.max_time_op > 0
        assert self.max_time_jobs > 0
        assert self.jobs > 0
        assert self.machines > 1, "We need at least 2 machines"
        assert self.instance_map is not None
        self.action_space = gym.spaces.Discrete(self.jobs + 1)
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
                "action_mask": gym.spaces.Box(0, 1, shape=(self.jobs + 1,)),
                "real_obs": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self.jobs, 7), dtype=float
                ),
            }
        )
    
    def _ids_to_key(self, id_job: int, id_activity : int, id_operation: int) -> str:
        try:
            return str(id_job) + str(id_activity) + str(id_operation)
        except KeyError:
            return None

    def _key_to_ids(self, key: str):
        """
        Order of IDs
        ------------
        id_job, id_activity, id_operation
        """
        return int(key[0]), int(key[1]), int(key[2])

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
        self.nb_legal_actions = self.jobs
        self.nb_machine_legal = 0
        # represent all the legal actions
        self.legal_actions = np.ones(self.actions + 1, dtype=bool)
        self.legal_actions[self.actions] = False
        # used to represent the solution
        self.solution = np.full((self.jobs, self.machines), -1, dtype=int)
        self.time_until_available_machine = np.zeros(self.machines, dtype=int)
        self.time_until_activity_finished_jobs = np.zeros(self.jobs, dtype=int) 
        self.todo_activity_jobs = np.zeros(self.jobs, dtype=int)
        self.total_perform_op_time_jobs = np.zeros(self.jobs, dtype=int)     
        self.needed_machine_operation = np.zeros(self.actions, dtype=int)           
        self.total_idle_time_jobs = np.zeros(self.jobs, dtype=int)      
        self.idle_time_jobs_last_op = np.zeros(self.jobs, dtype=int)
        self.illegal_actions = np.zeros((self.machines, self.actions), dtype=bool) # TODO: what is that for
        self.action_illegal_no_op = np.zeros(self.actions, dtype=bool) # TODO: what is that for
        self.machine_legal = np.zeros(self.machines, dtype=bool)
        for key, value in self.instance_map.items():
            id_job, _, id_operation = self._key_to_ids(key)
            needed_machine = value[0]
            self.needed_machine_operation[id_job + id_operation] = needed_machine
            if not self.machine_legal[needed_machine]:
                self.machine_legal[needed_machine - 1] = True
                self.nb_machine_legal += 1
        for operation, machine in enumerate(self.needed_machine_operation):
            if machine == 0:
                # This is just a placeholder. Thus this action must be illegal
                self.legal_actions[operation] = False
        self.state = np.zeros((self.actions, 7), dtype=float)
        return self._get_current_state_representation()

    def _prioritization_non_final(self):
        if self.nb_machine_legal < 1:
            return
        for machine in range(self.machines):
            if not self.machine_legal[machine]:
                continue
            keys_final_operations = list()
            non_final_operations = list()
            min_non_final = float("inf")
            for operation in range(self.actions):
                id_job = operation // self.max_alternatives
                if not (
                    self.needed_machine_operation[id_job] == machine
                    and self.legal_actions[operation]
                ):
                    continue
                id_activity = self.todo_activity_jobs[id_job]
                id_operation = operation % id_job
                key_this_step = self._ids_to_key(id_job, id_activity, id_operation)
                if key_this_step is None:
                    continue
                # TODO: check if it is possible if this can be called on the really last activity not the penultimate activity
                if id_activity == (self.last_activity_jobs[id_job] - 1):
                    keys_final_operations.append(key_this_step)
                else:
                    time_needed_legal = self.instance_map[key_this_step][1]
                    key_next_step = self._ids_to_key(id_job, id_activity + 1, id_operation)
                    machine_needed_nextstep = self.instance_map[key_next_step][0]
                    if (
                        self.time_until_available_machine[machine_needed_nextstep] == 0
                    ):
                        min_non_final = min(min_non_final, time_needed_legal)
                        non_final_operations.append(id_job)  
            if len(non_final_operations) > 0:
                for key in keys_final_operations:
                    operation = int(key[0]) + int(key[2])
                    time_needed_legal = self.instance_map[key][1]
                    if time_needed_legal > min_non_final:
                        self.legal_actions[operation] = False
                        self.nb_legal_actions -= 1

    def _check_no_op(self):
        """
        Restrictions:
        1. Disallow No-Op if: four or more machines with some allocatable job.
        2. Disallow No-Op if: five or more allocatable jobs.
        3. If we make a pause longer than D it would be better to 
            allocate a job with duration D.
        4. We do not wait for jobs that will be rejected by non-final.
        """
        self.legal_actions[self.actions] = False
        if (
            len(self.next_time_step) > 0
            and self.nb_machine_legal <= 3
            and self.nb_legal_actions <= 4
        ):
            machine_next = set()
            next_time_step = self.next_time_step[0]
            max_horizon = self.current_time_step
            max_horizon_machine = [
                self.current_time_step + self.max_time_op for _ in range(self.machines)
            ]
            # return if there is an legal job that ends before the next time step
            # max horizion is the min duration the machine is occupied.
            for operation in range(self.actions):
                if self.legal_actions[operation]:
                    id_job = operation // self.max_alternatives
                    id_activity = self.todo_activity_jobs[id_job]
                    id_operation = operation % id_job
                    key = self._ids_to_key(id_job, id_activity, id_operation)
                    if key is None:
                        continue
                    machine_needed, time_needed = self.instance_map[key]
                    end_job = self.current_time_step + time_needed
                    if end_job < next_time_step:
                        return
                    max_horizon_machine[machine_needed] = min(
                        max_horizon_machine[machine_needed], end_job
                    )
                    max_horizon = max(max_horizon, max_horizon_machine[machine_needed])
            # TODO: what is happening here
            for operation in range(self.actions):
                if self.legal_actions[operation]:
                    continue
                id_job = operation // self.max_alternatives
                id_activity = self.todo_activity_jobs[id_job]
                id_operation = operation % id_job
                if (
                    self.time_until_activity_finished_jobs[id_job] > 0
                    and id_activity + 1 < self.last_activity_jobs[id_job]
                ):
                    time_step = id_activity + 1
                    time_needed = (
                        self.current_time_step
                        + self.time_until_activity_finished_jobs[id_job]
                    )
                    while (
                        time_step < self.last_activity_jobs[id_job] - 1 and max_horizon > time_needed
                    ):
                        next_key = self._ids_to_key(id_operation, time_step, id_job)
                        if key is None:
                            continue
                        machine_needed = self.instance_map[next_key][0]
                        if (
                            max_horizon_machine[machine_needed] > time_needed
                            and self.machine_legal[machine_needed]
                        ):
                            machine_next.add(machine_needed)
                            # TODO: Why is that?
                            if len(machine_next) == self.nb_machine_legal:
                                self.legal_actions[self.actions] = True
                                return
                        time_needed += self.instance_map[next_key][1]
                        time_step += 1
                elif (
                    not self.action_illegal_no_op[operation]
                    and id_activity < self.last_activity_jobs[id_job]
                ):
                    time_step = self.todo_activity_jobs[id_job]
                    next_key = self._ids_to_key(id_job, time_step, id_operation)
                    machine_needed = self.instance_map[key][time_step][0]
                    time_needed = (
                        self.current_time_step
                        + self.time_until_available_machine[machine_needed]
                    )
                    while (
                        time_step < self.last_activity_jobs[id_job] - 1 and max_horizon > time_needed
                    ):
                        next_key = self._ids_to_key(id_operation, time_step, id_job)
                        if next_key is None:
                            continue
                        machine_needed = self.instance_map[next_key][0]
                        if (
                            max_horizon_machine[machine_needed] > time_needed
                            and self.machine_legal[machine_needed]
                        ):
                            machine_next.add(machine_needed)
                            if len(machine_next) == self.nb_machine_legal:
                                self.legal_actions[self.actions] = True
                                return
                        time_needed += self.instance_map[next_key][1]
                        time_step += 1
                            
    def step(self, action: int):
        reward = 0.0
        if action == self.actions:
            self.nb_machine_legal = 0
            self.nb_legal_actions = 0
            for operation in range(self.actions):
                if self.legal_actions[operation]:
                    self.legal_actions[operation] = False
                    needed_machine = self.needed_machine_operation[operation]
                    self.machine_legal[needed_machine] = False
                    self.illegal_actions[needed_machine][operation] = True
                    self.action_illegal_no_op[operation] = True
            while self.nb_machine_legal == 0:
                reward -= self.increase_time_step()
            scaled_reward = self._reward_scaler(reward)
            #self._prioritization_non_final()
            #self._check_no_op()
            return (
                self._get_current_state_representation(),
                scaled_reward,
                self._is_done(),
                {},
            )
        else:
            id_job = action // self.max_alternatives
            id_activity = self.todo_activity_jobs[id_job]
            id_operation = action % id_job
            key = self._ids_to_key(id_job, id_activity, id_operation)
            current_time_step_job = self.todo_activity_jobs[id_job]
            machine_needed, time_needed = self.instance_map[key]
            reward += time_needed
            self.time_until_available_machine[machine_needed] = time_needed
            self.time_until_activity_finished_jobs[id_job] = time_needed
            self.state[action][1] = time_needed / self.max_time_op
            to_add_time_step = self.current_time_step + time_needed
            # We can skip the timesteps where nothing happens in the env.
            if to_add_time_step not in self.next_time_step:
                index = bisect.bisect_left(self.next_time_step, to_add_time_step)
                self.next_time_step.insert(index, to_add_time_step)
                self.next_action.insert(index, action)
            self.solution[id_job][current_time_step_job] = self.current_time_step
            for operation in range(self.actions):
                if (
                    self.needed_machine_operation[operation] == machine_needed
                    and self.legal_actions[operation]
                ):
                    self.legal_actions[operation] = False
                    self.nb_legal_actions -= 1
                # All alternatives of the action taken are set to illegal
                elif (
                    self.legal_actions[operation] 
                    and (operation // self.max_alternatives) == id_job
                ):
                    self.legal_actions[operation] = False
                    self.nb_legal_actions -= 1
            self.nb_machine_legal -= 1
            self.machine_legal[machine_needed] = False
            # TODO: check this in the original code. what is it for
            for operation in range(self.actions):
                if self.illegal_actions[machine_needed][operation]:
                    self.action_illegal_no_op[operation] = False
            # if we can't allocate new job in the current timestep, we pass to the next one
            while self.nb_machine_legal == 0 and len(self.next_time_step) > 0:
                reward -= self.increase_time_step()
            #self._prioritization_non_final()
            #self._check_no_op()
            # we then need to scale the reward
            scaled_reward = self._reward_scaler(reward)
            return (
                self._get_current_state_representation(),
                scaled_reward,
                self._is_done(),
                {},
            )

    def _reward_scaler(self, reward):
        return reward / self.max_time_op

    def increase_time_step(self):
        hole_planning = 0
        next_time_step_to_pick = self.next_time_step.pop(0)
        self.next_action.pop(0)
        time_difference = next_time_step_to_pick - self.current_time_step
        self.current_time_step = next_time_step_to_pick
        for id_job in range(self.jobs):
            was_left_time = self.time_until_activity_finished_jobs[id_job]
            # This can only happen for one operation of the activity.
            if was_left_time > 0:
                performed_op_job = min(time_difference, was_left_time)
                self.time_until_activity_finished_jobs[id_job] = max(
                    0, self.time_until_activity_finished_jobs[id_job] - time_difference
                )
                self.state[id_job:id_job+self.max_alternatives][1] = (
                    self.time_until_activity_finished_jobs[id_job] / self.max_time_op
                )
                self.total_perform_op_time_jobs[id_job] += performed_op_job
                self.state[id_job:id_job+self.max_alternatives][3] = (
                    self.total_perform_op_time_jobs[id_job] / self.max_time_jobs
                )
                if self.time_until_activity_finished_jobs[id_job] == 0:
                    self.total_idle_time_jobs[id_job] += time_difference - was_left_time
                    self.state[id_job:id_job+self.max_alternatives][6] = self.total_idle_time_jobs[id_job] / self.sum_op
                    self.idle_time_jobs_last_op[id_job] = time_difference - was_left_time
                    self.state[id_job:id_job+self.max_alternatives][5] = self.idle_time_jobs_last_op[id_job] / self.sum_op
                    self.todo_activity_jobs[id_job] += 1
                    self.state[id_job:id_job+self.max_alternatives][2] = self.todo_activity_jobs[id_job] / self.machines
                    # TODO: job is not done
                    for id_operation in range(self.max_alternatives):
                        operation = id_operation + id_job
                        if self.todo_activity_jobs[id_job] <= self.last_activity_jobs[id_job]:
                            key = self._key_to_ids(id_job, self.todo_activity_jobs[id_job], id_operation)
                            if key is None:
                                # TODO: increase timestep for non existing activities
                                continue
                            needed_machine, _ = self.instance_map[key]
                            self.needed_machine_operation[operation] = needed_machine
                            self.state[operation][4] = (
                                max(
                                    0, self.time_until_available_machine[needed_machine] - time_difference,
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
                                self.nb_legal_actions -= 1
            elif self.todo_activity_jobs[id_job] <= self.last_activity_jobs[id_job]:
                self.total_idle_time_jobs[id_job] += time_difference
                self.idle_time_jobs_last_op[id_job] += time_difference
                self.state[id_job:id_job+self.max_alternatives][5] = self.idle_time_jobs_last_op[id_job] / self.sum_op
                self.state[id_job:id_job+self.max_alternatives][6] = self.total_idle_time_jobs[id_job] / self.sum_op
        for machine in range(1, self.machines+1):
            if self.time_until_available_machine[machine] < time_difference:
                empty = time_difference - self.time_until_available_machine[machine]
                hole_planning += empty
            self.time_until_available_machine[machine] = max(
                0, self.time_until_available_machine[machine] - time_difference
            )
            if self.time_until_available_machine[machine] == 0:
                # Set actions legal that need this machine.
                for operation in range(self.actions):
                    if (
                        self.needed_machine_operation[operation] == machine
                        and not self.legal_actions[operation]
                        and not self.illegal_actions[machine][operation]
                    ):
                        self.legal_actions[operation] = True
                        self.nb_legal_actions += 1
                        if not self.machine_legal[machine]:
                            self.machine_legal[machine] = True
                            self.nb_machine_legal += 1
        return hole_planning
    
    def _is_done(self):
        if self.nb_legal_actions == 0:
            self.last_time_step = self.current_time_step
            self.last_solution = self.solution
            return True
        return False
    

def main():
    env = FJSPEnv(env_config=None)
    env.reset()
    print("i am here")
    print(env.jobs)
    print(env.machines)
    print(env.legal_actions)
    print(env.needed_machine_operation)



if __name__ == "__main__":
    main()