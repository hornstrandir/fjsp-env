import gym
import random
import numpy as np
import datetime
from pathlib import Path


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
        self.activities = 0
        self.instance_map = {}
        self.jobs_length = None
        self.max_time_op = 0
        self.max_time_jobs = 0
        self.nb_legal_actions = 0
        self.nb_machine_legal = 0
        self.nb_activities_per_job = None
        self.nb_operations_per_activity = None
        # initial values for variables used for solving (to reinitialize when reset() is called)
        self.solution = None
        self.last_solution = None
        self.last_time_step = float("inf")
        self.current_time_step = float("inf")
        self.next_time_step = list()
        self.next_jobs = list()
        self.legal_actions = None
        self.time_until_available_machine = None
        self.time_until_finish_current_activity_jobs = None # changed from op to activity
        self.todo_activity_job = None
        self.total_perform_op_time_jobs = None
        self.needed_machine_next_activity = None
        self.total_idle_time_jobs = None
        self.idle_time_jobs_last_op = None
        self.state = None
        self.illegal_actions = None
        self.action_illegal_no_op = None
        self.machine_legal = None
        # initial values for variables used for representation
        self.start_timestamp = datetime.datetime.now().timestamp()


        with open(instance_path, 'r') as instance_file:
            for line_count, line in enumerate(instance_file):
                splitted_line = line.split()
                if line_count == 0:
                    self.jobs = int(splitted_line[0])
                    self.machines = int(splitted_line[1])
                    self.max_activities_jobs = int(splitted_line[2]) 
                    self.max_alternatives = int(splitted_line[3])
                    self.activities = self.jobs * self.max_alternatives
                    self.jobs_max_length = np.zeros(self.jobs, dtype=int)
                    self.nb_activities_per_job = np.zeros(self.jobs, dtype=int)
                    self.nb_operations_per_activity = np.zeros((self.jobs, self.machines), dtype=int)
                else:
                    idx = 1
                    # start counting jobs at null
                    job_nb = line_count - 1
                    id_activity = 0
                    while idx < len(splitted_line):
                        # TODO: Improvement: would be better to set number of activities just once
                        self.nb_activities_per_job[job_nb] = id_activity + 1 
                        number_operations = int(splitted_line[idx])
                        self.nb_operations_per_activity[job_nb][id_activity] = number_operations
                        max_time_activity = 0
                        for id_operation in range(1, number_operations+1):
                            machine, time = int(splitted_line[idx + 2 * id_operation - 1]), int(splitted_line[idx + 2 * id_operation])
                            if time > max_time_activity:
                                max_time_activity = time
                            key = self._ids_to_key(job_nb, id_activity, id_operation)
                            self.instance_map[key] = machine, time
                        self.jobs_max_length[job_nb] += max_time_activity
                        self.max_time_op = max(self.max_time_op, max_time_activity)
                        id_activity += 1
                        idx += 1 + 2 * number_operations
                        self.sum_time_activities += max_time_activity
        self.max_time_jobs = np.amax(self.jobs_max_length)
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
    
    def _ids_to_key(self, job_id: int, id_activity : int, id_operation : int) -> str:
        return str(job_id) + str(id_activity) + str(id_operation)

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
        self.next_jobs = list()
        self.nb_legal_actions = self.jobs
        self.nb_machine_legal = 0
        # represent all the legal actions
        self.legal_actions = np.ones(self.activities + 1, dtype=bool)
        self.legal_actions[self.activities] = False
        # used to represent the solution
        self.solution = np.full((self.jobs, self.machines), -1, dtype=int)
        self.time_until_available_machine = np.zeros(self.machines, dtype=int)
        self.time_until_finish_current_activity_jobs = np.zeros(self.jobs, dtype=int) # TODO: change to time of activity?
        self.todo_activity_job = np.zeros(self.jobs, dtype=int)
        self.total_perform_op_time_jobs = np.zeros(self.jobs, dtype=int)     
        self.needed_machine_next_activity = np.zeros(self.activities, dtype=int)           
        self.total_idle_time_jobs = np.zeros(self.jobs, dtype=int)      
        self.idle_time_jobs_last_op = np.zeros(self.jobs, dtype=int)
        self.illegal_actions = np.zeros((self.machines, self.jobs), dtype=bool)
        self.action_illegal_no_op = np.zeros(self.activities, dtype=bool) # TODO: what is that for
        self.machine_legal = np.zeros(self.machines, dtype=bool)
        for key, value in self.instance_map.items():
            id_machine, id_activity, id_operation = self._key_to_ids(key)
            needed_machine = value[0]
            self.needed_machine_next_activity[id_activity] = needed_machine
            if not self.machine_legal[needed_machine]:
                self.machine_legal[needed_machine - 1] = True
                self.nb_machine_legal += 1
        self.state = np.zeros((self.activities, 7), dtype=float) 
        return self._get_current_state_representation()
    """
    def _prioritization_non_final(self):
        if self.nb_machine_legal >= 1:
            for machine in range(self.machines):
                if self.machine_legal[machine]:
                    final_job = list()
                    non_final_job = list()
                    min_non_final = float("inf")
                    for activity in range(self.jobs*self.max_alternatives):
                        for needed_machine in self.needed_machine_next_activity[activity]:
                            if (
                                needed_machine == machine
                                and self.legal_actions[activity]
                            ):
                                if self.todo_activity_job[job] == (self.machines - 1):
                                    final_job.append(job)
                                else:
                                    current_time_step_non_final = self.todo_activity_job[
                                        job
                                    ]
                                    time_needed_legal = self.instance_matrix[job][
                                        current_time_step_non_final
                                    ][1]
                                    machine_needed_nextstep = self.instance_matrix[job][
                                        current_time_step_non_final + 1
                                    ][0]
                                    if (
                                        self.time_until_available_machine[
                                            machine_needed_nextstep
                                        ]
                                        == 0
                                    ):
                                        min_non_final = min(
                                            min_non_final, time_needed_legal
                                        )
                                        non_final_job.append(job)
                        if len(non_final_job) > 0:
                            for job in final_job:
                                current_time_step_final = self.todo_activity_job[job]
                                time_needed_legal = self.instance_matrix[job][
                                    current_time_step_final
                                ][1]
                                if time_needed_legal > min_non_final:
                                    self.legal_actions[job] = False
                                    self.nb_legal_actions -= 1

        """

def main():
    env = FJSPEnv(env_config=None)
    env.reset()
    print("i am here")
    print(env.jobs)
    print(env.machines)
    print(env.legal_actions)
    print(env.needed_machine_next_activity)



if __name__ == "__main__":
    main()