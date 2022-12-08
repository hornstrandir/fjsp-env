import gym
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
                + "/instances/Mk01.fjs"
            }
        instance_path = env_config["instance_path"]


        self.sum_time_activities = 0 # used to scale observations
        self.jobs = 0
        self.machines = 0
        self.instance_matrix = None
        self.jobs_length = None
        self.max_time_op = 0
        self.max_time_jobs = 0
        self.nb_legal_actions = 0
        self.nb_machine_legal = 0
        self.nb_activities_per_job = []
        self.nb_operations_per_activity = []
        # initial values for variables used for solving (to reinitialize when reset() is called)
        self.solution = None
        self.last_solution = None
        self.last_time_step = float("inf")
        self.current_time_step = float("inf")
        self.next_time_step = list()
        self.next_jobs = list()
        self.legal_actions = None
        self.time_until_available_machine = None
        self.time_until_finish_current_op_jobs = None
        self.todo_time_step_job = None
        self.total_perform_op_time_jobs = None
        self.needed_machine_jobs = None
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
                    self.jobs, self.machines = int(splitted_line[0]), int(splitted_line[1])
                    # matrix which store tuple of (machine, length of the job)
                    self.instance_matrix = np.zeros((self.jobs, self.machines, self.machines), dtype=(int, 2))
                    # contains all the time to complete jobs
                    self.jobs_max_length = np.zeros(self.jobs, dtype=int)
                else:
                    idx = 1
                    # start counting jobs at null
                    job_nb = line_count - 2
                    id_activity = 0
                    while idx < len(splitted_line):
                        self.nb_activities_per_job = id_activity + 1
                        number_operations = int(splitted_line[idx])
                        self.nb_operations_per_activity[id_activity] = number_operations
                        max_time_activity = 0
                        for id_operation in range(1, number_operations+1):
                            machine, time = int(splitted_line[idx + 2 * id_operation - 1]), int(splitted_line[idx + 2 * id_operation])
                            if time > max_time_activity:
                                max_time_activity = time
                            self.instance_matrix[job_nb][id_activity][id_operation-1] = machine, time
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
        assert self.instance_matrix is not None
        

if __name__ == "__main__":
    env = FJSPEnv(env_config=None)
    print(env.instance_matrix)
    print(env.jobs)
    print(env.machines)

