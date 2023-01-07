import os
from pathlib import Path

DIR_INFILE = Path("fjsp_env/envs/taillard_instances").absolute()
DIR_OUTFILE = Path("fjsp_env/envs/taillard_instances_preprocessed").absolute()
IN_FILES = [file for file in os.listdir(DIR_INFILE) if os.path.isfile(DIR_INFILE / file)]

def get_max_alternatives(splitted_line):
    max_number_operation = 0
    idx = 1
    number_activities = 0
    while idx < len(splitted_line):
        number_activities += 1
        number_operations = int(splitted_line[idx])
        max_number_operation = max(max_number_operation, number_operations)
        idx += 1 + 2 * number_operations
    return max_number_operation, number_activities



if __name__ == "__main__":
    for file in IN_FILES:
        infile_path = DIR_INFILE / file
        outfile_path = DIR_OUTFILE / file
        max_number_operation = 0
        number_activities = 0

        with open(infile_path, 'r') as infile:
            for line_count, line in enumerate(infile):
                if line_count == 0:
                    splitted_line = line.split()
                    splitted_line.extend([1,1])


                splitted_line = line.split()
                number_activities = max(number_activities, get_max_alternatives(splitted_line)[1])
                max_number_operation = max(max_number_operation, get_max_alternatives(splitted_line)[0])

        with open(infile_path, 'r') as infile, open(outfile_path, 'w') as outfile:
            for line_count, line in enumerate(infile):
                print(f"line before: {line}")
                if line_count == 0:
                    splitted_line = line.split()
                    # start counting at 0
                    print(type(splitted_line[0]))
                    machines = splitted_line[1]
                    splitted_line.append(machines) 
                    splitted_line.append(str(1))
                    number_machines = splitted_line[1] 
                    print("  ".join(map(str,splitted_line)) + "\n")
                    outfile.write("  ".join(map(str,splitted_line)) + "\n")
                else:
                    splitted_line = line.split()
                    #splitted_line = [str(int(item)-1) if (idx % 2 == 0) else item for idx, item in enumerate(splitted_line)]
                    #splitted_line.insert(0, number_machines)
                    new_line = []
                    for idx, item in enumerate(splitted_line):
                        if (idx) % 2 == 0:
                            new_line.append(str(1))
                            item = str(int(item) + 1)
                        new_line.append(item)
                    new_line.insert(0, number_machines)

                    new_line = "  ".join(map(str,new_line)) + "\n"
                    print(f" line after:  {new_line} ")
                    outfile.write(new_line)



