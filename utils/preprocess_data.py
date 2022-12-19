import os
from pathlib import Path

DIR_INFILE = Path("fjsp_env/envs/instances").absolute()
DIR_OUTFILE = Path("fjsp_env/envs/instances_preprocessed").absolute()
IN_FILES = [file for file in os.listdir(DIR_INFILE) if os.path.isfile(DIR_INFILE / file)]

def get_max_alternatives(splitted_line):
    max_number_operation = 0
    idx = 1
    while idx < len(splitted_line):
        number_operations = int(splitted_line[idx])
        max_number_operation = max(max_number_operation, number_operations)
        idx += 1 + 2 * number_operations
    return max_number_operation


if __name__ == "__main__":
    for file in IN_FILES:
        infile_path = DIR_INFILE / file
        outfile_path = DIR_OUTFILE / file
        max_number_operation = 0

        with open(infile_path, 'r') as infile:
            for line_count, line in enumerate(infile):
                if line_count == 0:
                    continue
                splitted_line = line.split()
                max_number_operation = max(max_number_operation, get_max_alternatives(splitted_line))
        
        with open(infile_path, 'r') as infile, open(outfile_path, 'w') as outfile:
            for line_count, line in enumerate(infile):
                print(f"line: {line}")
                if line_count == 0:
                    splitted_line = line.split()
                    splitted_line[-1] = max_number_operation
                    outfile.write("  ".join(map(str,splitted_line)) + "\n")
                else:
                    outfile.write(line)


