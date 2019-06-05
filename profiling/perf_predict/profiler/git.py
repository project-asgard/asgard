from subprocess import check_output
from options import DATA_FILE_DIR, ASGARD_PATH, USE_OLD_DATA, LEVELS, DEGREES


# Gets the commit hash from asgard output file
def get_commit_hash(pde):
    output_file_extention = f".out.l{LEVELS[0]}_d{DEGREES[0]}_p{pde}"
    asgard_output = DATA_FILE_DIR + "/asgard" + output_file_extention

    for line in open(asgard_output).readlines():
        if "Commit Summary" in line:
            # return commit hash
            return line.split(':')[1].strip()

    raise Exception(
        f"Commit hash not found in asgard output file '{asgard_output}'")


# Gets the commit hash from asgard output file
def get_commit_date(pde):
    output = check_output(
        ["git", "log", "-n" "1", "--format=\"%ads\""], shell=False).decode('utf-8')[1:-2]
    return output
