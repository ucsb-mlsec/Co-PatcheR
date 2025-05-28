import json
import logging
import os
import ast
import contextlib
import subprocess
import glob
import re
from collections import defaultdict
from os.path import dirname as pdirname
from os.path import join as pjoin
from pathlib import Path
from subprocess import CalledProcessError


def clean_poc_output(text):
    """
    Clean proof-of-concept output by removing memory addresses and warnings.
    
    Args:
        text (str): The text to clean.
        
    Returns:
        str: The cleaned text with memory addresses redacted and warnings removed.
    """
    # Step 1: Replace "at 0x...>" with "at 0xREDACTED>"
    text = re.sub(r'at 0x[0-9A-Fa-f]+>', 'at 0xREDACTED>', text)
    
    # Step 2: Remove lines with "Warning" and the following line
    text = re.sub(r'^.*Warning.*\n.*\n?', '', text, flags=re.MULTILINE)
    
    return text

def load_jsonl(filepath):
    """
    Load a JSONL file from the given filepath.

    Args:
        filepath (str): The path to the JSONL file to load.

    Returns:
        list: A list of dictionaries representing the data in each line of the JSONL file.
    """
    with open(filepath, "r") as file:
        return [json.loads(line) for line in file]


def write_jsonl(data, filepath):
    """
    Write data to a JSONL file at the given filepath.

    Args:
        data (list): A list of dictionaries to write to the JSONL file.
        filepath (str): The path to the JSONL file to write.
    """
    with open(filepath, "w") as file:
        for entry in data:
            file.write(json.dumps(entry) + "\n")


def load_json(filepath):
    return json.load(open(filepath, "r"))


def combine_by_instance_id(data):
    """
    Combine data entries by their instance ID.

    Args:
        data (list): A list of dictionaries with instance IDs and other information.

    Returns:
        list: A list of combined dictionaries by instance ID with all associated data.
    """
    combined_data = defaultdict(lambda: defaultdict(list))
    for item in data:
        instance_id = item.get("instance_id")
        if not instance_id:
            continue
        for key, value in item.items():
            if key != "instance_id":
                combined_data[instance_id][key].extend(
                    value if isinstance(value, list) else [value]
                )
    return [
        {**{"instance_id": iid}, **details} for iid, details in combined_data.items()
    ]


def setup_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger


def load_existing_instance_ids(output_file):
    instance_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    instance_ids.add(data["instance_id"])
                except json.JSONDecodeError:
                    continue
    return instance_ids


@contextlib.contextmanager
def cd(newdir):
    """
    Context manager for changing the current working directory
    :param newdir: path to the new directory
    :return: None
    """
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def run_command(cmd: list[str], cwd=None, **kwargs) -> subprocess.CompletedProcess:
    """
    Run a command in the shell.
    Args:
        - cmd: command to run
    """
    try:
        if cwd is not None:
            cp = subprocess.run(cmd, check=True, cwd=cwd, **kwargs)
        else:
            cp = subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}, {e}")
        print("Command failed with exit status:", e.returncode)
        print("stdout:", e.stdout)  
        print("stderr:", e.stderr) 
        if cwd is not None:
            print("#####################################################################")
            print(f"\nException at Working directory: {cwd}\n")
        raise e
    return cp


def is_git_repo() -> bool:
    """
    Check if the current directory is a git repo.
    """
    git_dir = ".git"
    return os.path.isdir(git_dir)


def clone_repo(clone_link: str, dest_dir: str, cloned_name: str):
    """
    Clone a repo to dest_dir.

    Returns:
        - path to the newly cloned directory.
    """
    clone_cmd = ["git", "clone", clone_link, cloned_name]
    create_dir_if_not_exists(dest_dir)
    run_command(clone_cmd, dest_dir)
    cloned_dir = pjoin(dest_dir, cloned_name)
    return cloned_dir


def apply_patch(model_patch: str, dest_dir: str):
    patch_file = os.path.join(dest_dir, "tmp.diff")
    with open(patch_file, "w") as f:
        f.write(model_patch)

    apply_cmd = ["git", "apply", "-v", patch_file]
    run_command(apply_cmd, cwd=dest_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("git patch applied!")


def repo_reset_and_clean_checkout(commit_hash: str, project_path: str) -> None:
    """
    Run commands to reset repo to the original commit state.
    Cleans both the uncommited changes and the untracked files, and submodule changes.
    Assumption: The current directory is the git repository.
    """
    # NOTE: do these before `git reset`. This is because some of the removed files below
    # may actually be in version control. So even if we deleted such files here, they
    # will be brought back by `git reset`.
    # Clean files that might be in .gitignore, but could have been created by previous runs
    coverage_folder = os.path.join(project_path, ".coverage")
    coveragerc_folder = os.path.join(project_path, "tests", ".coveragerc")
    if os.path.exists(coverage_folder):
        os.remove(coverage_folder)
    if os.path.exists(coveragerc_folder):
        os.remove(coveragerc_folder)
    other_cov_files = glob.glob(".coverage.TSS.*", root_dir=project_path, recursive=True)
    for f in other_cov_files:
        os.remove(f)

    remove_index_lock_cmd = ["rm", "-f", ".git/index.lock"]
    reset_cmd = ["git", "reset", "--hard", commit_hash]
    clean_cmd = ["git", "clean", "-fd"]
    checkout_cmd = ["git", "checkout", commit_hash]
    run_command(remove_index_lock_cmd, cwd=project_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run_command(reset_cmd, cwd=project_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run_command(clean_cmd, cwd=project_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # need to checkout before submodule init. Otherwise submodule may init to another version
    run_command(checkout_cmd, cwd=project_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # this is a fail-safe combo to reset any changes to the submodule: first unbind all submodules
    # and then make a fresh checkout of them.
    # Reference: https://stackoverflow.com/questions/10906554/how-do-i-revert-my-changes-to-a-git-submodule
    submodule_unbind_cmd = ["git", "submodule", "deinit", "-f", "."]
    submodule_init_cmd = ["git", "submodule", "update", "--init"]
    run_command(
        submodule_unbind_cmd, cwd=project_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    run_command(
        submodule_init_cmd, cwd=project_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def repo_clean_except_poc(project_path: str) -> None:
    clean_cmd = ["git", "clean", "-fd", "-e", "poc_code.py"]
    cp = run_command(clean_cmd, cwd=project_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if cp.returncode != 0:
        print(cp.stderr)
        raise RuntimeError(f"Command {clean_cmd} failed.")


def run_string_cmd_in_conda(
        command: str, env_name: str, cwd: str, **kwargs
) -> subprocess.CompletedProcess:
    """
    Run a complete command in a given conda environment, where the command is a string.

    This is useful when the command to be run contains &&, etc.

    NOTE: use `conda activate` instead of `conda run` in this verison, so that we can
          run commands that contain `&&`, etc.
    """
    conda_bin_path = os.getenv("CONDA_EXE")  # for calling conda
    if conda_bin_path is None:
        raise RuntimeError("Env variable CONDA_EXE is not set")
    conda_root_dir = pdirname(pdirname(conda_bin_path))
    conda_script_path = pjoin(conda_root_dir, "etc", "profile.d", "conda.sh")
    conda_cmd = f"source {conda_script_path} ; conda activate {env_name} ; {command} ; conda deactivate"
    print(f"Running command: {conda_cmd} in directory: {cwd}")
    try:
        return subprocess.run(conda_cmd, cwd=cwd, shell=True, **kwargs)
    except subprocess.TimeoutExpired as e:
        print(f"TimeoutError: {e}")
        raise e


def create_dir_if_not_exists(dir_path: str):
    """
    Create a directory if it does not exist.
    Args:
        dir_path (str): Path to the directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def to_relative_path(file_path: str, project_root: str) -> str:
    """Convert an absolute path to a path relative to the project root.

    Args:
        - file_path (str): The absolute path.
        - project_root (str): Absolute path of the project root dir.

    Returns:
        The relative path.
    """
    if Path(file_path).is_absolute():
        return str(Path(file_path).relative_to(project_root))
    else:
        return file_path


def to_absolute_path(file_path: str, project_root: str) -> str:
    """Convert a relative path to an absolute path.

    Args:
        - file_path (str): The relative path.
        - project_root (str): Absolute path of a root dir.
    """
    return pjoin(project_root, file_path)


def find_file(directory, filename) -> str | None:
    """
    Find a file in a directory. filename can be short name, relative path to the
    directory, or an incomplete relative path to the directory.
    Returns:
        - the relative path to the file if found; None otherwise.
    """

    # Helper method one
    def find_file_exact_relative(directory, filename) -> str | None:
        if os.path.isfile(os.path.join(directory, filename)):
            return filename
        return None

    # Helper method two
    def find_file_shortname(directory, filename) -> str | None:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file == filename:
                    return os.path.relpath(os.path.join(root, file), directory)
        return None

    # if the filename is exactly the relative path.
    found = find_file_exact_relative(directory, filename)
    if found is not None:
        return found

    # if the filename is a short name without any directory
    found = find_file_shortname(directory, filename)
    if found is not None:
        return found

    # if the filename has some directory, but is not a relative path to
    # the directory
    parts = filename.split(os.path.sep)
    shortname = parts[-1]
    found = find_file_shortname(directory, shortname)
    if found is None:
        # really cannot find this file
        return None
    # can find this shortname, but we also need to check whether the intermediate
    # directories match
    if filename in found:
        return found
    else:
        return None


def parse_function_invocation(
        invocation_str: str,
) -> tuple[str, list[str]]:
    try:
        tree = ast.parse(invocation_str)
        expr = tree.body[0]
        assert isinstance(expr, ast.Expr)
        call = expr.value
        assert isinstance(call, ast.Call)
        func = call.func
        assert isinstance(func, ast.Name)
        function_name = func.id
        raw_arguments = [ast.unparse(arg) for arg in call.args]
        # clean up spaces or quotes, just in case
        arguments = [arg.strip().strip("'").strip('"') for arg in raw_arguments]

        try:
            new_arguments = [ast.literal_eval(x) for x in raw_arguments]
            if new_arguments != arguments:
                print(
                    f"Refactored invocation argument parsing gives different result on "
                    f"{invocation_str!r}: old result is {arguments!r}, new result "
                    f" is {new_arguments!r}"
                )
        except Exception as e:
            print(
                f"Refactored invocation argument parsing failed on {invocation_str!r}: {e!s}"
            )
    except Exception as e:
        raise ValueError(f"Invalid function invocation: {invocation_str}") from e

    return function_name, arguments


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"create '{directory}' folder")


def parse_missing(missing_str):
    missing = []
    for part in missing_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            missing.extend(range(start, end + 1))
        else:
            missing.append(int(part))
    return missing


def coverage_to_dict(coverage_str):
    lines = coverage_str.strip().split('\n')

    data_lines = lines[2:]
    coverage_dict = {}

    for line in data_lines:
        if not line.strip():
            continue
        parts = line.split(maxsplit=4)
        if len(parts) < 4:
            continue

        filename = parts[0]
        stmts = parts[1]
        miss = parts[2]
        cover = parts[3]

        if len(parts) == 5:
            missing_str = parts[4]
            missing_list = parse_missing(missing_str)
        else:
            missing_list = []

        coverage_dict[filename] = missing_list
    
    return coverage_dict
