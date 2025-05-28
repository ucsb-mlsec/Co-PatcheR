import os
from abc import ABC, abstractmethod
from os.path import join as pjoin
import json
import subprocess

from patchpilot.util.utils import repo_reset_and_clean_checkout, run_string_cmd_in_conda, apply_patch, \
    repo_clean_except_poc, clean_poc_output
from patchpilot.util.utils_for_swe import get_container, reset, reset_and_clean, run_poc, apply_diff, delete_container


class Task(ABC):
    @property
    @abstractmethod
    def project_path(self) -> str:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def setup_project(self) -> None:
        """Set up the project before starting to resolve the task."""
        raise NotImplementedError("abstract method")

    @abstractmethod
    def reset_project(self) -> None:
        """Reset project to initial state."""
        raise NotImplementedError("abstract method")


class MockTask(Task):
    task_id: str

    def __init__(self, task_id):
        self.task_id = task_id

    @property
    def project_path(self) -> str:
        return f""

    def setup_project(self) -> None:
        print(f"Setting up project for task {self.task_id}")

    def reset_project(self) -> None:
        print(f"Resetting project for task {self.task_id}")


class SweTask(Task):
    task_id: str
    problem_statement: str
    repo_path: str
    commit: str
    env_name: str
    repo_name: str
    pre_install_cmds: list[str]
    install_cmd: str
    test_cmd: str
    test_patch: str
    testcases_passing: list[str]
    testcases_failing: list[str]
    patched_diff: str

    def __init__(self, task_id, problem_statement, repo_path, env_name, pre_install_cmds, install_cmd,
                 test_cmd, commit, repo_name, test_patch, testcases_passing, testcases_failing):
        super().__init__()
        self.task_id = task_id
        self.problem_statement = problem_statement
        self.repo_path = repo_path
        self.env_name = env_name
        self.pre_install_cmds = pre_install_cmds
        self.install_cmd = install_cmd
        # command to run the relevant tests,
        self.test_cmd = test_cmd
        self.commit = commit
        self.repo_name = repo_name
        # modifications to the test suite for this task instance,
        self.test_patch = test_patch
        self.testcases_passing = testcases_passing
        self.testcases_failing = testcases_failing
        self.patched_diff = ""

    @property
    def project_path(self) -> str:
        return self.repo_path

    @project_path.setter
    def project_path(self, value: str) -> None:
        self.repo_path = value

    def get_issue_statement(self) -> str:
        return self.problem_statement

    def setup_project(self):
        # get the correct version of the project and commit-specific pip install
        task = self
        repo_reset_and_clean_checkout(task.commit, task.project_path)
        return self._do_install()

    def dump_poc(self, poc_code) -> None:
        # setup the execution poc for the project
        task = self
        poc_file = os.path.join(task.project_path, "poc_code.py")
        with open(poc_file, "w") as f:
            f.write(poc_code)

    def is_execute(self, poc_info):
        poc_type = poc_info["result"]["poc"]["type"]
        poc_is_multi = poc_info["result"]["poc"].get("is_multi",False)
        if poc_type == "python" and not poc_is_multi:
            return True
        return False
    # start_commit is the commit that we start the backward search from
    # initial_commit is the commit in which we are fixing the bug
    def get_bug_introducing_commit_info(self, poc_info, start_commit, initial_commit) -> dict:
        task = self
        poc_type = poc_info["result"]["poc"]["type"]
        poc_is_multi = poc_info["result"]["poc"]["is_multi"]
        if poc_type == "python" and not poc_is_multi:
            repo_path = task.project_path

            # Use start_commit as the current commit
            current_commit = start_commit
            print(f"Starting search from commit: {current_commit}")

            # Checkout to start_commit
            cmd = f'git checkout -f {current_commit}'
            cp = run_string_cmd_in_conda(cmd, task.env_name, cwd=repo_path, capture_output=True, text=True)
            if cp.returncode != 0:
                print(cp.stderr)
                raise RuntimeError(f"Command {cmd} failed.")

            print('Running git clean -fd -e poc_code.py to remove unrelated files...')
            repo_clean_except_poc(self.project_path)

            # Run poc_code.py and save its output as reference
            print("Running poc_code.py at the start commit to get reference output...")
            cmd = "PYTHONWARNINGS=ignore timeout -s SIGKILL 60s python -W ignore poc_code.py"
            cp = run_string_cmd_in_conda(cmd, task.env_name, cwd=repo_path, capture_output=True, text=True)
            current_stdout, current_stderr = cp.stdout, cp.stderr
            # clean the output (remove warnings and other non-deterministic parts like addresses, even though we have -W ignore, some warnings are still printed)
            current_stdout = clean_poc_output(current_stdout)
            current_stderr = clean_poc_output(current_stderr)

            # Retrieve the list of commits for backward search starting from start_commit
            print("Retrieving the list of commits for backward search...")
            cmd = f'git rev-list --first-parent {current_commit}'
            cp = run_string_cmd_in_conda(cmd, task.env_name, cwd=repo_path, capture_output=True, text=True)
            if cp.returncode != 0:
                print(cp.stderr)
                raise RuntimeError(f"Command {cmd} failed.")
            else:
                commits = cp.stdout.strip().splitlines()
            commits.reverse()  # Order commits chronologically (from earliest to latest)

            # Initialize binary search boundaries
            left = 0
            right = len(commits) - 1
            earliest_commit = None

            print("Starting binary search to find the earliest commit with matching output...")

            while left <= right:
                mid = (left + right) // 2
                commit = commits[mid]
                print(f"Checking commit {commit} ({mid + 1}/{len(commits)})")

                # Checkout to the commit being tested
                cmd = f'git checkout -f {commit}'
                cp = run_string_cmd_in_conda(cmd, task.env_name, cwd=repo_path, capture_output=True, text=True)
                if cp.returncode != 0:
                    print(cp.stderr)
                    raise RuntimeError(f"Command {cmd} failed.")

                print('Running git clean -fd -e poc_code.py to remove unrelated files...')
                repo_clean_except_poc(self.project_path)

                # Run poc_code.py and capture its output
                cmd = "PYTHONWARNINGS=ignore timeout -s SIGKILL 60s python -W ignore poc_code.py"
                cp = run_string_cmd_in_conda(cmd, task.env_name, cwd=repo_path, capture_output=True, text=True)
                stdout, stderr = cp.stdout, cp.stderr
                # clean the output (remove warnings and other non-deterministic parts like addresses, even though we have -W ignore, some warnings are still printed)
                stdout = clean_poc_output(stdout)
                stderr = clean_poc_output(stderr)

                # Compare outputs with the reference outputs
                if stdout == current_stdout and stderr == current_stderr:
                    # Outputs match, continue searching earlier commits
                    if mid == 0:
                        # Earliest commit found
                        return {}
                    earliest_commit = commit
                    right = mid - 1
                else:
                    # Outputs differ, search later commits
                    left = mid + 1
            print('earliest_commit', earliest_commit)
            if not earliest_commit: # no commit is found to be the same as the start commit
                earliest_commit = start_commit
            print('earliest_commit', earliest_commit)                    
            try:
                print(f"Earliest commit with matching output: {earliest_commit}")
                result = {"earliest_commit": earliest_commit}
                
                # Get the parent commit of earliest_commit
                cmd = f'git rev-parse {earliest_commit}^'
                cp = run_string_cmd_in_conda(cmd, task.env_name, cwd=repo_path, capture_output=True, text=True)
                if cp.returncode != 0:
                    print(cp.stderr)
                    raise RuntimeError(f"Failed to get parent commit of {earliest_commit}.")
                parent_commit = cp.stdout.strip()
                result['parent_commit'] = parent_commit
                
                # Checkout to parent_commit
                cmd = f'git checkout -f {parent_commit}'
                cp = run_string_cmd_in_conda(cmd, task.env_name, cwd=repo_path, capture_output=True, text=True)
                if cp.returncode != 0:
                    print(cp.stderr)
                    raise RuntimeError(f"Command {cmd} failed.")

                print('Running git clean -fd -e poc_code.py to remove unrelated files...')
                repo_clean_except_poc(self.project_path)

                # Run poc_code.py and capture its output
                print(f"Running poc_code.py at the parent commit {parent_commit} to get outputs...")
                cmd = "PYTHONWARNINGS=ignore timeout -s SIGKILL 60s python -W ignore poc_code.py"
                cp = run_string_cmd_in_conda(cmd, task.env_name, cwd=repo_path, capture_output=True, text=True)
                parent_stdout, parent_stderr = cp.stdout, cp.stderr
                # clean the output (remove warnings and other non-deterministic parts like addresses, even though we have -W ignore, some warnings are still printed)
                parent_stdout = clean_poc_output(parent_stdout)
                parent_stderr = clean_poc_output(parent_stderr)
                result["parent_commit_stdout"] = parent_stdout
                result["parent_commit_stderr"] = parent_stderr

                # Get the list of files changed in earliest_commit relative to its parent
                cmd = f'git diff --name-only {parent_commit} {earliest_commit}'
                cp = run_string_cmd_in_conda(cmd, task.env_name, cwd=repo_path, capture_output=True, text=True)
                if cp.returncode != 0:
                    print(cp.stderr)
                    raise RuntimeError(f"Command to get changed files failed.")
                changed_files = cp.stdout.strip().splitlines()
                result["changed_files"] = changed_files

                # Get the git diff between parent_commit and earliest_commit
                cmd = f'git diff {parent_commit} {earliest_commit}'
                cp = run_string_cmd_in_conda(cmd, task.env_name, cwd=repo_path, capture_output=True, text=True)
                if cp.returncode != 0:
                    print(cp.stderr)
                    raise RuntimeError(f"Command to get git diff failed.")
                git_diff = cp.stdout
                result["git_diff"] = git_diff

                # Finally, restore to initial_commit
                print("Restoring to the initial commit...")
                cmd = f'git checkout -f {initial_commit}'
                cp = run_string_cmd_in_conda(cmd, task.env_name, cwd=repo_path, capture_output=True, text=True)
                if cp.returncode != 0:
                    print(cp.stderr)
                    raise RuntimeError(f"Command {cmd} failed.")

                print('Running git clean -fd -e poc_code.py to remove unrelated files...')
                repo_clean_except_poc(self.project_path)

                return result
            finally:
                # Ensure restoration to initial_commit regardless of outcome
                print("Ensuring the repository is restored to the initial commit...")
                cmd = f'git checkout -f {initial_commit}'
                cp = run_string_cmd_in_conda(cmd, task.env_name, cwd=repo_path, capture_output=True, text=True)
                if cp.returncode != 0:
                    print(cp.stderr)
                    raise RuntimeError(f"Command {cmd} failed.")

                print('Running git clean -fd -e poc_code.py to remove unrelated files...')
                repo_clean_except_poc(self.project_path)
        else:
            return {}

    def get_poc_coverage(self, poc_info) -> str:
        task = self
        # clean the project
        print('Running git clean -fd -e poc_code.py to unrealated files...')
        repo_clean_except_poc(self.project_path)

        poc_type = poc_info["result"]["poc"]["type"]
        poc_is_multi = poc_info["result"]["poc"]["is_multi"]
        print(f"Executing POC for task {task.task_id}")
        print(f"Poc code: {poc_info['result']['poc']['poc_code']}")
        if poc_type == "python":
            cmd = "timeout -s SIGKILL 60s  coverage run poc_code.py"
        else:
            cmd = ""
        coverage = ""
        if cmd != "" and poc_type == "python" and not poc_is_multi:
            # only run python now
            try:
                cp = run_string_cmd_in_conda(cmd, task.env_name, cwd=task.project_path, capture_output=True, text=True, timeout=40)
            except subprocess.TimeoutExpired as e:
                print(e)
                coverage = "No data, TimeoutError"
                return coverage
            if cp.returncode != 0:
                print(cp.stderr)
                raise RuntimeError(f"Command {cmd} failed.")
            cp = run_string_cmd_in_conda("coverage report -m -i", task.env_name, cwd=task.project_path, capture_output=True, text=True)
            if cp.returncode != 0:
                print(cp.stderr)
                raise RuntimeError(f"Command {cmd} failed.")
            else:
                coverage=cp.stdout
                #print(exec_output)
        return coverage

    def execute_poc(self, poc_info) -> dict:
        task = self
        exec_output = {
            "stdout": "",
            "stderr": "",
        }
        # clean the project
        print('Running git clean -fd -e poc_code.py to unrealated files...')
        repo_clean_except_poc(self.project_path)
        poc_type = poc_info["result"]["poc"]["type"]
        poc_is_multi = poc_info["result"]["poc"]["is_multi"]
        print(f"Executing POC for task {task.task_id}")
        print(f"Poc code: {poc_info['result']['poc']['poc_code']}")
        if poc_type == "python":
            cmd = "timeout -s SIGKILL 60s  python -W always poc_code.py"
        else:
            cmd = ""
        if cmd != "" and poc_type == "python" and not poc_is_multi:
            # only run python now
            try:
                cp = run_string_cmd_in_conda(cmd, task.env_name, cwd=task.project_path, capture_output=True, text=True, timeout=40)
            except subprocess.TimeoutExpired as e:
                print(e)
                exec_output = {
                    "stdout": "TimeoutError",
                    "stderr": "TimeoutError"
                }
                print(exec_output)
                return exec_output
            if cp.returncode != 0:
                print(cp.stderr)
                raise RuntimeError(f"Command {cmd} failed.")
            else:
                exec_output = {
                    "stdout": cp.stdout,
                    "stderr": cp.stderr
                }
                print(exec_output)
        return exec_output

    def apply_patch(self, patch="") -> dict:
        if patch != "":
            apply_patch(patch, self.repo_path)
            self._do_install()
        elif self.patched_diff != "":
            apply_patch(self.patched_diff, self.repo_path)
            self._do_install()

    def execute_functionality_test(self) -> dict:
        task = self
        cmd = task.test_cmd
        cmd = f"timeout -s SIGKILL 300s {cmd}"
        cp = run_string_cmd_in_conda(cmd, task.env_name, cwd=task.project_path, capture_output=True, text=True)
        if cp.returncode != 0:
            print(cp.stderr)
            raise RuntimeError(f"Command {cmd} failed.")
        else:
            execution_output = {
                "stdout": cp.stdout,
                "stderr": cp.stderr
            }
        return execution_output
    
    def execute_functionality_test_only_fail(self, cmd) -> dict:
        task = self
        cmd = f"timeout -s SIGKILL 300s {cmd}"
        cp = run_string_cmd_in_conda(cmd, task.env_name, cwd=task.project_path, capture_output=True, text=True)
        if cp.returncode != 0:
            print(cp.stderr)
            raise RuntimeError(f"Command {cmd} failed.")
        else:
            execution_output = {
                "stdout": cp.stdout,
                "stderr": cp.stderr
            }
        return execution_output

    def reset_project(self) -> None:
        repo_reset_and_clean_checkout(self.commit, self.repo_path)

    def _do_install(self):
        """Do left-over install commands after setting up.
        The commands being run here are 'pre_install' and 'install' defined in
        harness/constants.py file in SWE-bench.
        """
        task = self
        if not task.pre_install_cmds and not task.install_cmd:
            # no command for installation, skip
            return

        # (0) For matplotlib, qhull tarball download
        # just fails, so we need to pre-install the system version and use it
        if "matplotlib" in task.task_id:
            mplsetup = os.path.join(task.project_path, "mplsetup.cfg")
            with open(mplsetup, "w") as f:
                f.write("[libs]\nsystem_qhull = true")
        # (1) pre-install
        for cmd in task.pre_install_cmds:
            cp = run_string_cmd_in_conda(
                cmd, task.env_name, cwd=task.project_path, capture_output=True, text=True
            )
            if cp.returncode != 0:
                print(cp.stderr)
                raise RuntimeError(f"Command {cmd} failed.")

        # (2) install
        cp = run_string_cmd_in_conda(
            task.install_cmd, task.env_name, cwd=task.project_path, capture_output=True, text=True
        )
        if cp.returncode != 0:
            print(cp.stderr)
            raise RuntimeError(f"Command {task.install_cmd} failed.")
        else:
            cp.stderr = cp.stderr.replace("__pyx_L1_error", " ")# sklearn has warnings containing string "__pyx_L1_error"
            if "error" in cp.stderr or "Error" in cp.stderr:
                return False
        # (3) xmlrunner for our custom run_test; coverage required for fault localization
        other_install_cmd = (
            "python -m pip install xmlrunner coverage pytest pytest-cov"
        )
        cp = run_string_cmd_in_conda(
            other_install_cmd, task.env_name, cwd=task.project_path, capture_output=True, text=True
        )
        if cp.returncode != 0:
            print(cp.stderr)
            raise RuntimeError(f"Command {other_install_cmd} failed.")

        return True


class GymTask(Task):
    """
    For SWE-Gym
    """
    def __init__(self, task_id, problem_statement):

        # Call parent constructor for base functionality
        super().__init__()
        self.task_id = task_id
        self.problem_statement = problem_statement
        docker_tag = task_id.replace('__', '_s_')
        image_name = f"xingyaoww/sweb.eval.x86_64.{docker_tag}"
        self.image_name = image_name

    def project_path(self, value: str) -> None:
        self.repo_path = value

    def reset_project(self) -> None:
        reset_and_clean(self.container_id)

    def init_project(self):
        self.container_id = get_container(self.image_name)

    def setup_project(self):
        # get the correct version of the project and commit-specific pip install
        return reset(self.container_id)

    def is_execute(self, poc_info):
        poc_type = poc_info["result"]["poc"]["type"]
        poc_is_multi = poc_info["result"]["poc"].get("is_multi",False)
        if poc_type == "python" and not poc_is_multi:
            return True
        return False

    def execute_poc(self, poc_info) -> dict:
        task = self
        exec_output = {
            "stdout": "",
            "stderr": "",
        }
        # clean the project
        print('Running git clean -fd -e poc_code.py to unrealated files...')
        reset_and_clean(self.container_id)
        poc_type = poc_info["result"]["poc"]["type"]
        poc_is_multi = poc_info["result"]["poc"]["is_multi"]
        print(f"Executing POC for task {task.task_id}")
        print(f"Poc code: {poc_info['result']['poc']['poc_code']}")
        poc_code_dict = poc_info['result']['poc']['poc_code']
        _, poc_code = next(iter(poc_code_dict.items()))
        try:
            cp = run_poc(self.container_id, poc_code)
        except subprocess.TimeoutExpired as e:
            print(e)
            exec_output = {
                "stdout": "TimeoutError",
                "stderr": "TimeoutError"
            }
            print(exec_output)
            return exec_output

        exec_output = {
            "stdout": cp.stdout,
            "stderr": cp.stderr
        }
        print(exec_output)
        return exec_output

    def apply_patch(self, patch="") -> dict:
        if patch != "":
            apply_diff(patch, self.repo_path)
        elif self.patched_diff != "":
            apply_diff(self.patched_diff, self.repo_path)

    def clean(self):
        delete_container(self.container_id)


class RawTask(ABC):
    @property
    @abstractmethod
    def task_id(self) -> str:
        raise NotImplementedError("abstract base class")

    @abstractmethod
    def to_task(self) -> Task:
        raise NotImplementedError("abstract base class")

    @abstractmethod
    def dump_meta_data(self, output_dir: str) -> None:
        raise NotImplementedError("abstract base class")


class RawSweTask(RawTask):
    """
    Encapsulate everything required to run one task.
    """

    def __init__(self, task_id: str, setup_info: dict, task_info: dict):
        # a counter str, format "1/150", which means first task out of 150
        # id from the benchmark
        self._task_id = task_id
        # setup_info (Dict): keys: ['repo_path', 'env_name', 'pre_install', 'install','test_cmd']
        self.setup_info = setup_info
        # task_info (Dict): keys: ['base_commit', 'hints_text', 'created_at',
        # 'test_patch', 'repo', 'problem_statement', 'version', 'instance_id',
        # 'FAIL_TO_PASS', 'PASS_TO_PASS', 'environment_setup_commit']
        self.task_info = task_info

    @property
    def task_id(self) -> str:
        return self._task_id

    def to_task(self) -> SweTask:
        task_id = self.task_id
        setup_info = self.setup_info
        task_info = self.task_info

        return SweTask(
            task_id=task_id,
            problem_statement=task_info["problem_statement"],
            repo_path=setup_info["repo_path"],
            env_name=setup_info["env_name"],
            pre_install_cmds=setup_info["pre_install"],
            install_cmd=setup_info["install"],
            # command to run the relevant tests,
            test_cmd=setup_info["test_cmd"],
            commit=task_info["base_commit"],
            repo_name=task_info["repo"],
            # modifications to the test suite for this task instance,
            test_patch=task_info["test_patch"],
            testcases_passing=task_info["PASS_TO_PASS"],
            testcases_failing=task_info["FAIL_TO_PASS"],
        )

    def dump_meta_data(self, output_dir: str):
        meta = {
            "task_id": self.task_id,
            "setup_info": self.setup_info,
            "task_info": self.task_info,
        }
        with open(pjoin(output_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)
        with open(pjoin(output_dir, "problem_statement.txt"), "w") as f:
            f.write(self.task_info["problem_statement"])
        with open(pjoin(output_dir, "developer_patch.diff"), "w") as f:
            f.write(self.task_info["patch"])


def parse_task_list_file(task_list_file: str) -> list[str]:
    """
    Parse the task list file.
    The file should contain one task/instance id per line, without other characters.
    """
    with open(task_list_file) as f:
        task_ids = f.readlines()
    return [x.strip() for x in task_ids]


def make_swe_tasks(
    all_task_ids: list[str],
    setup_map_file: str,
    tasks_map_file: str,
) -> list[SweTask]:

    with open(setup_map_file) as f:
        setup_map = json.load(f)
    with open(tasks_map_file) as f:
        tasks_map = json.load(f)

    # Check if all task ids are in the setup and tasks map
    # This allows failing safely if some tasks are not set up properly
    missing_task_ids = [
        x for x in all_task_ids if not (x in setup_map and x in tasks_map)
    ]
    if missing_task_ids:
        # Log the tasks that are not in the setup or tasks map
        for task_id in sorted(missing_task_ids):
            print(
                f"Skipping task {task_id} which was not found in setup or tasks map."
            )
        # And drop them from the list of all task ids
        all_task_ids = filter(lambda x: x not in missing_task_ids, all_task_ids)

    all_task_ids = sorted(all_task_ids)

    # for each task in the list to run, create a Task instance
    all_tasks = []
    for task_id in all_task_ids:
        setup_info = setup_map[task_id]
        task_info = tasks_map[task_id]
        task = RawSweTask(task_id, setup_info, task_info)
        all_tasks.append(task.to_task())
    return all_tasks


def make_gym_tasks(
    swe_gym_data
):
    all_tasks = []
    for data in swe_gym_data:
        task_id = data["instance_id"]
        problem_statement = data["problem_statement"]
        task = GymTask(task_id, problem_statement)
        all_tasks.append(task)
    return all_tasks
