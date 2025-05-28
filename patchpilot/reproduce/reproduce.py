import argparse
import json
import os
import re
import concurrent.futures

from datasets import load_dataset

from patchpilot.reproduce.task import make_swe_tasks, make_gym_tasks
from patchpilot.util.utils import setup_logger, ensure_directory_exists
from patchpilot.util.model import make_model
from patchpilot.reproduce.task import parse_task_list_file


def check_existing_reproduce_ids(reproduce_path, num_samples):
    instance_ids = set()

    for root, _, files in os.walk(reproduce_path):
        for file in files:
            if file.endswith(f'issue_parsing_report_{num_samples-1}.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        return instance_ids
                if 'instance_id' in data:
                    instance_ids.add(data['instance_id'])
    return instance_ids

execution_output_template="""
Here are the stderr and stdout of the PoC you generated last time:
--- Begin Execution Output ---
{execution_output}
--- End Execution Output ---
"""

errors_template="""
### Errors ###
Here are some errors of the PoC you generated last time, please pay attention to them and do not make the same mistakes again.
--- Begin Errors ---
{reason}
--- End Errors ---
"""
last_time_poc_code_template = """
### PoC Generated Last Time ###
Here is the PoC you generated last time:
--- Begin PoC ---
{last_time_poc_code}
--- End PoC ---
"""

optimize_poc_prompt = """
When generating a PoC script, follow these steps **in order**:

**Try to extract an existing PoC from the issue description**
   * Scan the **GitHub issue description** for Python code blocks or inline snippets that appear to reproduce the bug.
   * If such a snippet exists, **use it verbatim as the base PoC** and only make the minimal edits needed to run:
       - Remove interactive prompts (`>>>`, `$`, `In [ ]:`) and any captured output lines.
       - Add any missing `import` statements.
       - Convert Python2 syntax to Python3, if present.
       - Merge multiple fragments into a single runnable file in their original order.

**If no valid PoC can be extracted, write one yourself**
   * Use the *specific* classes, functions, or code paths named in the issue to trigger the bug.
   * Keep the script minimal—just enough to demonstrate the failure (e.g., an `assert`, an expected exception, or a visible incorrect result).

**General rules for both cases**
   * The PoC **must be a single, self-contained Python3 file**.
   * If the issue description includes other languages or shell commands, recreate their behavior in Python (e.g., with `subprocess` or file operations).
   * If the snippet refers to external files, create them programmatically inside the script.
   * Always include `print()` or `assert` statements so the failure is obvious when the script is executed.

**Output format**
Return **exactly** python code wrapped in triple backticks, with no other text.

```python
{{poc_code here}}
```
### Context Provided to You
{last_time_poc_code}

{execution_output}

{reason}

### GitHub Issue Description
--- Begin Issue Description ---
{problem_statement}
--- End Issue Description ---
"""

judge_execution_result_prompt = """
You are a developer assigned to investigate a bug. You've been provided with a script intended to reproduce a specific wrong behavior. However, the script may contain errors, so its output may or may not reflect the wrong behavior described in the issue. Your task is to determine whether the Script Output manifests the wrong behavior as described in the Raw Issue Description.

When evaluating whether a script successfully reproduces an issue, ensure that the script uses the specific classes, functions, or code segments mentioned in the issue description from the original project to trigger the bug. You should explicitly reason whether the bug lies in code in the original project or just simulated by the script independently. If the bug is simulated by the script independently, you should not consider the output as relevant to the wrong behavior and output "No".
You should not assume that there is other output that is not shown in the Script Output. 

### Raw Issue Description ###
--- Begin Raw Issue Description ---
{issue_description}
--- End Raw Issue Description ---

### Script Code ###
--- Begin Script Code ---
{poc_code}
--- End Script Code ---

### Script Output ###
--- Begin Script Output ---
{execution_output}
--- End Script Output ---

Please analyze whether the Script Output manifests the issue, and provide your judgment along with a clear explanation of your reasoning.

### Output Format ###
Example 1:
<reasoning>Some reasoning process..</reasoning>
<judgement>No</judgement>

Example 2:
<reasoning>Some reasoning process..</reasoning>
<judgement>Yes</judgement>
"""


class LLMRP:
    def __init__(
            self,
            instance_id,
            problem_statement,
            model_name,
            backend,
            logger,
    ):
        self.instance_id = instance_id
        self.problem_statement = problem_statement
        self.max_tokens = 8192
        self.model_name = model_name
        self.backend = backend
        self.logger = logger    


    def clean_and_parse_response(self, text: str, default):
        poc_info = {
                "type": "python",
                "poc_code": {},
        }
        block_re = re.compile(r"```python\s*([\s\S]*?)\s*```", re.IGNORECASE)
        match = block_re.search(text)

        if match:
            poc_info["poc_code"]["poc_code.py"] = match.group(1).strip()
            return poc_info

        return default


def reproduce_instance(task, args, existing_instance_ids):
    instance_id = task.task_id
    log_file = os.path.join(
        args.reproduce_folder, "reproduce_logs", f"{instance_id}.log"
    )
    logger = setup_logger(log_file)
    logger.info(f"Processing bug {instance_id}")

    if instance_id in existing_instance_ids:
        print(f"Skip reproducing existing instance_id: {instance_id}")
        logger.info(f"Skip reproducing existing instance_id: {instance_id}")
        return

    logger.info(f"================ reproducing {instance_id} ================")
    if args.benchmark == "gym":
        task.init_project()
    problem_statement = task.problem_statement

    rp = LLMRP(
        instance_id,
        problem_statement,
        args.model,
        args.backend,
        logger,
    )
    
    # Create the issue folder
    issue_id_folder = os.path.join(args.reproduce_folder, task.task_id)
    ensure_directory_exists(issue_id_folder)
    
    # Generate multiple PoCs based on num_samples
    for sample_index in range(args.num_samples):
        logger.info(f"Generating PoC sample {sample_index} for instance {instance_id}")
        retry = 0
        reason = ""
        if_match = False
        last_time_poc_code = ""
        execution_output = {}
            
        # Generate POC with Retry
        while not if_match and retry < args.max_retries:
            print(f"Sample {sample_index}, Retry {retry} for instance {instance_id}")
            logger.info(f"Sample {sample_index}, Retry {retry} for instance {instance_id}")
            execution_output_prompt=""
            last_time_poc_code_prompt=""
            errors_prompt=""
            if last_time_poc_code:
                last_time_poc_code_prompt=last_time_poc_code_template.format(last_time_poc_code=json.dumps(last_time_poc_code, indent=4))
            if execution_output:
                execution_output_prompt=execution_output_template.format(execution_output=json.dumps(execution_output, indent=4))
            if reason:
                errors_prompt=errors_template.format(reason=reason)
            message = optimize_poc_prompt.format(
                problem_statement=problem_statement,
                last_time_poc_code=last_time_poc_code_prompt,
                execution_output=execution_output_prompt,
                reason=errors_prompt,
            ).strip()
            rp.logger.info(f"Instance: {instance_id}, prompting with message:\n{message}")
            print(f"Instance: {instance_id}, prompting with message:\n{message}")
            rp.logger.info("=" * 80)
            print("=" * 80)

            model = make_model(
                model=rp.model_name,
                backend=rp.backend,
                logger=rp.logger,
                max_tokens=rp.max_tokens,
                temperature=0.6 if sample_index > 0 else 0,  # Use some temperature for diversity in samples > 0
                batch_size=1,
            )
            if args.reasoning_mode:
                traj = model.codegen(message, num_samples=1, reasoning_mode=args.reasoning_mode, port=args.port, ip=args.ip)[0]
            else:
                traj = model.codegen(message, num_samples=1, port=args.port, ip=args.ip)[0]
            rp.logger.info(f"Got response:\n{traj}")
            print(f"Got response:\n{traj}")
            traj["prompt"] = message
            default_poc_description = {
                    "type": "unknown",
                    "poc_code": {},
                }
            poc_description = rp.clean_and_parse_response(traj["response"], default_poc_description)
            print("==========================")

            if args.reasoning_mode:
                reproduce_folder = args.reproduce_folder
                reasoning_output = os.path.join(reproduce_folder, f"reasoning_data.jsonl")
                reasoning_data = {
                    "instance_id": task.task_id,
                    "prompt": message,
                    "reasoning_content": traj["reasoning_content"],
                    "response": traj["response"],
                        "sample_index": sample_index
                }

                with open(reasoning_output, "a") as f:
                    f.write(json.dumps(reasoning_data) + "\n")

            print(f"poc_description: {poc_description}")       
            if len(poc_description["poc_code"]) != 1:
                poc_description["is_multi"] = True
            else:
                poc_description["is_multi"] = False
            poc_info = {
                "instance_id": instance_id,
                    "sample_index": sample_index,
                "result": {
                    "poc": poc_description,
                    "oracle": {
                        "issue_description": problem_statement,
                        "reasoning": "",
                        "execution_output": {
                            "stdout": "",
                            "stderr": "",
                        },
                    }
                }
            }
            poc_info, execution_output = execute_reproduce_instance(task, args, poc_info)
            if_match = poc_info["result"]["oracle"]["exec_match_wrong_behavior"]
            last_time_poc_code = poc_info["result"]["poc"]["poc_code"]
            reason = poc_info["result"]["oracle"]["if_match_reasoning"]
            retry += 1
        
        poc_info["result"]["retry"] = retry
                    
        # Save each PoC result in a separate file
        poc_output_file = os.path.join(issue_id_folder, f"issue_parsing_report_{sample_index}.json")
        with open(poc_output_file, "w") as ft:
            json.dump(poc_info, ft, indent=4)
            print(f"Execute reproducer for issue {task.task_id}, sample {sample_index} completed! The result is in {poc_output_file}")
        
    if args.benchmark == "gym":
        task.clean()

def execute_reproduce_instance(task, args, poc_info):
    is_setup = task.setup_project()
    if not is_setup:
        poc_info["result"]["setup"] = False
    else:
        poc_info["result"]["setup"] = True
    
    # poc is empty or could not be executed (not a single python file)
    if (not poc_info["result"]["poc"]["poc_code"]) or (not task.is_execute(poc_info)):
        poc_info["result"]["oracle"]["exec_match_wrong_behavior"] = False
        poc_info["result"]["oracle"]["execution_output"] = {
            "stdout": "",
            "stderr": "",
        }
        poc_info["result"]["oracle"]["if_match_reasoning"] = "poc code is empty or could not be executed, no match"
        print("instance {} poc code is empty or could not be executed, no match".format(task.task_id))
        print("issue description: {}".format(task.problem_statement))
        return poc_info, {}
    # poc is not empty
    else:
        poc_code_dict = poc_info["result"]["poc"]["poc_code"]
        _, poc_code = next(iter(poc_code_dict.items()))
        if args.benchmark != "gym":
            task.dump_poc(poc_code)
        execution_output = task.execute_poc(poc_info)
        poc_info["result"]["oracle"]["execution_output"] = execution_output
        log_file = os.path.join(
            args.reproduce_folder, "reproduce_logs", f"{task.task_id}.log"
        )
        logger = setup_logger(log_file)
        rp = LLMRP(
            task.task_id,
            task.problem_statement,
            args.model,
            args.backend,
            logger,
        )
        message = judge_execution_result_prompt.format(
            issue_description=task.problem_statement,
            poc_code=poc_info["result"]["poc"]["poc_code"],
            execution_output=execution_output
        ).strip()
        rp.logger.info(f"Instance: {task.task_id}, prompting with message:\n{message}")
        print(f"Instance: {task.task_id}, prompting with message:\n{message}")
        rp.logger.info("=" * 80)
        print("=" * 80)

        model = make_model(
            model=rp.model_name,
            backend=rp.backend,
            logger=rp.logger,
            max_tokens=rp.max_tokens,
            temperature=0,
            batch_size=1,
        )
        if args.reasoning_mode:
            traj = model.codegen(message, num_samples=1, reasoning_mode=args.reasoning_mode, port=args.port, ip=args.ip)[0]
        else:
            traj = model.codegen(message, num_samples=1, port=args.port, ip=args.ip)[0]
        rp.logger.info(f"Got response:\n{traj}")
        traj["prompt"] = message
        raw_output = traj["response"]

        if args.reasoning_mode:
            reproduce_folder = args.reproduce_folder
            reasoning_output = os.path.join(reproduce_folder, f"reasoning_data.jsonl")
            reasoning_data = {
                "instance_id": task.task_id,
                "prompt": message,
                "reasoning_content": traj["reasoning_content"],
                "response": traj["response"],
            }

            with open(reasoning_output, "a") as f:
                f.write(json.dumps(reasoning_data) + "\n")

        print(raw_output)
        judge_result = ""
        try:
            judge_result = raw_output.split("<judgement>")[1].split("</judgement>")[0]
        except:
            logger.error(f"Failed to parse judgement: {raw_output}")
            print(f"Failed to parse judgement: {raw_output}")
        if "yes" in judge_result.lower():
            poc_info["result"]["oracle"]["exec_match_wrong_behavior"] = True
        else:
            poc_info["result"]["oracle"]["exec_match_wrong_behavior"] = False
        reasoning = ""
        try:
            reasoning = raw_output.split("<reasoning>")[1].split("</reasoning>")[0]
        except:
            logger.error(f"Failed to parse reasoning: {raw_output}")
            print(f"Failed to parse reasoning: {raw_output}")
            reasoning = "Failed to parse reasoning"
        poc_info["result"]["oracle"]["if_match_reasoning"] = reasoning
    return poc_info, execution_output


def reproduce(args):
    existing_instance_ids = (
        check_existing_reproduce_ids(args.reproduce_folder, args.num_samples)
    )

    if args.num_threads == 1:
        for task in args.tasks_list:
            reproduce_instance(
                task, args, existing_instance_ids
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.num_threads
        ) as executor:
            futures = [
                executor.submit(
                    reproduce_instance,
                    task,
                    args,
                    existing_instance_ids
                )
                for task in args.tasks_list
            ]
            concurrent.futures.wait(futures)
            for fut in futures:
                # rasie exception if any
                result = fut.result()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_mode", action="store_true", default=False)
    parser.add_argument("--reproduce_folder", type=str, required=True)
    parser.add_argument(
        "--setup_map",
        type=str,
        required=True,
        help="Path to json file that contains the setup information of the projects.",
    )
    parser.add_argument(
        "--tasks_map",
        type=str,
        required=True,
        help="Path to json file that contains the tasks information.",
    )
    parser.add_argument(
        "--task_list_file",
        type=str,
        help="Path to the file that contains all tasks ids to be run.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for creating API requests",
    )
    parser.add_argument("--target_id", type=str)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument(
        "--model",
        type=str,
        default="UCSB-SURFI/Co-PatcheR-Val-no-assert-14B",
    )
    parser.add_argument(
        "--backend", type=str, default="opensource", choices=["openai", "deepseek", "claude", "opensource"]
    )
    parser.add_argument("--benchmark", default="lite", choices=["lite", "verified", "full", "gym"])
    parser.add_argument("--num_samples", type=int, default=1, help="Number of different PoC samples to generate")
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2951)

    args = parser.parse_args()

    os.makedirs(os.path.join(args.reproduce_folder, "reproduce_logs"), exist_ok=True)

    # write the arguments
    with open(f"{args.reproduce_folder}/reproduce_args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    assert not (args.target_id is not None and args.task_list_file is not None), "Cannot specify both task and task-list."
    all_task_ids = []
    if args.task_list_file is not None:
        all_task_ids = parse_task_list_file(args.task_list_file)
    elif args.target_id is not None:
        all_task_ids = [args.target_id]
    assert len(all_task_ids) > 0, "No task ids to run."

    if args.benchmark == "gym":
        swe_gym_data = load_dataset("SWE-Gym/SWE-Gym-Lite", split="train")
        args.tasks_list = make_gym_tasks(swe_gym_data)
    else:
        args.tasks_list = make_swe_tasks(all_task_ids, args.setup_map, args.tasks_map)

    reproduce(args)


if __name__ == "__main__":
    main()
