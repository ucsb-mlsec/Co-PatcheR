import argparse
import concurrent.futures
import json
import os
import random
import re
import ast
from datasets import load_dataset
from tqdm import tqdm
from filelock import FileLock
from patchpilot.util.model import make_model
from patchpilot.util.postprocess_data import fake_git_repo
from patchpilot.util.preprocess_data import (
    get_full_file_paths_and_classes_and_functions,
    get_repo_structure,
)
from patchpilot.util.utils import load_json, load_jsonl, setup_logger
from patchpilot.repair.utils import (
    construct_topn_file_context,
    apply_patch_with_indent_alignment
)
from patchpilot.reproduce.reproduce import ensure_directory_exists
from patchpilot.reproduce.task import parse_task_list_file

reloca_ids = []
reloca_locs = dict()
not_found_file_dict = dict()

locs_global = []

lock_path = '/tmp/lock_file'
output_file_lock = FileLock(lock_path)

PROJECT_STRUCTURE = os.environ.get("PROJECT_STRUCTURE", None)

num_generated_sample = 0
round_idx = 0
last_round = False

planning_example_format = """
Here is an example of the output format:

--- BEGIN REASON ---
The bug is caused by the function `foo` not returning the correct value.
--- END REASON ---

--- BEGIN EXPECTED BEHAVIOR ---
The function foo should return x+1
--- END EXPECTED BEHAVIOR  ---

--- BEGIN STEPS ---
<STEP> Check the input data </STEP> <Actions to be Taken> Go through the input data in input.py to identify any anomalies </Actions to be Taken>
<STEP> Modify the output data </STEP> <Actions to be Taken> Modify the output data in output.py to match the expected output </Actions to be Taken>
--- END STEPS ---
"""


whole_function_prompt = """
Cause I will give you the whole function context. You need to first return the whole function code after fixing
The format should be:
```python
def testing():
    ...
    ...
    return x
```
"""


apply_plan_prompt = """
We are currently solving the following issue within our repository. 
You are a maintainer of the project. Please analyze the bug as a maintainer, since the issue description might only describe the surface-level problem. Please analyze the bug thoroughly and infer the underlying real problem that needs to be addressed, using your inherit knowledge of the project. For example, if the goal is to fix an error or warning, focus on resolving the logic that causes the error or warning rather than simply suppressing or bypassing it.
Note that if a file name or argument is provided in the issue description as an example for reproduction, other arguments may also trigger the issue. Therefore, make the fix as general as possible. Don't restrict the fix to a specific set of arguments.
If the issue text includes a recommended fix, do not apply it directly. Instead, adapt it to align with the codebase's style and standards. Ensure that the patch considers interactions across different code sections, including nested structures, function calls, and data dependencies. The patch should maintain overall structural integrity, addressing the issue without unintended effects on other parts. Prefer solutions that are resilient to structural changes or future extensions.
You always need to adapt the code to the existing codebase's style and standards by considering the context of the code.
{whole_function_prompt}
Please generate code chunk edit to fix the issue, focusing only on the current step. I will supply one code snippet at a time.

Every *code chunk* edit must use this format:
1. The full replacement of the previous code chunk
2. The lines to replace into the source code
3. Do not miss any other lines you didn't change
4. FUNCTION REQUIRES PROPER INDENTATION

Wrap the *code chunk* edit in blocks ```python...```. Only the code chunk, no other information, no file path in the code.
Here is an example of the format:
```python
def testing():
    ...
    ...
    return x
```

## Now the issue is as follows:

Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are code segments from a relevant file which you need to judge and might need to fix.
--- BEGIN FILES ---
{content}
--- END FILES ---

Please note that the *code chunk* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Please make sure that the IDENTATION of the code in the *code chunk* edit is correct.
Do not include leading + or - signs in the *code chunk* edit.
Please not that the code segments I provide might NOT be the root cause code of the issue. So you need to judge by your knowledge whether need to fix it. If not, you should return the original code to me.
You need to be careful when choosing to modify the code, and don’t modify it unless necessary.
Please make sure that you have same IDENTATION of the code in the *code chunk* with before.
"""

apply_search_replace_plan_prompt = """
We are currently solving the following issue within our repository.

You are a maintainer of the project. Analyze the bug thoroughly and infer the underlying real problem, using your inherent knowledge of the project. Focus on resolving the root logic issue rather than suppressing symptoms.

Note that if the issue description mentions file names or arguments for reproduction, the fix must be generalized and not restricted to specific arguments. If the issue description includes a recommended fix, adapt it to align with the codebase's style and standards. Ensure your fix maintains structural integrity, considering interactions across code sections, nested structures, function calls, and data dependencies. Prefer solutions resilient to future structural changes or extensions.

The following is the issue description:

--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are the code segments from multiple files relevant to this issue. Each file is clearly marked. Decide carefully and only modify necessary segments. Preserve original indentation and formatting standards strictly.

--- BEGIN FILES ---
{content}
--- END FILES ---

Now, carefully analyze the files above. Determine which specific file segments require modifications and provide your edits using the following structured format for easy parsing:

<<< MODIFIED FILE: path/to/filename >>>
```python
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
<<< END MODIFIED FILE >>>
...

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""



reasoning_prompt = """
We are currently solving the following issue within our repository.
You are a maintainer of the project. Please analyze the bug as a maintainer, since the issue description might only describe the surface-level problem. Please analyze the bug thoroughly and infer the underlying real problem that needs to be addressed, focus on resolving the logic that causes the error or warning rather than simply suppressing or bypassing it.
Note that if a file name or argument is provided in the issue description as an example for reproduction, other arguments may also trigger the issue. Therefore, make the fix as general as possible. Don't restrict the fix to a specific set of arguments.
If the issue text includes a recommended fix, do not apply it directly. Instead, adapt it to align with the codebase's style and standards.
You always need to adapt the code to the existing codebase's style and standards by considering the context of the code.
{whole_function_prompt}
Please generate code chunk edit to fix the issue, focusing only on the current step. I will supply one code snippet at a time.

Every *code chunk* edit must use this format:
1. The full replacement of the previous code chunk
2. The lines to replace into the source code
3. Do not miss any other lines you didn't change
4. FUNCTION REQUIRES PROPER INDENTATION

Wrap the *code chunk* edit in blocks ```python...```. Only the code chunk, no other information, no file path in the code.
Here is an example of the format:
```python
def testing():
    ...
    ...
    return x
```

## Now the issue is as follows:

Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are code segments from a relevant file which you need to fix.
--- BEGIN FILES ---
{content}
--- END FILES ---

Please note that the *code chunk* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Please make sure that the IDENTATION of the code in the *code chunk* edit is correct.
Do not include leading + or - signs in the *code chunk* edit.
Please make sure that you have same IDENTATION of the code in the *code chunk* with before.
"""



vote_patch_prompt = """
We are currently addressing the following issue in our repository. Several candidate patches have been proposed to resolve this issue. Your task is to evaluate each patch in detail and select the one that offers the most effective and general solution.

Analyze the issue and provided patchs according to the following guidelines, you should look each patch at give each patch a score, output the score for each patch:

## Reason about the Scope (5 points):
Reason about the scope of the critical variable, considering the values that should and should not be affected. What situations should the patch handle, and what should it avoid? Ensure the patch correctly targets the issue without impacting unrelated code or values. Score based on the accuracy of the scope.
You should always explicitly infer the scope of the critical variable, output the exact scope of values that should and should not be affected.
It is not a negative factor if the patch introduces complexity logic.

Example:
For instance, if the issue can be triggered by an empty string, you need to explicitly consider whether it can also be triggered by None, an empty list, or other similar values. Prefer patches that only modify the variable triggering the issue. If None does not trigger the issue, the patch should not alter the behavior of None. 
Similarly, if an integer in the issue causes the problem, explicitly evaluate whether other integers can also trigger the issue. Prioritize patches that adjust the scope of the variable in a way that matches the specific values capable of triggering the issue, without impacting unrelated cases.

## Correctness (5 points):
Infer the logical root cause of the issue. Ensure the proposed patch fixes the issue as described in the problem statement and behaves as expected. 

## Reusability of Existing Functions (2 points):
Favor patches that reuse existing functions or utilities.

## Logic Changes(5 points):
If a patch reorders checks, it should get 0 points for this criteria.
You should always explicitly infer whether the checks are reordered and output the result.
If a patch broaden the scope of checks unnecessarily, it should get 0 points for this criteria. 
You should always explicitly infer whether the checks are broadened and output the result.
If a patch doesn't fix the issue completely, it should get 0 points for this criteria.

## Consideration of Structural Interactions (5 points):
Ensure that the patch handles interactions between different parts of the code, such as nested structures, function calls, or data dependencies.
The patch should maintain the integrity of the overall structure while addressing the issue, ensuring that changes in one part do not inadvertently affect other parts. 
Prefer solutions that are robust against changes in the structure or future extensions.

# Minimal Patch (2 points):
The patch should be minimal, only addressing the specific issue described in the problem statement. Avoid making unnecessary changes or introducing new functionality.

## Type (2 points):
If the patch involves checking or modifying the type of the variable, you should consider the context, and prefer types specific to the python project over general ones.

After evaluating each patch based on these criteria, conclude your analysis by stating:
"The best choice is s," where s is the integer ID of the patch you believe is the best option.

Your analysis should not involve copying any code from the patches.
Your analysis should not have any code snippets.
You should compare each patch and score each patch.

Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs.
--- BEGIN FILE ---
{content}
--- END FILE ---

Here are the candidate patches:
--- BEGIN PATCHES ---
{patches}
--- END PATCHES ---
"""

poc_info_prompt = """
Here is the code that reproduces the bug, called a proof of concept (POC).:
--- BEGIN POC ---
{poc_code}
--- END POC ---

--- BEGIN STDOUT ---
{stdout}
--- END STDOUT ---

--- BEGIN STDERR ---
{stderr}
--- END STDERR ---
"""


# a prompt for llm to attack the patch by edge cases
edge_case_prompt = """
You need to analyse and attack a patch aimed at fixing a bug in the codebase.

Here is the original issue that the patch wants to fix:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Here is the related code after applying the patch
--- BEGIN CODE ---
```
{content}
```
--- END CODE ---

Here is the patch that aims to fix the issue that may be incomplete or incorrect:
--- BEGIN PATCH ---
{diff}
--- END PATCH ---

You need to output: 
1. what edge cases can break the patch, consider complex cases such as nested structures and recursive patterns, for example, if the patch fixes an issue with an empty string, consider whether None, an empty list, or partially empty data structures might also trigger the bug.
2. why the patch is incomplete or incorrect, whether the interaction between the patched part and other parts of the codebase is not handled properly
3. whether the patch only fixes the issue for the specific case mentioned in the issue description or for all similar cases
4. whether the patch follows the codebase's style and standards, using the proper variable types, error or warning types, and adhering to the established format

"""

# The patch may have the following problems:
# -- BEGIN PROBLEMS --
# {problems}
# -- END PROBLEMS --


def weighted_sampling(models, weights):
    return random.choices(models, weights, k=1)[0]


def extract_diff_lines(patch_text):
    # Store line numbers for deleted and added lines
    # for deleted lines, we store the line number in the original file
    old_lines = set()
    new_lines = set()

    # Regex to match the hunk header (e.g., @@ -1018,8 +1018,9 @@)
    hunk_header_pattern = re.compile(r"@@ -(\d+),\d+ \+(\d+),\d+ @@")

    # Split the patch into lines for easier processing
    lines = patch_text.splitlines()

    # Initialize line counters
    old_line = new_line = None

    # Process each line in the patch
    for line in lines:
        # Match hunk headers to update line numbers
        match = hunk_header_pattern.match(line)
        if match:
            old_line = int(match.group(1))  # Start line number in the original file
            new_line = int(match.group(2))  # Start line number in the modified file
            continue

        if line.startswith('---') or line.startswith('+++'):
            continue

        # Process deletions
        if line.startswith('-'):
            old_lines.add(old_line)
            new_lines.add(new_line)
            old_line += 1  # Move to the next line in the original file

        # Process additions
        elif line.startswith('+'):
            new_lines.add(new_line)
            old_lines.add(old_line)
            new_line += 1  # Move to the next line in the modified file

        # Process context lines (not modified, so increase both counters)
        else:
            old_line += 1
            new_line += 1

    return old_lines, new_lines


def get_top_level_node(node):
    if hasattr(node, 'parent') and isinstance(node.parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return get_top_level_node(node.parent)
    return node


def get_node_intervals(node, changed_lines):
    if hasattr(node, 'body'):
        body = node.body
        start_line = getattr(node, 'lineno', None)

        # Check if body is a list to safely access body[-1]
        if isinstance(body, list) and body:
            end_line = getattr(body[-1], 'end_lineno', getattr(node, 'end_lineno', None))
        else:
            # If body is not a list, use end_lineno directly from the node if available
            end_line = getattr(node, 'end_lineno', None)

        # Check if any line in changed_lines is within the start and end lines of the node
        if start_line and end_line and any(start_line <= line <= end_line for line in changed_lines):
            top_level_node = get_top_level_node(node)
            top_start_line = getattr(top_level_node, 'lineno', None)

            # Check if top_level_node.body is a list to access top_level_node.body[-1] safely
            if hasattr(top_level_node, 'body') and isinstance(top_level_node.body, list) and top_level_node.body:
                top_end_line = getattr(top_level_node.body[-1], 'end_lineno', getattr(top_level_node, 'end_lineno', None))
            else:
                top_end_line = getattr(top_level_node, 'end_lineno', None)

            return (top_start_line-1, top_end_line)

    return None


def extract_top_level_intervals(source_code, changed_lines):
    tree = ast.parse(source_code)
    intervals = []

    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    for node in ast.walk(tree):
        interval = get_node_intervals(node, changed_lines)
        if interval:
            intervals.append(interval)

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = getattr(node, 'lineno', None)
            end_line = getattr(node, 'end_lineno', None)
            if start_line and end_line:
                intervals.append((start_line-1, end_line))

    unique_intervals = sorted(set(intervals))
    return unique_intervals


def merge_intervals(intervals):
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = []

    for interval in sorted_intervals:
        if not merged or merged[-1][1] < interval[0] - 1:
            merged.append(interval)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))

    return merged


def parse_git_diff_to_dict(diff_text: str) -> dict[str, str]:
    diff_dict = {}
    current_file = None
    current_diff = []

    # Regular expression to match the filename line in git diff output
    file_pattern = re.compile(r'^diff --git a\/(.+?) b\/(.+)$')

    for line in diff_text.splitlines():
        # Check if the line matches a new file entry
        file_match = file_pattern.match(line)
        if file_match:
            # Save the diff of the previous file (if any)
            if current_file and current_diff:
                diff_dict[current_file] = '\n'.join(current_diff)

            # Update the current file and reset the diff accumulator
            current_file = file_match.group(1)
            current_diff = [line]  # Start with the current line
        elif current_file:
            # If within a file diff, accumulate diff content
            current_diff.append(line)

    # Handle the last file's diff (if any)
    if current_file and current_diff:
        diff_dict[current_file] = '\n'.join(current_diff)

    return diff_dict


def process_loc(loc, args, swe_bench_data, prev_generations):
    instance_id = loc["instance_id"]
    log_file = os.path.join(
        args.output_folder, "repair_logs", f"{instance_id}.log"
    )
    logger = setup_logger(log_file)

    # check if the patch has been generated, skip if it has been generated
    found = False
    # we should just check the raw_output_file
    
    for entry in prev_generations:
        if entry["instance_id"] == instance_id:
            generated_for_instance = len(entry["git_diffs"])
            if generated_for_instance >= num_generated_sample + args.batch_size:
                found = True
            break

    if found:
        logger.info(f"skipping {instance_id} since patch already generated")
        print(f"skipping {instance_id} since patch already generated")
        return None

    logger.info(f"================ repairing {instance_id} ================")

    pred_files = loc["found_files"][: args.top_n]
    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = bench_data["problem_statement"]

    if PROJECT_STRUCTURE is not None:
        project_file = os.path.join(PROJECT_STRUCTURE, instance_id + ".jsonl")
        structure = load_json(project_file)
    else:
        structure = get_repo_structure(
            instance_id, bench_data["repo"], bench_data["base_commit"], "playground", "structure", args
        )
    files, classes, functions = get_full_file_paths_and_classes_and_functions(structure)

    file_contents = dict()
    for i, pred_file in enumerate(pred_files):
        content = None

        # files are all files in the repo, index 0 is the file name, index 1 is the content
        for file_content in files:
            if file_content[0] == pred_file:
                content = "\n".join(file_content[1])
                file_contents[pred_file] = content
                break

        assert content is not None, f"{pred_file} file not found"

    def extract_file_to_edit_locs(loc, pred_files):
        file_to_edit_locs = dict()
        for i, pred_file in enumerate(pred_files):
            if "found_edit_locs" in loc and len(loc["found_edit_locs"]) > i:
                file_to_edit_locs[pred_file] = loc["found_edit_locs"][i]
        return file_to_edit_locs

    # Construct top-n file context
    file_to_edit_locs = extract_file_to_edit_locs(loc, pred_files)

    topn_content, file_loc_intervals, _, _ = construct_topn_file_context(
        file_to_edit_locs,
        pred_files,
        file_contents,
        structure,
        context_window=args.context_window,
        loc_interval=args.loc_interval,
        fine_grain_loc_only=args.fine_grain_loc_only,
        add_space=args.add_space,
        sticky_scroll=args.sticky_scroll,
        no_line_number=True,
    )

    git_diff_patches = []
    prompt_file_contents = topn_content
    message_get_fix = apply_search_replace_plan_prompt.format(
        problem_statement=problem_statement,
        content=prompt_file_contents.rstrip(),
        whole_function_prompt=whole_function_prompt,
    ).strip()

    model_sample = make_model(
        model=args.model,
        logger=logger,
        max_tokens=8192,
        backend=args.backend,
        temperature=0.7,
        batch_size=args.batch_size,
    )
    logger.info(f"prompting with message:\n{message_get_fix}")


    if args.reasoning_mode:
        generating_trajs = model_sample.codegen(message_get_fix, num_samples=args.batch_size, reasoning_mode=args.reasoning_mode, port=args.port, ip=args.ip)
    else:
        generating_trajs = model_sample.codegen(message_get_fix, num_samples=args.batch_size, port=args.port, ip=args.ip)
    logger.info(f"Got response:\n{generating_trajs}")
    code_chunk = generating_trajs[0]["response"]

    if args.reasoning_mode:
        repair_folder = os.path.dirname(args.output_file)
        reasoning_output = os.path.join(repair_folder, f"reasoning_data_{num_generated_sample}.jsonl")
        reasoning_data = {
            "instance_id": instance_id,
            "file_info": file_loc_intervals,
            "prompt": message_get_fix,
            "reasoning_content": generating_trajs[0]["reasoning_content"],
            "response": generating_trajs[0]["response"],
        }

        with open(reasoning_output, "a") as f:
            f.write(json.dumps(reasoning_data) + "\n")

    file_pattern = re.compile(
        r"<<<\s*MODIFIED FILE:\s*(.*?)\s*>>>(.*?)<<<\s*END MODIFIED FILE\s*>>>",
        re.DOTALL | re.IGNORECASE
    )
    mod_pattern = re.compile(
        r"<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>[\s]*REPLACE",
        re.DOTALL
    )
    modify_info = []
    for file_match in file_pattern.finditer(code_chunk):
        file_path = file_match.group(1).strip()
        file_content = file_match.group(2)
        modifications = []
        for mod_match in mod_pattern.finditer(file_content):
            search_text = mod_match.group(1).rstrip()
            replace_text = mod_match.group(2).rstrip()
            modifications.append({
                "search": search_text,
                "replace": replace_text
            })
        found = False
        for item in modify_info:
            if item["file_path"] == file_path:
                item["modifications"].extend(modifications)
                found = True
                break
        if not found:
            modify_info.append({
                "file_path": file_path,
                "modifications": modifications
            })

    edited_files = []
    contents = []
    new_contents = []
    for modify_file_info in modify_info:
        vul_file = modify_file_info["file_path"]
        if vul_file not in file_contents:
            continue
        original_content = file_contents[vul_file]
        new_file_content = original_content
        for mod in modify_file_info.get("modifications", []):
            search_text = mod.get("search", "")
            replace_text = mod.get("replace", "")
            if search_text != "":
                new_file_content = apply_patch_with_indent_alignment(new_file_content, search_text, replace_text)

        edited_files.append(vul_file)
        contents.append(original_content)
        new_contents.append(new_file_content)

    merged_diff = fake_git_repo("playground", edited_files, contents, new_contents)

    patch_candidates = []
    git_diffs = []
    raw_git_diffs = []
    count = num_generated_sample
    # post process, generate patch and metadata
    for patch_candidate in patch_candidates:
        print(f"trying the {count + 1}-th sample ...")
        count += 1
        did_relocalization = False
        if args.best_patch_file and os.path.exists(args.best_patch_file):
            best_patches = load_jsonl(args.best_patch_file)
            for entry in best_patches:
                if entry["instance_id"] == instance_id:
                    base_patch_diff = entry["model_patch"]
                    did_relocalization = entry.get("reloca", False)
                    break
        print(f"The final patch for the {count}-th sample:")

    git_diffs.append(merged_diff)

    # save generated patches to file
    # use lock to prevent multiple threads from writing to the same file at the same time
    with output_file_lock:
        if os.path.exists(args.output_file):
            prev_generations = load_jsonl(args.output_file)
            found = False
            for entry in prev_generations:
                if entry["instance_id"] == instance_id:
                    found = True
                    entry["git_diffs"].extend(git_diffs)
                    entry["raw_git_diffs"].extend(raw_git_diffs)
                    break
            if found:
                with open(args.output_file, "w") as f:
                    for entry in prev_generations:
                        f.write(json.dumps(entry) + "\n")
                    f.flush()
            else:  # previous generations do not contain the current instance, add it
                with open(args.output_file, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "instance_id": instance_id,
                                "git_diffs": git_diffs,
                                "raw_git_diffs": raw_git_diffs,
                            }
                        )
                        + "\n"
                    )
                    f.flush()
        else:
            # write the first instance
            with open(args.output_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "instance_id": instance_id,
                            "git_diffs": git_diffs,
                            "raw_git_diffs": raw_git_diffs,
                        }
                    )
                    + "\n"
                )
                f.flush()


def repair(args):
    if args.benchmark == "lite":
        swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    elif args.benchmark == "verified":
        swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    elif args.benchmark == "full":
        swe_bench_data = load_dataset("princeton-nlp/SWE-bench", split="test")
    else:
        raise ValueError(f"benchmark {args.benchmark} not supported")
    task_ids_to_repair = args.task_ids_to_repair
    assert len(task_ids_to_repair) > 0, "No task ids to run."
    print(f"all task ids: {task_ids_to_repair}")

    locs = [loc for loc in locs_global if loc.get('instance_id') in task_ids_to_repair]
    # keys: "instance_id", "vul_files"
    # each vul_file has keys: 'file_path', 'modifications'
    # each modification has keys: "source_type", "extracted_source", "start_line", "end_line"
    # "source_type": for example, function
    # "extracted_source": content


    if len(locs) == 0:
        print("No task ids to run.")
        exit(0)
    
    with open(f"{args.output_folder}/used_locs.jsonl", "w") as f:
        for loc in locs:
            f.write(json.dumps(loc) + "\n")
            
    prev_generations = []
    if os.path.exists(args.raw_output_file):
        prev_generations = load_jsonl(args.raw_output_file)
        
    
    if args.num_threads == 1:
        for loc in tqdm(locs, total=len(locs)):
            process_loc(loc, args, swe_bench_data, prev_generations)
    else:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.num_threads
        ) as executor:
            futures = {
                executor.submit(process_loc, loc, args, swe_bench_data, prev_generations): loc
                for loc in locs
            }
            for future in tqdm(
                    concurrent.futures.as_completed(futures), total=len(locs)
            ):
                result = future.result()


def post_process_repair(args):
    """
    apply some diff formatting.
    """
    raw_outputs = load_jsonl(args.raw_output_file)

    for raw_output in raw_outputs:
        git_diff = ""
        instance_id = raw_output["instance_id"]
        if instance_id not in args.task_ids_to_repair:
            continue
        skip=False
        if os.path.exists(args.output_file):
            with open(args.output_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    if data.get("instance_id") == instance_id:
                        # If a match is found, skip further processing
                        skip=True
                        break
        if skip:
            continue

        if args.select_id == -1:
            # Use the last generation
            assert False, "not implemented for now"
        else:
            # Use the indexed generation
            generation_idx = args.select_id
            if generation_idx >= len(raw_output["git_diffs"]):
                continue
            git_diff = raw_output["git_diffs"][generation_idx]
        
        print(f"The patch for the {generation_idx}-th patch in post process:")
        print(f'model_patch: {git_diff.lstrip()}')
        
        with open(args.output_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "model_name_or_path": "PatchingPilot",
                        "instance_id": instance_id,
                        "model_patch": git_diff.lstrip(),
                    }
                )
                + "\n"
            )


def get_line_change_num(patch):
    lines = patch.split("\n")
    line_change_num = 0
    for line in lines:
        if line.startswith("+") or line.startswith("-"):
            line_change_num += 1
    return line_change_num


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc_file", type=str)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--loc_interval", action="store_true")
    parser.add_argument("--context_window", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=20, help="Sampling budget.")
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument(
        "--select_id",
        type=int,
        default=-1,
        help="Index the selected samples during post-processing.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="UCSB-SURFI/Co-PatcheR-Loc-Gen-14B",
    )
    parser.add_argument(
        "--backend", type=str, default="opensource", choices=["openai", "deepseek", "claude", "opensource"]
    )
    parser.add_argument("--add_space", action="store_true")
    parser.add_argument("--fine_grain_loc_only", action="store_true")
    parser.add_argument("--issue_summarize", action="store_true")
    parser.add_argument("--sticky_scroll", action="store_true")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--reasoning_mode", action="store_true", default=False)
    parser.add_argument(
        "--task_list_file",
        type=str,
        help="Path to the file that contains all tasks ids to be run.",
    )
    parser.add_argument("--target_id", type=str)

    # args for sampleing/refinement
    parser.add_argument("--sample_mod", action="store_true")
    parser.add_argument("--benchmark", type=str, default="lite", choices=["lite", "verified", "full"])
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for creating API requests",
    )
    parser.add_argument(
        "--save_repo_clone",
        action="store_true",
        help="Whether preserve the repo clone or delete it after use"
    )
    parser.add_argument(
        "--save_structure",
        action="store_true",
        help="Whether or not save the structure in a file"
    )
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2951)

    args = parser.parse_args()

    # if sample_mod and refine_mod are both true or both false, we will use sample_mod by default
    # args.sample_mod = True
    if args.batch_size > args.max_samples:
        args.batch_size = args.max_samples
    if args.batch_size < 1:
        args.batch_size = args.max_samples
    print("genearting samples:", args.max_samples)
    print("batch size:", args.batch_size)

    ensure_directory_exists(args.output_folder)
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    global locs_global
    locs_global = load_jsonl(args.loc_file)

    assert not (
                args.target_id is not None and args.task_list_file is not None), "Cannot specify both task and task-list."
    all_task_ids = []
    if args.task_list_file is not None:
        all_task_ids = parse_task_list_file(args.task_list_file)
    elif args.target_id is not None:
        all_task_ids = [args.target_id]
    else:
        all_task_ids = [loc["instance_id"] for loc in locs_global]
    assert len(all_task_ids) > 0, "No task ids to run."

    args.all_task_ids = all_task_ids
    args.task_ids_to_repair = all_task_ids

    args.patch_folder = args.output_folder
    args.num_samples = args.max_samples
    args.deduplicate = True
    args.plausible = True
    args.best_patch_file = None

    ensure_directory_exists(args.output_folder)
    ensure_directory_exists(os.path.join(args.output_folder, "repair_logs"))

    args.output_file = os.path.join(args.output_folder, "output.jsonl")
    args.raw_output_file = args.output_file

    return args


def main():
    args = args_parser()

    global round_idx
    global num_generated_sample
    global reloca_ids
    global reloca_locs
    global last_round

    while num_generated_sample < args.max_samples:
        if args.max_samples - num_generated_sample <= args.batch_size:
            args.batch_size = args.max_samples - num_generated_sample
            last_round = True
        print(f"already generated {num_generated_sample} examples")
        print(f"generating the {num_generated_sample + 1}th to {num_generated_sample + args.batch_size}th examples in round {round_idx}")

        repair(args)  # output.jsonl should have all generations in this round

        for i in range(args.batch_size):
            args.output_file = args.raw_output_file.replace(
                ".jsonl", f"_{num_generated_sample + i}_processed.jsonl"
            )
            args.select_id = num_generated_sample + i
            # do postprocess and save the processed output file to f"_{num_generated_sample+i}_processed.jsonl"
            post_process_repair(args)

        # for each output file in the round, do verification
            args.patch_file = args.output_file

        # update round_idx and num_generated_sample
        args.output_file = args.raw_output_file  # reset the output file

        num_generated_sample += args.batch_size

        if args.task_ids_to_repair == []:
            break
        round_idx += 1


if __name__ == "__main__":
    main()
