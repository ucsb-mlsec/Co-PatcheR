import itertools
import numpy as np
from functools import partial
from patchpilot.util.model import make_model
from patchpilot.util.utils import load_jsonl, setup_logger
from patchpilot.repair.utils import post_process_raw_output, construct_topn_file_context, apply_search_replace
import re
import copy 


whole_function_prompt = """
Cause I will give you the whole function context. You need to first return the whole function code after fixing
The format should be:
```python
def testing():
    ...
    ...
    return x
```
and then you should provide me a one *SEARCH/REPLACE* edit to fix the issue.
"""


apply_plan_prompt = """
We are currently solving the following issue within our repository. 
{whole_function_prompt}
Please follow the provided step of a plan to generate one *SEARCH/REPLACE* edit to fix the issue, focusing only on the current step.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example of the format:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```
Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
Please make sure that the IDENTATION of the code in the *SEARCH/REPLACE* edit is correct.
Please note that the *SEARCH/REPLACE* edit does not contain + or - signs. It only contains the lines to search for and the lines to replace. It is not a diff.
Please note that the *SEARCH/REPLACE* edit does not contain + or - signs. It only contains the lines to search for and the lines to replace. It is not a diff.
Please note that the *SEARCH/REPLACE* edit does not contain + or - signs. It only contains the lines to search for and the lines to replace. It is not a diff.
Do not include leading + or - signs in the *SEARCH/REPLACE* edit. The *SEARCH/REPLACE* edit should only contain the lines to search for and the lines to replace.

If the current step does not require a *SEARCH/REPLACE* edit, please write 'No *SEARCH/REPLACE* edit required.' and do not provide any code. But try to provide a valid edit if possible.

## Now the issue is as follows:

Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs.
--- BEGIN FILES ---
{content}
--- END FILES ---

Here is the whole plan for fixing the issue:
{planning}

Here is the current step for fixing the issue:
{step}

{errors}
You should import the necessary libraries if they are required by the code generated in the current step. 
You should make sure to import every library that is used in the code generated in the current step.
If there are some code segments in the description of the current step, do not apply it directly. Instead, adapt it to align with the codebase's style and standards. Ensure that the patch considers interactions across different code sections, including nested structures, function calls, and data dependencies. The patch should maintain overall structural integrity, addressing the issue without unintended effects on other parts. Prefer solutions that are resilient to structural changes or future extensions.
You always need to adapt the code to the existing codebase's style and standards by considering the context of the code.
You need to pay attention to the variable types and function return types. If the project uses a specific type, you should use the same type in the *SEARCH/REPLACE* edit.
Before generating the *SEARCH/REPLACE* edit, you should consider the following:
1.Study Existing Patterns: Before implementing, review similar functions in the same file or class. Note any patterns in output formatting to maintain consistency.
2.Check Issue Requirements Against Implementation: Always revisit the issue requirements to verify that each element is implemented as described.
3.Test for Consistency: Ensure that added functions align structurally and stylistically with similar functions already present, especially for return value formatting.

Please generate the *SEARCH/REPLACE* edit based on the current step.
For the *SEARCH/REPLACE* edit, the content to search for should be at least 2 lines, but not too long to be irrelevant, and the content to replace should be concise and minimal.
If you need to call a function from a library, you should import the library. 
Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.

*SEARCH/REPLACE* edits are not diffs. They are the lines to search for and the lines to replace. They should not contain leading + or - signs. They should only contain the lines to search for and the lines to replace.
You should ensure that the code that you search for is exactly the same as the code in the file. You should not change the code that you search for. Pay attention to the number of spaces, the number of tabs, and the number of newlines.
"""

evaluate_step_prompt = """
We are currently solving the following issue within our repository. 
Given several candidate *SEARCH/REPLACE* edits, decide which choice is most promising.
If the current step does not require a *SEARCH/REPLACE* edit, you should always prefer the choice saying "No *SEARCH/REPLACE* edit required." In any case, conclude in the last line with "The best choice is s", where s is the integer ID of the most promising candidate.
You should evaluate each candidate based on the following criteria:
1. Does the candidate perform the required action of the step? 5 points
2. Does the candidate effectively achieve the goal of the step? 5 points
3. Does the candidate maintain the existing style of the code? 2 points
4. Is the candidate minimal and concise? 2 points
Give each candidate a score based on the above criteria, do not output the analysis process.

Conclude in the last line with "The best choice is s", where s is the integer ID of the most promising candidate.
Do not copy the code of any of the candidates.

Here is the issue description text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs.
--- BEGIN FILE ---
{content}
--- END FILE ---

Here is the whole plan for fixing the issue:
{planning}
Here is the current step for fixing the issue:
{step}

Please analyze the following candidate SEARCH/REPLACE edits:
{candidates}
"""


def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
    vote_results = [0] * n_candidates
    for vote_output in vote_outputs:
        pattern = r"best choice is.*?(\d+)"
        matches = re.findall(pattern, vote_output, re.DOTALL)

        if matches:
            for match in matches:
                vote = int(match) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
        else:
            print(f'vote no match: {[vote_output]}')
    return vote_results



def apply_plan_step_by_step(log_file, model_version, plan, problem_statement, content, backend, file_loc_intervals, file_contents, granularity_sample=None, instance_id=None, feedback_prompt=None, not_found_file_dict=None):
    logger = setup_logger(log_file)
    # Use regular expression to match each <STEP> and <Actions to be Taken> block
    matches = re.findall(r'(<STEP>.*?</Actions to be Taken>)', plan, re.DOTALL)

    # Strip whitespace and create a list of steps and actions
    plan_list = [match.strip() for match in matches]
    if len(plan_list) == 0:
        print('The current plan has no steps.')
        logger.info('The current plan has no steps.')
        print("============== the current plan has no steps ===============")
        print("Here is the plan:")
        print(plan)
        print("============== End of the plan which has no steps ===============")
        return ['No *SEARCH/REPLACE* edit.']
        
    print("================ the current plan has {} steps ==================".format(len(plan_list)))
    print(plan_list)
    logger.info(f"current plan has {len(plan_list)} steps")
    for step in plan_list:
        logger.info(f"{step}")
    search_replace = ''
    model = make_model(
        model=model_version,
        logger=logger,
        max_tokens=4096,
        backend=backend,
        temperature=0.8,
        batch_size=1,
    )
    original_file_contents = copy.deepcopy(file_contents)
    file_contents_copy = copy.deepcopy(file_contents)

    for current_step in plan_list:
        print("================ the current step is ==================")
        print(current_step)
        if instance_id:
            print(f"================ the current instance id is {instance_id} ==================")
        logger.info(f"current step: {current_step}")

        sample_response = ""
        try_num = 0
        all_errors = set()
        success = False
        edited_files = []
        new_contents = []
        
        # apply the search replace edit to content

        while not success and try_num < 5:
            try_num += 1
            error_prompt = ''
            all_error_str = ', '.join(all_errors)
            if all_error_str:
                error_prompt = f"In the previous generations, we encountered the following errors: {all_error_str}. Please try to avoid these errors in the current generation."

            if granularity_sample:
                whole_prompt = whole_function_prompt
            else:
                whole_prompt = ""

            message = apply_plan_prompt.format(
                whole_function_prompt=whole_prompt,
                problem_statement=problem_statement,
                content=content,
                planning=plan,
                errors=error_prompt,
                step=current_step
            )
            if feedback_prompt:
                message += 'Here are some feedbacks from the previous generation, they are just for your reference. Do not search for the feedbacks in the codebase. \n'
                message += feedback_prompt

            logger.info(f'prompting with apply_plan_prompt {message}')
            sample_traj = model.codegen(message, num_samples=1)[0]
            sample_response = sample_traj['response']

            if 'No *SEARCH/REPLACE* edit required' in sample_response:
                new_search_replace = search_replace
                print('======== search replace for current step ========')
                print(sample_response)
                check_success = True
                errors = set()
                differ_by_empty_lines = False
            else:
                new_search_replace = search_replace + '\n' + sample_response

                print('======== search replace for current step ========')
                print(sample_response)

                _, _, _, check_success, errors, edited_files, new_contents, differ_by_empty_lines = post_process_raw_output(sample_response, file_contents_copy, logger, file_loc_intervals, True, not_found_file_dict=not_found_file_dict, instance_id=instance_id)

            if check_success and not differ_by_empty_lines:
                success = True
                search_replace = new_search_replace
                # update the file contents
                if edited_files:
                    for i, edited_file in enumerate(edited_files):
                        if i < len(new_contents) and edited_file in file_contents_copy:
                            file_contents_copy[edited_file] = new_contents[i]
                # update the contents in the prompt
                content = apply_search_replace(sample_response, content)
                
                
            else:
                if differ_by_empty_lines:
                    errors.add('Did not found the original code in the file. The search/replace edit was wrong. The search/replace edit should not differ by empty lines from the original code. Pay attention to the empty lines in the code.')
                    errors.add('Pay attentio to the format of the search/replace edit.')
                    errors.add('You can try to search for fewer lines in the search block, like 1 or 2 lines.')
                all_errors.update(errors)
                print('======== retry generating sample ========')
                print(f"======== retry generating sample for step {current_step}, try number is {try_num} ========")
                logger.info('retry generating sample')

        if not success:
            search_replace += '\nThe current step has no valid edit candidates, the fix may be incomplete.'
            print("The cuurent step has no valid edit candidates, the fix may be incomplete.")
            print("Current step is:")
            print(current_step)
                
            logger.info("The cuurent step has no valid edit candidates, the fix may be incomplete.")
            logger.info(f"Current step is:\n{current_step}")

            
    
    print("checking the consistency of the step-by-step file edit with the whole plan file edit")
    _, _, _, _, _, edited_files, new_contents, _ = post_process_raw_output(search_replace, original_file_contents, logger, file_loc_intervals, True)
    if edited_files:
        for i, edited_file in enumerate(edited_files):
            if i < len(new_contents) and edited_file in original_file_contents:
                original_file_contents[edited_file] = new_contents[i]
    
    if not original_file_contents == file_contents_copy:
        print("The step-by-step file edit is not consistent with the edit generated by the whole plan.")
        print("The step-by-step file edit is:")
        print(original_file_contents)
        print("The whole plan file edit is:")
        print(file_contents_copy)
        print("The search replace is:")
        print(search_replace)
        logger.info("The step-by-step file edit is not consistent with the edit generated by the whole plan.")
        logger.info("The step-by-step file edit is:")
        logger.info(original_file_contents)
        logger.info("The whole plan file edit is:")
        logger.info(file_contents_copy)
        logger.info("The search replace is:")
        logger.info(search_replace)
          
    
    return [search_replace]
