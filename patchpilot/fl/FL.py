from abc import ABC, abstractmethod
from collections import Counter
import json
import re
from patchpilot.repair.utils import construct_topn_file_context
from patchpilot.util.compress_file import get_skeleton
from patchpilot.util.get_function_interval import get_function_interval
from patchpilot.util.postprocess_data import extract_code_blocks, extract_locs_for_files
from patchpilot.util.search_tool import search_string, search_class_def, search_func_def, search_string_schema, search_class_def_schema, search_func_def_schema
from patchpilot.util.preprocess_data import (
    correct_file_paths,
    get_full_file_paths_and_classes_and_functions,
    get_repo_files,
    show_project_structure,
)

# Maximum context length for LLM processing
MAX_CONTEXT_LENGTH = 100000  # 128000


class FL(ABC):
    def __init__(self, instance_id, structure, problem_statement):
        self.structure = structure
        self.instance_id = instance_id
        self.problem_statement = problem_statement

    @abstractmethod
    def localize(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
        pass


class LLMFL(FL):
    search_str_with_file_prompt_template = """
### search results ###
Here are some files that contain the specific string, class, or function you searched for:
{search_results}
You may or may not need to edit these files to fix the problem. 
###

    """

    let_llm_search_prompt = """
Please look through the following GitHub issue description, decide whether there is a need to search for a specific string, class, or function, and call the appropriate function to search for the relevant information.
You can search for a specific string, class, or function to identify the file that need to be edited to fix the problem. 
If you search a class or function, please use only the name of the class or function as argument.
If you search for a string, please use the specific string as argument.
You should pay attention to error message or specific strings in the output or code snippets mentioned in the issue description. 
When you search for a string in the poc output, you should try to search for the descrptive sentence telling you what is happening, but remember to exclude variable names or details unique to individual instances. You should delete each part that may be dynamically generated, only keeping the general description of the error or issue.
You need to make the string as short as possible while still being specific enough to identify the relevant code, don't include any details, just search for the general error message or issue description.
For example, if the error message is "TypeError: 'int' object is not iterable", you should search for "object is not iterable". If the error message is "ValueError: invalid literal for int() with base 10: 'a'", you should search for "invalid literal".
You don't have to search for all three types of information, you can search any one or more of them as needed. 
Only search the string if you think it is unique enough to identify the relevant code. If you think the string is too generic and will return too many results, you should skip searching for the string.

### GitHub Issue Description ###
{problem_statement}
###

{poc_info}

Please perform all necessary tool calls in one step, as this is your only opportunity to execute searches. 
You may call multiple tools (e.g., search_func_def, search_class_def, search_string) as needed in this single operation.
"""

    obtain_relevant_files_prompt = """
Please look through the following GitHub problem description and Repository structure and provide a list of files that one would need to edit to fix the problem.

### GitHub Problem Description ###
{problem_statement}

###

### Repository Structure ###
{structure}

###

{search_str_with_file_prompt}

After analyzing the problem, provide the full path and return at most 5 files. 
The returned files should be separated by new lines ordered by most to least important and wrapped with ```
For example:
```
file1.py
file2.py
```
"""

    obtain_coverage_file_prompt = """
Please look through the following GitHub problem description and a list of coverage files would need to edit to fix the problem.

### GitHub Problem Description ###
{problem_statement}

###

You should choose files from the following files:
### Coverage files ###
{coverage_files}

###

{search_str_with_file_prompt}

After analyzing the problem, provide the full path and return at most 5 files. 
The returned files should be separated by new lines ordered by most to least important and wrapped with ```
For example:
```
file1.py
file2.py
```

"""

    obtain_relevant_code_prompt = """
Please look through the following GitHub problem description and file and provide a set of locations that one would need to edit to fix the problem.

### GitHub Problem Description ###
{problem_statement}

###

### File: {file_name} ###
{file_content}

###

After analyzing the problem, please provide either the class, the function name or line numbers that need to be edited. If you need to edit multiple places, please provide all of them. But if you want to provide the line number, please give me a number in the middle every time.
If you need to edit multiple classes or functions, please provide all the class or function names. 
You should always include a class or function, do not provide just the line numbers without the class or function name.
Here is the format you need to strictly follow, only the name of class and function, no body content or other suggestions, don't for get the "```":
### Example 1:
```
class: MyClass
```
### Example 2:
```
function: my_function
```
### Example 3:
```
function: my_function
line: 66
line: 99
```
### Example 4:
```
class: MyClass
line: 44
line: 55
```

"""
    #Return just the location(s)

    file_content_template = """
### File: {file_name} ###
{file_content}
"""

    file_content_in_block_template = """
### File: {file_name} ###
```python
{file_content}
```
"""

    obtain_relevant_code_combine_top_n_prompt = """
Please review the following GitHub problem description and relevant files, and provide a set of locations that need to be edited to fix the issue.
The locations should exact line numbers that require modification.
Pay attention! You should identify the method responsible for the core functionality of the issue. Focus on areas that define or enforce foundational behavior rather than case-specific in the issue.

### GitHub Problem Description ###
{problem_statement}

###
{file_contents}

###

{last_search_results}

After analyzing the problem, please provide the class name, function or method name, or the exact line numbers that need to be edited.
If you want to provide the line number, please give me a number in the middle every time.
If you need to edit multiple classes or functions, please provide all the function names or the line numbers in the class. 
You should always include a class or function, do not provide just the line numbers without the class or function name.
If you want to include a class rather than a function, you should always provide the line numbers for the class.
Here is the format you need to strictly follow, don't return any code content or other suggestions, don't for get the "```":
### Examples:
```
full_path1/file1.py
class: MyClass1
line: 51

full_path2/file2.py
function: MyClass2.my_method
line: 12

full_path3/file3.py
function: my_function
line: 24
line: 156
```

"""
    #Return just the location(s)

    obtain_relevant_code_graph_prompt = """
Please review the following GitHub problem description and relevant files, and provide a set of locations that need to be edited to fix the issue.
You will also be given a list of function/class dependencies to help you understand how functions/classes in relevant files fit into the rest of the codebase.
The locations can be specified as class names, function or method names, or exact line numbers that require modification.

### GitHub Problem Description ###
{problem_statement}

### Related Files ###
{file_contents}

### Function/Class Dependencies ###
{code_graph}

###

{last_search_results}

After analyzing the problem, please provide the class name, function or method name, or the exact line numbers that need to be edited.
Here is the format you need to strictly follow, don't return any code content or other suggestions, don't for get the "```":
### Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
line: 51

full_path2/file2.py
function: MyClass2.my_method
line: 12

full_path3/file3.py
function: my_function
line: 24
line: 156
```

"""
    # Return just the location(s)

    obtain_relevant_code_combine_top_n_no_line_number_prompt = """
Please review the following GitHub problem description and relevant files, and provide a set of locations that need to be edited to fix the issue.
The locations can be specified as method, or function names that require modification.

### GitHub Problem Description ###
{problem_statement}

###
{file_contents}

###

{last_search_results}

After analyzing the problem, please provide the method, or function names that need to be edited.
Here is the format you need to strictly follow, only the name of class and function, no body content or other suggestions, don't for get the "```":
### Examples:
```
full_path1/file1.py
function: my_function1

full_path2/file2.py
function: MyClass2.my_method

full_path3/file3.py
function: my_function2
```

"""
    # Return just the location(s)

    obtain_relevant_functions_from_compressed_files_prompt = """
Please look through the following GitHub problem description and the skeleton of relevant files.
Provide a thorough set of locations that need inspection or editing to fix the problem, including directly related areas as well as any potentially related functions and classes.

### GitHub Problem Description ###
{problem_statement}

###
{file_contents}

###

After analyzing the problem, please provide locations as either the class or the function name.
Here is the format you need to strictly follow, only the name of class and function, no body content or other suggestions, don't for get the "```":
### Examples:
```
full_path1/file1.py
class: MyClass1

full_path2/file2.py
function: MyClass2.my_method

full_path3/file3.py
function: my_function
```

"""
    # Return just the location(s)

    obtain_relevant_functions_and_vars_from_compressed_files_prompt_more = """
Please look through the following GitHub Problem Description and the Skeleton of Relevant Files.
Identify all locations that need inspection or editing to fix the problem, including directly related areas as well as any potentially related global variables, functions, and classes.
For each location you provide, either give the name of the class, the name of a method in a class, the name of a function, or the name of a global variable.
You should explicitly analyse whether a new function needs to be added, output whether a new function should be added and why. If a new function needs to be added, you should provide the class where the new function should be introduced as one of the locations, listing only the class name in this case. All other locations should be returned as usual, so do not return only this one class.

### GitHub Problem Description ###
{problem_statement}

### Skeleton of Relevant Files ###
{file_contents}

After analyzing the problem, please provide the complete set of locations as either a class name, a function name, or a variable name.
Note that if you include a class, you do not need to list its specific methods.
You can include either the entire class or don't include the class name and instead include specific methods in the class.
Here is the format you need to strictly follow, only the name of class and function, no body content or other suggestions, don't forget the "```":
### Examples:
```
full_path1/file1.py
function: my_function_1
class: MyClass1
function: MyClass2.my_method

full_path2/file2.py
variable: my_var
function: MyClass3.my_method

full_path3/file3.py
function: my_function_2
function: my_function_3
function: MyClass4.my_method_1
class: MyClass5
```

"""
    # Return just the locations.

    def __init__(
            self,
            instance_id,
            structure,
            problem_statement,
            model_name,
            backend,
            logger,
            match_partial_paths,
            temperature,
            port=2951,
    ):
        super().__init__(instance_id, structure, problem_statement)
        self.max_tokens = 4240
        self.model_name = model_name
        self.backend = backend
        self.logger = logger
        self.match_partial_paths = match_partial_paths
        self.temperature = temperature
        self.port = port

    def _parse_model_return_lines(self, content: str) -> list[str]:
        if content:
            return content.strip().split("\n")

    def search_in_problem_statement(self, reproduce_info) -> dict[str, str]:
        from patchpilot.util.model import make_model
        """
        Ask the LLM what it wants to search for and then perform the search
        """
        search_str_with_file = dict()
        message = self.let_llm_search_prompt.format(problem_statement=self.problem_statement, poc_info=reproduce_info)
        self.logger.info(f"prompting with message for search:\n{message}")
        # print(f"prompting with message for search:\n{message}")
        self.logger.info("=" * 80)
        # print("=" * 80)
        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0.3,
            batch_size=1,
        )
        if self.backend == "openai":
            traj = model.codegen(message, num_samples=1, tools=[search_string_schema, search_class_def_schema, search_func_def_schema], port=self.port)[0]
        elif self.backend == "claude":
            traj = model.codegen_litellm(message, num_samples=1, tools=[search_string_schema, search_class_def_schema, search_func_def_schema])[0]
        elif self.backend == "deepseek":
            # directly use openai to finish tool_call
            tool_model = make_model(
                model="o3-mini-2025-01-14",
                backend="openai",
                logger=self.logger,
                max_tokens=self.max_tokens,
                temperature=0.3,
                batch_size=1,
            )
            traj = tool_model.codegen(message, num_samples=1,
                                      tools=[search_string_schema, search_class_def_schema, search_func_def_schema], port=self.port)[0]
        else:
            raise ValueError(f"Backend {self.backend} is not supported")
        if traj:
            self.logger.info(f"Response for search:\n{str(traj)}")
            if "tool_call" in traj and traj["tool_call"]:
                for tool_call in traj["tool_call"]:
                    if tool_call.function.name == "search_string":
                        try:
                            arguments = tool_call.function.arguments
                            argument_dict = json.loads(arguments)
                        except Exception as e:
                            raise e
                        if argument_dict and isinstance(argument_dict, dict) and "query_string" in argument_dict:
                            query_string = argument_dict["query_string"]
                            pattern = r"(?i)\b\w*warning\w*\b|\b\w*error\w*\b"
                            cleaned_text = re.sub(pattern, "", query_string)
                            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                            search_results = " ".join(search_string(cleaned_text, structure=self.structure))
                            if search_results:
                                search_str_with_file[query_string] = search_results
                                self.logger.info(f'search result for string {query_string}: {search_results}')
                            else:
                                self.logger.info(f'search result for string {query_string}: not found')
                    elif tool_call.function.name == "search_class_def":
                        try:
                            arguments = tool_call.function.arguments
                            argument_dict = json.loads(arguments)
                        except Exception as e:
                            raise e
                        if argument_dict and isinstance(argument_dict, dict) and "class_name" in argument_dict:
                            class_name = argument_dict["class_name"]
                            search_results = " ".join(search_class_def(class_name, structure=self.structure))
                            if search_results:
                                search_str_with_file[class_name] = search_results
                                self.logger.info(f'search result for class {class_name}: {search_results}')
                            else:
                                self.logger.info(f'search result for class {class_name}: not found')
                    elif tool_call.function.name == "search_func_def":
                        try:
                            arguments = tool_call.function.arguments
                            argument_dict = json.loads(arguments)
                        except Exception as e:
                            raise e
                        if argument_dict and isinstance(argument_dict, dict) and "function_name" in argument_dict:
                            function_name = argument_dict["function_name"]
                            search_results = " ".join(search_func_def(function_name, structure=self.structure))
                            if search_results:
                                search_str_with_file[function_name] = search_results
                                self.logger.info(f'search result for func {function_name}: {search_results}')
                            else:
                                self.logger.info(f'search result for func {function_name}: not found')
        return search_str_with_file

    def localize(
            self, top_n=1, mock=False, match_partial_paths=False,  search_res_files=None, num_samples=1, coverage_info=None, additional_info=None, reasoning_mode=False
    ):
        # lazy import, not sure if this is actually better?
        from patchpilot.util.api_requests import num_tokens_from_messages
        from patchpilot.util.model import make_model

        found_files = []
        search_str_with_files = ''
        for search_str, file_path in search_res_files.items():
            search_str_with_files += f"{search_str} is in: {file_path}\n"
        if search_str_with_files:
            search_str_with_file_prompt = self.search_str_with_file_prompt_template.format(search_results=search_str_with_files)
            self.logger.info(f"prompting with message:\n{search_str_with_file_prompt}")
        else:
            search_str_with_file_prompt = ''

        if coverage_info and coverage_info.get("coverage_dict", None) and len(coverage_info["coverage_dict"]) > 2:
            coverage_dict = coverage_info["coverage_dict"]
            coverage_files = []
            for coverage_file in coverage_dict:
                coverage_files.append(coverage_file)
            message = self.obtain_coverage_file_prompt.format(
                problem_statement=self.problem_statement,
                coverage_files=coverage_files,
                search_str_with_file_prompt=search_str_with_file_prompt,
            ).strip()
        else:
            message = self.obtain_relevant_files_prompt.format(
                problem_statement=self.problem_statement,
                structure=show_project_structure(self.structure).strip(),
                search_str_with_file_prompt=search_str_with_file_prompt,
            ).strip()

        if coverage_info and coverage_info.get("commit_info", None):
            change_files = coverage_info["commit_info"].get('changed_files', {})
            bug_fixed = coverage_info["commit_info"].get('bug_fixed', False)
            if change_files and bug_fixed:
                change_files_prompt = ("\nPlease pay attention here: We have found the commit id that may be related to this issue, and found the change files at that time.\n"
                                       "These files may have caused this issue, so please pay more attention to these files:")
                change_files_prompt = change_files_prompt + str(change_files)
                message = message + change_files_prompt
        if additional_info:
            message = message + additional_info

        self.logger.info(f"prompting with message:\n{message}")
        # print(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)
        # print("=" * 80)
        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            batch_size=num_samples,
        )
        raw_trajs = model.codegen(message, num_samples=num_samples, reasoning_mode=reasoning_mode, port=self.port, ip=self.ip)
        # traj = model.codegen(message, num_samples=num_samples)[0]
        raw_outputs = [raw_traj["response"] for raw_traj in raw_trajs]
        traj = {
            "prompt": message,
            "response": raw_outputs,
        }
        model_found_files_raw = []
        for raw_output in raw_outputs:
            model_found_files_raw.extend(self._parse_model_return_lines(raw_output))
        element_count = Counter(model_found_files_raw)
        model_found_files = [item for item, count in element_count.most_common()]

        files, classes, functions = get_full_file_paths_and_classes_and_functions(
            self.structure
        )

        # sort based on order of appearance in model_found_files
        found_files = correct_file_paths(model_found_files, files, match_partial_paths)
        found_files = found_files[:top_n]

        self.logger.info(raw_outputs)
        # print(raw_outputs)

        return (
            found_files,
            {"raw_output_files": raw_output},
            traj,
            raw_trajs,
            message
        )

    def localize_function_from_compressed_files(self, file_names, mock=False, num_samples=1, coverage_info=None, additional_info=None, reasoning_mode=False, args=None):
        from patchpilot.util.api_requests import num_tokens_from_messages
        from patchpilot.util.model import make_model
        file_contents = get_repo_files(self.structure, file_names)
        coverage_dict = {}
        if coverage_info and "coverage_dict" in coverage_info:
            coverage_dict = coverage_info["coverage_dict"]
        file_to_delete_functions_start_lines = {}
        for file_name in file_names:
            file_to_delete_functions_start_lines[file_name] = []
            uncovered_lines = []
            if coverage_dict:
                if file_name not in coverage_dict:
                    continue
                else:
                    uncovered_lines = coverage_dict[file_name]
            func_to_interval = get_function_interval(file_contents[file_name])
            for func_name, interval in func_to_interval.items():
                covered = False
                for line in range(interval[0]+1, interval[1]+1): # +1 to exclude the function definition line
                    if line not in uncovered_lines:
                        covered = True
                        break
                if not covered:
                    file_to_delete_functions_start_lines[file_name].append(func_to_interval[func_name][0])
        compressed_file_contents = {
            fn: get_skeleton(code, True, file_to_delete_functions_start_lines.get(fn, None)) for fn, code in file_contents.items()
        }
        contents = [
            self.file_content_in_block_template.format(file_name=fn, file_content=code)
            for fn, code in compressed_file_contents.items()
        ]
        file_contents = "".join(contents)
        template = (
            self.obtain_relevant_functions_and_vars_from_compressed_files_prompt_more
        )
        message = template.format(
            problem_statement=self.problem_statement, file_contents=file_contents
        )

        def message_too_long(message):
            return (
                    num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH
            )

        while message_too_long(message) and len(contents) > 1:
            self.logger.info(f"reducing to \n{len(contents)} files")
            contents = contents[:-1]
            file_contents = "".join(contents)
            message = template.format(
                problem_statement=self.problem_statement, file_contents=file_contents
            )  # Recreate message

        if message_too_long(message):
            raise ValueError(
                "The remaining file content is too long to fit within the context length"
            )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(
                        message,
                        self.model_name,
                    ),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            batch_size=num_samples,
        )
        # traj = model.codegen(message, num_samples=num_samples)[0]
        if additional_info:
            message = message + additional_info
        raw_trajs = model.codegen(message, num_samples=num_samples, reasoning_mode=reasoning_mode, port=self.port, ip=self.ip)
        raw_outputs = [raw_traj["response"] for raw_traj in raw_trajs]
        answer_contents = [raw_traj.get("answer_content", raw_traj["response"]) for raw_traj in raw_trajs]
        traj = {
            "prompt": message,
            "response": raw_outputs,
        }
        model_found_locs_union = []
        for answer_content in answer_contents:
            model_found_locs = extract_code_blocks(answer_content)
            # model_found_locs_separated = extract_locs_for_files(
            #     model_found_locs, file_names
            # )
            model_found_locs_separated = extract_locs_for_files(
                model_found_locs, file_names
            )
            model_found_locs_union.append(model_found_locs_separated)
        model_found_locs_merged_result = [[''] for _ in range(len(file_names))]
        for list0 in model_found_locs_union:
            for index, list1 in enumerate(list0):
                if list1[0] not in model_found_locs_merged_result[index][0]:
                    model_found_locs_merged_result[index][0] += list1[0]
                    model_found_locs_merged_result[index][0] += '\n'

        self.logger.info(f"==== raw output ====")
        self.logger.info(raw_outputs)
        self.logger.info("=" * 80)
        self.logger.info(f"==== extracted locs ====")
        for loc in model_found_locs_merged_result:
            self.logger.info(loc)
        self.logger.info("=" * 80)

        # print(raw_outputs)

        return model_found_locs_merged_result, {"raw_output_loc": raw_outputs}, traj, raw_trajs, message


    def localize_line_from_files(
            self,
            file_names,
            num_samples: int = 1,
            args = None,
    ):
        from patchpilot.util.api_requests import num_tokens_from_messages
        from patchpilot.util.model import make_model

        def message_too_long(msg: str) -> bool:
            return (
                    num_tokens_from_messages(
                        [{"role": "user", "content": msg}], self.model_name
                    )
                    >= MAX_CONTEXT_LENGTH
            )

        # read repo files
        file_contents = get_repo_files(self.structure, file_names)

        # containers
        results = [[[] for _ in file_names] for _ in range(num_samples)]
        raw_outputs_per_sample: list[list[str]] = [[] for _ in range(num_samples)]
        traj = {"prompt": [], "response": []}

        # iterate over files
        all_blocks = ["" for _ in range(num_samples)]
        for file_idx, fn in enumerate(file_names):
            code = file_contents.get(fn, "")

            # number each line
            numbered = [
                f"{idx}: {line}"
                for idx, line in enumerate(code.splitlines(keepends=True), start=1)
            ]

            # chunk long files
            chunks, current = [], []
            for line in numbered:
                current.append(line)
                content_block = self.file_content_in_block_template.format(
                    file_name=fn, file_content="".join(current)
                )
                prompt = self.obtain_relevant_code_combine_top_n_prompt.format(
                    problem_statement=self.problem_statement,
                    file_contents=content_block,
                    last_search_results="",
                )
                if message_too_long(prompt):
                    last = current.pop()
                    if current:
                        chunks.append("".join(current))
                    current = [last]
            if current:
                chunks.append("".join(current))

            # query model per chunk
            model = make_model(
                model=self.model_name,
                backend=self.backend,
                logger=self.logger,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                batch_size=num_samples,
            )

            raw_trajs_list = []
            message_list = []
            for chunk in chunks:
                content_block = self.file_content_in_block_template.format(
                    file_name=fn, file_content=chunk
                )
                message = self.obtain_relevant_code_combine_top_n_prompt.format(
                    problem_statement=self.problem_statement,
                    file_contents=content_block,
                    last_search_results="",
                )

                raw_trajs = model.codegen(message, num_samples=num_samples, port=self.port, ip=self.ip)
                raw_trajs_list.append(raw_trajs)
                message_list.append(message)
                traj["prompt"].append(message)
                traj["response"].extend([rt["response"] for rt in raw_trajs])

                # parse outputs
                for samp_id, rt in enumerate(raw_trajs):
                    raw = rt["response"]
                    raw_outputs_per_sample[samp_id].append(raw)

                    blocks = extract_code_blocks(raw) or [raw]
                    for block in blocks:
                        all_blocks[samp_id] += block + "\n"
                    locs = extract_locs_for_files(blocks, [fn])[0][0]
                    results[samp_id][file_idx].append(locs)
        meta = {"raw_output_loc": raw_outputs_per_sample}
        return results, meta, traj, raw_trajs, message, raw_trajs_list, message_list



    def localize_line_from_coarse_function_locs(
            self,
            file_names,
            coarse_locs,
            context_window: int,
            add_space: bool,
            sticky_scroll: bool,
            no_line_number: bool,
            code_graph: bool,
            code_graph_context: str,
            num_samples: int = 1,
            mock=False,
            coverage_info=None,
            last_search_results: str = "",
            reasoning_mode=False,
    ):
        if coverage_info is None:
            coverage_info = {}
        from patchpilot.util.api_requests import num_tokens_from_messages
        from patchpilot.util.model import make_model

        file_contents = get_repo_files(self.structure, file_names)
        topn_content, file_loc_intervals, _, _ = construct_topn_file_context(
            coarse_locs,
            file_names,
            file_contents,
            self.structure,
            context_window=context_window,
            loc_interval=True,
            add_space=add_space,
            sticky_scroll=sticky_scroll,
            no_line_number=no_line_number,
        )
        if no_line_number:
            template = self.obtain_relevant_code_combine_top_n_no_line_number_prompt
            message = template.format(
                problem_statement=self.problem_statement, file_contents=topn_content, last_search_results=last_search_results
            )
        elif code_graph:
            template = self.obtain_relevant_code_graph_prompt
            message = template.format(
                problem_statement=self.problem_statement, file_contents=topn_content, code_graph=code_graph_context, last_search_results=last_search_results
            )
            if num_tokens_from_messages(message, "gpt-4o-2024-05-13") > 128000:
                template = self.obtain_relevant_code_combine_top_n_prompt
                message = template.format(
                    problem_statement=self.problem_statement, file_contents=topn_content, last_search_results=last_search_results
                )
        else:
            template = self.obtain_relevant_code_combine_top_n_prompt
            message = template.format(
                problem_statement=self.problem_statement, file_contents=topn_content, last_search_results=last_search_results
            )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)
        assert num_tokens_from_messages(message, self.model_name) < MAX_CONTEXT_LENGTH
        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            batch_size=num_samples,
        )
        raw_trajs = model.codegen(message, num_samples=num_samples, reasoning_mode=reasoning_mode, port=self.port, ip=self.ip)

        # Merge trajectories
        raw_outputs = [raw_traj["response"] for raw_traj in raw_trajs]
        answer_contents = [raw_traj.get("answer_content", raw_traj["response"]) for raw_traj in raw_trajs]
        traj = {
            "prompt": message,
            "response": raw_outputs,
            "usage": {  # merge token usage
                "completion_tokens": sum(
                    raw_traj["usage"]["completion_tokens"] for raw_traj in raw_trajs
                ),
                "prompt_tokens": sum(
                    raw_traj["usage"]["prompt_tokens"] for raw_traj in raw_trajs
                ),
            },
        }
        model_found_locs_separated_in_samples = []
        for answer_content in answer_contents:
            model_found_locs = extract_code_blocks(answer_content)
            model_found_locs_separated = extract_locs_for_files(
                model_found_locs, file_names
            )
            model_found_locs_separated_in_samples.append(model_found_locs_separated)

            self.logger.info(f"==== raw output ====")
            self.logger.info(answer_content)
            self.logger.info("=" * 80)
            # print(answer_content)
            # print("=" * 80)
            self.logger.info(f"==== extracted locs ====")
            for loc in model_found_locs_separated:
                self.logger.info(loc)
            self.logger.info("=" * 80)
        self.logger.info("==== Input coarse_locs")
        coarse_info = ""
        for fn, found_locs in coarse_locs.items():
            coarse_info += f"### {fn}\n"
            if isinstance(found_locs, str):
                coarse_info += found_locs + "\n"
            else:
                coarse_info += "\n".join(found_locs) + "\n"
        self.logger.info("\n" + coarse_info)
        if len(model_found_locs_separated_in_samples) == 1:
            model_found_locs_separated_in_samples = (
                model_found_locs_separated_in_samples[0]
            )

        return (
            model_found_locs_separated_in_samples,
            {"raw_output_loc": raw_outputs},
            traj,
            raw_trajs,
            message,
        )