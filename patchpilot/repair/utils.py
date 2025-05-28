from patchpilot.util.postprocess_data import (
    check_code_differ_by_just_empty_lines,
    check_syntax,
    extract_python_blocks,
    fake_git_repo,
    get_diff_real_git_repo,
    lint_code,
    parse_diff_edit_commands,
    parse_edit_commands,
    remove_empty_lines,
    split_edit_multifile_commands
)
from patchpilot.util.preprocess_data import (
    line_wrap_content,
    transfer_arb_locs_to_locs,
    get_extended_context_intervals
)
import json
import traceback
from difflib import unified_diff


infer_intended_behavior_prompt = """
You are an assistant for analyzing the description of an issue. You need to infer the expected behavior of the provided code snippet based on the issue description and the code.
Here is the issue text:
--- BEGIN ISSUE ---
{issue_description}
--- END ISSUE ---

Here is the code snippet you need to analyze and infer the expected behavior:
--- BEGIN CODE ---
{code}
--- END CODE ---

{extended_code}

You should output the expected behavior of the code snippet wrapped in --- BEGIN CODE --- and --- END CODE --- in a concise manner. Here is an example of the output format:
--- BEGIN EXPECTED BEHAVIOR ---
The function foo is expected to add 1 to the argument passed in and return the result. 
The function bar is expected to return the sum of the two arguments passed in.
--- END EXPECTED BEHAVIOR  ---

"""

def construct_topn_file_context(
        file_to_locs,
        pred_files,
        file_contents,
        structure,
        context_window: int,
        loc_interval: bool = True,
        fine_grain_loc_only: bool = False,
        add_space: bool = False,
        sticky_scroll: bool = False,
        no_line_number: bool = True,
):
    """Concatenate provided locations to form a context.

    loc: {"file_name_1": ["loc_str_1"], ...}
    """
    file_loc_intervals = dict()
    file_import_intervals = dict()
    file_used_globals = dict()
    topn_content = ""

    for pred_file, locs in file_to_locs.items():
        content = file_contents[pred_file]
        line_locs, context_intervals, import_intervals, used_globals = transfer_arb_locs_to_locs(
            locs,
            structure,
            pred_file,
            context_window,
            loc_interval,
            fine_grain_loc_only,
            file_content=file_contents[pred_file] if pred_file in file_contents else "",
        )

        if len(line_locs) > 0:
        # Note that if no location is predicted, we exclude this file.
            file_loc_content = line_wrap_content(
                content,
                context_intervals,
                add_space=add_space,
                no_line_number=no_line_number,
                sticky_scroll=sticky_scroll,
            )
            if used_globals:
                global_vars_str = "\n".join(used_globals)
                topn_content += ">>> FILE " + pred_file + "\nHere are some global variables that may or may not be relevant; they are provided as context, do not modify them." + global_vars_str + "\n>>> FILE END\n"

            for sub_loc_conent in file_loc_content:
                topn_content += ">>> FILE " + pred_file + "\n" + sub_loc_conent + "\n" + ">>> FILE END" + "\n\n"

            file_loc_intervals[pred_file] = context_intervals
            file_import_intervals[pred_file] = import_intervals
            file_used_globals[pred_file] = used_globals

    return topn_content, file_loc_intervals, file_import_intervals, file_used_globals


def get_content_from_one_interval(
        file_contents,
        pred_file,
        interval,
        add_space: bool = False,
        sticky_scroll: bool = False,
        no_line_number: bool = True,
):
    content = file_contents[pred_file]
    file_loc_content = line_wrap_content(
        content,
        [interval],
        add_space=add_space,
        no_line_number=no_line_number,
        sticky_scroll=sticky_scroll,
    )
    return file_loc_content


# This function is used for refine mode where the file contents are already modified by previous patches.
# We need to use the original github repo to get the actual git diff relative to the original content.
def post_process_raw_output_refine(
    raw_output_text, file_contents, logger, file_loc_intervals, repo, base_commit, base_patch_diff):
    """
    Post-process the raw output text from the repair tool.

    Arguments:
    raw_output_text: The raw output text from the repair tool. A string.
    file_contents: A dictionary with the modified file by the patch as key and file content as values. It's the file content before applying the patch.
    logger: A logger object.
    file_loc_intervals: A dictionary with file paths as keys and lists of line intervals as values.
    repo: A string with the name of the github repo.
    base_commit: A string with the commit hash of the base commit.
    base_patch_diff: A string with the diff of the base commit.

    Returns:
    git_diffs: A string with the git diff of the proposed changes.
    raw_git_diffs: A string with the raw git diff of the proposed changes.
    content: A string with the content of the edited file.
    check_success: A boolean indicating whether the linting and syntax checking were successful.
    errors: A set of error messages.
    """
    git_diffs = ""
    raw_git_diffs = ""
    lint_success = False
    check_success = False
    errors = set()
    prev_errors = set()
    edited_files, new_contents, contents = [], [], []
    differ_by_empty_lines = False
    try:
        file_to_contents = {}
        for file in file_loc_intervals:
            file_to_contents[file] = file_contents[file]
        edited_files, new_contents = _post_process_multifile_repair(
            raw_output_text,
            file_contents,
            logger,
            file_loc_intervals,
            diff_format=True,
        )
        contents = [file_contents[edited_file] for edited_file in edited_files]
        assert len(edited_files) == len(new_contents)
        assert len(edited_files) == len(contents)
        for i, edited_file in enumerate(edited_files):
            file_to_contents[edited_file] = new_contents[i]
                     
        # file_to_contents only records modification of the edited file by the current patch, we need to apply base_patch_diff to maintain the changes in other files.
        git_diff = get_diff_real_git_repo("playground", file_to_contents, repo, base_commit, base_patch_diff)

        raw_git_diffs += "\n" + git_diff.replace("\\ No newline at end of file\n", "")

        syntax_success = True
        syntax_errors = set()
        for new_content in new_contents:
            syntax_success_i, syntax_error = check_syntax(new_content)
            syntax_success = syntax_success and syntax_success_i
            if syntax_error:
                syntax_errors.add(syntax_error)
        if not syntax_success:
            print("Syntax checking failed.")
            errors = syntax_errors
            return git_diffs, raw_git_diffs, contents, check_success, errors, edited_files, new_contents, differ_by_empty_lines
        
        lint_success = True
        for i in range(len(contents)):
            if i < len(new_contents):
                lint_success_i, prev_errors_i, errors_i = lint_code(
                    "playground", "test.py", new_contents[i], contents[i]
                )
                lint_success = lint_success and lint_success_i
                prev_errors.update(prev_errors_i)
                errors.update(errors_i)

        differ_by_empty_lines = check_code_differ_by_just_empty_lines(
            new_contents, contents
        )
        print(git_diff)
        print(lint_success, prev_errors, errors, differ_by_empty_lines)
        
        logger.info(f"git diff: {git_diff}")
        logger.info(f"{lint_success}, {prev_errors}, {errors}, {differ_by_empty_lines}")

        logger.info(f"{differ_by_empty_lines = }")
        if syntax_success and not differ_by_empty_lines:
            git_diffs = raw_git_diffs
        else:
            git_diffs = ""  # no need to evaluate
    except Exception as e:
        print(raw_output_text)
        print(e)
    
    if lint_success and syntax_success:
        check_success = True
        
    errors.difference_update(prev_errors)

    return git_diffs, raw_git_diffs, contents, check_success, errors, edited_files, new_contents, differ_by_empty_lines


def post_process_raw_output(
    raw_output_text, file_contents, logger, file_loc_intervals, if_diff_format=False, not_found_file_dict=None, instance_id=None
):
    """
    Post-process the raw output text from the repair tool.

    Arguments:
    raw_output_text: The raw output text from the repair tool. A string.
    file_contents: A dictionary with file paths as keys and file contents as values.
    logger: A logger object.
    file_loc_intervals: A dictionary with file paths as keys and lists of line intervals as values.
    args: A Namespace object with the following attributes:
        diff_format: A boolean indicating whether the repair tool uses diff format.

    Returns:
    git_diffs: A string with the git diff of the proposed changes.
    raw_git_diffs: A string with the raw git diff of the proposed changes.
    content: A string with the content of the edited file.
    check_success: A boolean indicating whether the linting and syntax checking were successful.
    """
    git_diffs = ""
    raw_git_diffs = ""
    lint_success = False
    check_success = False
    errors = set()
    prev_errors = set()
    differ_by_empty_lines = False
    edited_files, new_contents, contents = [], [], []
    try:
        edited_files, new_contents = _post_process_multifile_repair(
            raw_output_text,
            file_contents,
            logger,
            file_loc_intervals,
            diff_format=if_diff_format,
        )
        contents = [file_contents[edited_file] for edited_file in edited_files]
        if contents:
            git_diff = fake_git_repo("playground", edited_files, contents, new_contents)
            
            raw_git_diffs += "\n" + git_diff.replace("\\ No newline at end of file\n", "")

            syntax_success = True
            syntax_errors = set()
            for new_content in new_contents:
                syntax_success_i, syntax_error = check_syntax(new_content)
                syntax_success = syntax_success and syntax_success_i
                if syntax_error:
                    syntax_errors.add(syntax_error)
            if not syntax_success:
                print("Syntax checking failed.")
                errors = syntax_errors
                return git_diffs, raw_git_diffs, contents, check_success, errors, edited_files, new_contents, differ_by_empty_lines
            
            lint_success = True
            for i in range(len(contents)):
                if i < len(new_contents):
                    lint_success_i, prev_errors_i, errors_i = lint_code(
                        "playground", "test.py", new_contents[i], contents[i]
                    )
                    lint_success = lint_success and lint_success_i
                    prev_errors.update(prev_errors_i)
                    errors.update(errors_i)

            differ_by_empty_lines = check_code_differ_by_just_empty_lines(
                new_contents, contents
            )
            print(git_diff)
            print(lint_success, prev_errors, errors, differ_by_empty_lines)
            
            logger.info(f"git diff: {git_diff}")
            logger.info(f"{lint_success}, {prev_errors}, {errors}, {differ_by_empty_lines}")

            logger.info(f"{differ_by_empty_lines = }")
            if syntax_success and not differ_by_empty_lines:
                git_diffs = raw_git_diffs
            else:
                git_diffs = ""  # no need to evaluate
        else:
            print("Failed to extract the edited file.")
            errors.add("Failed to extract the edited file.")
            print(f'raw_output_text: {raw_output_text}')
            if isinstance(not_found_file_dict, dict) and instance_id:
                if instance_id in not_found_file_dict:
                    not_found_file_dict[instance_id] += "\n" + "\n".join([edited_file for edited_file in edited_files])
                else:
                    not_found_file_dict[instance_id] = "\n" + "\n".join([edited_file for edited_file in edited_files])
    except Exception as e:
        print(raw_output_text)
        print(e)
    
    if lint_success and syntax_success:
        check_success = True
        
    errors.difference_update(prev_errors)

    return git_diffs, raw_git_diffs, contents, check_success, errors, edited_files, new_contents, differ_by_empty_lines


def _post_process_multifile_repair(
    raw_output: str,
    file_contents: dict[str, str],
    logger,
    file_loc_intervals: dict[str, list],
    diff_format=False,
)-> tuple[list[str], list[str]]:
    edit_multifile_commands = extract_python_blocks(raw_output)
    edited_files = []
    new_contents = []
    try:
        file_to_commands = split_edit_multifile_commands(edit_multifile_commands, diff_format=diff_format)
    except Exception as e:
        logger.error(e)
        return edited_files, new_contents
    logger.info("=== file_to_commands: ===")
    logger.info(json.dumps(file_to_commands, indent=2))

    for edited_file_key in file_to_commands:
        edited_file = ""
        new_content = ""
        try:
            logger.info(f"=== edited_file: {edited_file_key} ===")
            edit_commands = file_to_commands[edited_file_key]
            logger.info("=== edit_commands: ===")
            for c in edit_commands:
                logger.info(c)
                logger.info("\n" + "-" * 40)
            edited_file = eval(edited_file_key)  # convert '"file.py"' to 'file.py'
            content = file_contents[edited_file]
            if diff_format:
                new_content, replaced = parse_diff_edit_commands(
                    edit_commands, content, file_loc_intervals[edited_file]
                )
            else:
                new_content = parse_edit_commands(edit_commands, content)
        except Exception as e:
            logger.error(e)
            edited_file = ""
            new_content = ""

        if edited_file == "" or new_content == "":
            continue
        edited_files.append(edited_file)
        new_contents.append(new_content)
        diff = list(
            unified_diff(
                content.split("\n"),
                new_content.split("\n"),
                fromfile=edited_file,
                tofile=edited_file,
                lineterm="",
            )
        )

        logger.info(f"extracted patch:")
        logger.info("\n".join(diff))
        # print("\n".join(diff))

    return edited_files, new_contents


def apply_search_replace(
    raw_output: str,
    content: str,
)-> str:
    edit_multifile_commands = extract_python_blocks(raw_output)
    try:
        file_to_commands = split_edit_multifile_commands(edit_multifile_commands, diff_format=True)
    except Exception as e:
        return content
    all_edit_commands = []
    for edited_file_key in file_to_commands:
        all_edit_commands += file_to_commands[edited_file_key]
        
    content_interval = [(0, len(content.splitlines()))]
    new_content, replaced = parse_diff_edit_commands(
        all_edit_commands, content, content_interval
    )
    return new_content


def fix_patch_indentation(patch_text: str) -> str:
    """
    Fix patch indentation by comparing the indentation of the first '-' line
    and the first '+' line in each hunk, then adding the indentation difference
    to every '+' line in that hunk on top of their existing indentation.

    For each hunk (indicated by lines starting with '@@'), we:
      - Record the leading space count of the first deletion line.
      - Record the leading space count of the first addition line.
      - Compute the difference: (indentation of '-' line) - (indentation of '+' line).
      - For every '+' line in the hunk, prepend extra spaces equal to that difference
        (if the difference is positive) or remove spaces if the difference is negative.

    :param patch_text: The original patch diff text.
    :return: The patch diff text with fixed indentation for added lines.
    """
    lines = patch_text.splitlines()
    fixed_lines = []

    in_hunk = False
    hunk_diff = None  # The indentation difference for the current hunk
    first_minus_indent = None  # Leading spaces count from the first '-' line in the hunk
    first_plus_indent = None  # Leading spaces count from the first '+' line in the hunk

    for line in lines:
        # Detect hunk header (e.g., @@ -1476,10 +1476,19 @@)
        if line.startswith("@@"):
            in_hunk = True
            # Reset hunk-specific variables for each new hunk.
            hunk_diff = None
            first_minus_indent = None
            first_plus_indent = None
            fixed_lines.append(line)
            continue

        # For diff metadata lines (e.g., diff, --- or +++), output them unchanged.
        if line.startswith("diff ") or line.startswith("---") or line.startswith("+++"):
            fixed_lines.append(line)
            continue

        if in_hunk:
            if line.startswith("-"):
                # Process deletion line.
                content = line[1:]
                indent_count = len(content) - len(content.lstrip(" "))
                if first_minus_indent is None:
                    first_minus_indent = indent_count
                    # If we already have a first addition indent, compute the difference.
                    if first_plus_indent is not None:
                        hunk_diff = first_minus_indent - first_plus_indent
                fixed_lines.append(line)
            elif line.startswith("+"):
                # Process addition line.
                content = line[1:]
                indent_count = len(content) - len(content.lstrip(" "))
                if first_plus_indent is None:
                    first_plus_indent = indent_count
                    # If we already have a first deletion indent, compute the difference.
                    if first_minus_indent is not None:
                        hunk_diff = first_minus_indent - first_plus_indent
                # If we have computed a hunk indentation difference, adjust this added line.
                if hunk_diff is not None:
                    if hunk_diff > 0:
                        # Prepend extra spaces equal to the difference.
                        content = " " * hunk_diff + content
                    elif hunk_diff < 0:
                        # Remove spaces if the difference is negative, but not below zero.
                        remove_spaces = min(-hunk_diff, len(content) - len(content.lstrip(" ")))
                        content = content[remove_spaces:]
                fixed_lines.append("+" + content)
            else:
                # For context lines within a hunk, leave them unchanged.
                fixed_lines.append(line)
        else:
            # Outside a hunk, just output the line as is.
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def strip_indent(lines: list[str]) -> list[str]:
    return [line.lstrip() for line in lines]

def find_matching_block(source_lines: list[str], search_lines: list[str]) -> int:
    stripped_source = strip_indent(source_lines)
    stripped_search = strip_indent(search_lines)

    search_len = len(stripped_search)

    for i in range(len(stripped_source) - search_len + 1):
        if stripped_source[i:i + search_len] == stripped_search:
            return i
    return -1

def adjust_patch_indentation(extracted_source: str, patch_chunk: str) -> str:
    old_lines = extracted_source.expandtabs(4).splitlines()
    new_lines = patch_chunk.expandtabs(4).splitlines()

    if not new_lines or extracted_source.strip() == "":
        return ""

    def first_meaningful_line(lines):
        for line in lines:
            if line.strip() and not line.strip().startswith("#"):
                return line
        return lines[0] if lines else ""

    old_first_line = first_meaningful_line(old_lines)
    new_first_line = first_meaningful_line(new_lines)

    old_indent_count = len(old_first_line) - len(old_first_line.lstrip())
    new_indent_count = len(new_first_line) - len(new_first_line.lstrip())

    indent_diff = old_indent_count - new_indent_count

    fixed_lines = []
    for line in new_lines:
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)
        new_indent = max(current_indent + indent_diff, 0)
        fixed_lines.append(" " * new_indent + stripped)

    return "\n".join(fixed_lines)

def apply_patch_with_indent_alignment(source_code: str, search_code: str, replace_code: str) -> str:
    source_lines = source_code.splitlines()
    search_lines = search_code.splitlines()

    match_start = find_matching_block(source_lines, search_lines)
    if match_start == -1:
        print("Search block could not be matched in the source.")
        return source_code

    match_end = match_start + len(search_lines)
    true_search_block = "\n".join(source_lines[match_start:match_end])

    # 调整 replace 缩进
    fixed_replace = adjust_patch_indentation(true_search_block, replace_code)

    # 替换操作（注意只替换第一次匹配）
    before = "\n".join(source_lines[:match_start])
    after = "\n".join(source_lines[match_end:])
    updated = "\n".join(filter(None, [before, fixed_replace, after]))

    return updated


def replace_function_in_file(file_content: str, extracted_source: str, patch_chunk: str) -> str:
    if extracted_source not in file_content:
        return file_content

    updated_content = file_content.replace(extracted_source, patch_chunk, 1)
    return updated_content


def generate_git_diff(file_path: str, old_content: str, new_content: str) -> str:
    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()

    diff_lines = unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm=""
    )

    return "\n".join(diff_lines)

# def generate_git_diff(file_path, original_content, old_code, new_code):
#     """
#     Generates a git-style diff without requiring an actual git repository.
#
#     Parameters:
#     - file_path (str): Path to the file being modified.
#     - original_content (str): The full content of the file before modification.
#     - old_code (str): The code snippet that needs to be replaced.
#     - new_code (str): The new code snippet replacing the old one.
#
#     Returns:
#     - str: A formatted git diff output.
#     """
#
#     # Split the original content into lines, preserving line endings
#     original_lines = original_content.splitlines()
#
#     # Split old and new code snippets into lines, preserving line endings
#     old_code_lines = old_code.splitlines()
#     new_code_lines = new_code.splitlines()
#
#     try:
#         # Find the starting index of the old code snippet within the file
#         start_index = original_lines.index(old_code_lines[0])
#         # Determine the ending index based on the number of lines in the old snippet
#         end_index = start_index + len(old_code_lines)
#     except Exception:
#         print("The old code snippet was not found in the original file content.")
#         return ""
#
#     # Replace the old code with the new code in the modified content
#     modified_lines = original_lines[:start_index] + new_code_lines + original_lines[end_index:]
#
#     # Generate a unified diff in git format
#     diff = list(difflib.unified_diff(
#         original_lines, modified_lines,
#         fromfile=f"a/{file_path}",
#         tofile=f"b/{file_path}",
#         lineterm=""  # Ensures correct line endings in the diff output
#     ))
#
#     return "\n".join(diff)
