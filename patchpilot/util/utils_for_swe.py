import docker
import time
import io
import tarfile
import os
import shutil
import re
import ast
import shlex
import tempfile
from pathlib import Path

from collections import namedtuple
ExecResult = namedtuple("ExecResult", "returncode stdout stderr")


client = docker.from_env(timeout=300)

def copy_directory_from_container(container_id, src_path, dst_path):
    container = client.containers.get(container_id)
    exec_result = container.exec_run(f"ls -la {src_path}")
    if exec_result.exit_code != 0:
        raise ValueError(f"Path {src_path} does not exist in the container.")

    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    os.makedirs(dst_path, exist_ok=True)

    stream, stats = container.get_archive(src_path)
    temp_tar = os.path.join(dst_path, "temp_archive.tar")

    with open(temp_tar, "wb") as f:
        for chunk in stream:
            f.write(chunk)

    with tarfile.open(temp_tar) as tar:
        tar.extractall(path=dst_path, numeric_owner=True)

    os.remove(temp_tar)


def copy_directory_to_container(container_id, src_path, dst_path):
    """
    Copies a directory from the host (src_path) into the container (dst_path).
    If dst_path does not exist, it is created. Contents are overwritten.
    """
    container = client.containers.get(container_id)

    mkdir_cmd = f"mkdir -p {dst_path}"
    exit_code, output = container.exec_run(mkdir_cmd)
    if exit_code != 0:
        raise RuntimeError(f"Failed to create directory {dst_path} in container: {output.decode()}")

    mem_tar = io.BytesIO()
    with tarfile.open(fileobj=mem_tar, mode='w') as tar:
        tar.add(src_path, arcname="")
    mem_tar.seek(0)

    container.put_archive(dst_path, mem_tar.getvalue())

def get_instance_docker_image(instance_id: str) -> str:
    image_name = 'sweb.eval.x86_64.' + instance_id
    image_name = "swebench/"+image_name
    image_name = image_name.replace("__", "_1776_")
    return image_name

def get_container(image_name):
    """Create and start a new container from the specified image."""
    container = client.containers.run(image_name, command="bash", stdin_open=True, tty=True, detach=True)
    print(f"Container {container.id} started from image {image_name}")
    return container.id

def run_poc(container_id, poc_code):
    """
    Writes poc_code into workdir/poc_code.py inside the container,
    then executes 'python poc_code.py' and captures the output.
    """
    # Get the container object.
    container = client.containers.get(container_id)

    # Get the container's working directory (assumes get_work_dir is defined)
    work_dir = get_work_dir(container_id)

    # Define the target file name.
    target_file = "poc_code.py"

    # Create a tar archive containing the target file with poc_code as its content.
    archive_data = create_tar_bytes(poc_code, arcname=target_file)

    # Place the archive into the working directory of the container.
    container.put_archive(work_dir, archive_data)

    # Execute the command "python poc_code.py" in the container.
    exec_result = run_command_in_container(container_id, f"python {target_file}")
    output = exec_result
    return output

def checkout(container_id, commit, dir=""):
    """Checkout to the specified commit inside the container.
    If `dir` is provided, run the command inside that directory.
    """
    container = client.containers.get(container_id)

    if dir:
        checkout_command = f"bash -c 'cd {dir} && git checkout {commit}'"
    else:
        checkout_command = f"git checkout {commit}"

    result = container.exec_run(checkout_command)
    print(f"Checked out commit {commit} in container {container_id}")
    return result

def reset(container_id):
    base_dir = get_work_dir(container_id)
    container = client.containers.get(container_id)
    reset_command = f"bash -c 'cd {base_dir} && git reset --hard'"
    result = container.exec_run(reset_command)
    print(f"Reset completed in container {container_id} at {base_dir}, output: {result.output.decode()}")
    return result

def reset_and_clean(container_id):
    """Perform git reset and clean inside the container at the specified base_dir."""
    base_dir = get_work_dir(container_id)
    container = client.containers.get(container_id)
    reset_command = f"bash -c 'cd {base_dir} && git reset --hard && git clean -f -d -x'"
    result = container.exec_run(reset_command)
    print(f"Reset completed in container {container_id} at {base_dir}, output: {result.output.decode()}")
    return result


def delete_container(container_id):
    container = client.containers.get(container_id)
    container.remove(force=True)
    print(f"Container {container_id} has been removed")


def extract_file_from_container(container, filepath):
    """
    Extracts the content of the specified file from the container.
    Returns the file content as a decoded string.
    """
    stream, _ = container.get_archive(filepath)
    file_data = b""
    for chunk in stream:
        file_data += chunk
    tar_stream = io.BytesIO(file_data)
    with tarfile.open(fileobj=tar_stream) as tar:
        member = tar.getmembers()[0]
        f = tar.extractfile(member)
        content = f.read().decode()
    return content

def extract_file_from_container_bytes(container, filepath):
    """
    Extracts the content of the specified file from the container.
    Returns the file content as bytes.
    """
    stream, _ = container.get_archive(filepath)
    file_data = b""
    for chunk in stream:
        file_data += chunk
    tar_stream = io.BytesIO(file_data)
    with tarfile.open(fileobj=tar_stream) as tar:
        member = tar.getmembers()[0]
        f = tar.extractfile(member)
        content = f.read()
    return content

def create_tar_bytes(file_content, arcname):
    """
    Packs the given file content into a tar archive.

    :param file_content: The file content as a string.
    :param arcname: The name of the file inside the archive.
    :return: The tar archive as a byte string.
    """
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode='w') as tar:
        file_bytes = file_content.encode()
        tarinfo = tarfile.TarInfo(name=arcname)
        tarinfo.size = len(file_bytes)
        tar.addfile(tarinfo, io.BytesIO(file_bytes))
    tar_stream.seek(0)
    return tar_stream.read()

def get_function_info(source_code):
    tree = ast.parse(source_code)
    functions = {}

    class FunctionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_FunctionDef(self, node):
            full_name = ".".join(self.stack + [node.name])
            end_lineno = getattr(node, "end_lineno", None)
            if end_lineno is None:
                end_lineno = node.body[-1].lineno if node.body else node.lineno
            functions[full_name] = (node.lineno, end_lineno)

            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)

        def visit_ClassDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

    FunctionVisitor().visit(tree)
    return functions

def adjust_indent_to_match(old_code: str, new_code: str) -> list[str]:
    def get_indent(line: str) -> int:
        return len(line) - len(line.lstrip())

    old_first_indent = get_indent(old_code.splitlines()[0])
    new_first_indent = get_indent(new_code.splitlines()[0])
    indent_diff = old_first_indent - new_first_indent

    adjusted_lines = []
    for line in new_code.splitlines():
        if not line.strip():
            adjusted_lines.append("")  # preserve blank lines
            continue

        if indent_diff > 0:
            adjusted_line = " " * indent_diff + line
        else:
            current_indent = get_indent(line)
            remove_n = min(abs(indent_diff), current_indent)
            adjusted_line = line[remove_n:]

        adjusted_lines.append(adjusted_line)
    return adjusted_lines

def patch_function(container_id, file_path, function_name, new_code):
    # Get the container object
    container = client.containers.get(container_id)

    # Extract file content from the container (assumes extract_file_from_container is defined)
    file_content = extract_file_from_container(container, file_path)

    # Get line range of the target function (assumes get_function_info is defined)
    try:
        functions = get_function_info(file_content)
    except Exception as e:
        return f"Error parsing file {file_path}: {e}"
    if function_name not in functions:
        print(f"Function {function_name} not found in file {file_path}")
        return

    start_line, end_line = functions[function_name]
    lines = file_content.splitlines()

    # Validate line range
    if start_line < 1 or end_line > len(lines):
        print(f"Invalid line range for function {function_name}: {start_line}-{end_line}")
        return

    # Prepare new code lines (with proper indentation)
    new_code_lines = adjust_indent_to_match(lines[start_line - 1], new_code)

    # Replace the old function with the new one
    new_lines = lines[:start_line - 1] + new_code_lines + lines[end_line:]


    # Join back into full file content
    new_file_content = "\n".join(new_lines) + "\n"

    # Create tar archive data (assumes create_tar_bytes is defined)
    archive_data = create_tar_bytes(new_file_content, arcname=file_path.split("/")[-1])
    destination_dir = "/".join(file_path.split("/")[:-1]) or "/"

    # Put updated file back into container
    container.put_archive(destination_dir, archive_data)
    print(f"Updated function {function_name} in file {file_path} within container {container_id}")




def get_file_content(container_id, filepath):
    """
    Retrieves the content of a file inside the container.

    Input:
      - container_id: The ID of the Docker container.
      - filepath: The path to the file inside the container.

    Returns:
      The content of the file as a string, or None if the file is not found.
    """
    container = client.containers.get(container_id)

    # Extract the file content from the container
    file_content = extract_file_from_container(container, filepath)

    return file_content

def run_command_in_container(container_id, command):
    container = client.containers.get(container_id)
    conda_prefix = "source ~/.bashrc && "
    full_command = f"/bin/bash -c {shlex.quote(conda_prefix + command)}"

    exit_code, (stdout, stderr) = container.exec_run(full_command, demux=True)
    stdout = stdout.decode() if stdout else ""
    stderr = stderr.decode() if stderr else ""

    return ExecResult(returncode=exit_code, stdout=stdout, stderr=stderr)

def get_work_dir(container_id):
    """
    Get the working directory of the container.
    """
    work_dir = run_command_in_container(container_id, "pwd").stdout.strip()
    return work_dir


def wrap_in_cd(command, basedir):
    if basedir:
        return f"bash -c 'cd {basedir} && {command}'"
    return command

def get_python_functions(source_code):
    """
    Parses Python source to extract function names and their (start_line, end_line),
    including class methods and nested functions.
    Returns a dict: full_function_name -> (start_line, end_line)
    """
    tree = ast.parse(source_code)
    functions = {}

    class FunctionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_FunctionDef(self, node):
            full_name = ".".join(self.stack + [node.name])
            end_lineno = getattr(node, "end_lineno", None)
            if end_lineno is None:
                end_lineno = node.body[-1].lineno if node.body else node.lineno
            functions[full_name] = (node.lineno, end_lineno)

            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)

        def visit_ClassDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

    FunctionVisitor().visit(tree)
    return functions


def parse_modified_functions_from_diff(container_id, diff_text):
    file_changes = {}
    current_file = None

    # Regular expressions for diff headers and hunks
    file_header_regex = re.compile(r'^diff --git a/(.*?) b/')
    hunk_header_regex = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@')

    work_dir = get_work_dir(container_id)
    lines = diff_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        file_match = file_header_regex.match(line)
        if file_match:
            current_file = file_match.group(1)
            file_changes[current_file] = set()
            i += 1
            continue

        hunk_match = hunk_header_regex.match(line)
        if hunk_match and current_file:
            new_start = int(hunk_match.group(1))
            # new_count is not used, but we can extract it
            new_count = int(hunk_match.group(2)) if hunk_match.group(2) else 1
            new_line_num = new_start
            i += 1
            while i < len(lines) and not lines[i].startswith("diff --git") and not lines[i].startswith("@@"):
                l = lines[i]
                if l.startswith(" "):
                    new_line_num += 1
                elif l.startswith("+"):
                    file_changes[current_file].add(new_line_num)
                    new_line_num += 1
                elif l.startswith("-"):
                    file_changes[current_file].add(new_line_num)
                i += 1
            continue

        i += 1

    result = []
    seen = set()

    for file_path, changed_lines in file_changes.items():
        full_path = os.path.join(work_dir, file_path)
        try:
            file_content = get_file_content(container_id, full_path)
            if not file_content:
                continue
            # Use splitlines(keepends=True) to preserve original formatting
            lines_with_newline = file_content.splitlines(keepends=True)
            # Parse file content with AST
            tree = ast.parse(file_content)

            # Visitor to extract functions along with their qualified name and location
            class FunctionVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.scope = []
                    self.functions = []
                def visit_Module(self, node):
                    for child in node.body:
                        self.visit(child)
                def visit_ClassDef(self, node):
                    self.scope.append(node.name)
                    for child in node.body:
                        self.visit(child)
                    self.scope.pop()
                def visit_FunctionDef(self, node):
                    qualified_name = ".".join(self.scope + [node.name])
                    start = node.lineno
                    # Use end_lineno attribute if available (Python 3.8+), else fallback
                    end = getattr(node, "end_lineno", None)
                    if end is None:
                        end = find_function_end_lineno(lines_with_newline, start)
                    self.functions.append((qualified_name, start, end))
                    # Continue visiting nested functions if any
                    for child in node.body:
                        self.visit(child)
                def visit_AsyncFunctionDef(self, node):
                    self.visit_FunctionDef(node)

            def find_function_end_lineno(lines, start):
                """
                Naively finds the end line number of a function definition based on indentation.
                This is a fallback if ast does not provide end_lineno.
                """
                def_indent = len(lines[start-1]) - len(lines[start-1].lstrip())
                for idx in range(start, len(lines)):
                    line = lines[idx]
                    # If a non-empty line has an indentation less or equal to the def line, we assume function body ended.
                    if line.strip() and (len(line) - len(line.lstrip())) <= def_indent:
                        return idx
                return len(lines)

            visitor = FunctionVisitor()
            visitor.visit(tree)
            functions = visitor.functions

            # Check each function to see if any changed line falls within its range
            for qualified_name, start_line, end_line in functions:
                if any(start_line <= line <= end_line for line in changed_lines):
                    function_code = ''.join(lines_with_newline[start_line-1:end_line])
                    key = (qualified_name, full_path, start_line, end_line, function_code)
                    if key not in seen:
                        seen.add(key)
                        result.append(key)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    return result

def get_function_from_file(container_id, file_path, function_name):
    # Retrieve file content from the container.
    file_content = get_file_content(container_id, file_path)
    if not file_content:
        print(f"File not found or is empty: {file_path}")
        return None

    # Get a mapping from qualified function names to (start_line, end_line)
    function_mapping = get_python_functions(file_content)
    if function_name not in function_mapping:
        print(f"Function {function_name} not found in file {file_path}")
        return None

    start_line, end_line = function_mapping[function_name]
    # Use splitlines(keepends=True) to preserve the original line endings.
    lines = file_content.splitlines(keepends=True)
    func_code = "".join(lines[start_line - 1:end_line])
    return func_code

def apply_diff(container_id, diff_text):
    container = client.containers.get(container_id)
    temp_diff_filename = "temp_diff.patch"
    dest_dir = "/tmp"
    dest_path = f"{dest_dir}/{temp_diff_filename}"

    # Create a temporary file locally in a temporary folder and write diff_text to it
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        temp_file.write(diff_text)
        temp_file.flush()
        temp_file_path = temp_file.name

    # Create a tar archive containing the temporary diff file
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        tar.add(temp_file_path, arcname=temp_diff_filename)
    tar_stream.seek(0)

    # Copy the tar archive into the container's /tmp folder
    container.put_archive(dest_dir, tar_stream.read())

    # Apply the diff using git
    result = container.exec_run(f"git apply {dest_path}")

    # Remove the temporary diff file from the container
    container.exec_run(f"rm {dest_path}")

    # Clean up the local temporary file
    os.unlink(temp_file_path)

    return result


def path_to_module_name(container_id, path, base_dir=None):
    if not base_dir:
        base_dir = get_work_dir(container_id)
    path = str(Path(path).relative_to(base_dir))

    # Replace slashes with dots
    module_name = path.replace("/", ".")

    # Remove the file extension if present
    if ".py" in module_name:
        module_name = module_name.rsplit(".py", 1)[0]

    return module_name