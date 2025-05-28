import libcst as cst
import libcst.matchers as m
from typing import Union, List
from libcst.metadata import PositionProvider, MetadataWrapper


class CompressTransformer(cst.CSTTransformer):
    DESCRIPTION = str = "Replaces function body with ..."
    replacement_string = '"$$FUNC_BODY_REPLACEMENT_STRING$$"'

    def __init__(self, keep_constant=True, functions_to_delete=None):
        self.keep_constant = keep_constant
        self.functions_to_delete = functions_to_delete or []
        self.function_info = []  # store function info

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        new_body = []
        for stmt in updated_node.body:
            if m.matches(stmt, m.ClassDef()):
                new_body.append(stmt)
            elif m.matches(stmt, m.FunctionDef()):
                if stmt.name.value not in self.functions_to_delete:
                    new_body.append(stmt)
            elif (
                self.keep_constant
                and m.matches(stmt, m.SimpleStatementLine())
                and m.matches(stmt.body[0], m.Assign())
            ):
                new_body.append(stmt)
        return updated_node.with_changes(body=new_body)

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        # Remove docstring in the class body and functions to delete
        new_body = []
        for stmt in updated_node.body.body:
            if (
                m.matches(stmt, m.SimpleStatementLine())
                and m.matches(stmt.body[0], m.Expr())
                and m.matches(stmt.body[0].value, m.SimpleString())
            ):
                continue  # Remove docstring
            if isinstance(stmt, cst.FunctionDef) and stmt.name.value in self.functions_to_delete:
                continue  # Remove functions to delete
            new_body.append(stmt)
        return updated_node.with_changes(body=cst.IndentedBlock(body=new_body))

    def leave_FunctionDef(
            self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> Union[cst.FunctionDef, cst.RemovalSentinel]:
        # If current function is in the set to delete, remove it
        if original_node in self.functions_to_delete:
            return cst.RemoveFromParent()
        # Initialize list to collect strings
        strings = []

        # Collect all string literals in the function body
        class StringCollector(cst.CSTVisitor):
            def visit_SimpleString(self, node: cst.SimpleString):
                if isinstance(node.evaluated_value, str):
                    if "div>" not in node.evaluated_value and "<div" not in node.evaluated_value and "</div" not in node.evaluated_value:
                        strings.append(node.evaluated_value)
                else:
                    strings.append(node.evaluated_value)

            def visit_ConcatenatedString(self, node: cst.ConcatenatedString):
                # Recursively collect strings
                node.left.visit(self)
                node.right.visit(self)

        updated_node.body.visit(StringCollector())

        # Remove code logic, keep only comments and docstrings
        class CodeRemover(cst.CSTTransformer):
            def leave_SimpleStatementLine(self, original_node, updated_node):
                # Keep comments and docstrings
                has_comment = any(line.comment for line in original_node.leading_lines) or \
                              original_node.trailing_whitespace.comment
                is_docstring = (
                        len(original_node.body) == 1 and isinstance(original_node.body[0], cst.Expr) and
                        isinstance(original_node.body[0].value, cst.SimpleString)
                )
                if has_comment or is_docstring:
                    return original_node
                else:
                    # Remove other statements
                    return cst.RemoveFromParent()

            def leave_Assign(self, original_node, updated_node):
                # Remove assignment statements
                return cst.RemoveFromParent()

            def leave_AugAssign(self, original_node, updated_node):
                # Remove augmented assignment statements (e.g., +=)
                return cst.RemoveFromParent()

            def leave_AnnAssign(self, original_node, updated_node):
                # Remove annotated assignments
                return cst.RemoveFromParent()

            def leave_For(self, original_node, updated_node):
                return cst.RemoveFromParent()

            def leave_While(self, original_node, updated_node):
                return cst.RemoveFromParent()

            def leave_If(self, original_node, updated_node):
                return cst.RemoveFromParent()

            def leave_With(self, original_node, updated_node):
                return cst.RemoveFromParent()

            def leave_Try(self, original_node, updated_node):
                return cst.RemoveFromParent()

            def leave_FunctionDef(self, original_node, updated_node):
                # Do not traverse nested functions
                return original_node

            def leave_ClassDef(self, original_node, updated_node):
                # Do not traverse nested classes
                return original_node

            def leave_Return(self, original_node, updated_node):
                return cst.RemoveFromParent()

            def leave_Expr(self, original_node, updated_node):
                # Remove expressions unless they are docstrings
                if isinstance(original_node.value, cst.SimpleString):
                    # Keep docstrings
                    return original_node
                else:
                    return cst.RemoveFromParent()

        # Apply CodeRemover to the function body
        new_body = updated_node.body.visit(CodeRemover())

        # Create the strings assignment
        if strings:
            strings_assignment = cst.SimpleStatementLine(
                [
                    cst.Assign(
                        targets=[cst.AssignTarget(target=cst.Name(value='strings'))],
                        value=cst.List(
                            elements=[cst.Element(value=cst.SimpleString(repr(s))) for s in strings]
                        ),
                    )
                ]
            )
        else:
            strings_assignment = None

        # Build the new function body
        body_statements = []

        if strings_assignment:
            body_statements.append(strings_assignment)

        # If new_body is an IndentedBlock, extract its statements
        if isinstance(new_body, cst.IndentedBlock):
            body_statements.extend(new_body.body)
        elif new_body is not None:
            body_statements.append(new_body)

        # Remove any EmptyLines from body_statements
        body_statements = [stmt for stmt in body_statements if not isinstance(stmt, cst.EmptyLine)]

        # Ensure the function body is not empty; if empty, add a 'pass' statement
        if not body_statements:
            pass_stmt = cst.SimpleStatementLine([cst.Pass()])
            body_statements.append(pass_stmt)

        # Create the new function body
        new_body = cst.IndentedBlock(body=body_statements)

        # Return the updated function node
        return updated_node.with_changes(body=new_body)

class FunctionCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, functions_to_delete_func_start_lines: List[int]):
        self.functions_to_delete_func_start_lines = set(functions_to_delete_func_start_lines)
        self.functions_to_delete = set()

    def visit_FunctionDef(self, node: cst.FunctionDef):
        start_pos = self.get_metadata(PositionProvider, node).start
        if start_pos.line in self.functions_to_delete_func_start_lines:
            self.functions_to_delete.add(node)

class CommentAndStringCollector(cst.CSTVisitor):
    def __init__(self):
        super().__init__()
        self.comments = []
        self.strings = []

    def visit_Comment(self, node: cst.Comment):
        self.comments.append(node.value.strip())

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine):
        for line in node.leading_lines:
            if line.comment:
                self.comments.append(line.comment.value.strip())

    def visit_SimpleString(self, node: cst.SimpleString):
        self.strings.append(node.evaluated_value)

    def visit_ConcatenatedString(self, node: cst.ConcatenatedString):
        node.left.visit(self)
        node.right.visit(self)


code = """
\"\"\"
this is a module
...
\"\"\"
const = {1,2,3}
import os

class fooClass:
    '''this is a class'''

    def __init__(self, x):
        '''initialization.'''
        self.x = x

    def print(self):
        print(self.x)

def test():
    a = fooClass(3)
    a.print()

"""

def get_skeleton(raw_code, keep_constant: bool = True, delete_func_start_lines: List[int] = None):
    try:
        tree = cst.parse_module(raw_code)
    except:
        return raw_code

    wrapper = MetadataWrapper(tree)
    
    # Step 1: Collect functions to delete
    collector = FunctionCollector(delete_func_start_lines)
    wrapper.visit(collector)
    functions_to_delete_nodes = collector.functions_to_delete

    # Step 2: Transform the tree
    transformer = CompressTransformer(
        keep_constant=keep_constant,
        functions_to_delete=functions_to_delete_nodes
    )
    modified_tree = wrapper.visit(transformer)
    code = modified_tree.code
    code = code.replace(CompressTransformer.replacement_string + "\n", "...\n")
    code = code.replace(CompressTransformer.replacement_string, "...\n")
    return code


def test_compress():
    functions_to_delete = ['test'] 
    skeleton = get_skeleton(code, True, functions_to_delete)
    print(skeleton)


if __name__ == "__main__":
    test_compress()
