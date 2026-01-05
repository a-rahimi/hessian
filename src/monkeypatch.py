"""This file isn't necessary for the Hessian-inverse-vector product algorithm.

It's used only in the demo notebook to save the intermediate matrices to disk.
The alternative to this is to clutter the code with debugging statements.
"""

from typing import Callable
import ast
import inspect
import textwrap


def monkey_patch_function(
    func: Callable, line_number_in_file: int, code_to_insert: str
) -> Callable:
    """
    Monkey patches a function by inserting arbitrary code at a specific line number.

    Args:
        func: The function to patch.
        line_number: The absolute line number in the file where the code should be inserted (before the existing line).
        code_to_insert: A string of Python code to insert.
    """
    # Get source and location
    source = inspect.getsource(func)
    lines, start_line = inspect.getsourcelines(func)
    source = textwrap.dedent(source)

    # Calculate relative line number.
    # Note: If we dedented, the line numbers in AST are still relative to the start of the string.
    # The `start_line` from `getsourcelines` is the absolute line number of the first line of `source`.
    target_relative_line = line_number_in_file - start_line + 1

    if target_relative_line < 1 or target_relative_line > len(lines):
        raise ValueError(
            f"Line number {line_number_in_file} is out of range for function {func.__name__} (starts at {start_line}, length {len(lines)})"
        )

    # Parse code to insert
    try:
        insert_tree = ast.parse(textwrap.dedent(code_to_insert))
        inserted_nodes = insert_tree.body
    except SyntaxError as e:
        raise ValueError(f"Invalid code to insert: {e}")

    # Parse function source
    tree = ast.parse(source)

    class InsertTransformer(ast.NodeTransformer):
        def __init__(self):
            self.inserted = False

        def visit(self, node):
            if (
                self.inserted
                or not isinstance(node, ast.stmt)
                or getattr(node, "lineno", -1) != target_relative_line
            ):
                return super().visit(node)

            self.inserted = True
            for new_node in inserted_nodes:
                ast.copy_location(new_node, node)

            return inserted_nodes + [node]

    transformer = InsertTransformer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    if not transformer.inserted:
        print(
            f"Warning: Could not find statement at line {line_number_in_file} to insert code. "
            "It's possible the line number corresponds to a blank line or comment which AST doesn't represent directly as a statement. "
            "In that case, we might miss."
        )

    # Compile the modified AST.
    code_obj = compile(tree, filename=inspect.getfile(func), mode="exec")

    # Execute to create new function
    # We use the original globals to ensure the function has access to the same environment
    namespace = func.__globals__.copy() if hasattr(func, "__globals__") else {}

    # We might need to handle closure variables if the function uses them.
    # This simple version does not handle closures that are not in globals.
    exec(code_obj, namespace)

    # Retrieve the new function
    func_name = tree.body[0].name
    new_func = namespace[func_name]

    # Replace in module
    module = inspect.getmodule(func)
    if module:
        setattr(module, func_name, new_func)

    return new_func


def find_line_number(search_string: str, file_path: str) -> int:
    with open(file_path, "r") as f:
        return next(i for i, line in enumerate(f, 1) if search_string in line)
