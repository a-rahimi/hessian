#!/usr/bin/env python3

import os
import re


def main():
    # Determine paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    readme_path = os.path.join(project_root, "README.md")
    source_path = os.path.join(project_root, "src", "hessian.py")

    function_name = "hessian_inverse_product"

    if not os.path.exists(source_path):
        print(f"Error: Source file not found at {source_path}")
        return

    # 1. Find line number in source
    line_number = None
    with open(source_path, "r") as f:
        for i, line in enumerate(f, 1):
            # Match "def hessian_inverse_product(" with optional whitespace
            if re.search(rf"def\s+{function_name}\s*\(", line):
                line_number = i
                break

    if line_number is None:
        print(f"Error: Could not find function '{function_name}' in {source_path}")
        return

    if not os.path.exists(readme_path):
        print(f"Error: README not found at {readme_path}")
        return

    # 2. Update README.md
    with open(readme_path, "r") as f:
        content = f.read()

    # Regex to find the link: [hessian_inverse_product](src/hessian.py#L<number>)
    pattern = rf"(\[{function_name}\]\(src/hessian\.py#L)(\d+)(\))"

    def replacement(match):
        prefix = match.group(1)
        old_line = match.group(2)
        suffix = match.group(3)
        if str(line_number) != old_line:
            print(f"Updating line number from {old_line} to {line_number}")
        else:
            print(f"Line number {line_number} is already correct.")
        return f"{prefix}{line_number}{suffix}"

    new_content, count = re.subn(pattern, replacement, content)

    if count == 0:
        print(f"Warning: Could not find link for '{function_name}' in README.md")
    else:
        with open(readme_path, "w") as f:
            f.write(new_content)
        print("README.md check complete.")


if __name__ == "__main__":
    main()
