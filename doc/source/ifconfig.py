"""Add ifconfig directive to doctests in docstrings."""

from __future__ import annotations

import ast
import doctest
from pathlib import Path


def _add_ifconfig_to_file(file_path: str | Path):
    # Read the source code from the file
    with Path.open(file_path) as source_file:
        source_code = source_file.read()

    # Parse the source code into an AST
    parsed_tree = ast.parse(source_code)

    # Split the source code into lines for modification
    source_lines = source_code.splitlines()

    # List to store docstring replacements
    docstring_replacements = []

    # Traverse the AST and find docstrings to replace code snippets
    for ast_node in ast.walk(parsed_tree):
        if isinstance(ast_node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
            original_docstring = ast.get_docstring(ast_node, clean=False)

            if original_docstring:
                # Parse the docstring using doctest to extract code examples
                parsed_examples = doctest.DocTestParser().parse(original_docstring)
                transformed_docstring_parts = []

                # Process each part of the docstring
                for doc_part in parsed_examples:
                    if isinstance(doc_part, doctest.Example):
                        # Replace doctest code examples with the :ifconfig: directive
                        # Add a placehold so it can be replaced with indentation later
                        indent = doc_part.indent * ' '

                        code = ''.join(
                            [f'{indent}``>>> {line}``\n' for line in doc_part.source.splitlines()]
                        )
                        want = ''.join(
                            [f'{indent}``{line}``\n' for line in doc_part.want.splitlines()]
                        )

                        transformed_docstring_parts.append(code + want)
                    else:
                        # Append non-example parts unchanged
                        transformed_docstring_parts.append(str(doc_part))

                # Reassemble the transformed docstring
                transformed_docstring = ''.join(transformed_docstring_parts)

                # Store the start and end line numbers for the docstring replacement
                start_line_number = ast_node.body[0].lineno - 1
                end_line_number = ast_node.body[0].end_lineno
                docstring_replacements.append(
                    (start_line_number, end_line_number, transformed_docstring)
                )

    # Apply the replacements to the original source code lines
    for start_line, end_line, new_docstring in sorted(docstring_replacements, reverse=True):
        # Calculate the indentation of the docstring
        docstring_indent = ' ' * (
            len(source_lines[start_line]) - len(source_lines[start_line].lstrip())
        )

        # Add indentation to new directive lines using the placeholder
        new_doc_lines = new_docstring.splitlines()

        # Add triple quotes to docstring
        new_doc_lines[0] = f'{docstring_indent}"""' + new_doc_lines[0].lstrip()
        new_doc_lines[-1] = new_doc_lines[-1].rstrip() + f'{docstring_indent}"""'

        # Replace the original docstring lines with the new ones
        source_lines[start_line:end_line] = new_doc_lines

    # Write the updated source code back to the file
    with Path.open(file_path, 'w') as output_file:
        output_file.write('\n'.join(source_lines))


def _add_ifconfig_to_files_recursively(package):
    def get_python_files(dir_: str):
        return [str(file) for file in Path(dir_).rglob('*.py')]

    for file in get_python_files(Path(package.__file__).parent):
        _add_ifconfig_to_file(file)
