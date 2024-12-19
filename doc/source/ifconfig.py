from __future__ import annotations

import ast
import doctest


def replace_snippets_with_ifconfig(file_path):
    # Read the source code
    with open(file_path) as file:
        source = file.read()

    # Parse the source code into an AST
    tree = ast.parse(source)

    # Function to transform docstrings
    def transform_docstring(docstring):
        examples = doctest.DocTestParser().parse(docstring)
        updated_docstring = []
        for part in examples:
            if isinstance(part, doctest.Example):
                source = part.source
                indent = ' ' * (len(source) - len(source.lstrip()))
                directive = f'{indent}:ifconfig: show\n\n{indent}    {source.strip()}\n'
                updated_docstring.append(directive)
            else:
                updated_docstring.append(part)
        return ''.join(str(p) for p in updated_docstring)

    # List to store replacements
    replacements = []

    # Traverse the AST and collect docstring updates
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
            if docstring := ast.get_docstring(node, clean=False):
                # Transform the docstring
                new_docstring = transform_docstring(docstring)
                # Record the old and new docstrings
                start_lineno = node.body[0].lineno - 1
                end_lineno = node.body[0].end_lineno
                replacements.append((start_lineno, end_lineno, new_docstring))

    # Apply replacements to the original source
    lines = source.splitlines()
    for start, end, new_docstring in sorted(replacements, reverse=True):
        indent = ' ' * (len(lines[start]) - len(lines[start].lstrip()))
        new_lines = new_docstring.splitlines()
        new_lines[0] = f'{indent}"""{new_lines[0]}'
        new_lines[-1] = f'{new_lines[-1]}\n{indent}"""'
        lines[start:end] = new_lines

    # Write the updated source back to the file
    with open(file_path, 'w') as file:
        file.write('\n'.join(lines))

    print(f'Updated file: {file_path}')

