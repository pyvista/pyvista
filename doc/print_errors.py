"""Read errors output from a sphinx build and remove duplicate groups"""
import os
import pathlib
import sys

sys.tracebacklimit = 0
my_path = pathlib.Path(__file__).parent.resolve()

errors = set()
error_file = os.path.join(my_path, 'build_errors.txt')
if os.path.isfile(error_file):
    with open(error_file) as fid:
        group = []
        for line in fid.readlines():
            line = line.strip()
            if line:
                group.append(line)
            else:
                errors.add('\n'.join(group))
                group = []

    for error in list(errors):
        print(error)
        print()

    # There should be no errors here since sphinx will have exited
    print()
    if errors:
        raise Exception(f'Sphinx reported unique {len(errors)} warnings\n\n')
else:
    print(f'build_errors.txt not found at {my_path}')

print('Sphinx Reported no warnings\n\n')
