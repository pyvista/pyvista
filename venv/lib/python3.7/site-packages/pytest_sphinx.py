# -*- coding: utf-8 -*-
"""
http://www.sphinx-doc.org/en/stable/ext/doctest.html
https://github.com/sphinx-doc/sphinx/blob/master/sphinx/ext/doctest.py

* TODO
** CLEANUP: use the sphinx directive parser from the sphinx project
** support for :options: in testoutput (see sphinx-doc)
"""

import doctest
import enum
import itertools
import re
import textwrap
import sys
import traceback

import _pytest.doctest
import pytest


def pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return list(zip(a, b))


class SphinxDoctestDirectives(enum.Enum):
    TESTCODE = 1
    TESTOUTPUT = 2
    TESTSETUP = 3
    TESTCLEANUP = 4
    DOCTEST = 5


def pytest_collect_file(path, parent):
    config = parent.config
    if path.ext == ".py":
        if config.option.doctestmodules:
            return SphinxDoctestModule(path, parent)
    elif _is_doctest(config, path, parent):
        return SphinxDoctestTextfile(path, parent)


def _is_doctest(config, path, parent):
    if path.ext in ('.txt', '.rst') and parent.session.isinitpath(path):
        return True
    globs = config.getoption("doctestglob") or ['test*.txt']
    for glob in globs:
        if path.check(fnmatch=glob):
            return True
    return False


# This regular expression looks for option directives in the expected output
# (testoutput) code of an example.  Option directives are comments starting
# with ":options:".
_OPTION_DIRECTIVE_RE = re.compile(r':options:\s*([^\n\'"]*)$',
                                  re.MULTILINE)


# In order to compare the testoutput, containing Option directives we have
# to remove the optiondirectives from the string after parsing.
_OPTION_DIRECTIVE_RE_SUB = (
    re.compile(r':options:\s*([^\n\'"]*)\n').sub)


def _find_options(want, name, lineno):
    """
    Return a dictionary containing option overrides extracted from option
    directives in the given `want` string.

    `name` is the string's name, and `lineno` is the line number where
    the example starts; both are used for error messages.

    """
    options = {}
    # (note: with the current regexp, this will match at most once:)
    for m in _OPTION_DIRECTIVE_RE.finditer(want):
        option_strings = m.group(1).replace(',', ' ').split()
        for option in option_strings:
            if (option[0] not in '+-' or
                    option[1:] not in doctest.OPTIONFLAGS_BY_NAME):
                raise ValueError('line %r of the doctest for %s '
                                 'has an invalid option: %r' %
                                 (lineno + 1, name, option))
            flag = doctest.OPTIONFLAGS_BY_NAME[option[1:]]
            options[flag] = (option[0] == '+')
    if options and doctest.DocTestParser._IS_BLANK_OR_COMMENT(want):
        raise ValueError('line %r of the doctest for %s has an option '
                         'directive on a line with no example: %r' %
                         (lineno, name, want))
    return options


def docstring2examples(docstring):
    """
    Parse all sphinx test directives in the docstring and create a
    list of examples.
    """
    # TODO subclass doctest.DocTestParser instead?

    lines = textwrap.dedent(docstring).splitlines()
    matches = [i for i, line in enumerate(lines) if
               any(line.strip().startswith('.. ' + d.name.lower() + '::')
                   for d in SphinxDoctestDirectives)]
    if not matches:
        return []

    matches.append(len(lines))

    class Section(object):
        def __init__(self, name, content, lineno):
            super(Section, self).__init__()
            self.name = name
            self.lineno = lineno
            if name in (SphinxDoctestDirectives.TESTCODE,
                        SphinxDoctestDirectives.TESTOUTPUT):
                # remove empty lines
                self.content = '\n'.join(
                    [line for line in content.splitlines()
                     if not re.match(r'^\s*$', line)])
            else:
                self.content = content

    def is_empty_of_indented(line):
        return not line or line.startswith('   ')

    sections = []
    for x, y in pairwise(matches):
        section = lines[x:y]
        header = section[0]
        directive = next(d for d in SphinxDoctestDirectives
                         if d.name.lower() in header)
        out = '\n'.join(itertools.takewhile(
            is_empty_of_indented, section[1:]))
        sections.append(Section(
            directive,
            textwrap.dedent(out),
            lineno=x))

    examples = []
    for x, y in pairwise(sections):
        # TODO support SphinxDoctestDirectives.TESTSETUP, ...
        if (x.name == SphinxDoctestDirectives.TESTCODE and
                y.name == SphinxDoctestDirectives.TESTOUTPUT):

            want = y.content
            m = doctest.DocTestParser._EXCEPTION_RE.match(want)
            if m:
                exc_msg = m.group('msg')
            else:
                exc_msg = None

            options = _find_options(want, 'dummy', y.lineno)

            # where should the :options: string be removed?
            # (only in the OutputChecker?, but then it is visible in the
            # pytest output in the "EXPECTED" section....
            want = _OPTION_DIRECTIVE_RE_SUB('', want)

            examples.append(
                doctest.Example(source=x.content, want=want,
                                exc_msg=exc_msg,
                                # we want to see the ..testcode lines in the
                                # console output but not the ..testoutput
                                # lines
                                lineno=y.lineno - 1,
                                options=options))

    return examples


class SphinxDocTestRunner(doctest.DebugRunner):
    """
    overwrite doctest.DocTestRunner.__run, since it uses 'single' for the
    `compile` function instead of 'exec'.
    """
    def _DocTestRunner__run(self, test, compileflags, out):
        """
        Run the examples in `test`.

        Write the outcome of each example with one of the
        `DocTestRunner.report_*` methods, using the writer function
        `out`.  `compileflags` is the set of compiler flags that should
        be used to execute examples.  Return a tuple `(f, t)`, where `t`
        is the number of examples tried, and `f` is the number of
        examples that failed.  The examples are run in the namespace
        `test.globs`.

        """
        # Keep track of the number of failures and tries.
        failures = tries = 0

        # Save the option flags (since option directives can be used
        # to modify them).
        original_optionflags = self.optionflags

        SUCCESS, FAILURE, BOOM = range(3)  # `outcome` state

        check = self._checker.check_output

        # Process each example.
        for examplenum, example in enumerate(test.examples):

            # If REPORT_ONLY_FIRST_FAILURE is set, then suppress
            # reporting after the first failure.
            quiet = (self.optionflags & doctest.REPORT_ONLY_FIRST_FAILURE and
                     failures > 0)

            # Merge in the example's options.
            self.optionflags = original_optionflags
            if example.options:
                for (optionflag, val) in example.options.items():
                    if val:
                        self.optionflags |= optionflag
                    else:
                        self.optionflags &= ~optionflag

            # If 'SKIP' is set, then skip this example.
            if self.optionflags & doctest.SKIP:
                continue

            # Record that we started this example.
            tries += 1
            if not quiet:
                self.report_start(out, test, example)

            # Use a special filename for compile(), so we can retrieve
            # the source code during interactive debugging (see
            # __patched_linecache_getlines).
            filename = '<doctest %s[%d]>' % (test.name, examplenum)

            # Run the example in the given context (globs), and record
            # any exception that gets raised.  (But don't intercept
            # keyboard interrupts.)
            try:
                # Don't blink!  This is where the user's code gets run.
                exec(compile(example.source, filename, "exec",
                             compileflags, 1), test.globs)
                self.debugger.set_continue()  # ==== Example Finished ====
                exception = None
            except KeyboardInterrupt:
                raise
            except Exception:
                exception = sys.exc_info()
                self.debugger.set_continue()  # ==== Example Finished ====

            got = self._fakeout.getvalue()  # the actual output
            self._fakeout.truncate(0)
            outcome = FAILURE   # guilty until proved innocent or insane

            # If the example executed without raising any exceptions,
            # verify its output.
            if exception is None:
                if check(example.want, got, self.optionflags):
                    outcome = SUCCESS

            # The example raised an exception:  check if it was expected.
            else:
                exc_msg = traceback.format_exception_only(*exception[:2])[-1]
                if not quiet:
                    got += doctest._exception_traceback(exception)

                # If `example.exc_msg` is None, then we weren't expecting
                # an exception.
                if example.exc_msg is None:
                    outcome = BOOM

                # We expected an exception:  see whether it matches.
                elif check(example.exc_msg, exc_msg, self.optionflags):
                    outcome = SUCCESS

                # Another chance if they didn't care about the detail.
                elif self.optionflags & doctest.IGNORE_EXCEPTION_DETAIL:
                    if check(doctest._strip_exception_details(example.exc_msg),
                             doctest._strip_exception_details(exc_msg),
                             self.optionflags):
                        outcome = SUCCESS

            # Report the outcome.
            if outcome is SUCCESS:
                if not quiet:
                    self.report_success(out, test, example, got)
            elif outcome is FAILURE:
                if not quiet:
                    self.report_failure(out, test, example, got)
                failures += 1
            elif outcome is BOOM:
                if not quiet:
                    self.report_unexpected_exception(out, test, example,
                                                     exception)
                failures += 1
            else:
                assert False, ("unknown outcome", outcome)

            if failures and self.optionflags & doctest.FAIL_FAST:
                break

        # Restore the option flags (in case they were modified)
        self.optionflags = original_optionflags

        # Record and return the number of failures and tries.
        self._DocTestRunner__record_outcome(test, failures, tries)
        return doctest.TestResults(failures, tries)


class SphinxDocTestParser(object):
    def get_doctest(self, docstring, globs, name, filename, lineno):
        # TODO document why we need to overwrite? get_doctest
        return doctest.DocTest(examples=docstring2examples(docstring),
                               globs=globs,
                               name=name,
                               filename=filename,
                               lineno=lineno,
                               docstring=docstring)


class SphinxDoctestTextfile(pytest.Module):
    obj = None

    def collect(self):
        # inspired by doctest.testfile; ideally we would use it directly,
        # but it doesn't support passing a custom checker
        encoding = self.config.getini("doctest_encoding")
        text = self.fspath.read_text(encoding)
        name = self.fspath.basename

        optionflags = _pytest.doctest.get_optionflags(self)
        runner = SphinxDocTestRunner(verbose=0,
                                     optionflags=optionflags,
                                     checker=_pytest.doctest._get_checker())

        test = doctest.DocTest(examples=docstring2examples(text),
                               globs={},
                               name=name,
                               filename=name,
                               lineno=0,
                               docstring=text)

        if test.examples:
            yield _pytest.doctest.DoctestItem(
                test.name, self, runner, test)


class SphinxDoctestModule(pytest.Module):
    def collect(self):
        if self.fspath.basename == "conftest.py":
            module = self.config.pluginmanager._importconftest(self.fspath)
        else:
            try:
                module = self.fspath.pyimport()
            except ImportError:
                if self.config.getvalue('doctest_ignore_import_errors'):
                    pytest.skip('unable to import module %r' % self.fspath)
                else:
                    raise
        optionflags = _pytest.doctest.get_optionflags(self)

        class MockAwareDocTestFinder(doctest.DocTestFinder):
            """
            a hackish doctest finder that overrides stdlib internals to fix
            a stdlib bug
            https://github.com/pytest-dev/pytest/issues/3456
            https://bugs.python.org/issue25532

            fix taken from https://github.com/pytest-dev/pytest/pull/4212/
            """

            def _find(self, tests, obj, name, module, source_lines,
                      globs, seen):
                if _is_mocked(obj):
                    return
                with _patch_unwrap_mock_aware():
                    doctest.DocTestFinder._find(
                        self, tests, obj, name, module, source_lines, globs,
                        seen
                    )

        try:
            from _pytest.doctest import _is_mocked
            from _pytest.doctest import _patch_unwrap_mock_aware
        except ImportError:
            finder = doctest.DocTestFinder(parser=SphinxDocTestParser())
        else:
            finder = MockAwareDocTestFinder(parser=SphinxDocTestParser())

        runner = SphinxDocTestRunner(verbose=0,
                                     optionflags=optionflags,
                                     checker=_pytest.doctest._get_checker())

        for test in finder.find(module, module.__name__):
            if test.examples:
                yield _pytest.doctest.DoctestItem(
                    test.name, self, runner, test)
