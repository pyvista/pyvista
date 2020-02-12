# Contributing

We absolutely welcome contributions and we hope that this guide will facilitate
an understanding of the PyVista code repository. It is important to note that
the  PyVista software package is maintained on a volunteer basis and thus we
need to foster a community that can support user questions and develop new
features to make this software a useful tool for all users.

This page is dedicated to outline where you should start with your question,
concern, feature request, or desire to contribute.


## Cloning the Source Repository

You can clone the source repository from `https://github.com/pyvista/pyvista`
and install the latest version by running:

```bash
git clone https://github.com/pyvista/pyvista.git
cd pyvista
python -m pip install -e .
```


## Questions

For general questions about the project, its applications, or about software
usage, please create an issue in the [pyvista/pyvista-support](https://github.com/pyvista/pyvista-support)
repository where the community can collectively address your questions.
You are also welcome to join us on [Slack](http://slack.pyvista.org)
or send one of the developers an email.
The project support team can be reached at [info@pyvista.org](mailto:info@pyvista.org)


For more technical questions, you are welcome to create an issue on the
[issues page](https://github.com/pyvista/pyvista/issues) which we will address promptly.
Through posting on the issues page, your question can be addressed by community
members with the needed expertise and the information gained will remain
available on the issues page for other users.


## Reporting Bugs

If you stumble across any bugs, crashes, or concerning quirks while using code
distributed here, please report it on the [issues page](https://github.com/pyvista/pyvista/issues)
with an appropriate label so we can promptly address it.
When reporting an issue, please be overly descriptive so that we may reproduce
it. Whenever possible, please provide tracebacks, screenshots, and sample files
to help us address the issue.

## Feature Requests

We encourage users to submit ideas for improvements to PyVista code base!
Please create an issue on the [issues page](https://github.com/pyvista/pyvista/issues)
with a *Feature Request* label to suggest an improvement.
Please use a descriptive title and provide ample background information to help
the community implement that functionality. For example, if you would like a
reader for a specific file format, please provide a link to documentation of
that file format and possibly provide some sample files with screenshots to work
with. We will use the issue thread as a place to discuss and provide feedback.

## Contributing New Code

If you have an idea for how to improve PyVista, please first create an issue as
a feature request which we can use as a discussion thread to work through how to
implement the contribution.

Once you are ready to start coding and develop for PyVista, please take a look
at the remainder of the pages in this *Development Guide*.

## Licensing

All contributed code will be licensed under The MIT License found in the
repository. If you did not write the code yourself, it is your responsibility
to ensure that the existing license is compatible and included in the
contributed files or you can obtain permission from the original author to
relicense the code.


## Guidelines

Through direct access to the Visualization Toolkit (VTK) via direct array
access and intuitive Python properties, we hope to make the entire VTK library
easily accessible to researchers of all disciplines. To further PyVista towards
being the de facto Python interface to VTK, we need your help to make it even
better!

If you want to add one or two interesting analysis algorithms as filters,
implement a new plotting routine, or just fix 1-2 typos - your efforts are
welcome!


There are three general coding paradigms that we believe in:

1. **Make it intuitive**. PyVista's goal is to create an intuitive and easy
   to use interface back to the VTK library. Any new features should have
   intuitive naming conventions and explicit keyword arguments for users to
   make the bulk of the library accessible to novice users.

2. **Document everything!** At the least, include a docstring for any method
   or class added. Do not describe what you are doing but why you are doing
   it and provide a for simple use cases for the new features.

3. **Keep it tested**. We aim for a high test coverage. See
   testing for more details.



There are two important copyright guidelines:

4. Please do not include any data sets for which a license is not available
   or commercial use is prohibited. Those can undermine the license of
   the whole projects.

5. Do not use code snippets for which a license is not available (e.g. from
   stackoverflow) or commercial use is prohibited. Those can undermine
   the license of the whole projects.

Please also take a look at our [Code of Conduct](https://github.com/pyvista/pyvista/blob/master/CODE_OF_CONDUCT.md)

## Testing

After making changes, please test changes locally before creating a pull
request. The following tests will be executed after any commit or pull request,
so we ask that you perform the following sequence locally to track down any new
issues from your changes.
To run our comprehensive suite of unit tests, install all the dependencies
listed in ``requirements.txt``, ``requirements_docs.txt``:


```bash
pip install -r requirements.txt
pip install -r requirements_docs.txt
```

Then, if you have everything installed, you can run the various test suites:


Run the primary test suite and generate coverage report:

```bash
python -m pytest -v --cov pyvista
```

Run all code examples in the docstrings:

```bash
python -m pytest -v --doctest-modules pyvista
```


And finally, test the documentation examples:

```bash
cd docs
make clean
make doctest
make html -b linkcheck
```

The finished documentation can be found in the `docs/_build/html` directory.


## Release Checklist

The following is a checklist to complete before taging a release.
This helps us ensure everything goes smoothely before our army of robots send
PyVista out to PyPI and Conda-Forge for deployment. Please checkout the
latest state of the `master` branch locally and:

1. Run all tests as outlined in the [Testing Section](#testing) and ensure
all are passing.

2. Ensure all builds and tests on Travis and Azure are passing for the
latest commits on the `master` branch

3. Locally test and build the documentation with link checking to make sure
no links are outdated. Be sure to run `make clean` to ensure no results are
cached.
    ```bash
    cd docs
    make clean  # deletes the sphinx-gallery cache
    make doctest
    make html -b linkcheck
    ```

4. After building the documentation, open the local build and examine the
examples gallery for any obvious issues. If issues are present, abort the
release.


5. Update the version numbers in `pyvista/_version.py`, commit those
changes, and then tag with the same version. Push the changes *and* the tags:
    ```bash
    git push origin master
    git push origin --tags
    ```

6. Create a list of all changes for the release. It is often helpful to
leverage [GitHub's *compare* feature](https://github.com/pyvista/pyvista/compare)
to see the differences from the last tag and the `master` branch.
Be sure to acknowledge new contributors by their GitHub username and place
mentions where appropriate if a specific contributor is to thank for a new
feature.

7. Place your release notes from step 6 in the description for
[the new release on GitHub](https://github.com/pyvista/pyvista/releases/new)

8. Go grab a beer/coffee and wait for
[@regro-cf-autotick-bot](https://github.com/regro-cf-autotick-bot) to open a
pull reequest on the conda-forge
[PyVista feedstock](https://github.com/conda-forge/pyvista-feedstock).
Merge that pull request.

9. Announce the new release in the PyVista Slack workspace and celebrate!
