"""Regression testing."""

import os
import re
import warnings

import pyvista


class VerifyImageCache:
    """Control image caching for testing.

    Image cache files are names according to ``test_name``.
    Multiple calls to an instance of this class will append
    `_X` to the name after the first one.  That is, files
    ``{test_name}``, ``{test_name}_1``, and ``{test_name}_2``
    will be saved if called 3 times.

    Parameters
    ----------
    test_name : str
        Name of test to save.  Sets name of image cache file.

    """

    reset_image_cache = False
    ignore_image_cache = False
    fail_extra_image_cache = False

    def __init__(
        self,
        test_name,
        cache_dir,
        *,
        error_value=500,
        warning_value=200,
        var_error_value=1000,
        var_warning_value=1000,
        high_variance_tests=None,
        skip_tests=None,
    ):
        """Init VerifyImageCache."""
        self.test_name = test_name

        self.cache_dir = cache_dir

        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

        self.error_value = error_value
        self.warning_value = warning_value

        self.var_error_value = var_error_value
        self.var_warning_value = var_warning_value

        if high_variance_tests is not None:
            self.high_variance_tester = re.compile("|".join(high_variance_tests))
        else:
            self.high_variance_tester = None

        if skip_tests is not None:
            self.skip_tester = re.compile("|".join(skip_tests))
        else:
            self.skip_tester = None

        self.skip = False
        self.n_calls = 0

    def __call__(self, plotter):
        """Either store or validate an image.

        Parameters
        ----------
        plotter : pyvista.Plotter
            The Plotter object that is being closed.

        """
        if self.skip:
            return

        if self.ignore_image_cache:
            return

        if self.skip_tester is not None and self.skip_tester.search(self.test_name):
            return

        if self.n_calls > 0:
            test_name = f"{self.test_name}_{self.n_calls}"
        else:
            test_name = self.test_name
        self.n_calls += 1

        if self.high_variance_tester is not None and self.high_variance_tester.search(
            self.test_name
        ):
            allowed_error = self.var_error_value
            allowed_warning = self.var_warning_value
        else:
            allowed_error = self.error_value
            allowed_warning = self.warning_value

        # cached image name
        image_filename = os.path.join(self.cache_dir, test_name[5:] + '.png')

        if not os.path.isfile(image_filename) and self.fail_extra_image_cache:
            raise RuntimeError(f"{image_filename} does not exist in image cache")
        # simply save the last screenshot if it doesn't exist or the cache
        # is being reset.
        if self.reset_image_cache or not os.path.isfile(image_filename):
            return plotter.screenshot(image_filename)

        # otherwise, compare with the existing cached image
        error = pyvista.compare_images(image_filename, plotter)
        if error > allowed_error:
            raise RuntimeError(
                f'{test_name} Exceeded image regression error of '
                f'{allowed_error} with an image error of '
                f'{error}'
            )
        if error > allowed_warning:
            warnings.warn(
                f'{test_name} Exceeded image regression warning of '
                f'{allowed_warning} with an image error of '
                f'{error}'
            )
