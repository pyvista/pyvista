"""Utilities for using pyvista with sphinx-gallery."""

import shutil

import pyvista


def _get_sg_image_scraper():
    """Return the callable scraper to be used by Sphinx-Gallery.

    It allows PyVista users to just use strings as they already can for
    'matplotlib' and 'mayavi'. Details on this implementation can be found in
    `sphinx-gallery/sphinx-gallery/494`_

    .. _sphinx-gallery/sphinx-gallery/494: https://github.com/sphinx-gallery/sphinx-gallery/pull/494
    """
    return Scraper()


class Scraper:
    """
    Save ``pyvista.Plotter`` objects.

    Used by sphinx-gallery to generate the plots from the code in the examples.

    Pass an instance of this class to ``sphinx_gallery_conf`` in your
    ``conf.py`` as the ``"image_scrapers"`` argument.

    Be sure to set ``pyvista.BUILDING_GALLERY = True`` in your ``conf.py``.

    """

    def __call__(self, block, block_vars, gallery_conf):
        """Save the figures generated after running example code.

        Called by sphinx-gallery.

        """
        try:
            from sphinx_gallery.scrapers import figure_rst
        except ImportError:  # pragma: no cover
            raise ImportError('You must install `sphinx_gallery` to use this feature.')

        if not pyvista.BUILDING_GALLERY:
            raise RuntimeError(
                'pyvista.BUILDING_GALLERY must be set to True in your conf.py to capture '
                'images within sphinx_gallery or when building documentation using the '
                'pyvista-plot directive.'
            )

        image_names = list()
        image_path_iterator = block_vars["image_path_iterator"]
        figures = pyvista.plotting._ALL_PLOTTERS
        for _, plotter in figures.items():
            fname = next(image_path_iterator)
            if hasattr(plotter, '_gif_filename'):
                # move gif to fname
                shutil.move(plotter._gif_filename, fname)
            else:
                plotter.screenshot(fname)
            image_names.append(fname)
        pyvista.close_all()  # close and clear all plotters
        return figure_rst(image_names, gallery_conf["src_dir"])
