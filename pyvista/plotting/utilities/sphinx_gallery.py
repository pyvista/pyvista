"""Utilities for using pyvista with sphinx-gallery."""

import shutil

import pyvista

BUILDING_GALLERY_ERROR_MSG = (
    "pyvista.BUILDING_GALLERY must be set to True in your conf.py to capture "
    "images within sphinx_gallery or when building documentation using the "
    "pyvista-plot directive."
)


def _get_sg_image_scraper():
    """Return the callable scraper to be used by Sphinx-Gallery.

    It allows PyVista users to just use strings as they already can for
    'matplotlib' and 'mayavi'. Details on this implementation can be found in
    `sphinx-gallery/sphinx-gallery/494`_

    This must be imported into the top level namespace of PyVista.

    .. _sphinx-gallery/sphinx-gallery/494: https://github.com/sphinx-gallery/sphinx-gallery/pull/494
    """
    return Scraper()


def html_rst(
    figure_list, sources_dir, fig_titles='', srcsetpaths=None
):  # pragma: no cover  # numpydoc ignore=PR01,RT01
    """Generate reST for viewer with exported scene."""
    from sphinx_gallery.scrapers import _get_srcset_st, figure_rst

    if srcsetpaths is None:
        # this should never happen, but figure_rst is public, so
        # this has to be a kwarg...
        srcsetpaths = [{0: fl} for fl in figure_list]

    images_rst = ""
    for i, hinnames in enumerate(srcsetpaths):
        srcset = _get_srcset_st(sources_dir, hinnames)
        if srcset[-5:] == "vtksz":
            png_file = figure_list[i][:-5] + "png"

            indented_firgure_rst = "\n".join(
                " " * 5 + line for line in figure_rst([png_file], sources_dir).split("\n")
            )
            images_rst += f"""
\n
\n
.. tab-set::\n
\n
   .. tab-item:: Static Scene\n
\n
       {indented_firgure_rst}
\n
   .. tab-item:: Interactive Scene\n
\n
       .. offlineviewer:: {figure_list[i]}\n\n"""

        else:
            images_rst += "\n" + figure_rst([figure_list[i]], sources_dir) + "\n\n"

    return images_rst


def _process_events_before_scraping(plotter):
    """Process events such as changing the camera or an object before scraping."""
    if plotter.iren is not None and plotter.iren.initialized:
        # check for pyvistaqt app which can be specifically bound to pyvista plotter
        # objects in order to interact with qt, then process the events from qt
        if hasattr(plotter, "app") and plotter.app is not None:
            plotter.app.processEvents()
        plotter.update()


class Scraper:
    """
    Save ``pyvista.Plotter`` objects.

    Used by sphinx-gallery to generate the plots from the code in the examples.

    Pass an instance of this class to ``sphinx_gallery_conf`` in your
    ``conf.py`` as the ``"image_scrapers"`` argument.

    Be sure to set ``pyvista.BUILDING_GALLERY = True`` in your ``conf.py``.

    """

    def __repr__(self):
        """Return a stable representation of the class instance."""
        return f"<{type(self).__name__} object>"

    def __call__(self, block, block_vars, gallery_conf):
        """Save the figures generated after running example code.

        Called by sphinx-gallery.

        """
        from sphinx_gallery.scrapers import figure_rst

        if not pyvista.BUILDING_GALLERY:
            raise RuntimeError(BUILDING_GALLERY_ERROR_MSG)

        image_names = list()
        image_path_iterator = block_vars["image_path_iterator"]
        figures = pyvista.plotting.plotter._ALL_PLOTTERS
        for plotter in figures.values():
            _process_events_before_scraping(plotter)
            fname = next(image_path_iterator)
            if hasattr(plotter, "_gif_filename"):
                # move gif to fname
                fname = fname[:-3] + "gif"
                shutil.move(plotter._gif_filename, fname)
            else:
                plotter.screenshot(fname)
            image_names.append(fname)
        pyvista.close_all()  # close and clear all plotters
        return figure_rst(image_names, gallery_conf["src_dir"])


class DynamicScraper:  # pragma: no cover
    """
    Save ``pyvista.Plotter`` objects dynamically.

    Used by sphinx-gallery to generate the plots from the code in the examples.

    Pass an instance of this class to ``sphinx_gallery_conf`` in your
    ``conf.py`` as the ``"image_scrapers"`` argument.

    Be sure to set ``pyvista.BUILDING_GALLERY = True`` in your ``conf.py``.

    If the boolean variable ``PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True/False`` is set as a global
    variable in the document then its value will be used as default for the
    force_static argument of the pyvista-plot command. see also the notes at :func:plot_directive

    To alter the global value behavior just for some plots you may set the
    boolean variable ``PYVISTA_GALLERY_FORCE_STATIC = True``/
    ``PYVISTA_GALLERY_FORCE_STATIC = False`` just before the appropriate ``plot`` command.

    The default behavior of this scraper is to create interactive plots.

    """

    def __repr__(self):
        """Return a stable representation of the class instance."""
        return f"<{type(self).__name__} object>"

    def __call__(self, block, block_vars, gallery_conf):  # pragma: no cover
        """Save the figures generated after running example code.

        Called by sphinx-gallery.

        """
        if not pyvista.BUILDING_GALLERY:
            raise RuntimeError(BUILDING_GALLERY_ERROR_MSG)

        image_names = list()
        image_path_iterator = block_vars["image_path_iterator"]
        figures = pyvista.plotting.plotter._ALL_PLOTTERS
        # read global option  if it exists
        force_static = block_vars['example_globals'].get(
            "PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT", False
        )
        # override with block specific value if it exists
        if "PYVISTA_GALLERY_FORCE_STATIC = True" in block[1].split('\n'):
            force_static = True
        elif "PYVISTA_GALLERY_FORCE_STATIC = False" in block[1].split('\n'):
            force_static = False
        for plotter in figures.values():
            _process_events_before_scraping(plotter)
            fname = next(image_path_iterator)
            if hasattr(plotter, '_gif_filename'):
                fname = fname[:-3] + "gif"
                shutil.move(plotter._gif_filename, fname)
                image_names.append(fname)
            else:
                plotter.screenshot(fname)  # produce PNG for thumbnail
                if force_static or plotter.last_vtksz is None:
                    image_names.append(fname)
                else:
                    fname = fname[:-3] + "vtksz"
                    with open(fname, "wb") as f:
                        f.write(plotter.last_vtksz)
                        image_names.append(fname)
        pyvista.close_all()  # close and clear all plotters
        return html_rst(image_names, gallery_conf["src_dir"])
