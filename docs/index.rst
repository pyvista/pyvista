.. title:: PyVista

.. jupyter-execute::
   :hide-code:

   from pyvista.demos import logo
   logo._for_landing_page(height='200px')

.. raw:: html

    <div class="banner">
        <h2>3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK)</h2>
        <a href="./examples/index.html"><img src="_static/pyvista_banner_small.png" alt="pyvista" width="100%"/></a>
    </div>


PyVista is...

* *"VTK for humans"*: a high-level API to the `Visualization Toolkit`_ (VTK)
* mesh data structures and filtering methods for spatial datasets
* 3D plotting made simple and built for large/complex data geometries

.. _Visualization Toolkit: https://vtk.org


PyVista (formerly ``vtki``) is a helper module for the Visualization Toolkit
(VTK) that takes a different approach on interfacing with VTK through NumPy and
direct array access.
This package provides a Pythonic, well-documented interface exposing
VTK's powerful visualization backend to facilitate rapid prototyping, analysis,
and visual integration of spatially referenced datasets.

This module can be used for scientific plotting for presentations and research
papers as well as a supporting module for other mesh dependent Python modules.

.. |tweet| image:: https://img.shields.io/twitter/url.svg?style=social&url=http%3A%2F%2Fshields.io
   :target: https://twitter.com/intent/tweet?text=Check%20out%20this%20project%20for%203D%20visualization%20in%20Python&url=https://github.com/pyvista/pyvista&hashtags=3D,visualization,Python,vtk,mesh,plotting,PyVista

Share this project on Twitter: |tweet|


.. |binder| image:: https://static.mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/pyvista/pyvista-examples/master
   :alt: Launch on Binder

Want to test-drive PyVista? Check out our live examples on MyBinder: |binder|


.. toctree::
   :hidden:

   getting-started/index
   user-guide/index
   examples/index
   api
   extras/index



Translating
***********

The recommended way for new contributors to translate ``pyvista``'s documentation is to
join the translation team on Transifex.

There is a `pyvista translation page`_ for pyvista (master) documentation.

1. Login to transifex_ service.
2. Go to `pyvista translation page`_.
3. Click ``Request language`` and fill form.
4. Wait acceptance by transifex pyvista translation maintainers.
5. (After acceptance) Translate on transifex.
6. You can see the translated document in `Read The Docs`_.

Details can be found here: https://docs.transifex.com/getting-started-1/translators

.. _`pyvista translation page`: https://www.transifex.com/getfem-doc/pyvista-doc/
.. _Transifex: https://www.transifex.com/
.. _`Read The Docs`: https://pyvista-doc.readthedocs.io/en/latest


Status
******

.. |pypi| image:: https://img.shields.io/pypi/v/pyvista.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/pyvista/

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/pyvista.svg?logo=conda-forge&logoColor=white
   :target: https://anaconda.org/conda-forge/pyvista

.. |azure| image:: https://dev.azure.com/pyvista/PyVista/_apis/build/status/pyvista.pyvista?branchName=master
   :target: https://dev.azure.com/pyvista/PyVista/_build/latest?definitionId=3&branchName=master

.. |codecov| image:: https://codecov.io/gh/pyvista/pyvista/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/pyvista/pyvista

.. |codacy| image:: https://api.codacy.com/project/badge/Grade/e927f0afec7e4b51aeb7785847d0fd47
   :target: https://www.codacy.com/app/banesullivan/pyvista?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=akaszynski/pyvista&amp;utm_campaign=Badge_Grade

.. |contributors| image:: https://img.shields.io/github/contributors/pyvista/pyvista.svg?logo=github&logoColor=white
   :target: https://github.com/pyvista/pyvista/graphs/contributors/

.. |stars| image:: https://img.shields.io/github/stars/pyvista/pyvista.svg?style=social&label=Stars
   :target: https://github.com/pyvista/pyvista
   :alt: GitHub

.. |zenodo| image:: https://zenodo.org/badge/92974124.svg
   :target: https://zenodo.org/badge/latestdoi/92974124

.. |joss| image:: https://joss.theoj.org/papers/78f2901bbdfbd2a6070ec41e8282d978/status.svg
   :target: https://joss.theoj.org/papers/10.21105/joss.01450

.. |MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

.. |slack| image:: https://img.shields.io/badge/Slack-PyVista-green.svg?logo=slack
   :target: http://slack.pyvista.org

.. |gitter| image:: https://img.shields.io/gitter/room/pyvista/community?color=darkviolet
   :target: https://gitter.im/pyvista/community


+----------------------+------------------------+
| Deployment           | |pypi| |conda|         |
+----------------------+------------------------+
| Build Status         | |azure|                |
+----------------------+------------------------+
| Metrics              | |codacy| |codecov|     |
+----------------------+------------------------+
| GitHub               | |contributors| |stars| |
+----------------------+------------------------+
| Citation             | |joss| |zenodo|        |
+----------------------+------------------------+
| License              | |MIT|                  |
+----------------------+------------------------+
| Community            | |slack| |gitter|       |
+----------------------+------------------------+


Project Index
*************

* :ref:`genindex`
