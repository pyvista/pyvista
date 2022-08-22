import datetime

from packaging.version import parse as parse_version
import sphinx

import pyvista

# -- General configuration ------------------------------------------------

extensions = ['pyvista.ext.plot_directive']
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = 'tinypages'
year = datetime.date.today().year
copyright = f"2021-{year}, PyVista developers"
version = '0.1'
release = '0.1'
exclude_patterns = ['_build']
pygments_style = 'sphinx'

# -- Plot directive specific configuration --------------------------------
plot_setup = plot_cleanup = 'import pyvista'

# -- Options for HTML output ----------------------------------------------

if parse_version(sphinx.__version__) >= parse_version('1.3'):
    html_theme = 'classic'
else:
    html_theme = 'default'

html_static_path = ['_static']

# -- pyvista configuration ------------------------------------------------
pyvista.BUILDING_GALLERY = True
