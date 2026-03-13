{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoenum:: {{ objname }}

{# Manually document enum properties since autoenum does not enable use of template variables #}
{% if objname == 'CellType' %}
{{ _('Attributes') }}
{{ '-' * _('Attributes')|length }}

.. autosummary::
   :toctree: _autosummary

   CellType.dimension
   CellType.is_linear
   CellType.is_composite
   CellType.n_points
   CellType.n_edges
   CellType.n_faces
   CellType.vtk_class

{% endif %}
