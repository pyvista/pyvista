{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoenum:: {{ objname }}

# Manually document enum properties since autoenum does not enable use of template variables
{% if objname == 'CellType' %}
{{ _('Attributes') }}
{{ '-' * _('Attributes')|length }}

.. autosummary::
   :toctree: _autosummary

   {{ objname }}.dimension
   {{ objname }}.is_linear
   {{ objname }}.is_primary
   {{ objname }}.n_points
   {{ objname }}.n_edges
   {{ objname }}.n_faces
   {{ objname }}.vtk_class

{% endif %}
