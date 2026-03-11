{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
{# autodoc does not document enum properties so we need to special case these #}
{% set is_celltype_enum = (objname == "CellType") %}
{% set celltype_properties = ["vtk_class", "dimension", "is_linear", "is_primary", "n_points", "n_edges", "n_faces"] %}

{% block methods %}
{% if methods %}

{{ _('Methods') }}
{{ '-' * _('Methods')|length }}

.. autosummary::
   :toctree:
{% for item in methods %}
   {% if not item in skipmethods %}
     {% if name == 'Plotter' or item not in inherited_members %}
       {{ name }}.{{ item }}
     {% endif %}
   {% endif %}
{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes or is_celltype_enum %}

{{ _('Attributes') }}
{{ '-' * _('Attributes')|length }}

.. autosummary::
   :toctree:
{% for item in attributes %}
   {% if name == 'Plotter' or name == 'DataSetMapper' or name == 'ImageData' or item not in inherited_members %}
     {% if item.0 != item.upper().0 %}
       {{ name }}.{{ item }}
     {% endif %}
   {% endif %}
{% endfor %}

{# Special-case CellType properties #}
{% if is_celltype_enum %}
   {% for prop in celltype_properties %}
       {{ name }}.{{ prop }}
   {% endfor %}
{% endif %}

{% endif %}
{% endblock %}
