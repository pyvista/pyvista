{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoenum:: {{ objname }}

{# No `:toctree:`: each property is already documented in full inline by EnumDocumenter
   (pyvista/ext/_autoenum.py), so this just links to that in-page entry. #}
{% set properties = metaclass_property_names(module, objname) %}
{% if properties %}
{{ _('Attributes') }}
{{ '-' * _('Attributes')|length }}

.. autosummary::
{% for item in properties %}
   {{ objname }}.{{ item }}
{%- endfor %}

{% endif %}
