{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoenum:: {{ objname }}

{# autosummary does not populate ``attributes`` for the ``enum`` objtype the way it does
   for ``class`` (see pyvista/ext/_autoenum.py), so both lists below come from our own
   introspection helpers instead, via ``autosummary_context`` in conf.py. #}
{% set properties = instance_property_names(module, objname) %}
{% set class_properties = metaclass_property_names(module, objname) %}
{% if properties or class_properties %}
{{ _('Attributes') }}
{{ '-' * _('Attributes')|length }}

{% if properties %}
.. autosummary::
   :toctree:
{% for item in properties %}
   {{ objname }}.{{ item }}
{%- endfor %}

{% endif %}
{% if class_properties %}
{# Metaclass properties (e.g. dimension_map) evaluate eagerly through a plain getattr,
   so the usual per-item dispatch can't tell they came from a property at all -- forcing
   ``:template:`` here routes every item through our own directive instead. #}
.. autosummary::
   :toctree:
   :template: metaclassproperty
{% for item in class_properties %}
   {{ objname }}.{{ item }}
{%- endfor %}

{% endif %}
{% endif %}
