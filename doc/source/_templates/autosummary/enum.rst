{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoenum:: {{ objname }}

{# autosummary does not populate ``attributes`` for the ``enum`` objtype the way it does
   for ``class`` (see pyvista/ext/_autoenum.py), so both lists below come from our own
   introspection helpers instead, via ``autosummary_context`` in conf.py. #}
{% set properties = instance_property_names(module, objname) %}
{% set class_properties = metaclass_property_descriptions(module, objname) %}
{% if properties %}
{{ _('Attributes') }}
{{ '-' * _('Attributes')|length }}

.. autosummary::
   :toctree:
{% for item in properties %}
   {{ objname }}.{{ item }}
{%- endfor %}

{% endif %}
{% if class_properties %}
{{ _('Class Attributes') }}
{{ '-' * _('Class Attributes')|length }}

{# The autosummary directive gets each entry's description the same eagerly-evaluated way
   it gets everything else about a metaclass property wrong, so this is a hand-written table
   instead, using metaclass_property_descriptions(). The ``.. only::`` block below still
   generates each one's page (autosummary_generate scans the raw text for ``:toctree:``,
   regardless of ``only::``) -- it's just not rendered here, since its own table would have
   the same blank descriptions this table exists to avoid. ``never`` is never a defined tag,
   so the block is always excluded. #}
.. only:: never

   .. autosummary::
      :toctree:
      :template: metaclassproperty
{% for item, description in class_properties %}
      {{ objname }}.{{ item }}
{%- endfor %}

.. list-table::
   :class: autosummary longtable
   :widths: 10 90

{% for item, description in class_properties %}
   * - :py:obj:`{{ objname }}.{{ item }} <{{ module }}.{{ objname }}.{{ item }}>`
     - {{ description }}
{%- endfor %}

{% endif %}
