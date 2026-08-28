{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{# ``methods`` and ``attributes`` include inherited members, so this template asks
   pyvista/ext/_autoinherit.py which class each member's page belongs to: this one, a
   documented base, or -- for the members VTK implements -- nowhere at all. Only the
   members homed here get ``:toctree:``; the rest are listed with the class that defines
   them, so each member is documented once and no class is missing any of its API.

   The inherited tables are hand-built because ``autosummary`` renders an entry exactly
   as written, which cannot show ``_BaseMapper.bounds`` without also spelling out its
   module. ``enum.rst`` builds a table this way for the same reason. #}
{%- set documented_methods = methods | reject('in', skipmethods) | list %}
{%- set documented_attributes = attributes | reject('in', skipmethods) | list %}
{%- set own_methods = own_members(module, objname, documented_methods) %}
{%- set own_attributes = own_members(module, objname, documented_attributes) %}
{%- set inherited_methods = inherited_member_rows(module, objname, documented_methods) %}
{%- set inherited_attributes = inherited_member_rows(module, objname, documented_attributes) %}

{% block methods %}
{% if own_methods %}

{{ _('Methods') }}
{{ '-' * _('Methods')|length }}

.. autosummary::
   :toctree:
{% for item in own_methods %}
   {{ name }}.{{ item }}
{%- endfor %}
{% endif %}

{% if inherited_methods %}

{{ _('Inherited Methods') }}
{{ '-' * _('Inherited Methods')|length }}

.. list-table::
   :class: autosummary longtable
   :widths: 10 90

{% for label, target, summary in inherited_methods %}
   * - :py:obj:`{{ label }} <{{ target }}>`
     - {{ summary }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes %}
{% if own_attributes %}

{{ _('Attributes') }}
{{ '-' * _('Attributes')|length }}

.. autosummary::
   :toctree:
{% for item in own_attributes %}
   {{ name }}.{{ item }}
{%- endfor %}
{% endif %}

{% if inherited_attributes %}

{{ _('Inherited Attributes') }}
{{ '-' * _('Inherited Attributes')|length }}

.. list-table::
   :class: autosummary longtable
   :widths: 10 90

{% for label, target, summary in inherited_attributes %}
   * - :py:obj:`{{ label }} <{{ target }}>`
     - {{ summary }}
{%- endfor %}
{% endif %}
{% endblock %}
