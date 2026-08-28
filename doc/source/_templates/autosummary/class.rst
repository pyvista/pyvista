{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{# ``methods`` and ``attributes`` include inherited members, so this template asks
   pyvista/ext/_autoinherit.py which class each member's page belongs to: this one, a
   documented base, or -- for the members VTK implements -- nowhere at all. Only the
   members homed here get ``:toctree:``; the rest are linked where they are documented,
   so each member is documented exactly once and no class is missing any of its API. #}
{%- set documented_methods = methods | reject('in', skipmethods) | list %}
{%- set documented_attributes = attributes | reject('in', skipmethods) | list %}
{%- set own_methods = own_members(module, objname, documented_methods) %}
{%- set own_attributes = own_members(module, objname, documented_attributes) %}
{%- set inherited = inherited_member_groups(module, objname, documented_methods + documented_attributes) %}

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
{% endblock %}

{% block inherited %}
{% if inherited %}

{{ _('Inherited Members') }}
{{ '-' * _('Inherited Members')|length }}

These are also available on {{ objname }}, and are documented with the class that
defines them.
{% for base_module, base_name, items in inherited %}
.. currentmodule:: {{ base_module }}

.. dropdown:: From :class:`~{{ base_module }}.{{ base_name }}` ({{ items | length }})

   .. autosummary::
{% for item in items %}
      ~{{ base_name }}.{{ item }}
{%- endfor %}
{% endfor %}
.. currentmodule:: {{ module }}
{% endif %}
{% endblock %}
