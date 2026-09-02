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
   module. ``enum.rst`` builds a table this way for the same reason.

   The inheritance section leads. It names the classes the sections below link to, not
   every base: an undocumented base has no page to point at, so it says where the
   inherited members are documented rather than claiming to be the full hierarchy. The
   VTK class the page's own class wraps is linked out to VTK's docs by the ``:vtk:`` role.
   Its links to the member sections are implicit references to their titles, so it may
   only name a section this page actually renders.

   Filters get their own section at the end: every dataset mixes in the filter classes,
   and for ``PolyData`` they are 142 of the 226 members it inherits. Attributes come
   before the three method sections so those stay together, ending with those filters. #}
{%- set documented_methods = methods | reject('in', skipmethods) | list %}
{%- set documented_attributes = attributes | reject('in', skipmethods) | list %}
{%- set own_methods = own_members(module, objname, documented_methods) %}
{%- set own_attributes = own_members(module, objname, documented_attributes) %}
{%- set inherited_methods = inherited_member_rows(module, objname, documented_methods) %}
{%- set inherited_attributes = inherited_member_rows(module, objname, documented_attributes) %}
{%- set filters = filter_member_rows(module, objname, documented_methods + documented_attributes) %}
{%- set base_classes = inherited_classes(module, objname) %}
{%- set vtk_classes = vtk_bases(module, objname) %}

{% block inheritance %}
{%- set member_sections = [
      _('Inherited Attributes') if inherited_attributes else '',
      _('Inherited Methods') if inherited_methods else '',
      _('Filters') if filters else '',
   ] | select | list %}
{% if base_classes or vtk_classes %}

{{ _('Inheritance') }}
{{ '-' * _('Inheritance')|length }}
{% if base_classes %}
Inherited members are documented on {% for item in base_classes[:-1] %}:py:obj:`~{{ item }}`, {% endfor %}:py:obj:`~{{ base_classes[-1] }}`.
{% endif %}
{% if member_sections %}
See them all under {% for section in member_sections[:-1] %}`{{ section }}`_{{ ' and ' if loop.last else ', ' }}{% endfor %}`{{ member_sections[-1] }}`_.
{% endif %}
{% if vtk_classes %}
Wraps {% for item in vtk_classes[:-1] %}:vtk:`{{ item }}`, {% endfor %}:vtk:`{{ vtk_classes[-1] }}`.
{% endif %}
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

{% block filters %}
{% if filters %}

{{ _('Filters') }}
{{ '-' * _('Filters')|length }}

.. list-table::
   :class: autosummary longtable
   :widths: 10 90

{% for label, target, summary in filters %}
   * - :py:obj:`{{ label }} <{{ target }}>`
     - {{ summary }}
{%- endfor %}
{% endif %}
{% endblock %}
