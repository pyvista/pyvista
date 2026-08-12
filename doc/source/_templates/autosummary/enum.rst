{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoenum:: {{ objname }}

{# autosummary does not populate `attributes` for the `enum` objtype the way it does for
   `class` (see pyvista/ext/autoenum.py), so the metaclass-property names below come from
   our own introspection helper instead, via `autosummary_context` in conf.py. This works
   for any enum -- unlike the per-class hardcoded list this template used to have.

   No `:toctree:` here, unlike class.rst's own Attributes/Methods blocks: autosummary's
   stub-page generation always resolves an item through a plain `getattr`, which would
   just reproduce the original bug for a metaclass property (see EnumDocumenter in
   pyvista/ext/autoenum.py, which already documents each one properly, in full, on this
   same page). This table links straight to that in-page entry instead of a separate page. #}
{% set properties = metaclass_property_names(module, objname) %}
{% if properties %}
{{ _('Attributes') }}
{{ '-' * _('Attributes')|length }}

.. autosummary::
{% for item in properties %}
   {{ objname }}.{{ item }}
{%- endfor %}

{% endif %}
