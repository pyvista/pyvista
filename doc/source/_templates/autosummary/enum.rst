{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoenum:: {{ objname }}

{% block methods %}
{% if methods %}

{{ _('Methods') }}
{{ '-' * _('Methods')|length }}

.. autosummary::
   :toctree:
{% for item in methods %}
   {% if not item in skipmethods %}
     {% if item not in inherited_members %}
       {{ name }}.{{ item }}
     {% endif %}
   {% endif %}
{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}

{{ _('Attributes') }}
{{ '-' * _('Attributes')|length }}

.. autosummary::
   :toctree:
{% for item in attributes %}
   {% if item not in inherited_members %}
     {% if item.0 != item.upper().0 %}
       {{ name }}.{{ item }}
     {% endif %}
   {% endif %}
{%- endfor %}
{% endif %}
{% endblock %}
