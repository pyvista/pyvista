{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree: _autosummary
   {% for item in methods %}
      {% if item != "__init__" %}
      {{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :toctree: _autosummary
   {% for item in attributes %}
      {% if item.isupper() %}
      {{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
