{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

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
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :toctree:
   {% for item in attributes %}
      {% if name == 'Plotter' or item not in inherited_members %}
        {% if item.0 != item.upper().0 %}
          {{ name }}.{{ item }}
        {% endif %}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
