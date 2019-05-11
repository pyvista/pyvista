.. _guidelines_ref:

Guidelines
==========

Through direct access to the Visualization Toolkit (VTK) via direct array
access and intuitive Python properties, we hope to make the entire VTK library
easily accessible to researchers of all disciplines. To further PyVista towards
being the de facto Python interface to VTK, we need your help to make it even
better!

If you want to add one or two interesting analysis algorithms as filters,
implement a new plotting routine, or just fix 1-2 typos - your efforts are
welcome!


There are three general coding paradigms that we believe in:

    1. **Make it intuitive**. PyVista's goal is to create an intuitive and easy
       to use interface back to the VTK library. Any new features should have
       intuitive naming conventions and explicit keyword arguments for users to
       make the bulk of the library accessible to novice users.

    2. **Document everything!** At the least, include a docstring for any method
       or class added. Do not describe what you are doing but why you are doing
       it and provide a for simple use cases for the new features.

    3. **Keep it tested**. We aim for a high test coverage. See
       :ref:`testing_ref` for more details.



There are two important copyright guidelines:

    4. Please do not include any data sets for which a license is not available
       or commercial use is prohibited. Those can undermine the license of
       the whole projects.

    5. Do not use code snippets for which a license is not available (e.g. from
       stackoverflow) or commercial use is prohibited. Those can undermine
       the license of the whole projects.
