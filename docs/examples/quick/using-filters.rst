
Using Common Filters
~~~~~~~~~~~~~~~~~~~~

``vtki`` wrapped data objects have a suite of common filters ready for immediate
use directly on the object (see :ref:`filters_ref`). These filters include:

* ``slice``: creates a single slice through the input dataset on a user defined plane
* ``slice_orthogonal``: creates a ``MultiBlock`` dataset of three orthogonal slices
* ``slice_along_axis``: creates a ``MultiBlock`` dataset of many slices along a specified axis
* ``threshold``: Thresholds a dataset by a single value or range of values
* ``threshold_percent``: Threshold by percentages of the scalar range
* ``clip``: Clips the dataset by a user defined plane
* ``outline_corners``: Outlines the corners of the data extent
* ``extract_geometry``: Extract surface geometry


To use these filters, call the method of your choice directly on your data object:


.. testcode:: python

    from vtki import examples

    dataset = examples.load_uniform()

    # Apply a threshold over a data range
    result = dataset.threshold([300, 500])
