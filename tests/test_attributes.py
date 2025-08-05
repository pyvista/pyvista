from __future__ import annotations

from enum import Enum
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import pyvista as pv
from pyvista.core._vtk_core import DisableVtkSnakeCase
from pyvista.core._vtk_core import VTKObjectWrapperCheckSnakeCase
from pyvista.core.errors import PyVistaAttributeError
from pyvista.core.errors import VTKVersionError
from pyvista.core.utilities.misc import _NoNewAttrMixin
from pyvista.plotting.charts import _vtkWrapper


def get_all_pyvista_classes() -> tuple[tuple[str, ...], tuple[type, ...]]:
    """Return all classes defined in the pyvista package."""
    class_types: set[type] = set()

    package_path = Path(pv.__path__[0])  # path to pyvista package

    def find_py_files(path: Path):
        return [p for p in path.rglob('*.py') if p.name != '__init__.py']

    core_py_files = find_py_files(package_path / 'core')
    plotting_py_files = find_py_files(package_path / 'plotting')

    for file_path in [*core_py_files, *plotting_py_files]:
        # Convert path to dotted module name
        rel_path = file_path.relative_to(package_path.parent)
        module_name = '.'.join(rel_path.with_suffix('').parts)

        module = importlib.import_module(module_name)

        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and getattr(obj, '__module__', '') == module_name:
                class_types.add(obj)

    # Sort and return names and types as separate tuples
    sorted_classes = sorted(class_types, key=lambda cls: cls.__name__)
    return tuple(cls.__name__ for cls in sorted_classes), tuple(sorted_classes)


def pytest_generate_tests(metafunc):
    """Generate parametrized tests."""
    if 'vtk_subclass' in metafunc.fixturenames:
        # This test automatically collects all pyvista classes that _inherit_ from
        # vtk classes. For classes that wrap vtk classes through composition, we
        # manually add these here
        VTK_SUBCLASS_BY_COMPOSITION: set[type] = {pv.pyvista_ndarray}

        def get_vtk_subclasses_through_composition() -> dict[str, type]:
            return {cls.__name__: cls for cls in VTK_SUBCLASS_BY_COMPOSITION}

        def get_vtk_subclasses_through_inheritance() -> dict[str, type]:
            class_names, class_types = get_all_pyvista_classes()

            def inherits_from_vtk(klass):
                bases = klass.__mro__[1:]
                return any(base.__name__[:3].lower() == 'vtk' for base in bases)

            inherits_from_vtk = {
                name: cls
                for name, cls in zip(class_names, class_types, strict=True)
                if inherits_from_vtk(cls)
            }
            assert inherits_from_vtk
            return inherits_from_vtk

        inheritance = get_vtk_subclasses_through_inheritance()
        composition = get_vtk_subclasses_through_composition()
        inheritance.update(composition)

        names = inheritance.keys()
        types = inheritance.values()
        metafunc.parametrize('vtk_subclass', types, ids=names)

    if 'pyvista_class' in metafunc.fixturenames:
        class_names, class_types = get_all_pyvista_classes()

        SKIP_SUBCLASS = {Warning, Exception, tuple, Enum}

        class_map = {
            name: cls
            for name, cls in zip(class_names, class_types, strict=True)
            if not name.startswith('_') and not issubclass(cls, tuple(SKIP_SUBCLASS))
        }
        metafunc.parametrize('pyvista_class', list(class_map.values()), ids=list(class_map.keys()))


def try_init_pyvista_object(class_):
    # Init object but skip if abstract
    kwargs = get_default_class_init_kwargs(class_)
    try:
        instance = class_(**kwargs)
    except (ImportError, VTKVersionError):
        pytest.skip('VTK Version not supported.')
    except TypeError as e:
        if 'abstract' in repr(e):
            pytest.skip('Class is abstract.')
        raise
    return instance


@pytest.mark.needs_vtk_version(9, 2)
def get_default_class_init_kwargs(pyvista_class):
    # Define kwargs as required for initializing some classes
    kwargs = {}
    if pyvista_class is pv.CubeAxesActor:
        kwargs['camera'] = pv.Camera()
    elif pyvista_class is pv.Renderer:
        kwargs['parent'] = pv.Plotter()
    elif pyvista_class in [pv.ChartBox, pv.ChartPie]:
        kwargs['data'] = list(range(10))
    elif pyvista_class is pv.CompositeAttributes:
        kwargs['mapper'] = pv.CompositePolyDataMapper()
        kwargs['dataset'] = pv.MultiBlock()
    elif pyvista_class is pv.CornerAnnotation:
        kwargs['position'] = 'top'
        kwargs['text'] = 'text'
    elif pyvista_class is pv.plotting.utilities.algorithms.ActiveScalarsAlgorithm:
        kwargs['name'] = 'name'
    elif pyvista_class is pv.charts.Charts:
        kwargs['renderer'] = pv.Renderer(pv.Plotter())
    elif pyvista_class is pv.charts.AreaPlot:
        kwargs['chart'] = pv.charts.Chart2D()
        kwargs['x'] = (0, 0, 0)
        kwargs['y1'] = (1, 0, 0)
    elif pyvista_class in [
        pv.charts.BarPlot,
        pv.charts.LinePlot2D,
        pv.charts.ScatterPlot2D,
    ]:
        kwargs['chart'] = pv.charts.Chart2D()
        kwargs['x'] = (0, 0, 0)
        kwargs['y'] = (1, 0, 0)
    elif pyvista_class is pv.charts.StackPlot:
        kwargs['chart'] = pv.charts.Chart2D()
        kwargs['x'] = (0, 0, 0)
        kwargs['ys'] = (1, 0, 0)
    elif pyvista_class in [pv.charts.BoxPlot, pv.charts.PiePlot]:
        kwargs['chart'] = pv.charts.Chart2D()
        kwargs['data'] = [0, 0, 0]
    elif pyvista_class is pv.charts._ChartBackground:
        kwargs['chart'] = pv.charts.Chart2D()
    elif pyvista_class is pv.plotting.background_renderer.BackgroundRenderer:
        kwargs['parent'] = pv.Plotter()
        kwargs['image_path'] = pv.examples.logofile
    elif pyvista_class is pv.pyvista_ndarray:
        kwargs['array'] = pv.vtk_points(np.eye(3)).GetData()
    elif pyvista_class is pv.ActorProperties:
        kwargs['properties'] = pv.Property()
    elif pyvista_class is pv.AffineWidget3D:
        kwargs['plotter'] = pv.Plotter()
        kwargs['actor'] = pv.Actor()
    elif pyvista_class is pv.BlockAttributes:
        dataset = pv.ImageData()
        kwargs['block'] = dataset
        kwargs['attr'] = pv.CompositeAttributes(
            pv.plotting.composite_mapper.CompositePolyDataMapper(), dataset
        )
    elif pyvista_class is pv.CameraPosition:
        kwargs['position'] = (1, 2, 3)
        kwargs['focal_point'] = (1, 2, 3)
        kwargs['viewup'] = (1, 2, 3)
    elif pyvista_class is pv.plotting.renderer.RenderPasses:
        kwargs['renderer'] = pv.Renderer(pv.Plotter())
    elif (
        pyvista_class is pv.plotting.plotter.Renderers
        or pyvista_class is pv.RenderWindowInteractor
        or pyvista_class is pv.plotting.plotter.ScalarBars
    ):
        kwargs['plotter'] = pv.Plotter()
    elif pyvista_class is pv.Timer:
        kwargs['max_steps'] = 0
        kwargs['callback'] = lambda x: x
    elif pyvista_class is pv.DataSetAttributes:
        dataset = pv.ImageData()
        kwargs['vtkobject'] = dataset.GetPointData()
        kwargs['dataset'] = dataset
        kwargs['association'] = pv.FieldAssociation.POINT
    elif pyvista_class is pv.ProgressMonitor:
        kwargs['algorithm'] = pv.plotting.utilities.algorithms.CrinkleAlgorithm
    elif pyvista_class is pv.plotting.lookup_table.lookup_table_ndarray:
        kwargs['array'] = pv.convert_array(np.array((1, 2, 3)))
    elif pyvista_class is pv.plotting.picking.RectangleSelection:
        kwargs['frustum'] = None
        kwargs['viewport'] = None
    elif issubclass(
        pyvista_class, pv.plotting.render_window_interactor.InteractorStyleCaptureMixin
    ):
        kwargs['render_window_interactor'] = pv.Plotter().iren
    elif issubclass(pyvista_class, pv.BaseReader):
        kwargs['path'] = __file__  # Dummy file to pass is_file() checks
    return kwargs


def test_vtk_snake_case_api_is_disabled(vtk_subclass):
    if vtk_subclass is VTKObjectWrapperCheckSnakeCase:
        pytest.skip('Class is effectively abstract.')

    assert pv.vtk_snake_case() == 'error'

    # Default test values for classes
    vtk_attr_camel_case = 'GetGlobalWarningDisplay'
    vtk_attr_snake_case = 'global_warning_display'

    if vtk_subclass is pv.pyvista_ndarray:
        vtk_attr_camel_case = 'GetName'
        vtk_attr_snake_case = 'name'

    instance = try_init_pyvista_object(vtk_subclass)

    # Make sure the CamelCase attribute exists and can be accessed
    assert hasattr(instance, vtk_attr_camel_case)

    if pv.vtk_version_info >= (9, 4):
        # Test getting the snake_case equivalent raises error

        try:
            getattr(instance, vtk_attr_snake_case)
        except PyVistaAttributeError as e:
            # Test passes, we want an error to be raised
            # Confirm error message is correct
            match = (
                f"The attribute '{vtk_attr_snake_case}' is defined by VTK and is not part of "
                f'the PyVista API'
            )
            assert match in repr(e)  # noqa: PT017
        else:
            if DisableVtkSnakeCase not in vtk_subclass.__mro__:
                msg = (
                    f'The class {vtk_subclass.__name__!r} in {vtk_subclass.__module__!r}\n'
                    f'must inherit from {DisableVtkSnakeCase.__name__!r} in '
                    f'{DisableVtkSnakeCase.__module__!r}'
                )
            else:
                msg = f'{PyVistaAttributeError.__name__} was NOT raised (but was expected).'
            pytest.fail(msg)
    else:
        assert not hasattr(instance, vtk_attr_snake_case)


def test_pyvista_class_no_new_attributes(pyvista_class):
    def skip_test_for_some_classes():
        if pyvista_class in (
            pv.MFIXReader,
            pv.Nek5000Reader,
            pv.XdmfReader,
            pv.PVDReader,
            pv.CGNSReader,
            pv.ExodusIIBlockSet,
        ):
            assert issubclass(pyvista_class, _NoNewAttrMixin)
            pytest.skip('Test fails without proper dataset files.')
        elif pyvista_class is pv.core.dataset.ActiveArrayInfo:
            pytest.skip('Deprecated.')
        elif pyvista_class in (pv.PVDDataSet, pv.core.utilities.cell_quality.CellQualityInfo):
            assert issubclass(pyvista_class, _NoNewAttrMixin)
            pytest.skip('Dataclass, no test required.')
        elif pyvista_class in (
            pv.core.utilities.misc.conditional_decorator,
            pv.plotting.utilities.sphinx_gallery.Scraper,
            pv.plotting.utilities.sphinx_gallery.DynamicScraper,
            pv.core._vtk_core.DisableVtkSnakeCase,
            pv.core._vtk_core.vtkPyVistaOverride,
            pv.core._vtk_core.VTKObjectWrapperCheckSnakeCase,
            pv.VtkErrorCatcher,
        ):
            assert not issubclass(pyvista_class, _NoNewAttrMixin)
            pytest.skip('Specialized class with no real risk of new attributes being added.')
        elif pyvista_class in [pv.charts.PiePlot, pv.charts.Pen, pv.charts.Brush, pv.charts.Axis]:
            assert not issubclass(pyvista_class, _NoNewAttrMixin)
            assert issubclass(pyvista_class, _vtkWrapper)
            pytest.skip(
                f'Parent {_vtkWrapper.__name__} is not compatible with {_NoNewAttrMixin.__name__}.'
            )

    skip_test_for_some_classes()
    instance = try_init_pyvista_object(pyvista_class)
    try:
        instance.new_attribute = 'foo'
    except PyVistaAttributeError as e:
        # Test passes, we want an error to be raised
        match = 'does not exist and cannot be added to class'
        assert match in repr(e)  # noqa: PT017
    except AttributeError as e:
        if 'dict' in repr(e):
            pytest.skip('Skip dict classes')
    else:
        if not issubclass(pyvista_class, _NoNewAttrMixin):
            msg = (
                f'The class {pyvista_class.__name__!r} in {pyvista_class.__module__!r}'
                f'must inherit from \n{_NoNewAttrMixin.__name__!r} '
                f'in {_NoNewAttrMixin.__module__!r}'
            )
        else:
            msg = f'{PyVistaAttributeError.__name__} was NOT raised (but was expected).'
        pytest.fail(msg)
