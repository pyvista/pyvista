from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from IPython.core.guarded_eval import EVALUATION_POLICIES
from IPython.core.guarded_eval import EvaluationContext
from IPython.core.guarded_eval import guarded_eval
import numpy as np
import pytest

import pyvista as pv
from pyvista.core.utilities.misc import _allow_ipython_completion

LIMITED = EVALUATION_POLICIES['limited']


class Mesh(pv.PolyData):
    """Defined after IPython is imported, so class creation registers it."""


class Blocks(pv.MultiBlock):
    """Defined after IPython is imported, so class creation registers it."""


@pytest.fixture
def namespace():
    # pytest imported pyvista before IPython, so register the classes created then
    _allow_ipython_completion(pv.DataSetAttributes)
    _allow_ipython_completion(pv.pyvista_ndarray)
    mesh = Mesh(pv.Sphere())
    mesh['scalars'] = np.arange(mesh.n_points)
    return {'mesh': mesh, 'blocks': Blocks({'a': mesh})}


@pytest.mark.parametrize('cls', [Mesh, Blocks])
def test_subclass_registered(cls):
    assert cls in LIMITED.allowed_getattr
    assert cls in LIMITED.allowed_getitem


def test_subclass_not_registered_without_ipython(monkeypatch):
    monkeypatch.delitem(sys.modules, 'IPython')

    class Unregistered(pv.PolyData): ...

    assert Unregistered not in LIMITED.allowed_getattr
    assert Unregistered not in LIMITED.allowed_getitem


@pytest.mark.parametrize(
    'expr',
    [
        'mesh.n_points',
        'mesh.points.shape',
        'mesh.point_data',
        "mesh['scalars'].dtype",
        "mesh.point_data['scalars'].shape",
        "blocks['a'].active_scalars_name",
        'blocks[0].n_points',
    ],
)
def test_guarded_eval_limited(namespace, expr):
    context = EvaluationContext(globals={}, locals=namespace, evaluation='limited')
    assert guarded_eval(expr, context) == eval(expr, {}, namespace)  # noqa: S307


def test_dataset_attributes_key_completions(namespace):
    point_data = namespace['mesh'].point_data
    assert point_data._ipython_key_completions_() == ['Normals', 'scalars']


def test_ipython_completer(tmp_path):
    code = textwrap.dedent(
        """
        import IPython
        import numpy as np
        import pyvista as pv
        import pyvista.plotting
        from IPython.core.completer import provisionalcompleter
        from IPython.core.guarded_eval import EVALUATION_POLICIES
        from IPython.core.interactiveshell import InteractiveShell
        from pyvista.core.utilities.misc import _NoNewAttrMixin

        def subclasses(cls):
            for sub in cls.__subclasses__():
                yield sub
                yield from subclasses(sub)

        allowed = EVALUATION_POLICIES['limited'].allowed_getattr
        missing = {cls for cls in subclasses(_NoNewAttrMixin) if cls not in allowed}
        assert not missing, missing

        ip = InteractiveShell.instance()
        mesh = pv.Sphere()
        mesh['scalars'] = np.arange(mesh.n_points)
        ip.user_ns['mesh'] = mesh

        def complete(text):
            with provisionalcompleter():
                completions = ip.Completer.completions(text, len(text))
                return {c.text.lstrip('.') for c in completions}

        assert 'scalars' in complete("mesh.point_data['")
        ip.Completer.use_jedi = False
        assert 'scalars' in complete("mesh.point_data['")
        assert 'active_scalars' in complete('mesh.point_data.act')
        assert 'shape' in complete("mesh.point_data['scalars'].sh")
        """
    )
    env = {**os.environ, 'IPYTHONDIR': str(tmp_path)}
    subprocess.run([sys.executable, '-c', code], check=True, env=env)
