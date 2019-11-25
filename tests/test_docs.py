import pytest
from pyvista.utilities import copy_function_doc



class Foo:
    def goo(self, *args):
        """This has some documentation.

        Parameters
        ----------
        arg0 : int
            This is an argument.

        """
        pass

    @copy_function_doc(goo, alias=True)
    def goo_alias(self, *args):
        pass

    @copy_function_doc(goo, deprecated=True)
    def goo_depr(self, *args):
        pass


def test_copy_function_doc():
    """Test copying function documentation."""

    def blank():
        pass

    def foo():
        """Single line."""
        pass

    def foo2():
        """Multiple.

        Lines.
        """
        pass

    @copy_function_doc(foo)
    def bar():
        pass

    @copy_function_doc(foo)
    def bar2():
        """Single line."""
        pass

    @copy_function_doc(foo2)
    def bar3():
        """Single line."""
        pass

    @copy_function_doc(foo2)
    def bar4():
        """Multiple.

        Lines.
        """
        pass

    @copy_function_doc(foo, deprecated=True)
    def dep_fun():
        pass

    @copy_function_doc(foo2, alias=True)
    def alias_fun():
        """Multiple.

        Lines.
        """
        pass

    with pytest.raises(ValueError):
        @copy_function_doc(blank)
        def blank2():
            pass

    assert bar.__doc__ == foo.__doc__
    assert bar2.__doc__ == foo.__doc__ + '\nSingle line.'
    assert bar3.__doc__ == foo2.__doc__ + 'Single line.'
    assert bar4.__doc__ == foo2.__doc__ + foo2.__doc__
    assert dep_fun.__doc__ == foo.__doc__ + '\nDEPRECATED: Please use' + \
        ' ``foo`` instead.\n'
    assert alias_fun.__doc__ == bar4.__doc__ + 'Alias for: ``foo2``.\n'
    # Test a class
    assert Foo.goo_alias.__doc__ == Foo.goo.__doc__ + '\nAlias for: ``goo``.\n'
    assert Foo.goo_depr.__doc__ == Foo.goo.__doc__ + '\nDEPRECATED: Please use' + \
        ' ``goo`` instead.\n'
