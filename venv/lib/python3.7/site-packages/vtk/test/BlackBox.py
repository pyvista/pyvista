from vtk.util import vtkMethodParser


class Tester:
    def __init__(self, debug=0):
        self.setDebug(debug)
        self.parser = vtkMethodParser.VtkDirMethodParser()
        self.obj = None

    def setDebug(self, val):
        """Sets debug value of the vtkMethodParser.  1 is verbose and
        0 is not.  0 is default."""
        vtkMethodParser.DEBUG = val

    def testParse(self, obj):
        """ Testing if the object is parseable."""
        self.parser.parse_methods(obj)
        self.obj = obj

    def testGetSet(self, obj, excluded_methods=[]):
        """ Testing Get/Set methods."""
        if obj != self.obj:
            self.testParse(obj)
        methods = self.parser.get_set_methods()
        toggle = [x[:-2] for x in self.parser.toggle_methods()]
        methods.extend(toggle)
        for method in methods:
            if method in excluded_methods:
                continue
            setm = "Set%s"%method
            getm = "Get%s"%method
            val = eval("obj.%s()"%getm)
            try:
                 eval("obj.%s"%setm)(*val)
            except TypeError:
                eval("obj.%s"%setm)(*(val,))

            val1 = eval("obj.%s()"%getm)

            if val1 != val:
                name = obj.GetClassName()
                msg = "Failed test for %(name)s.Get/Set%(method)s\n"\
                      "Before Set, value = %(val)s; "\
                      "After Set, value = %(val1)s"%locals()
                raise AssertionError(msg)

    def testBoolean(self, obj, excluded_methods=[]):
        """ Testing boolean (On/Off) methods."""
        if obj != self.obj:
            self.testParse(obj)
        methods = self.parser.toggle_methods()
        for method1 in methods:
            method = method1[:-2]

            if method in excluded_methods:
                continue

            getm = "Get%s"%method

            orig_val = eval("obj.%s()"%getm)

            # Turn on
            eval("obj.%sOn()"%method)
            val = eval("obj.%s()"%getm)

            if val != 1:
                name = obj.GetClassName()
                msg = "Failed test for %(name)s.%(method)sOn\n"\
                      "Result not equal to 1 "%locals()
                raise AssertionError(msg)

            # Turn on
            eval("obj.%sOff()"%method)
            val = eval("obj.%s()"%getm)

            if val != 0:
                name = obj.GetClassName()
                msg = "Failed test for %(name)s.%(method)sOff\n"\
                      "Result not equal to 0 "%locals()
                raise AssertionError(msg)

            # set the value back to the original value.
            eval("obj.Set%s(orig_val)"%method)


    def test(self, obj):
        """Test the given vtk object."""

        # first try parsing the object.
        self.testParse(obj)

        # test the get/set methods
        self.testGetSet(obj)

        # test the boolean methods
        self.testBoolean(obj)
