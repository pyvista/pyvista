"""
This python module provides functionality to parse the methods of a
VTK object.

Created by Prabhu Ramachandran.  Committed in Apr, 2002.

"""

import string, re, sys
import types

# set this to 1 if you want to see debugging messages - very useful if
# you have problems
DEBUG=0

def debug (msg):
    if DEBUG:
        print(msg)

class VtkDirMethodParser:
    """ Parses the methods from dir(vtk_obj). """
    def initialize_methods (self, vtk_obj):
        debug ("VtkDirMethodParser:: initialize_methods ()")

        self.methods = dir (vtk_obj)[:]
        # stores the <blah>On methods
        self.toggle_meths = []
        # stores the Set<blah>To<blah> methods
        self.state_meths = []
        # stores the methods that have a Get<blah> and Set<blah>
        # only the <blah> is stored
        self.get_set_meths = []
        # pure get methods
        self.get_meths = []
        self.state_patn = re.compile ("To[A-Z0-9]")

    def parse_methods (self, vtk_obj):
        debug ("VtkDirMethodParser:: parse_methods()")
        self.initialize_methods (vtk_obj)
        debug ("VtkDirMethodParser:: parse_methods() - initialized methods")

        for method in self.methods[:]:
            # finding all the methods that set the state.
            if method[:3].find("Set") >= 0 and \
                 self.state_patn.search (method) is not None :
                try:
                    eval ("vtk_obj.Get%s"%method[3:])
                except AttributeError:
                    self.state_meths.append (method)
                    self.methods.remove (method)
            # finding all the On/Off toggle methods
            elif method[-2:].find("On") >= 0:
                try:
                    self.methods.index ("%sOff"%method[:-2])
                except ValueError:
                    pass
                else:
                    self.toggle_meths.append (method)
                    self.methods.remove (method)
                    self.methods.remove ("%sOff"%method[:-2])
            # finding the Get/Set methods.
            elif method[:3].find("Get") == 0:
                set_m = "Set"+method[3:]
                try:
                    self.methods.index (set_m)
                except ValueError:
                    pass
                else:
                    self.get_set_meths.append (method[3:])
                    self.methods.remove (method)
                    self.methods.remove (set_m)

        self.clean_up_methods (vtk_obj)

    def clean_up_methods (self, vtk_obj):
        self.clean_get_set (vtk_obj)
        self.clean_state_methods (vtk_obj)
        self.clean_get_methods (vtk_obj)

    def clean_get_set (self, vtk_obj):
        debug ("VtkDirMethodParser:: clean_get_set()")
        # cleaning up the Get/Set methods by removing the toggle funcs.
        for method in self.toggle_meths:
            try:
                self.get_set_meths.remove (method[:-2])
            except ValueError:
                pass

        # cleaning them up by removing any methods that are responsible for
        # other vtkObjects
        for method in self.get_set_meths[:]:
            try:
                eval ("vtk_obj.Get%s ().GetClassName ()"%method)
            except (TypeError, AttributeError):
                pass
            else:
                self.get_set_meths.remove (method)
                continue
            try:
                val = eval ("vtk_obj.Get%s ()"%method)
            except (TypeError, AttributeError):
                self.get_set_meths.remove (method)
            else:
                if val is None:
                    self.get_set_meths.remove (method)

    def clean_state_methods (self, vtk_obj):
        debug ("VtkDirMethodParser:: clean_state_methods()")
        # Getting the remaining pure GetMethods
        for method in self.methods[:]:
            if method[:3].find("Get") == 0:
                self.get_meths.append (method)
                self.methods.remove (method)

        # Grouping similar state methods
        if len (self.state_meths) != 0:
            tmp = self.state_meths[:]
            self.state_meths = []
            state_group = [tmp[0]]
            end = self.state_patn.search (tmp[0]).start ()
            # stores the method type common to all similar methods
            m = tmp[0][3:end]
            for i in range (1, len (tmp)):
                if tmp[i].find(m) >= 0:
                    state_group.append (tmp[i])
                else:
                    self.state_meths.append (state_group)
                    state_group = [tmp[i]]
                    end = self.state_patn.search (tmp[i]).start ()
                    m = tmp[i][3:end]
                try: # remove the corresponding set method in get_set
                    val = self.get_set_meths.index (m)
                except ValueError:
                    pass
                else:
                    del self.get_set_meths[val]
                    #self.get_meths.append ("Get"+m)
                clamp_m = "Get" + m + "MinValue"
                try: # remove the GetNameMax/MinValue in get_meths
                    val = self.get_meths.index (clamp_m)
                except ValueError:
                    pass
                else:
                    del self.get_meths[val]
                    val = self.get_meths.index ("Get" + m + "MaxValue")
                    del self.get_meths[val]

            if len (state_group) > 0:
                self.state_meths.append (state_group)

    def clean_get_methods (self, vtk_obj):
        debug ("VtkDirMethodParser:: clean_get_methods()")
        for method in self.get_meths[:]:
            debug (method)
            try:
                res = eval ("vtk_obj.%s ()"%method)
            except (TypeError, AttributeError):
                self.get_meths.remove (method)
                continue
            else:
                try:
                    eval ("vtk_obj.%s ().GetClassName ()"%method)
                except AttributeError:
                    pass
                else:
                    self.get_meths.remove (method)
                    continue
            if method[-8:].find("MaxValue") > -1:
                self.get_meths.remove( method)
            elif method[-8:].find("MinValue") > -1:
                self.get_meths.remove( method)

        self.get_meths.sort ()

    def toggle_methods (self):
        return self.toggle_meths

    def state_methods (self):
        return self.state_meths

    def get_set_methods (self):
        return self.get_set_meths

    def get_methods (self):
        return self.get_meths


class VtkPrintMethodParser:

    """ This class finds the methods for a given vtkObject.  It uses
    the output from vtkObject->Print() (or in Python str(vtkObject))
    and output from the VtkDirMethodParser to obtain the methods. """

    def parse_methods (self, vtk_obj):
        "Parse for the methods."
        debug ("VtkPrintMethodParser:: parse_methods()")

        if self._initialize_methods (vtk_obj):
            # if David Gobbi's improvements are in this version of VTK
            # then I need to go no further.
            return

        for method in self.methods[:]:
            # removing methods that have nothing to the right of the ':'
            if (method[1] == '') or \
               (method[1].find("none") > -1) :
                self.methods.remove (method)

        for method in self.methods:
            # toggle methods are first identified
            if (method[1] == "On") or (method[1] == "Off"):
                try:
                    val = eval ("vtk_obj.Get%s ()"%method[0])
                    if val == 1:
                        eval ("vtk_obj.%sOn ()"%method[0])
                    elif val == 0:
                        eval ("vtk_obj.%sOff ()"%method[0])
                except AttributeError:
                    pass
                else:
                    self.toggle_meths.append (method[0]+"On")
            else: # see it it is get_set or get or a state method
                found = 0
                # checking if it is a state func.
                # figure out the long names from the dir_state_meths
                for sms in self.dir_state_meths[:]:
                    if sms[0].find(method[0]) >= 0:
                        self.state_meths.append (sms)
                        self.dir_state_meths.remove (sms)
                        found = 1
                if found:
                    self.get_meths.append ("Get"+method[0])
                    try:
                        t = eval ("vtk_obj.Get%sAsString ()"%method[0])
                    except AttributeError:
                        pass
                    else:
                        self.get_meths.append ("Get"+method[0]+"AsString")
                else:
                    # the long name is inherited or it is not a state method
                    try:
                        t = eval ("vtk_obj.Get%s ().GetClassName ()"%
                                  method[0])
                    except AttributeError:
                        pass
                    else:
                        continue
                    val = 0
                    try:
                        val = eval ("vtk_obj.Get%s ()"%method[0])
                    except (TypeError, AttributeError):
                        pass
                    else:
                        try:
                            f = eval ("vtk_obj.Set%s"%method[0])
                        except AttributeError:
                            self.get_meths.append ("Get"+method[0])
                        else:
                            try:
                                f(*val)
                            except TypeError:
                                try:
                                    f(*(val, ))
                                except TypeError:
                                    self.get_meths.append ("Get"+method[0])
                                else:
                                    self.get_set_meths.append (method[0])
                            else:
                                self.get_set_meths.append (method[0])

        self._clean_up_methods (vtk_obj)

    def _get_str_obj (self, vtk_obj):
        debug ("VtkPrintMethodParser:: _get_str_obj()")
        self.methods = str (vtk_obj)
        self.methods = self.methods.split ("\n")
        del self.methods[0]

    def _initialize_methods (self, vtk_obj):
        "Do the basic parsing and setting up"
        debug ("VtkPrintMethodParser:: _initialize_methods()")
        dir_p = VtkDirMethodParser ()
        dir_p.parse_methods (vtk_obj)

        # testing if this version of vtk has David Gobbi's cool
        # stuff. If it does then no need to do other things.
        try:
            junk = vtk_obj.__class__
        except AttributeError:
            pass
        else:
            self.toggle_meths = dir_p.toggle_methods ()
            self.state_meths = dir_p.state_methods ()
            self.get_set_meths = dir_p.get_set_methods ()
            self.get_meths = dir_p.get_methods ()
            return 1

        self.dir_toggle_meths = dir_p.toggle_methods ()
        self.dir_state_meths = dir_p.state_methods ()
        self.dir_get_set_meths = dir_p.get_set_methods ()
        self.dir_get_meths = dir_p.get_methods ()

        self._get_str_obj (vtk_obj)
        patn = re.compile ("  \S")

        for method in self.methods[:]:
            if not patn.match (method):
                self.methods.remove (method)

        for method in self.methods[:]:
            if method.find(":") == -1:
                self.methods.remove (method)

        for i in range (0, len (self.methods)):
            strng = self.methods[i]
            strng = strng.replace (" ", "")
            self.methods[i] = strng.split (":")


        self.toggle_meths = []
        self.state_meths = []
        self.get_set_meths = []
        self.get_meths = []

        return 0

    def _clean_up_methods (self, vtk_obj):
        "Merge dir and str methods.  Finish up."
        debug ("VtkPrintMethodParser:: _clean_up_methods()")
        for meth_list in ((self.dir_toggle_meths, self.toggle_meths),\
                          (self.dir_get_set_meths, self.get_set_meths),\
                          (self.dir_get_meths, self.get_meths)):
            for method in meth_list[0]:
                try:
                    meth_list[1].index (method)
                except ValueError:
                    meth_list[1].append (method)

        # Remove all get_set methods that are already in toggle_meths
        # This case can happen if the str produces no "On/Off" but
        # dir does and str produces a get_set instead.
        for method in self.toggle_meths:
            try:
                self.get_set_meths.remove (method[:-2])
            except ValueError:
                pass

        self.toggle_meths.sort ()
        self.state_meths.sort ()
        self.get_set_meths.sort ()
        self.get_meths.sort ()

    def toggle_methods (self):
        return self.toggle_meths

    def state_methods (self):
        return self.state_meths

    def get_set_methods (self):
        return self.get_set_meths

    def get_methods (self):
        return self.get_meths
