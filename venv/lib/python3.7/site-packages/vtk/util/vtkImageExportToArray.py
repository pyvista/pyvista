"""
vtkImageExportToArray - a NumPy front-end to vtkImageExport

This class converts a VTK image to a numpy array.  The output
array will always have 3 dimensions (or 4, if the image had
multiple scalar components).

To use this class, you must have numpy installed (http://numpy.scipy.org)

Methods

  SetInputConnection(vtkAlgorithmOutput) -- connect to VTK image pipeline
  SetInputData(vtkImageData) -- set an vtkImageData to export
  GetArray() -- execute pipeline and return a numpy array

Methods from vtkImageExport

  GetDataExtent()
  GetDataSpacing()
  GetDataOrigin()
"""

import numpy
import numpy.core.umath as umath

from vtk import vtkImageExport
from vtk import vtkStreamingDemandDrivenPipeline
from vtk import VTK_SIGNED_CHAR
from vtk import VTK_UNSIGNED_CHAR
from vtk import VTK_SHORT
from vtk import VTK_UNSIGNED_SHORT
from vtk import VTK_INT
from vtk import VTK_UNSIGNED_INT
from vtk import VTK_LONG
from vtk import VTK_UNSIGNED_LONG
from vtk import VTK_FLOAT
from vtk import VTK_DOUBLE


class vtkImageExportToArray:
    def __init__(self):
        self.__export = vtkImageExport()
        self.__ConvertUnsignedShortToInt = False

    # type dictionary

    __typeDict = { VTK_SIGNED_CHAR:'b',
                   VTK_UNSIGNED_CHAR:'B',
                   VTK_SHORT:'h',
                   VTK_UNSIGNED_SHORT:'H',
                   VTK_INT:'i',
                   VTK_UNSIGNED_INT:'I',
                   VTK_FLOAT:'f',
                   VTK_DOUBLE:'d'}

    __sizeDict = { VTK_SIGNED_CHAR:1,
                   VTK_UNSIGNED_CHAR:1,
                   VTK_SHORT:2,
                   VTK_UNSIGNED_SHORT:2,
                   VTK_INT:4,
                   VTK_UNSIGNED_INT:4,
                   VTK_FLOAT:4,
                   VTK_DOUBLE:8 }

    # convert unsigned shorts to ints, to avoid sign problems
    def SetConvertUnsignedShortToInt(self,yesno):
        self.__ConvertUnsignedShortToInt = yesno

    def GetConvertUnsignedShortToInt(self):
        return self.__ConvertUnsignedShortToInt

    def ConvertUnsignedShortToIntOn(self):
        self.__ConvertUnsignedShortToInt = True

    def ConvertUnsignedShortToIntOff(self):
        self.__ConvertUnsignedShortToInt = False

    # set the input
    def SetInputConnection(self,input):
        return self.__export.SetInputConnection(input)

    def SetInputData(self,input):
        return self.__export.SetInputData(input)

    def GetInput(self):
        return self.__export.GetInput()

    def GetArray(self):
        self.__export.Update()
        input = self.__export.GetInput()
        extent = input.GetExtent()
        type = input.GetScalarType()
        numComponents = input.GetNumberOfScalarComponents()
        dim = (extent[5]-extent[4]+1,
               extent[3]-extent[2]+1,
               extent[1]-extent[0]+1)
        if (numComponents > 1):
            dim = dim + (numComponents,)

        imArray = numpy.zeros(dim, self.__typeDict[type])
        self.__export.Export(imArray)

        # convert unsigned short to int to avoid sign issues
        if (type == VTK_UNSIGNED_SHORT and self.__ConvertUnsignedShortToInt):
            imArray = umath.bitwise_and(imArray.astype('i'),0xffff)

        return imArray

    def GetDataExtent(self):
        return self.__export.GetDataExtent()

    def GetDataSpacing(self):
        return self.__export.GetDataSpacing()

    def GetDataOrigin(self):
        return self.__export.GetDataOrigin()
