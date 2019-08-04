"""
Utility module to make it easier to create new keys.
"""
from vtk.vtkCommonCore import vtkInformationDataObjectKey as DataaObjectKey
from vtk.vtkCommonCore import vtkInformationDoubleKey as DoubleKey
from vtk.vtkCommonCore import vtkInformationDoubleVectorKey as DoubleVectorKey
from vtk.vtkCommonCore import vtkInformationIdTypeKey as IdTypeKey
from vtk.vtkCommonCore import vtkInformationInformationKey as InformationKey
from vtk.vtkCommonCore import vtkInformationInformationVectorKey as InformationVectorKey
from vtk.vtkCommonCore import vtkInformationIntegerKey as IntegerKey
from vtk.vtkCommonCore import vtkInformationIntegerVectorKey as IntegerVectorKey
from vtk.vtkCommonCore import vtkInformationKeyVectorKey as KeyVectorKey
from vtk.vtkCommonCore import vtkInformationObjectBaseKey as ObjectBaseKey
from vtk.vtkCommonCore import vtkInformationObjectBaseVectorKey as ObjectBaseVectorKey
from vtk.vtkCommonCore import vtkInformationRequestKey as RequestKey
from vtk.vtkCommonCore import vtkInformationStringKey as StringKey
from vtk.vtkCommonCore import vtkInformationStringVectorKey as StringVectorKey
from vtk.vtkCommonCore import vtkInformationUnsignedLongKey as UnsignedLongKey
from vtk.vtkCommonCore import vtkInformationVariantKey as VariantKey
from vtk.vtkCommonCore import vtkInformationVariantVectorKey as VariantVectorKey
from vtk.vtkCommonExecutionModel import vtkInformationDataObjectMetaDataKey as DataObjectMetaDataKey
from vtk.vtkCommonExecutionModel import vtkInformationExecutivePortKey as ExecutivePortKey
from vtk.vtkCommonExecutionModel import vtkInformationExecutivePortVectorKey as ExecutivePortVectorKey
from vtk.vtkCommonExecutionModel import vtkInformationIntegerRequestKey as IntegerRequestKey

def MakeKey(key_type, name, location, *args):
    """Given a key type, make a new key of given name
    and location."""
    return key_type.MakeKey(name, location, *args)
