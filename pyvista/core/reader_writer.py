import os
import pyvista
import pyvista.utilities.fileio as fileio
import vtk


class ReaderWriter:
    __slots__ = ('valid_reader_extensions', 'valid_writer_extensions')

    def __init__(self, valid_read_extensions, valid_write_extensions):
        self.valid_reader_extensions = tuple(valid_read_extensions)
        self.valid_writer_extensions = tuple(valid_write_extensions)

    def load(self, filename):
        """Load a surface mesh from a mesh file.

        Mesh file may be an ASCII or binary ply, stl, or vtk mesh file.

        Parameters
        ----------
        filename : str
            Filename of mesh to be loaded.  File type is inferred from the
            extension of the filename

        Notes
        -----
        Binary files load much faster than ASCII.

        """
        filename = os.path.abspath(os.path.expanduser(filename))
        if not os.path.isfile(filename):
            raise Exception('File %s does not exist' % filename)

        try:
            reader = fileio.get_reader(filename)
        except KeyError:
            valid_extensions = self.valid_reader_extensions
            raise ValueError('Extension must be {}'.format(self._comma_or(valid_extensions)))

        reader.SetFileName(filename)
        reader.Update()
        return reader.GetOutput()

    def save(self, vtk_object, filename, binary=True):
        """Write a structured grid to disk.

        Parameters
        ----------
        filename : str
            Filename of grid to be written.  The file extension will select the
            type of writer to use.  ".vtk" will use the legacy writer, while
            ".vts" will select the VTK XML writer.

        binary : bool, optional
            Writes as a binary file by default.  Set to False to write ASCII.


        Notes
        -----
        Binary files write much faster than ASCII, but binary files written on
        one system may not be readable on other systems.  Binary can be used
        only with the legacy writer.

        """
        filename = os.path.abspath(os.path.expanduser(filename))
        try:
            writer = fileio.get_writer(filename)
        except KeyError:
            valid_extensions = self.valid_writer_extensions
            raise Exception('Extension must be {}'.format(self._comma_or(valid_extensions)))

        self._set_vtkwriter_mode(vtk_writer=writer, use_binary=binary)
        writer.SetFileName(filename)
        writer.SetInputData(vtk_object)
        writer.Write()

    def _comma_or(self, iterable):
        """Joins an iterable as a 'comma or' list
        For example:
            (1, 2, 3) -> '1, 2, or 3'
            (1) -> '1'
        """
        if len(iterable) > 1:
            nameslist = '{} or {}'.format(
                ', '.join(iterable[:-1]), iterable[-1])
        else:
            nameslist = iterable[0]
        return nameslist

    def _set_vtkwriter_mode(self, vtk_writer, use_binary=True):
        if isinstance(vtk_writer, vtk.vtkDataWriter):
            if use_binary:
                vtk_writer.SetFileTypeToBinary()
            else:
                vtk_writer.SetFileTypeToASCII()
        elif isinstance(vtk_writer, vtk.vtkXMLWriter):
            if use_binary:
                vtk_writer.SetDataModeToBinary()
            else:
                vtk_writer.SetDataModeToAscii()
        return vtk_writer

