// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
#include "TestImageDataLIC2D.h"

#include "vtkDataSetWriter.h"
#include "vtkFloatArray.h"
#include "vtkGenericDataObjectReader.h"
#include "vtkImageData.h"
#include "vtkImageDataLIC2D.h"
#include "vtkImageIterator.h"
#include "vtkImageMapToColors.h"
#include "vtkImagePermute.h"
#include "vtkImageShiftScale.h"
#include "vtkLookupTable.h"
#include "vtkPNGReader.h"
#include "vtkPNGWriter.h"
#include "vtkPixelExtent.h"
#include "vtkPixelTransfer.h"
#include "vtkPointData.h"
#include "vtkProbeFilter.h"
#include "vtkRegressionTestImage.h"
#include "vtkRenderWindow.h"
#include "vtkSmartPointer.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkStructuredData.h"
#include "vtkTestUtilities.h"
#include "vtkTesting.h"
#include "vtkTimerLog.h"
#include "vtkTrivialProducer.h"
#include "vtkUnsignedCharArray.h"
#include "vtkUnstructuredGrid.h"
#include <sstream>
#include <vtksys/CommandLineArguments.hxx>
#include <vtksys/SystemTools.hxx>
using std::ostringstream;

//------------------------------------------------------------------------------
int TestImageDataLIC2D(int argc, char* argv[])
{
  char* fname = vtkTestUtilities::ExpandDataFileName(argc, argv, "Data/SurfaceVectors.vtk");
  std::string filename = fname;
  filename = "--data=" + filename;
  delete[] fname;

  fname = vtkTestUtilities::ExpandDataFileName(argc, argv, "Data/noise.png");
  std::string noise = fname;
  noise = "--noise=" + noise;
  delete[] fname;

  char** new_argv = new char*[argc + 10];
  for (int cc = 0; cc < argc; cc++)
  {
    new_argv[cc] = vtksys::SystemTools::DuplicateString(argv[cc]);
  }
  new_argv[argc++] = vtksys::SystemTools::DuplicateString(filename.c_str());
  new_argv[argc++] = vtksys::SystemTools::DuplicateString(noise.c_str());
  new_argv[argc++] = vtksys::SystemTools::DuplicateString("--mag=5");
  new_argv[argc++] = vtksys::SystemTools::DuplicateString("--partitions=5");
  int status = ImageDataLIC2D(argc, new_argv);
  for (int kk = 0; kk < argc; kk++)
  {
    delete[] new_argv[kk];
  }
  delete[] new_argv;
  return status;
}

// Example demonstrating use of vtkImageDataLIC2D filter.
// Typical usage:
// ./bin/ImageDataLIC2D --data=<vtk file> --output=<png file>
//------------------------------------------------------------------------------
int ImageDataLIC2D(int argc, char* argv[])
{
  std::string filename;
  std::string noise_filename;
  int resolution = 10;
  int magnification = 1;
  std::string outputpath;
  int num_partitions = 1;
  int num_steps = 40;

  vtksys::CommandLineArguments arg;
  arg.StoreUnusedArguments(true);
  arg.Initialize(argc, argv);

  typedef vtksys::CommandLineArguments argT;
  arg.AddArgument("--data", argT::EQUAL_ARGUMENT, &filename,
    "(required) Enter dataset to load (currently only *.vtk files are supported");
  arg.AddArgument("--res", argT::EQUAL_ARGUMENT, &resolution,
    "(optional: default 10) Number of sample per unit distance");
  arg.AddArgument(
    "--mag", argT::EQUAL_ARGUMENT, &magnification, "(optional: default 1) Magnification");
  arg.AddArgument("--output", argT::EQUAL_ARGUMENT, &outputpath, "(optional) Output png image");
  arg.AddArgument("--partitions", argT::EQUAL_ARGUMENT, &num_partitions,
    "(optional: default 1) Number of partitions");
  arg.AddArgument("--num-steps", argT::EQUAL_ARGUMENT, &num_steps,
    "(optional: default 40) Number of steps in each direction");
  arg.AddArgument("--noise", argT::EQUAL_ARGUMENT, &noise_filename,
    "(optional) Specify the filename to a png image file to use as the noise texture.");

  if (!arg.Parse() || filename.empty())
  {
    cerr << "Problem parsing arguments." << endl;
    cerr << arg.GetHelp() << endl;
    return -1;
  }

  if (magnification < 1)
  {
    cerr << "WARNING: Magnification cannot be less than 1. Using 1" << endl;
    magnification = 1;
  }

  if (num_steps < 0)
  {
    cerr << "WARNING: Number of steps cannot be less than 0. Forcing 0." << endl;
    num_steps = 0;
  }

  // set up test helper
  vtkSmartPointer<vtkTesting> tester = vtkSmartPointer<vtkTesting>::New();

  for (int cc = 0; cc < argc; cc++)
  {
    tester->AddArgument(argv[cc]);
  }
  if (!tester->IsValidImageSpecified())
  {
    cerr << "ERROR: Valid image not specified." << endl;
    return -2;
  }

  // load noise
  vtkSmartPointer<vtkImageData> noise;
  if (!noise_filename.empty())
  {
    vtkSmartPointer<vtkPNGReader> pngReader = vtkSmartPointer<vtkPNGReader>::New();

    pngReader->SetFileName(noise_filename.c_str());
    pngReader->Update();

    noise = pngReader->GetOutput();

    vtkUnsignedCharArray* cVals =
      vtkArrayDownCast<vtkUnsignedCharArray>(noise->GetPointData()->GetScalars());
    if (!cVals)
    {
      cerr << "Error: expected unsigned chars, test fails" << endl;
      return 1;
    }

    unsigned char* pCVals = cVals->GetPointer(0);
    vtkIdType cTups = cVals->GetNumberOfTuples();

    vtkFloatArray* fVals = vtkFloatArray::New();
    fVals->SetNumberOfComponents(2);
    fVals->SetNumberOfTuples(cTups);
    fVals->SetName("noise");
    float* pFVals = fVals->GetPointer(0);

    size_t nVals = 2 * cTups;
    for (size_t i = 0; i < nVals; ++i)
    {
      pFVals[i] = pCVals[i] / 255.0;
    }

    noise->GetPointData()->RemoveArray(0);
    noise->GetPointData()->SetScalars(fVals);
    fVals->Delete();
  }

  // load vectors
  vtkSmartPointer<vtkGenericDataObjectReader> reader =
    vtkSmartPointer<vtkGenericDataObjectReader>::New();

  reader->SetFileName(filename.c_str());
  reader->Update();

  vtkDataSet* dataset = vtkDataSet::SafeDownCast(reader->GetOutput());
  if (!dataset)
  {
    cerr << "Error: expected dataset, test fails" << endl;
    return 1;
  }
  double bounds[6];
  dataset->GetBounds(bounds);

  // If 3D use XY slice, otherwise use non-trivial slice.
  int dataDesc = VTK_XY_PLANE;
  if (bounds[0] == bounds[1])
  {
    dataDesc = VTK_YZ_PLANE;
  }
  else if (bounds[2] == bounds[3])
  {
    dataDesc = VTK_XZ_PLANE;
  }
  else if (bounds[4] == bounds[5])
  {
    dataDesc = VTK_XY_PLANE;
  }

  int comp[3] = { 0, 1, 2 };
  switch (dataDesc)
  {
    case VTK_XY_PLANE:
      comp[0] = 0;
      comp[1] = 1;
      comp[2] = 2;
      break;

    case VTK_YZ_PLANE:
      comp[0] = 1;
      comp[1] = 2;
      comp[2] = 0;
      break;

    case VTK_XZ_PLANE:
      comp[0] = 0;
      comp[1] = 2;
      comp[2] = 1;
      break;
  }

  int width = static_cast<int>(ceil((bounds[2 * comp[0] + 1] - bounds[2 * comp[0]]) * resolution));
  int height = static_cast<int>(ceil((bounds[2 * comp[1] + 1] - bounds[2 * comp[1]]) * resolution));

  int dims[3];
  dims[comp[0]] = width;
  dims[comp[1]] = height;
  dims[comp[2]] = 1;

  double spacing[3];
  spacing[comp[0]] = (bounds[2 * comp[0] + 1] - bounds[2 * comp[0]]) / double(width);
  spacing[comp[1]] = (bounds[2 * comp[1] + 1] - bounds[2 * comp[1]]) / double(height);
  spacing[comp[2]] = 1.0;

  double origin[3] = { bounds[0], bounds[2], bounds[4] };

  int outWidth = magnification * width;
  int outHeight = magnification * height;

  double outSpacing[3];
  outSpacing[0] = spacing[comp[0]] / magnification;
  outSpacing[1] = spacing[comp[1]] / magnification;
  outSpacing[2] = 1.0;

  // convert input dataset to an image data
  vtkSmartPointer<vtkImageData> probeData = vtkSmartPointer<vtkImageData>::New();

  probeData->SetOrigin(origin);
  probeData->SetDimensions(dims);
  probeData->SetSpacing(spacing);

  vtkSmartPointer<vtkProbeFilter> probe = vtkSmartPointer<vtkProbeFilter>::New();

  probe->SetSourceConnection(reader->GetOutputPort());
  probe->SetInputData(probeData);
  probe->Update();
  probeData = nullptr;

  // create and initialize a rendering context
  vtkSmartPointer<vtkRenderWindow> renWin = vtkSmartPointer<vtkRenderWindow>::New();
  renWin->Render();

  // create and initialize the image lic'er
  vtkSmartPointer<vtkImageDataLIC2D> filter = vtkSmartPointer<vtkImageDataLIC2D>::New();

  if (filter->SetContext(renWin) == 0)
  {
    cerr << "WARNING: Required OpenGL not supported, test passes." << endl;
    return 0;
  }
  filter->SetSteps(num_steps);
  filter->SetStepSize(0.8 / magnification);
  filter->SetMagnification(magnification);
  filter->SetInputConnection(0, probe->GetOutputPort(0));
  if (noise)
  {
    filter->SetInputData(1, noise);
  }
  filter->UpdateInformation();
  noise = nullptr;

  // array to hold the results
  vtkPixelExtent licDataExt(outWidth, outHeight);

  vtkIdType licDataSize = static_cast<vtkIdType>(licDataExt.Size());

  vtkSmartPointer<vtkFloatArray> licData = vtkSmartPointer<vtkFloatArray>::New();

  licData->SetNumberOfComponents(3);
  licData->SetNumberOfTuples(licDataSize);

  // for each piece in the partitioned dataset compute lic and
  // copy into the output.
  for (int kk = 0; kk < num_partitions; kk++)
  {
    filter->UpdatePiece(kk, num_partitions, 0);

    vtkImageData* licPieceDataSet = filter->GetOutput();
    vtkDataArray* licPiece = licPieceDataSet->GetPointData()->GetScalars();

    int tmp[6];
    licPieceDataSet->GetExtent(tmp);

    vtkPixelExtent licPieceExt(
      tmp[2 * comp[0]], tmp[2 * comp[0] + 1], tmp[2 * comp[1]], tmp[2 * comp[1] + 1]);

    vtkPixelTransfer::Blit(licPieceExt, licPieceExt, licDataExt, licPieceExt,
      licPiece->GetNumberOfComponents(), licPiece->GetDataType(), licPiece->GetVoidPointer(0),
      licData->GetNumberOfComponents(), licData->GetDataType(), licData->GetVoidPointer(0));
  }
  probe = nullptr;
  filter = nullptr;
  renWin = nullptr;

  // convert from float to u char for png
  vtkSmartPointer<vtkUnsignedCharArray> licPng = vtkSmartPointer<vtkUnsignedCharArray>::New();

  licPng->SetNumberOfComponents(3);
  licPng->SetNumberOfTuples(licDataSize);
  unsigned char* pPng = licPng->GetPointer(0);
  float* pData = licData->GetPointer(0);
  size_t n = 3 * licDataSize;
  for (size_t i = 0; i < n; ++i)
  {
    pPng[i] = static_cast<unsigned char>(pData[i] * 255.0f);
  }
  licData = nullptr;

  // wrap the result into an image data for the png writer
  vtkSmartPointer<vtkImageData> pngDataSet = vtkSmartPointer<vtkImageData>::New();

  pngDataSet->SetDimensions(outWidth, outHeight, 1);
  pngDataSet->SetSpacing(outSpacing);
  pngDataSet->SetOrigin(origin);
  pngDataSet->GetPointData()->SetScalars(licPng);
  licPng = nullptr;

  // save a png
  if (!outputpath.empty())
  {
    vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();

    writer->SetFileName(outputpath.c_str());
    writer->SetInputData(pngDataSet);
    writer->Write();
    writer = nullptr;
  }

  // run the test
  vtkSmartPointer<vtkTrivialProducer> tp = vtkSmartPointer<vtkTrivialProducer>::New();

  tp->SetOutput(pngDataSet);
  int retVal = (tester->RegressionTest(tp, 10) == vtkTesting::PASSED) ? 0 : -4;
  if (retVal)
  {
    cerr << "ERROR: test failed." << endl;
  }

  tp = nullptr;
  pngDataSet = nullptr;

  return retVal;
}
