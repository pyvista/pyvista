// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkBoundaryMeshQuality.h"
#include "vtkCellData.h"
#include "vtkDoubleArray.h"
#include "vtkLogger.h"
#include "vtkMathUtilities.h"
#include "vtkTestUtilities.h"
#include "vtkXMLUnstructuredGridReader.h"

int TestBoundaryMeshQuality(int argc, char* argv[])
{
  char* fname = vtkTestUtilities::ExpandDataFileName(argc, argv, "Data/can.vtu");
  vtkNew<vtkXMLUnstructuredGridReader> reader;
  reader->SetFileName(fname);
  reader->Update();
  delete[] fname;

  vtkNew<vtkBoundaryMeshQuality> boundaryMeshQuality;
  boundaryMeshQuality->SetInputConnection(reader->GetOutputPort());
  boundaryMeshQuality->Update();
  auto output = boundaryMeshQuality->GetOutput();

  auto distance1 = vtkDoubleArray::SafeDownCast(
    output->GetCellData()->GetArray("DistanceFromCellCenterToFaceCenter"));
  if (!distance1)
  {
    vtkLog(ERROR, "DistanceFromCellCenterToFaceCenter array not found");
    return EXIT_FAILURE;
  }
  auto distance2 = vtkDoubleArray::SafeDownCast(
    output->GetCellData()->GetArray("DistanceFromCellCenterToFacePlane"));
  if (!distance2)
  {
    vtkLog(ERROR, "DistanceFromCellCenterToFacePlane array not found");
    return EXIT_FAILURE;
  }
  auto angle = vtkDoubleArray::SafeDownCast(
    output->GetCellData()->GetArray("AngleFaceNormalAndCellCenterToFaceCenterVector"));
  if (!angle)
  {
    vtkLog(ERROR, "AngleFaceNormalAndCellCenterToFaceCenterVector array not found");
    return EXIT_FAILURE;
  }

  const auto distanceEpsilon = 1e-6;
  double range1[2];
  distance1->GetRange(range1);
  if (!vtkMathUtilities::FuzzyCompare(range1[0], 0.00951085, distanceEpsilon) ||
    !vtkMathUtilities::FuzzyCompare(range1[1], 0.237624, distanceEpsilon))
  {
    vtkLog(ERROR, "DistanceFromCellCenterToFaceCenter: " << range1[0] << " " << range1[1]);
    return EXIT_FAILURE;
  }

  double range2[2];
  distance2->GetRange(range2);
  if (!vtkMathUtilities::FuzzyCompare(range2[0], 0.00203259, distanceEpsilon) ||
    !vtkMathUtilities::FuzzyCompare(range2[1], 0.235969, distanceEpsilon))
  {
    vtkLog(ERROR, "DistanceFromCellCenterToFacePlane: " << range2[0] << " " << range2[1]);
    return EXIT_FAILURE;
  }

  const auto angleEpsilon = 1e-4;
  double range3[2];
  angle->GetRange(range3);
  if (!vtkMathUtilities::FuzzyCompare(range3[0], 0.0569455, angleEpsilon) ||
    !vtkMathUtilities::FuzzyCompare(range3[1], 98.6947, angleEpsilon))
  {
    vtkLog(
      ERROR, "AngleFaceNormalAndCellCenterToFaceCenterVector: " << range3[0] << " " << range3[1]);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
