#!/usr/bin/env bash
# Configure, build, package and repair a VTK wheel, one stage per invocation.
# Runs natively on macOS and Windows and inside a manylinux container on Linux, so it
# reads its inputs from the environment: RUNNER_OS, VTK_NIGHTLY_SHA, COMPILER_LAUNCHER and
# PYTHON (defaults to the `python` on PATH).
set -euo pipefail

stage="${1:?usage: build-vtk-wheel.sh configure|build|wheel|repair}"
python="${PYTHON:-python}"
# pip-installed cmake and ninja live next to the interpreter.
PATH="$(dirname "$(command -v "$python")"):$PATH"

case "$stage" in
configure)
	mkdir -p vtk-build
	cd vtk-build
	cmake_args=(
		-GNinja
		-DCMAKE_BUILD_TYPE=Release
		-DVTK_WHEEL_BUILD=ON
		-DVTK_WRAP_PYTHON=ON
		-DVTK_BUILD_TESTING=OFF
		-DVTK_BUILD_DOCUMENTATION=OFF
		-DVTK_BUILD_EXAMPLES=OFF
		# Off in VTK's own wheel builds too; a precompiled header defeats the compiler cache.
		-DVTK_USE_PCH=OFF
		-DVTK_VERSION_SUFFIX="dev0+g${VTK_NIGHTLY_SHA:0:8}"
		-DPython3_EXECUTABLE="$("$python" -c 'import sys; print(sys.executable)')"
		-DCMAKE_C_COMPILER_LAUNCHER="$COMPILER_LAUNCHER"
		-DCMAKE_CXX_COMPILER_LAUNCHER="$COMPILER_LAUNCHER"
		-DVTK_MODULE_ENABLE_VTK_FiltersParallelDIY2=YES
		-DVTK_MODULE_ENABLE_VTK_RenderingMatplotlib=YES
		-DVTK_MODULE_ENABLE_VTK_IOImport=YES
		-DVTK_MODULE_ENABLE_VTK_IOParallelExodus=YES
		-DVTK_MODULE_ENABLE_VTK_IOXdmf2=YES
		-DVTK_MODULE_ENABLE_VTK_WebCore=YES
		-DVTK_MODULE_ENABLE_VTK_WebGLExporter=YES
		-DVTK_MODULE_ENABLE_VTK_WebPython=YES
		../vtk-src
	)
	if [ "$RUNNER_OS" = "macOS" ]; then
		# Same target as VTK's own arm64 wheels. Without it delocate tags the wheel
		# with the runner's macOS release, which older macOS cannot install.
		cmake_args+=(-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0)
	fi
	cmake "${cmake_args[@]}"
	;;

build)
	# MSVC in particular dumps huge multi-line warning/template-instantiation
	# traces for VTK's third-party code that overwhelm the Actions log viewer.
	# Stream a filtered view live, but keep the full log to print on failure.
	noisy_pattern=': warning C[0-9]+:|: note:|^[[:space:]]+(with|\[|\])[[:space:]]*$|^[[:space:]]+[A-Za-z_][A-Za-z0-9_]*='
	set +e
	cmake --build vtk-build --parallel 2>&1 | tee build.log | grep -Ev --line-buffered "$noisy_pattern"
	build_status=${PIPESTATUS[0]}
	set -e
	if [ "$build_status" -ne 0 ]; then
		echo "::error::VTK build failed (exit $build_status); full unfiltered log follows"
		cat build.log
		exit "$build_status"
	fi
	;;

wheel)
	cd vtk-build
	if [ "$RUNNER_OS" = "macOS" ]; then
		export _PYTHON_HOST_PLATFORM="macosx-11.0-arm64"
		export ARCHFLAGS="-arch arm64"
	fi
	"$python" setup.py bdist_wheel
	;;

repair)
	cd vtk-build
	mkdir -p wheelhouse
	case "$RUNNER_OS" in
	Linux)
		auditwheel repair dist/*.whl -w wheelhouse/
		;;
	macOS)
		dylib="$(find . -name 'libvtksys*.dylib' -print -quit)"
		if [ -z "$dylib" ]; then
			echo "::error::could not locate libvtksys*.dylib anywhere under vtk-build"
			exit 1
		fi
		libdir="$(dirname "$dylib")"
		echo "Found VTK dylibs in: $libdir"
		DYLD_LIBRARY_PATH="$PWD/$libdir" delocate-wheel --require-archs arm64 -w wheelhouse -v dist/*.whl
		;;
	Windows)
		delvewheel repair --add-path bin -w wheelhouse dist/*.whl
		;;
	esac
	;;

*)
	echo "unknown stage: $stage" >&2
	exit 1
	;;
esac
