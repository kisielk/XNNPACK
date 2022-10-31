mkdir build\local

set CMAKE_ARGS=-DXNNPACK_LIBRARY_TYPE=static

rem We run out of disk space and timeout on Windows, so build less.
set CMAKE_ARGS=%CMAKE_ARGS% -DXNNPACK_BUILD_BENCHMARKS=OFF
set CMAKE_ARGS=%CMAKE_ARGS% -DXNNPACK_BUILD_TESTS=OFF

rem Use-specified CMake arguments go last to allow overridding defaults
set CMAKE_ARGS=%CMAKE_ARGS% %*

echo %CMAKE_ARGS%

cd build\local && cmake ../.. %CMAKE_ARGS%
cmake --build . -j %NUMBER_OF_PROCESSORS%
