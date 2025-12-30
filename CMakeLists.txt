cmake_minimum_required(VERSION 3.24)

##
# vcpkg portion
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
  set(bootstrap_vcpkg_ext "bat")
else()
  set(bootstrap_vcpkg_ext "sh")
endif()

execute_process(
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/vcpkg/bootstrap-vcpkg.${bootstrap_vcpkg_ext}"
  RESULT_VARIABLE bootstrap_vcpkg_result
)

if(NOT bootstrap_vcpkg_result EQUAL 0)
  message(FATAL_ERROR "Bootstrapping failed with code: ${bootstrap_vcpkg_result}")
else()
  message(STATUS "Bootstrapping succeeded.")
endif()

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/vcpkg/scripts/buildsystems/vcpkg.cmake"
  CACHE STRING "vcpkg toolchain file")

# end vcpkg portion
##

project(mandelbrot VERSION 0.1.0)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

# Folders for IDEs that support it
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(CTest)
enable_testing()

find_package(OpenCV CONFIG REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(mandelbrot main.cpp)
target_link_libraries( mandelbrot ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
