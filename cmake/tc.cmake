if(NOT WITH_GPU)
    return()
endif()

set(TC_ROOT "/paddle/tc/install" CACHE PATH "TC ROOT")
find_path(TC_INCLUDE_DIR Halide.h
    PATHS ${TC_ROOT} ${TC_ROOT}/include
    $ENV{TC_ROOT} $ENV{TC_ROOT}/include
    NO_DEFAULT_PATH
)

#get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)

set(TARGET_ARCH "x86_64")
if(NOT ${CMAKE_SYSTEM_PROCESSOR})
    set(TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()

list(APPEND TC_CHECK_LIBRARY_DIRS
    ${TC_ROOT}
    ${TC_ROOT}/lib
    $ENV{TC_ROOT}
    $ENV{TC_ROOT}/lib
	)
set(TC_LIB_NAME "")

if (LINUX)
    set(TC_LIB_NAME "llibHalide.so")
endif(LINUX)

find_library(TC_LIBRARY NAMES ${TC_LIB_NAME} # libcudnn_static.a
    PATHS ${TC_CHECK_LIBRARY_DIRS} ${TC_INCLUDE_DIR} ${__libpath_hist}
          NO_DEFAULT_PATH
    DOC "Path to cuDNN library.")

if(TC_INCLUDE_DIR AND TC_LIBRARY)
    message("TC FOUND")
    message(${TC_INCLUDE_DIR})
    message(${TC_LIBRARY})
else()
    message("TC NOT FOUND")
endif()

