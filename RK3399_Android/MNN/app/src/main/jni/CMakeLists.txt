cmake_minimum_required(VERSION 3.10.2)

set(lib_DIR ${CMAKE_SOURCE_DIR}/libs)
include_directories(${CMAKE_SOURCE_DIR}/includes)

add_library( MNN SHARED IMPORTED GLOBAL )
set_target_properties( MNN
        PROPERTIES IMPORTED_LOCATION
        ${lib_DIR}/${ANDROID_ABI}/libMNN.so
        )

add_library( MNN_OpenCL SHARED IMPORTED GLOBAL )
set_target_properties( MNN_OpenCL
        PROPERTIES IMPORTED_LOCATION
        ${lib_DIR}/${ANDROID_ABI}/libMNN_CL.so
        )

file(GLOB_RECURSE CPP_SRCS ${CMAKE_SOURCE_DIR}/*.cpp )
add_library( mnncore SHARED ${CPP_SRCS} )

find_library( log-lib log )
find_library( jnigraphics-lib jnigraphics )

add_definitions(-DMNN_USE_LOGCAT)
target_link_libraries( mnncore MNN MNN_OpenCL ${log-lib} ${jnigraphics-lib})
