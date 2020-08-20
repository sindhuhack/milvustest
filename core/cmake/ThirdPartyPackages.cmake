# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.

set(MILVUS_THIRDPARTY_DEPENDENCIES

        Prometheus
        fiu
        AWS
        oatpp)

message(STATUS "Using ${MILVUS_DEPENDENCY_SOURCE} approach to find dependencies")

# For each dependency, set dependency source to global default, if unset
foreach (DEPENDENCY ${MILVUS_THIRDPARTY_DEPENDENCIES})
    if ("${${DEPENDENCY}_SOURCE}" STREQUAL "")
        set(${DEPENDENCY}_SOURCE ${MILVUS_DEPENDENCY_SOURCE})
    endif ()
endforeach ()

macro(build_dependency DEPENDENCY_NAME)
    if ("${DEPENDENCY_NAME}" STREQUAL "Prometheus")
        build_prometheus()
    elseif ("${DEPENDENCY_NAME}" STREQUAL "SQLite")
        build_sqlite()
    elseif ("${DEPENDENCY_NAME}" STREQUAL "fiu")
        build_fiu()
    elseif ("${DEPENDENCY_NAME}" STREQUAL "oatpp")
        build_oatpp()
    elseif("${DEPENDENCY_NAME}" STREQUAL "AWS")
        build_aws()
    else ()
        message(FATAL_ERROR "Unknown thirdparty dependency to build: ${DEPENDENCY_NAME}")
    endif ()
endmacro()

# ----------------------------------------------------------------------
# Identify OS
if (UNIX)
    if (APPLE)
        set(CMAKE_OS_NAME "osx" CACHE STRING "Operating system name" FORCE)
    else (APPLE)
        ## Check for Debian GNU/Linux ________________
        find_file(DEBIAN_FOUND debian_version debconf.conf
                PATHS /etc
                )
        if (DEBIAN_FOUND)
            set(CMAKE_OS_NAME "debian" CACHE STRING "Operating system name" FORCE)
        endif (DEBIAN_FOUND)
        ##  Check for Fedora _________________________
        find_file(FEDORA_FOUND fedora-release
                PATHS /etc
                )
        if (FEDORA_FOUND)
            set(CMAKE_OS_NAME "fedora" CACHE STRING "Operating system name" FORCE)
        endif (FEDORA_FOUND)
        ##  Check for RedHat _________________________
        find_file(REDHAT_FOUND redhat-release inittab.RH
                PATHS /etc
                )
        if (REDHAT_FOUND)
            set(CMAKE_OS_NAME "redhat" CACHE STRING "Operating system name" FORCE)
        endif (REDHAT_FOUND)
        ## Extra check for Ubuntu ____________________
        if (DEBIAN_FOUND)
            ## At its core Ubuntu is a Debian system, with
            ## a slightly altered configuration; hence from
            ## a first superficial inspection a system will
            ## be considered as Debian, which signifies an
            ## extra check is required.
            find_file(UBUNTU_EXTRA legal issue
                    PATHS /etc
                    )
            if (UBUNTU_EXTRA)
                ## Scan contents of file
                file(STRINGS ${UBUNTU_EXTRA} UBUNTU_FOUND
                        REGEX Ubuntu
                        )
                ## Check result of string search
                if (UBUNTU_FOUND)
                    set(CMAKE_OS_NAME "ubuntu" CACHE STRING "Operating system name" FORCE)
                    set(DEBIAN_FOUND FALSE)

                    find_program(LSB_RELEASE_EXEC lsb_release)
                    execute_process(COMMAND ${LSB_RELEASE_EXEC} -rs
                            OUTPUT_VARIABLE LSB_RELEASE_ID_SHORT
                            OUTPUT_STRIP_TRAILING_WHITESPACE
                            )
                    STRING(REGEX REPLACE "\\." "_" UBUNTU_VERSION "${LSB_RELEASE_ID_SHORT}")
                endif (UBUNTU_FOUND)
            endif (UBUNTU_EXTRA)
        endif (DEBIAN_FOUND)
    endif (APPLE)
endif (UNIX)

# ----------------------------------------------------------------------
# thirdparty directory
set(THIRDPARTY_DIR "${MILVUS_SOURCE_DIR}/thirdparty")

macro(resolve_dependency DEPENDENCY_NAME)
    if (${DEPENDENCY_NAME}_SOURCE STREQUAL "AUTO")
        find_package(${DEPENDENCY_NAME} MODULE)
        if (NOT ${${DEPENDENCY_NAME}_FOUND})
            build_dependency(${DEPENDENCY_NAME})
        endif ()
    elseif (${DEPENDENCY_NAME}_SOURCE STREQUAL "BUNDLED")
        build_dependency(${DEPENDENCY_NAME})
    elseif (${DEPENDENCY_NAME}_SOURCE STREQUAL "SYSTEM")
        find_package(${DEPENDENCY_NAME} REQUIRED)
    endif ()
endmacro()

# ----------------------------------------------------------------------
# ExternalProject options

string(TOUPPER ${CMAKE_BUILD_TYPE} UPPERCASE_BUILD_TYPE)

set(EP_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${UPPERCASE_BUILD_TYPE}}")
set(EP_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${UPPERCASE_BUILD_TYPE}}")

# Set -fPIC on all external projects
set(EP_CXX_FLAGS "${EP_CXX_FLAGS} -fPIC")
set(EP_C_FLAGS "${EP_C_FLAGS} -fPIC")

# CC/CXX environment variables are captured on the first invocation of the
# builder (e.g make or ninja) instead of when CMake is invoked into to build
# directory. This leads to issues if the variables are exported in a subshell
# and the invocation of make/ninja is in distinct subshell without the same
# environment (CC/CXX).
set(EP_COMMON_TOOLCHAIN -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})

if (CMAKE_AR)
    set(EP_COMMON_TOOLCHAIN ${EP_COMMON_TOOLCHAIN} -DCMAKE_AR=${CMAKE_AR})
endif ()

if (CMAKE_RANLIB)
    set(EP_COMMON_TOOLCHAIN ${EP_COMMON_TOOLCHAIN} -DCMAKE_RANLIB=${CMAKE_RANLIB})
endif ()

# External projects are still able to override the following declarations.
# cmake command line will favor the last defined variable when a duplicate is
# encountered. This requires that `EP_COMMON_CMAKE_ARGS` is always the first
# argument.
set(EP_COMMON_CMAKE_ARGS
        ${EP_COMMON_TOOLCHAIN}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_FLAGS=${EP_C_FLAGS}
        -DCMAKE_C_FLAGS_${UPPERCASE_BUILD_TYPE}=${EP_C_FLAGS}
        -DCMAKE_CXX_FLAGS=${EP_CXX_FLAGS}
        -DCMAKE_CXX_FLAGS_${UPPERCASE_BUILD_TYPE}=${EP_CXX_FLAGS})

if (NOT MILVUS_VERBOSE_THIRDPARTY_BUILD)
    set(EP_LOG_OPTIONS LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 LOG_DOWNLOAD 1)
else ()
    set(EP_LOG_OPTIONS)
endif ()

# Ensure that a default make is set
if ("${MAKE}" STREQUAL "")
    find_program(MAKE make)
endif ()

if (NOT DEFINED MAKE_BUILD_ARGS)
    set(MAKE_BUILD_ARGS "-j8")
endif ()
message(STATUS "Third Party MAKE_BUILD_ARGS = ${MAKE_BUILD_ARGS}")

# ----------------------------------------------------------------------
# Find pthreads

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ----------------------------------------------------------------------
# Versions and URLs for toolchain builds, which also can be used to configure
# offline builds

# Read toolchain versions from cpp/thirdparty/versions.txt
file(STRINGS "${THIRDPARTY_DIR}/versions.txt" TOOLCHAIN_VERSIONS_TXT)
foreach (_VERSION_ENTRY ${TOOLCHAIN_VERSIONS_TXT})
    # Exclude comments
    if (NOT _VERSION_ENTRY MATCHES "^[^#][A-Za-z0-9-_]+_VERSION=")
        continue()
    endif ()

    string(REGEX MATCH "^[^=]*" _LIB_NAME ${_VERSION_ENTRY})
    string(REPLACE "${_LIB_NAME}=" "" _LIB_VERSION ${_VERSION_ENTRY})

    # Skip blank or malformed lines
    if (${_LIB_VERSION} STREQUAL "")
        continue()
    endif ()

    # For debugging
    #message(STATUS "${_LIB_NAME}: ${_LIB_VERSION}")

    set(${_LIB_NAME} "${_LIB_VERSION}")
endforeach ()


if (DEFINED ENV{MILVUS_PROMETHEUS_URL})
    set(PROMETHEUS_SOURCE_URL "$ENV{PROMETHEUS_OPENBLAS_URL}")
else ()
    set(PROMETHEUS_SOURCE_URL
        "https://github.com/milvus-io/prometheus-cpp/archive/${PROMETHEUS_VERSION}.zip")
endif ()

if (DEFINED ENV{MILVUS_FIU_URL})
    set(FIU_SOURCE_URL "$ENV{MILVUS_FIU_URL}")
else ()
    set(FIU_SOURCE_URL "https://github.com/albertito/libfiu/archive/${FIU_VERSION}.tar.gz"
                       "https://gitee.com/quicksilver/libfiu/repository/archive/${FIU_VERSION}.zip")
endif ()

if (DEFINED ENV{MILVUS_OATPP_URL})
    set(OATPP_SOURCE_URL "$ENV{MILVUS_OATPP_URL}")
else ()
    set(OATPP_SOURCE_URL "https://github.com/oatpp/oatpp/archive/${OATPP_VERSION}.tar.gz")
#   set(OATPP_SOURCE_URL "https://github.com/BossZou/oatpp/archive/${OATPP_VERSION}.zip")
endif ()

if (DEFINED ENV{MILVUS_AWS_URL})
    set(AWS_SOURCE_URL "$ENV{MILVUS_AWS_URL}")
else ()
    set(AWS_SOURCE_URL "https://github.com/aws/aws-sdk-cpp/archive/${AWS_VERSION}.tar.gz")
endif ()

# ----------------------------------------------------------------------
# Prometheus

macro(build_prometheus)
    message(STATUS "Building Prometheus-${PROMETHEUS_VERSION} from source")
    set(PROMETHEUS_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/prometheus_ep-prefix/src/prometheus_ep")
    set(PROMETHEUS_STATIC_LIB_NAME prometheus-cpp)
    set(PROMETHEUS_CORE_STATIC_LIB
            "${PROMETHEUS_PREFIX}/core/${CMAKE_STATIC_LIBRARY_PREFIX}${PROMETHEUS_STATIC_LIB_NAME}-core${CMAKE_STATIC_LIBRARY_SUFFIX}"
            )
    set(PROMETHEUS_PUSH_STATIC_LIB
            "${PROMETHEUS_PREFIX}/push/${CMAKE_STATIC_LIBRARY_PREFIX}${PROMETHEUS_STATIC_LIB_NAME}-push${CMAKE_STATIC_LIBRARY_SUFFIX}"
            )
    set(PROMETHEUS_PULL_STATIC_LIB
            "${PROMETHEUS_PREFIX}/pull/${CMAKE_STATIC_LIBRARY_PREFIX}${PROMETHEUS_STATIC_LIB_NAME}-pull${CMAKE_STATIC_LIBRARY_SUFFIX}"
            )

    set(PROMETHEUS_CMAKE_ARGS
            ${EP_COMMON_CMAKE_ARGS}
            -DCMAKE_INSTALL_LIBDIR=lib
            -DBUILD_SHARED_LIBS=OFF
            "-DCMAKE_INSTALL_PREFIX=${PROMETHEUS_PREFIX}"
            -DCMAKE_BUILD_TYPE=Release)

    ExternalProject_Add(prometheus_ep
            URL
            ${PROMETHEUS_SOURCE_URL}
            ${EP_LOG_OPTIONS}
            URL_MD5
            "6550819ae4d61c480a55a69f08159413"
            CMAKE_ARGS
            ${PROMETHEUS_CMAKE_ARGS}
            UPDATE_COMMAND
            ""
            BUILD_COMMAND
            ${MAKE}
            ${MAKE_BUILD_ARGS}
            BUILD_COMMAND
            ${MAKE}
            ${MAKE_BUILD_ARGS}
            BUILD_IN_SOURCE
            1
            INSTALL_COMMAND
            ${MAKE}
            "DESTDIR=${PROMETHEUS_PREFIX}"
            install
            BUILD_BYPRODUCTS
            "${PROMETHEUS_CORE_STATIC_LIB}"
            "${PROMETHEUS_PUSH_STATIC_LIB}"
            "${PROMETHEUS_PULL_STATIC_LIB}")

    file(MAKE_DIRECTORY "${PROMETHEUS_PREFIX}/push/include")
    add_library(prometheus-cpp-push STATIC IMPORTED)
    set_target_properties(prometheus-cpp-push
            PROPERTIES IMPORTED_LOCATION "${PROMETHEUS_PUSH_STATIC_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${PROMETHEUS_PREFIX}/push/include")
    add_dependencies(prometheus-cpp-push prometheus_ep)

    file(MAKE_DIRECTORY "${PROMETHEUS_PREFIX}/pull/include")
    add_library(prometheus-cpp-pull STATIC IMPORTED)
    set_target_properties(prometheus-cpp-pull
            PROPERTIES IMPORTED_LOCATION "${PROMETHEUS_PULL_STATIC_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${PROMETHEUS_PREFIX}/pull/include")
    add_dependencies(prometheus-cpp-pull prometheus_ep)

    file(MAKE_DIRECTORY "${PROMETHEUS_PREFIX}/core/include")
    add_library(prometheus-cpp-core STATIC IMPORTED)
    set_target_properties(prometheus-cpp-core
            PROPERTIES IMPORTED_LOCATION "${PROMETHEUS_CORE_STATIC_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${PROMETHEUS_PREFIX}/core/include")
    add_dependencies(prometheus-cpp-core prometheus_ep)
endmacro()

if (MILVUS_WITH_PROMETHEUS)

    resolve_dependency(Prometheus)

    link_directories(SYSTEM ${PROMETHEUS_PREFIX}/push/)
    include_directories(SYSTEM ${PROMETHEUS_PREFIX}/push/include)

    link_directories(SYSTEM ${PROMETHEUS_PREFIX}/pull/)
    include_directories(SYSTEM ${PROMETHEUS_PREFIX}/pull/include)

    link_directories(SYSTEM ${PROMETHEUS_PREFIX}/core/)
    include_directories(SYSTEM ${PROMETHEUS_PREFIX}/core/include)

endif ()

# ----------------------------------------------------------------------
# fiu
macro(build_fiu)
    message(STATUS "Building FIU-${FIU_VERSION} from source")
    set(FIU_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/fiu_ep-prefix/src/fiu_ep")
    set(FIU_SHARED_LIB "${FIU_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}fiu${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(FIU_INCLUDE_DIR "${FIU_PREFIX}/include")

    ExternalProject_Add(fiu_ep
            URL
            ${FIU_SOURCE_URL}
            ${EP_LOG_OPTIONS}
            URL_MD5
            "75f9d076daf964c9410611701f07c61b"
            CONFIGURE_COMMAND
            ""
            BUILD_IN_SOURCE
            1
            BUILD_COMMAND
            ${MAKE}
            ${MAKE_BUILD_ARGS}
            INSTALL_COMMAND
            ${MAKE}
            "PREFIX=${FIU_PREFIX}"
            install
            BUILD_BYPRODUCTS
            ${FIU_SHARED_LIB}
            )

        file(MAKE_DIRECTORY "${FIU_INCLUDE_DIR}")
        add_library(fiu SHARED IMPORTED)
    set_target_properties(fiu
        PROPERTIES IMPORTED_LOCATION "${FIU_SHARED_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${FIU_INCLUDE_DIR}")

    add_dependencies(fiu fiu_ep)
endmacro()

resolve_dependency(fiu)

get_target_property(FIU_INCLUDE_DIR fiu INTERFACE_INCLUDE_DIRECTORIES)
include_directories(SYSTEM ${FIU_INCLUDE_DIR})

# ----------------------------------------------------------------------
# oatpp
macro(build_oatpp)
    message(STATUS "Building oatpp-${OATPP_VERSION} from source")
    set(OATPP_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/oatpp_ep-prefix/src/oatpp_ep")
    set(OATPP_STATIC_LIB "${OATPP_PREFIX}/lib/oatpp-${OATPP_VERSION}/${CMAKE_STATIC_LIBRARY_PREFIX}oatpp${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(OATPP_INCLUDE_DIR "${OATPP_PREFIX}/include/oatpp-${OATPP_VERSION}/oatpp")
    set(OATPP_DIR_SRC "${OATPP_PREFIX}/src")
    set(OATPP_DIR_LIB "${OATPP_PREFIX}/lib")

    set(OATPP_CMAKE_ARGS
            ${EP_COMMON_CMAKE_ARGS}
            "-DCMAKE_INSTALL_PREFIX=${OATPP_PREFIX}"
            -DCMAKE_INSTALL_LIBDIR=lib
            -DBUILD_SHARED_LIBS=OFF
            -DOATPP_BUILD_TESTS=OFF
            -DOATPP_DISABLE_LOGV=ON
            -DOATPP_DISABLE_LOGD=ON
            -DOATPP_DISABLE_LOGI=ON
            )


    ExternalProject_Add(oatpp_ep
            URL
            ${OATPP_SOURCE_URL}
            ${EP_LOG_OPTIONS}
            URL_MD5
            "396350ca4fe5bedab3769e09eee2cc9f"
            CMAKE_ARGS
            ${OATPP_CMAKE_ARGS}
            BUILD_COMMAND
            ${MAKE}
            ${MAKE_BUILD_ARGS}
            BUILD_BYPRODUCTS
            ${OATPP_STATIC_LIB}
            )

    file(MAKE_DIRECTORY "${OATPP_INCLUDE_DIR}")
    add_library(oatpp STATIC IMPORTED)
    set_target_properties(oatpp
            PROPERTIES IMPORTED_LOCATION "${OATPP_STATIC_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${OATPP_INCLUDE_DIR}")

    add_dependencies(oatpp oatpp_ep)
endmacro()

if (MILVUS_WITH_OATPP)
    resolve_dependency(oatpp)

    get_target_property(OATPP_INCLUDE_DIR oatpp INTERFACE_INCLUDE_DIRECTORIES)
    include_directories(SYSTEM ${OATPP_INCLUDE_DIR})
endif ()

# ----------------------------------------------------------------------
# aws
macro(build_aws)
    message(STATUS "Building aws-${AWS_VERSION} from source")
    set(AWS_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/aws_ep-prefix/src/aws_ep")

    set(AWS_CMAKE_ARGS
            ${EP_COMMON_TOOLCHAIN}
            "-DCMAKE_INSTALL_PREFIX=${AWS_PREFIX}"
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_INSTALL_LIBDIR=lib
            -DBUILD_ONLY=s3
            -DBUILD_SHARED_LIBS=off
            -DENABLE_TESTING=off
            -DENABLE_UNITY_BUILD=on
            -DNO_ENCRYPTION=off)

    set(AWS_CPP_SDK_CORE_STATIC_LIB
            "${AWS_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}aws-cpp-sdk-core${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(AWS_CPP_SDK_S3_STATIC_LIB
            "${AWS_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}aws-cpp-sdk-s3${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(AWS_INCLUDE_DIR "${AWS_PREFIX}/include")
    set(AWS_CMAKE_ARGS
            ${AWS_CMAKE_ARGS}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_FLAGS=${EP_C_FLAGS}
            -DCMAKE_CXX_FLAGS=${EP_CXX_FLAGS})

    ExternalProject_Add(aws_ep
            ${EP_LOG_OPTIONS}
            CMAKE_ARGS
            ${AWS_CMAKE_ARGS}
            BUILD_COMMAND
            ${MAKE}
            ${MAKE_BUILD_ARGS}
            INSTALL_DIR
            ${AWS_PREFIX}
            URL
            ${AWS_SOURCE_URL}
            BUILD_BYPRODUCTS
            "${AWS_CPP_SDK_S3_STATIC_LIB}"
            "${AWS_CPP_SDK_CORE_STATIC_LIB}")

    file(MAKE_DIRECTORY "${AWS_INCLUDE_DIR}")
    add_library(aws-cpp-sdk-s3 STATIC IMPORTED)
    add_library(aws-cpp-sdk-core STATIC IMPORTED)

    set_target_properties(aws-cpp-sdk-s3
            PROPERTIES
            IMPORTED_LOCATION "${AWS_CPP_SDK_S3_STATIC_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${AWS_INCLUDE_DIR}"
            )

    set_target_properties(aws-cpp-sdk-core
            PROPERTIES
            IMPORTED_LOCATION "${AWS_CPP_SDK_CORE_STATIC_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${AWS_INCLUDE_DIR}"
            )

    if(REDHAT_FOUND)
        set_target_properties(aws-cpp-sdk-s3
                PROPERTIES
                INTERFACE_LINK_LIBRARIES
                "${AWS_PREFIX}/lib64/libaws-c-event-stream.a;${AWS_PREFIX}/lib64/libaws-checksums.a;${AWS_PREFIX}/lib64/libaws-c-common.a")
        set_target_properties(aws-cpp-sdk-core
                PROPERTIES
                INTERFACE_LINK_LIBRARIES
                "${AWS_PREFIX}/lib64/libaws-c-event-stream.a;${AWS_PREFIX}/lib64/libaws-checksums.a;${AWS_PREFIX}/lib64/libaws-c-common.a")
    else()
        set_target_properties(aws-cpp-sdk-s3
                PROPERTIES
                INTERFACE_LINK_LIBRARIES
                "${AWS_PREFIX}/lib/libaws-c-event-stream.a;${AWS_PREFIX}/lib/libaws-checksums.a;${AWS_PREFIX}/lib/libaws-c-common.a")
        set_target_properties(aws-cpp-sdk-core
                PROPERTIES
                INTERFACE_LINK_LIBRARIES
                "${AWS_PREFIX}/lib/libaws-c-event-stream.a;${AWS_PREFIX}/lib/libaws-checksums.a;${AWS_PREFIX}/lib/libaws-c-common.a")
    endif()

    add_dependencies(aws-cpp-sdk-s3 aws_ep)
    add_dependencies(aws-cpp-sdk-core aws_ep)

endmacro()

if(MILVUS_WITH_AWS)
    resolve_dependency(AWS)

    link_directories(SYSTEM ${AWS_PREFIX}/lib)

    get_target_property(AWS_CPP_SDK_S3_INCLUDE_DIR aws-cpp-sdk-s3 INTERFACE_INCLUDE_DIRECTORIES)
    include_directories(SYSTEM ${AWS_CPP_SDK_S3_INCLUDE_DIR})

    get_target_property(AWS_CPP_SDK_CORE_INCLUDE_DIR aws-cpp-sdk-core INTERFACE_INCLUDE_DIRECTORIES)
    include_directories(SYSTEM ${AWS_CPP_SDK_CORE_INCLUDE_DIR})

endif()
