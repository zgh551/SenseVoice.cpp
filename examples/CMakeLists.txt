add_library(common STATIC
        common-ggml.h
        common-ggml.cc
)

if (SENSEVOICE_TINYALSA)
    # common-tinyalsa

    set(TARGET common-alsa)

    add_library(${TARGET} STATIC
        common-alsa.h
        common-alsa.cpp
    )

    include(DefaultTargetOptions)
    include_directories(SYSTEM /usr/aarch64-android-ndk/include)
    include_directories(SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../sense-voice/csrc)
    target_include_directories(${TARGET} PUBLIC  /usr/aarch64-android-ndk/include)
    target_link_libraries     (${TARGET} PRIVATE /usr/aarch64-android-ndk/lib/libtinyalsa.so)

    message(STATUS "Tinyalsa_LIBRARIES    = /usr/aarch64-android-ndk/lib/libtinyalsa.so")

    # set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # set_target_properties(${TARGET} PROPERTIES FOLDER "libs")
endif(SENSEVOICE_TINYALSA)

if (SENSEVOICE_TINYALSA)
    add_subdirectory(stream)
    set_target_properties(stream PROPERTIES FOLDER "examples")
endif (SENSEVOICE_TINYALSA)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(common ggml)
add_subdirectory(quantize)