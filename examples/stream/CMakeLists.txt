if (SENSEVOICE_TINYALSA)
    # add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../sense-voice/csrc)
    # include(${CMAKE_CURRENT_LIST_DIR}/../../sense-voice/csrc/CMakeLists.txt)
    # stream
    set(TARGET stream)
    add_executable(${TARGET} stream.cc)

    include(DefaultTargetOptions)
    
    include_directories(SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../../sense-voice/csrc)
    include_directories(SYSTEM ${CMAKE_CURRENT_LIST_DIR}/..)
    target_link_libraries(${TARGET} PRIVATE common common-alsa sense-voice-core ggml ${CMAKE_THREAD_LIBS_INIT})
endif (SENSEVOICE_TINYALSA)
