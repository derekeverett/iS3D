set (SOURCES
    deltaf_table.cpp
    freearray.cpp
    gauss_integration.cpp
    readindata.cpp
    thermal_integrands.cpp
    )

add_executable(deltaf_table ${SOURCES})

target_link_libraries (deltaf_table ${GSL_LIBRARIES})

install(TARGETS deltaf_table DESTINATION ${CMAKE_HOME_DIRECTORY})
