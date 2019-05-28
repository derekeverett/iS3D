# Install script for directory: /work/05780/everett/stampede2/JETSCAPE-COMP-TEST-EBE/external_packages/iS3D/generate_delta_f_coefficients/smash_box/df_vh_dimensionless/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/work/05780/everett/stampede2/JETSCAPE-COMP-TEST-EBE/external_packages/iS3D/generate_delta_f_coefficients/smash_box/df_vh_dimensionless/deltaf_table" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/work/05780/everett/stampede2/JETSCAPE-COMP-TEST-EBE/external_packages/iS3D/generate_delta_f_coefficients/smash_box/df_vh_dimensionless/deltaf_table")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/work/05780/everett/stampede2/JETSCAPE-COMP-TEST-EBE/external_packages/iS3D/generate_delta_f_coefficients/smash_box/df_vh_dimensionless/deltaf_table"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/work/05780/everett/stampede2/JETSCAPE-COMP-TEST-EBE/external_packages/iS3D/generate_delta_f_coefficients/smash_box/df_vh_dimensionless/deltaf_table")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/work/05780/everett/stampede2/JETSCAPE-COMP-TEST-EBE/external_packages/iS3D/generate_delta_f_coefficients/smash_box/df_vh_dimensionless" TYPE EXECUTABLE FILES "/work/05780/everett/stampede2/JETSCAPE-COMP-TEST-EBE/external_packages/iS3D/generate_delta_f_coefficients/smash_box/df_vh_dimensionless/build/src/deltaf_table")
  if(EXISTS "$ENV{DESTDIR}/work/05780/everett/stampede2/JETSCAPE-COMP-TEST-EBE/external_packages/iS3D/generate_delta_f_coefficients/smash_box/df_vh_dimensionless/deltaf_table" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/work/05780/everett/stampede2/JETSCAPE-COMP-TEST-EBE/external_packages/iS3D/generate_delta_f_coefficients/smash_box/df_vh_dimensionless/deltaf_table")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/apps/gcc/6.3.0/bin/strip" "$ENV{DESTDIR}/work/05780/everett/stampede2/JETSCAPE-COMP-TEST-EBE/external_packages/iS3D/generate_delta_f_coefficients/smash_box/df_vh_dimensionless/deltaf_table")
    endif()
  endif()
endif()

