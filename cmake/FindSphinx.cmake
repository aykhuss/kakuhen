# FindSphinx.cmake
#
# Finds the Sphinx documentation generator.
#
# This module defines:
#  Sphinx_FOUND - True if Sphinx was found
#  SPHINX_EXECUTABLE - The path to the sphinx-build executable
#  Sphinx::Build - Imported target for sphinx-build
#
# Usage:
#  find_package(Sphinx [REQUIRED] [QUIET])

include(FindPackageHandleStandardArgs)

# Try to find sphinx-build
find_program(SPHINX_EXECUTABLE
    NAMES sphinx-build sphinx-build.exe
    DOC "Path to the sphinx-build executable"
)

# Handle the standard arguments (REQUIRED, QUIET, etc.)
find_package_handle_standard_args(Sphinx
    REQUIRED_VARS SPHINX_EXECUTABLE
    FAIL_MESSAGE "Failed to find sphinx-build executable"
)

# Create an imported target if found
if(Sphinx_FOUND)
  if(NOT TARGET Sphinx::Build)
    add_executable(Sphinx::Build IMPORTED)
    set_target_properties(Sphinx::Build PROPERTIES
      IMPORTED_LOCATION "${SPHINX_EXECUTABLE}"
    )
  endif()
endif()

mark_as_advanced(SPHINX_EXECUTABLE)
