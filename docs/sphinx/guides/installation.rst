Installation
============

This guide summarizes the recommended ways to consume `kakuhen` from a CMake
project.

Requirements
------------

* **CMake**: 3.18 or newer.
* **C++ compiler**: C++20 capable (GCC 10+, Clang 10+, MSVC 2019+).
* **Git**: For cloning the repository.

Option A: Add as subdirectory
-----------------------------

This is usually the simplest option while developing both your application and
the library side by side.

1. **Vendor or add a submodule**, for example:

   .. code-block:: bash

       git submodule add https://github.com/aykhuss/kakuhen.git external/kakuhen

2. **Add to your build**:

   .. code-block:: cmake

       add_subdirectory(external/kakuhen)
       target_link_libraries(my_app PRIVATE kakuhen::kakuhen)

Option B: Install + find_package
--------------------------------

This option is a better fit when you want a reusable installation or package
manager style workflow.

1. **Clone and configure**:

   .. code-block:: bash

       git clone https://github.com/aykhuss/kakuhen.git
       cd kakuhen
       cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

2. **Build and install**:

   .. code-block:: bash

       cmake --build build
       cmake --install build

   On Windows (multi-config generators), specify the configuration:

   .. code-block:: bash

       cmake --build build --config Release
       cmake --install build --config Release

3. **Use in your project**:

   .. code-block:: cmake

       find_package(kakuhen CONFIG REQUIRED)
       target_link_libraries(my_app PRIVATE kakuhen::kakuhen)

   If CMake cannot find the package, set one of:

   * ``CMAKE_PREFIX_PATH`` to the install prefix
   * ``kakuhen_DIR`` to the package config directory

Build options
-------------

Common CMake options:

* ``KAKUHEN_BUILD_TESTING``: Build tests.
* ``KAKUHEN_BUILD_CLI``: Build the CLI tool.
* ``KAKUHEN_BUILD_EXAMPLES``: Build examples.
* ``KAKUHEN_BUILD_DOCS``: Build documentation.
* ``KAKUHEN_ENABLE_COVERAGE``: Enable coverage (GCC/Clang only).
