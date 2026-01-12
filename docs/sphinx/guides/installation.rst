Installation
============

This guide provides instructions on how to set up and install the `kakuhen` library.

Prerequisites
-------------
Before you begin, ensure you have the following installed on your system:

*   **CMake**: Version 3.18 or higher. You can download it from the official CMake website or install via your system's package manager (e.g., ``brew install cmake`` on macOS, ``sudo apt-get install cmake`` on Ubuntu).
*   **C++ Compiler**: A C++20 compliant compiler (e.g., GCC 10+, Clang 10+, MSVC 2019+).
*   **Git**: For cloning the repository.

Building and Installing
-----------------------

Follow these steps to build and install the `kakuhen` library:

1.  **Clone the repository**:

    .. code-block:: bash

        git clone https://github.com/aykhuss/kakuhen.git
        cd kakuhen

2.  **Create a build directory and configure CMake**:

    It's recommended to build out-of-source.

    .. code-block:: bash

        mkdir build
        cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release

    *   `-DCMAKE_BUILD_TYPE=Release`: Specifies a Release build for optimized performance. You can use ``Debug`` for development.
    *   To enable building tests, add ``-DKAKUHEN_BUILD_TESTING=ON``.
    *   To enable building the CLI tool, add ``-DKAKUHEN_BUILD_CLI=ON``.

3.  **Build the library**:

    .. code-block:: bash

        cmake --build .

    This command will compile the library and any enabled examples or tools.

4.  **Install the library (Optional)**:

    If you want to install `kakuhen` to your system's default installation paths (e.g., ``/usr/local`` on Linux/macOS), run:

    .. code-block:: bash

        cmake --install .

    You can specify a custom installation prefix using ``cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install`` during the configuration step.

Using kakuhen in your project
-----------------------------

Once installed, you can use ``find_package(kakuhen CONFIG REQUIRED)`` in your own CMake projects to link against ``kakuhen::kakuhen``.

For example, in your ``CMakeLists.txt``::

    cmake_minimum_required(VERSION 3.18)
    project(MyProject LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(kakuhen CONFIG REQUIRED)

    add_executable(my_app main.cpp)
    target_link_libraries(my_app PRIVATE kakuhen::kakuhen)

And in ``main.cpp``::

    #include <kakuhen/kakuhen.h> // Or specific sub-headers

    int main() {
        // Use kakuhen library functions
        return 0;
    }
