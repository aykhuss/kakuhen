Quickstart
==========

This guide provides a quick introduction to using the `kakuhen` library for
multidimensional Monte Carlo integration.

Basic usage
-----------

Let's integrate a simple function, for example, ``f(x, y) = x^2 + y^2`` over a 2D domain.

First, define your function and use the ``Plain`` integrator.

.. literalinclude:: ../example_quickstart.cpp
   :language: cpp
   :start-after: // [example-start]
   :end-before: // [example-end]
   :dedent: 0

Explanation:
*   We include necessary headers, specifically ``kakuhen/integrator/plain.h``.
*   We define our integrand function ``func`` which takes a ``Point<>`` object. The coordinates are accessed via ``p.x[i]``.
*   We create an instance of ``kakuhen::integrator::Plain`` for a 2-dimensional integration.
*   We define ``num_evaluations`` and ``num_iterations`` for the Monte Carlo process.
*   The ``integrate`` method performs the integration and returns a ``Result`` object.
*   Finally, we print the estimated integral value, its error, and the chi-squared per degree of freedom.

Build and run
-------------

To compile and run this example:

1.  Save the code above (from the ``literalinclude`` block) as ``my_integration_app.cpp``.
2.  Assuming `kakuhen` is installed and found by CMake, you can compile and link your application using CMake:

    .. code-block:: cmake

        # CMakeLists.txt for your application
        cmake_minimum_required(VERSION 3.18)
        project(MyIntegrationApp LANGUAGES CXX)

        set(CMAKE_CXX_STANDARD 20)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)

        find_package(kakuhen CONFIG REQUIRED)

        add_executable(my_integration_app my_integration_app.cpp)
        target_link_libraries(my_integration_app PRIVATE kakuhen::kakuhen)

    Then, from your build directory:

    .. code-block:: bash

        cmake -S . -B build
        cmake --build build

3.  Execute the application:

    .. code-block:: bash

        ./my_integration_app # On Linux/macOS
        .\\my_integration_app.exe # On Windows

This should print the estimated integral value and error to the console.

For more advanced usage and different integration algorithms (e.g., ``Vegas``, ``Basin Hopping``), refer to the :doc:`API Reference <../api/index>` and other guides.
