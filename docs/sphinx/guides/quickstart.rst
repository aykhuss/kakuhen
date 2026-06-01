Quickstart
==========

This guide shows the smallest complete `kakuhen` program and explains the
pieces you need for a first multidimensional Monte Carlo integration.

Basic usage
-----------

As a concrete example, integrate

.. math::

   f(x, y) = x^2 + y^2

over the two-dimensional unit square.

The example below uses the ``Plain`` integrator, which samples uniformly and is
usually the easiest entry point when you just want a baseline result.

.. literalinclude:: ../example_quickstart.cpp
   :language: cpp
   :start-after: // [example-start]
   :end-before: // [example-end]
   :dedent: 0

Explanation:

* We include ``kakuhen/integrator/plain.h``.
* The integrand receives a ``Point<>`` object containing the sample coordinates
  in ``p.x``, the sampling weight in ``p.weight``, and metadata such as
  ``sample_index`` and ``user_data``.
* ``Plain(2)`` creates a non-adaptive integrator for a two-dimensional unit
  hypercube.
* ``neval`` is the number of samples per iteration and ``niter`` is the number
  of statistically independent iterations.
* ``integrate()`` returns a ``Result`` object that stores both the aggregate
  estimate and the individual per-iteration entries.
* ``value()``, ``error()``, and ``chi2dof()`` summarize the final estimate.

Choosing ``neval`` and ``niter``
--------------------------------

The usual pattern is:

* increase ``neval`` to reduce the Monte Carlo error of each iteration
* use ``niter > 1`` when you want a meaningful cross-iteration error estimate
  and ``chi2/dof``

For exploratory runs, small values such as ``neval = 10000`` and ``niter = 5``
are often enough. For production runs, it is common to warm up an adaptive
integrator first and then perform a larger frozen run.

Reading per-iteration entries
-----------------------------

``Result`` also exposes the stored per-iteration accumulators:

.. code-block:: cpp

    auto result = plain_integrator.integrate(
        func,
        {.neval = num_evaluations, .niter = num_iterations, .verbosity = 0});

    for (const auto& entry : result.entries()) {
      std::cout << entry.value() << " +/- " << entry.error() << '\n';
    }

You can also use indexed access:

.. code-block:: cpp

    const auto& first_iteration = result[0];
    std::cout << first_iteration.value() << '\n';

Progress output
---------------

The two-argument ``integrate()`` overload can show a built-in terminal progress
bar when ``verbosity > 0``. For quiet runs, pass ``verbosity = 0``. For custom
reporting, use the three-argument overload with a progress callback as shown in
the :doc:`Progress Reporting <progress>` guide.

Build and run
-------------

To compile and run this example:

1. Save the code above as ``my_integration_app.cpp``.
2. Assuming `kakuhen` is installed and found by CMake, compile and link your
   application with:

   .. code-block:: cmake

      cmake_minimum_required(VERSION 3.18)
      project(MyIntegrationApp LANGUAGES CXX)

      set(CMAKE_CXX_STANDARD 20)
      set(CMAKE_CXX_STANDARD_REQUIRED ON)

      find_package(kakuhen CONFIG REQUIRED)

      add_executable(my_integration_app my_integration_app.cpp)
      target_link_libraries(my_integration_app PRIVATE kakuhen::kakuhen)

   Then configure and build:

   .. code-block:: bash

      cmake -S . -B build
      cmake --build build

3. Run the executable:

   .. code-block:: bash

      ./my_integration_app
      .\\my_integration_app.exe

This should print the estimated integral value and error to the console.

Next steps
----------

* See :doc:`Progress Reporting <progress>` for built-in and custom progress
  updates.
* See :doc:`API Reference <../api/index>` for the integrator classes and core
  data structures.
* Move on to ``Vegas`` or ``Basin`` when you want adaptive importance sampling.
