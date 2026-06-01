
kakuhen documentation
=====================

A header-only C++20 library for multidimensional Monte Carlo integration,
adaptive importance sampling, and high-throughput histogramming.

Key features
------------

* Header-only library with modern C++20 interfaces.
* Multiple integrators with different tradeoffs: ``Plain``, ``Vegas``, and ``Basin``.
* Built-in progress reporting via callbacks or a default terminal progress bar.
* Histogramming utilities designed for Monte Carlo workflows.
* Header-only consumption via ``add_subdirectory`` or ``find_package``.

Getting started
---------------

Start with the :doc:`Quickstart <guides/quickstart>` to run your first integration,
then see :doc:`Installation <guides/installation>` for recommended CMake usage.
If you want live progress updates or custom reporting, continue with
:doc:`Progress Reporting <guides/progress>`.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guides/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
