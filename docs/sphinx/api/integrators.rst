Integrators
===========

The ``kakuhen::integrator`` namespace contains the numerical integration
algorithms, common configuration types, progress-reporting types, and result
containers.

Main classes
------------

Use ``Plain`` for simple uniform Monte Carlo sampling, ``Vegas`` for adaptive
one-dimensional importance sampling, and ``Basin`` for a more aggressive
adaptive scheme that also models inter-dimensional structure.

.. doxygenclass:: kakuhen::integrator::Plain
   :members:

.. doxygenclass:: kakuhen::integrator::Vegas
   :members:

.. doxygenclass:: kakuhen::integrator::Basin
   :members:

Common support types
--------------------

These types are shared by all integrators:

.. doxygenstruct:: kakuhen::integrator::Point
   :members:

.. doxygenclass:: kakuhen::integrator::Result
   :members:

.. doxygenstruct:: kakuhen::integrator::ProgressEvent
   :members:

.. doxygenenum:: kakuhen::integrator::ProgressEventKind

.. doxygenenum:: kakuhen::integrator::EventSignal

Options and configuration
-------------------------

``Options`` is the common configuration object accepted by all integrators.
Not every field is meaningful for every integrator, but the type is shared so
that code can switch integrator implementations without rewriting option
handling.

.. doxygenstruct:: kakuhen::integrator::Options
   :members:
