Progress Reporting
==================

`kakuhen` supports two styles of progress reporting during integration:

* a built-in terminal progress bar for the two-argument ``integrate()`` overload
* a user-supplied callback that receives structured progress events

The callback interface is the more general mechanism. The built-in progress bar
is a thin wrapper around that same event stream.

Default progress bar
--------------------

If you call the two-argument overload

.. code-block:: cpp

    auto result = integrator.integrate(func, {.neval = 100000, .niter = 5, .verbosity = 1});

then `kakuhen` installs a default progress bar automatically when all of the
following are true:

* ``progress_bar`` is unset or ``true``
* ``verbosity > 0`` for that integration call
* you are using the overload without an explicit callback

To suppress the default bar, set ``progress_bar = false`` or pass an explicit
callback (including ``nullptr``).

.. code-block:: cpp

    auto result = integrator.integrate(
        func,
        {.neval = 100000, .niter = 5, .verbosity = 0, .progress_bar = false});

Progress callbacks
------------------

For custom reporting, pass a callback as the third argument to ``integrate()``.
The callback receives a ``ProgressEvent<T, U>`` and returns an
``EventSignal``.

.. code-block:: cpp

    using namespace kakuhen::integrator;

    auto callback = [](const ProgressEvent<double, std::uint64_t>& ev) {
      if (ev.kind == ProgressEventKind::ITER_END) {
        std::cout << "iteration " << (ev.current_iter + 1)
                  << "/" << ev.niter
                  << ": value = " << ev.value
                  << " +/- " << ev.error << '\n';
      }
      return EventSignal::NONE;
    };

    auto result = integrator.integrate(
        func,
        {.neval = 100000, .niter = 5, .verbosity = 0},
        callback);

Event kinds
-----------

The callback can receive the following events:

* ``START``: emitted once before the first iteration begins
* ``ITER_START``: emitted before each iteration
* ``EVAL_MILESTONE``: emitted when the current iteration crosses the next progress threshold
* ``ITER_END``: emitted after the current iteration is accumulated into the final ``Result``
* ``END``: emitted once after integration finishes or is cancelled

Milestone spacing is controlled by ``progress_step``, which is interpreted per
iteration rather than over the entire run.

.. code-block:: cpp

    integrator.set_options({.progress_step = 0.1});  // 10 milestones per iteration

Important fields in ``ProgressEvent``
-------------------------------------

The most commonly used fields are:

* ``current_iter``: zero-based iteration index
* ``niter``: total number of iterations
* ``current_eval``: completed evaluations in the current iteration
* ``neval``: requested evaluations per iteration
* ``fraction``: per-iteration progress in ``[0, 1]``
* ``value`` and ``error``: current aggregate estimate from completed iterations
* ``elapsed_start`` and ``elapsed_iter``: elapsed wall-clock time in seconds

One subtle but important detail is that ``value`` and ``error`` are
cross-iteration aggregates. Mid-iteration milestone events therefore report the
state accumulated from completed iterations, not the partially completed
current iteration.

Cancelling an integration
-------------------------

The callback can stop integration early by returning ``EventSignal::CANCEL``.

.. code-block:: cpp

    auto stop_early = [](const ProgressEvent<double, std::uint64_t>& ev) {
      if (ev.kind == ProgressEventKind::EVAL_MILESTONE && ev.fraction >= 0.5) {
        return EventSignal::CANCEL;
      }
      return EventSignal::NONE;
    };

Integration then returns the partial ``Result`` accumulated so far.

Live terminal example
---------------------

See ``examples/example_live_progress.cpp`` for a complete example that wires
progress events into ``kakuhen::util::ProgressBar``.
