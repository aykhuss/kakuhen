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

Distributed envelope training
-----------------------------

``VegasGenerator`` and ``BasinGenerator`` support immediate envelope training
(``raise_envelope``, ``optimize_envelope``) and deferred observation collection
(``collect_envelope``). Collection leaves the envelope unchanged and records
every violating observation, in evaluation order, as the complete tuple of
factor indices and the absolute transformed contribution ``|J f(x)|``.

``generate_trials()`` checks that the sampled product of envelope factors
``R`` is finite and positive. Otherwise it throws ``std::runtime_error``
before evaluating the integrand.

For distributed collection, first adapt an integration grid, freeze it,
initialize an envelope, and save a common ``.khs`` checkpoint. Each worker
loads that checkpoint and uses its own RNG seed:

.. code-block:: cpp

   using Generator = kakuhen::integrator::BasinGenerator<>;
   Generator worker(2);
   worker.load("campaign.khs");
   // Options are not stored in .khs; set them explicitly after loading.
   worker.set_options({.frozen = true});
   worker.set_seed(101);
   auto result = worker.collect_envelope(integrand, 800000, /*max_records=*/100000);
   worker.save_envelope("worker-101.khe");

Collection uses the requested evaluation budget; the fixed envelope gives no
convergence measure for early stopping. The optional ``max_records`` bounds the
total pending record count across calls. It stops collection immediately after
the observation that fills the limit; an already full buffer performs no further
evaluations, and zero prevents collection. The default is unlimited (subject to
available memory). There is no disk spilling or silent record dropping.

The result's ``status()`` is ``EnvelopeStatus::BUDGET_EXHAUSTED``,
``TARGET_REACHED`` (``optimize_envelope`` only), or ``RECORD_LIMIT_REACHED``
(``collect_envelope`` only). ``n_evaluations()``, ``n_violations()``, ``n_raised()``
and ``n_nonfinite()`` describe the whole call. A violation is an evaluation whose
``|f|`` exceeded the envelope as it stood when checked, so
``n_violations() / n_evaluations()`` is the violation rate of every call;
``n_raised()`` counts envelope updates (equal to ``n_violations()`` for raising
passes, zero for collection, at most ``n_violations()`` on replay);
``adapt_envelope`` and ``merge_envelope`` do not sample and report no non-finite samples.
``count()`` continues to describe the cumulative absolute-integral estimate.
``optimize_envelope`` tests convergence using the last pass's violation fraction.

The coordinator loads the same grid and imports the worker files:

.. code-block:: cpp

   Generator combined(2);
   combined.load("campaign.khs");
   combined.set_options({.frozen = true});
   combined.merge_envelope("worker-101.khe");
   combined.merge_envelope("worker-102.khe");
   combined.save("combined.khs");

Every merge takes the maximum of corresponding factors, including the
coordinator's current envelope if it is ready, then replays the incoming observations
once in file order. It returns an ``EnvelopeResult`` for the incoming batch, like
``adapt_envelope``: its evaluations, its records (``n_violations``) and the
records that raised the envelope (``n_raised``). Each observation is checked against the current bound
and triggers at most one standard MINT raise. Covered observations are skipped.
The proposal caches are rebuilt after replay.
The result is immediately usable for generation or further raising.
Without a ready envelope (uninitialized or stale), the
import supplies the factors; such an envelope must not have pending data.
Envelope-only merging is order independent and idempotent. With observations,
import order matters and importing a file twice repeats its observations and
statistics; preventing accidental duplicates is the caller's responsibility.
Workers may have different initialization seeds, training histories, and budgets;
the common starting checkpoint is a reproducibility convention. Factorwise
maximum ensures that observations omitted below each worker's collection bound
remain covered when that worker's records are replayed.

Imports validate the generator and numeric types, table shape, and full
sampling-map fingerprint, including BASIN's sampling order. Callers must
ensure workers evaluated the same integrand and configuration. A failed
import leaves the generator unchanged, including its proposal caches.

Saving reserves a unique staging directory beside the destination and
renames its payload into place only once writing and closing succeed.
Concurrent processes use independent staging directories, so each publishes
a complete file; the last successful rename wins. Failed saves clean up
only their own staging directory. An interrupted process can leave its
staging directory behind, but does not truncate an existing destination.
A destination directory that is missing or not writable is reported as
``std::filesystem::filesystem_error`` rather than ``std::ios_base::failure``.
This holds for ``.khs`` and ``.khd`` files as well.

``.khe`` files contain factors, compatibility metadata, and a batch-presence
flag. A pending batch adds ordered observations and statistics for all its
evaluations, including those below the bound and non-finite values counted as
zero. Initialization history and immediate-training statistics are not exported.
Imports add these batch statistics once to the receiver's absolute-integral
estimate. If the merged evaluation count does not fit ``count_type``, the import
throws ``std::overflow_error`` and leaves the generator unchanged.
Envelope-only imports leave that estimate unchanged. Merging into a
ready envelope preserves the local seed; an import that replaces the envelope
sets ``envelope_seed()`` to zero. ``predicted_efficiency()`` needs a positive
seed or absolute-integral estimate.

Saving does not consume a batch. A batch with no violating records still
carries its evaluation statistics. ``has_envelope_data()`` distinguishes it
from no batch; ``n_envelope_records()`` reports the pending record count.
Full ``.khs`` checkpoints also preserve pending batches, as long as the envelope
is ready; loading drops the batch of an envelope that is not.

``adapt_envelope()`` replays and consumes the local batch without evaluating
the integrand or adding its statistics again. Its ``n_evaluations()`` describes
the evaluations represented by the batch, ``n_violations()`` its records and
``n_raised()`` the actual raises. File merging never consumes the receiver's local pending batch.
Both operations build a replacement envelope before committing, so a failure
preserves the original factors, proposal caches, diagnostics, and local batch.

``clear_envelope_data()`` discards pending observations and export statistics,
retaining the envelope and cumulative diagnostics. Use it after saving to start
another collection batch. While data are pending, envelope increases and
immediate training are allowed, but reinitialization and scaling below one are
rejected. A changed sampling map invalidates the envelope and its observations;
clear pending data before reinitializing on the new map.

Factorwise maximum dominates each worker's envelope but can exceed their
pointwise maximum, reducing acceptance efficiency. Training and merging
do not guarantee a true upper bound; overweight events still use the
generator's weighted correction. Merging does not reproduce serial training.

The no-argument ``save_envelope()`` and ``merge_envelope()`` overloads use
``file_envelope()``, the ``.khd`` naming convention with a ``.khe`` extension:
grid prefix plus RNG seed, or ``<file_path stem>.s<seed>.khe`` when overridden.
Use explicit paths for multiple snapshots with the same seed. Distribute
the combined ``.khs`` to production workers and explicitly enable frozen mode.

Generator checkpoints and standalone ``.khe`` payloads keep version 1, with
the new batch fields always present. Earlier development layouts are not
supported. Fingerprints cover the full sampling map, including BASIN's order.
BASIN ``.khd`` fingerprints cover the sampling map, rejecting data from a
different grid or sampling order; VEGAS ``.khd`` files keep the grid hash.
Data-file layouts may change during development; files must use the current
layout.
BASIN collects raw ``G`` without reversal reweighting or per-sample range bounds. Before
refinement, ``adapt()`` checks all grid-cell sums and throws
``std::overflow_error`` if any is non-finite. This failure leaves the grids,
sampling order and pending data unchanged; call ``clear_data()`` to discard
the invalid batch. Importing data with non-finite cell sums is also rejected.
Data imports check both the integral and adaptation counts, throwing
``std::overflow_error`` before changing the integrator if either merged count
does not fit ``count_type``. Local sampling requires the configured count type to
hold all cumulative evaluation counts.
VEGAS and BASIN data include cumulative integral statistics for all pending
collection batches. Snapshots saved during an iteration include all completed
samples. Callbacks may save data or call ``clear_data()`` between samples.
Clearing discards the pending grid data and their integral statistics; later
collection starts fresh, while the result returned by ``integrate()`` still
includes all samples evaluated in that call. Callbacks must not adapt or
reload the integrator while it collects. A non-finite sample in any pending
batch carries over into these statistics, so merging the file fails. Calls with
``adapt = false`` continue collecting, while ``frozen = true`` leaves pending
data unchanged. BASIN ranks candidate parent choices by local factor gains
strictly above ``min_gain`` (default ``0.03``). It assembles an acyclic temporary
forest, then restores compatible original parents for unselected dimensions.
Parents can themselves acquire a parent later in the same adaptation; reversed
choices do not force them to remain roots. After each accepted choice, BASIN
scores the completed forest using every changed factor, including any negative
gain for a fallback root. It keeps the best completed forest whose summed gain
exceeds ``min_gain``. Equal local gains retain dimension scan order.

The sum is a first-order variance model: it omits interactions between factor
changes and uses the grids before refinement. It does not guarantee lower
variance on the next iteration or a globally best forest. Selection uses the
existing grid sums without extra aggregate dimensions or retained samples.

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
