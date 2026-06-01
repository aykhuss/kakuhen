Histograms Guide
================

This guide explains how to use the histogramming facilities in the `kakuhen`
library. The histogram subsystem is designed for Monte Carlo workloads where
fills are frequent and buffering intermediate data is useful.

Core components
---------------

- **HistogramRegistry**: A central manager for histogram definitions and global storage.
- **HistogramBuffer**: A temporary fill buffer used to batch histogram updates before flushing.
- **Axes**: Definitions of how continuous coordinates map to discrete bins (Uniform or Variable).

Basic workflow
--------------

1. **Initialize the registry**: create a ``HistogramRegistry`` instance.
2. **Define axes**: create self-contained axis objects, either uniform or variable.
3. **Book histograms**: register named histograms with ``book()``.
4. **Fill via buffers**: create a ``HistogramBuffer`` and use it during your event loop.
5. **Flush results**: merge the buffered contributions into the registry.

Example usage
-------------

.. code-block:: cpp

    #include "kakuhen/histogram/axis.h"
    #include "kakuhen/histogram/histogram_registry.h"
    #include <iostream>

    using namespace kakuhen::histogram;

    int main() {
        // 1. Create the registry
        HistogramRegistry<> registry;

        // 2. Define a uniform axis: 10 bins from 0.0 to 100.0
        UniformAxis<> x_axis(10, 0.0, 100.0);

        // 3. Book a histogram using this axis
        auto hist_id = registry.book("my_histogram", 1, x_axis);

        // 4. Create a fill buffer
        auto buffer = registry.create_buffer();

        // 5. Fill the histogram (e.g., in a simulation loop)
        for (int i = 0; i < 1000; ++i) {
            double x = 50.0;
            double weight = 1.0;
            registry.fill(buffer, hist_id, weight, x);
        }

        // 6. Flush the buffer to the global storage
        registry.flush(buffer);

        // 7. Access results
        std::cout << "Value at 50.0: " << registry.value(hist_id, 1) << std::endl;

        return 0;
    }

Numerical stability
-------------------

The underlying accumulators use compensated summation so that fills remain
stable even when weights exhibit strong cancellations, for example from
positive and negative Monte Carlo contributions.

Usage notes
-----------

``HistogramBuffer`` is useful when you want to batch many fills and flush them
into the registry later. This reduces per-fill overhead in tight loops and
keeps histogram update logic separate from final aggregation.

Performance tips
----------------

* Reuse a buffer across the event loop instead of recreating it for every fill.
* Flush in batches (e.g., end of a loop or chunk) instead of after every fill.
* Prefer uniform axes where possible for lower overhead.
