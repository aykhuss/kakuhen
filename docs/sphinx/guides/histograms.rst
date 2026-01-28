Histograms Guide
================

This guide explains how to use the histogramming facilities in the `kakuhen` library.
The library provides a high-performance, thread-safe histogramming system optimized
for Monte Carlo simulations.

Core Components
---------------

- **HistogramRegistry**: A central manager for histogram definitions and global storage.
- **HistogramBuffer**: A thread-local buffer used for high-frequency filling to avoid memory contention.
- **Axes**: Definitions of how continuous coordinates map to discrete bins (Uniform or Variable).

Basic Workflow
--------------

1.  **Initialize the Registry**: Create a `HistogramRegistry` instance.
2.  **Define Axes**: Create self-contained `Axis` objects (Uniform or Variable).
3.  **Book Histograms**: Register named histograms using the `Axis` objects via `book()`.
4.  **Fill via Buffers**: Create a `HistogramBuffer` for each thread and fill it during your event loop.
5.  **Flush Results**: Flush the buffers into the registry to aggregate global results.

Example Usage
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
        Uniform<> x_axis(10, 0.0, 100.0);

        // 3. Book a histogram using this axis
        auto hist_id = registry.book("my_histogram", 1, x_axis);

        // 4. Create a thread-local buffer
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

Handling Cancellations
----------------------

The `BinAccumulator` defaults to using `TwoSum` summation, which provides high
numerical stability even when accumulating weights with large cancellations
(e.g., +W and -W). This is crucial for precise Monte Carlo error estimation.

Thread Safety
-------------

The `HistogramRegistry` itself is intended to be used for setup and final aggregation.
Filling is performed via `HistogramBuffer` objects, which are designed to be
**thread-local**. Multiple threads can fill their own buffers concurrently without
locks. A synchronization point is required only when calling `flush()`.
