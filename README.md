# kakuhen[^1]

> A modern C++20 header-only Monte Carlo integration library.

![Standard](https://img.shields.io/badge/standard-C%2B%2B20-blue.svg?logo=c%2B%2B)
![License](https://img.shields.io/badge/License-Apache_2.0-green)
[![codecov](https://codecov.io/github/aykhuss/kakuhen/graph/badge.svg?token=RMQXCWQ4XL)](https://codecov.io/github/aykhuss/kakuhen)

**kakuhen** provides efficient, type-safe implementations of adaptive Monte Carlo integration algorithms for high-dimensional functions. Designed for performance and ease of use, it allows seamless state serialization, enabling checkpointing and parallelization of integration tasks.

## Features

- **Header-only**: No separate library build required; include the headers in your C++20 application.
- **Modern C++**: Built with C++20 concepts and features.
- **Algorithms**:
  - **Plain**: Naive Monte Carlo sampling.
  - **VEGAS**: Classic adaptive importance sampling algorithm.
  - **BASIN**: Blockwise Adaptive Sampling with Interdimensional Nesting (for complex correlations).
- **Event generation**: `VegasGenerator` and `BasinGenerator` generate events using trained sampling envelopes, with support for signed integrands and overweight corrections.
- **Serialization**: Save/load grids, integration data, and generator checkpoints; merge data and envelope files from independent workers.
- **Type-safe**: Strongly typed interfaces to prevent configuration errors.

## Integration

### CMake FetchContent (Recommended)

You can include `kakuhen` directly in your project using `FetchContent`:

```cmake
include(FetchContent)

FetchContent_Declare(
  kakuhen
  GIT_REPOSITORY https://github.com/aykhuss/kakuhen.git
  GIT_TAG main
)
FetchContent_MakeAvailable(kakuhen)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE kakuhen::kakuhen)
```

### Add as Subdirectory

If you vendor or add the repo as a submodule:

```cmake
add_subdirectory(external/kakuhen)
target_link_libraries(my_app PRIVATE kakuhen::kakuhen)
```

### Install + find_package

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build
```

```cmake
find_package(kakuhen CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE kakuhen::kakuhen)
```

If CMake cannot find the package, set one of:

- `CMAKE_PREFIX_PATH` to the install prefix
- `kakuhen_DIR` to the package config directory

### Manual

Since it is header-only, you can simply copy the `include/kakuhen` directory to your project's include path or install it globally:

```bash
cmake -S . -B build
cmake --install build
```

## Quick Start

Here is a simple example integrating a 2D function using the BASIN algorithm:

```cpp
#include "kakuhen/kakuhen.h"
#include <iostream>

int main() {
  using namespace kakuhen::integrator;

  // 1. Define the integrand
  // Input: Point struct containing coordinates (x) and weight
  // Output: double (function value)
  auto func = [](const Point<>& p) {
    const auto& x = p.x;
    // Example: Integrate x^2 + y^2 over [0, 1]^2
    return x[0] * x[0] + x[1] * x[1];
  };

  // 2. Initialize Integrator (2 Dimensions)
  // Basin(ndim, ndiv1, ndiv2)
  auto basin = Basin(2, 8, 16);
  basin.set_seed(42); // Reproducibility

  // 3. Warmup (Adapt grid; discard the warmup result)
  std::cout << "Warming up...\n";
  basin.integrate(func, {
      .neval = 1000,
      .niter = 5,
      .adapt = true  // Update grid
  });

  // 4. Production Run (Fix/freeze grid, accumulate results)
  std::cout << "Running integration...\n";
  basin.set_options({.frozen = true}); // Freeze grid

  auto result = basin.integrate(func, {
      .neval = 10000,
      .niter = 10
  });

  // 5. Output results
  std::cout << "Result: " << result.value() << " +/- " << result.error() << "\n";
  std::cout << "Chi2/dof: " << result.chi2dof() << "\n";

  return 0;
}
```

`frozen = true` is the recommended option for production phases: it disables
further grid adaptation and skips collecting adaptation data.

## Advanced Usage

### Serialization & Checkpointing

`kakuhen` uses three file types for adaptive integrators and generators:

| File | Contents | Save / restore |
| --- | --- | --- |
| `.khs` | Integration grid; generator checkpoints also include the envelope, its diagnostics, and pending envelope data | `save()` / `load()` |
| `.khd` | Current integral accumulator and collected grid-adaptation statistics | `save_data()` / `append_data()` |
| `.khe` | Envelope factors and any pending observations with their batch statistics | `save_envelope()` / `merge_envelope()` |

An integrator's `.khs` file saves its grid, not accumulated integration data or
previous `Result` objects. Options and RNG state are not included: reapply the
options and seed after loading. Each `integrate()` call returns a new result.

```cpp
// Save the grid
basin.save("checkpoint.khs");

// ... application restart ...

// Restore the grid and configure a new production run
auto basin_resumed = Basin("checkpoint.khs");
basin_resumed.set_options({.frozen = true});
basin_resumed.set_seed(43);
auto resumed_result = basin_resumed.integrate(func, {.neval = 5000, .niter = 5});
```

### Distributed Data Collection

To refine a grid using independent runs, give every worker the same grid and
integrand, but a different seed. Workers must integrate with **`.adapt = false`
and `.frozen = false`**: keep the grid unchanged while still collecting adaptation
data. Enabling adaptation changes the grid and clears those data; freezing the
grid skips their collection entirely.

The following snippets use `func` and the namespace from Quick Start.

1. Save a common starting grid after warmup:

   ```cpp
   basin.save("grid.khs");
   ```

2. Run N workers, each loading that grid and saving its own data. For worker 1:

   ```cpp
   auto worker = Basin("grid.khs");
   worker.set_seed(101); // Use a different seed for each worker
   worker.integrate(func, {
       .neval = 10000,
       .niter = 1,
       .adapt = false,  // Defer adaptation until all worker data are merged
       .frozen = false  // Keep collecting adaptation data
   });
   worker.save_data("run_1.khd"); // Worker 2 writes run_2.khd, etc.
   ```

3. Load the same grid in a coordinator and append each worker's file once.
   Adapt only after all files have been appended:

   ```cpp
   auto combined = Basin("grid.khs");
   combined.append_data("run_1.khd");
   combined.append_data("run_2.khd");
   combined.adapt(); // Refine the grid using the combined data
   combined.save("refined_grid.khs");
   ```

Distribute the refined grid for the next round. See
[`examples/example.cpp`](examples/example.cpp) for a runnable VEGAS example.

### Distributed Envelope Training & Event Generation

`BasinGenerator<>` and `VegasGenerator<>` can collect envelope-training data in
independent runs and combine their `.khe` files. This workflow requires a
**frozen integration grid**. `collect_envelope()` records observations that exceed
the current envelope without changing it; `merge_envelope()` applies the imported
observations to the combined envelope.

Using the same `func` as above:

1. Adapt the integration grid, freeze it, initialize an envelope, and save a
   common generator checkpoint:

   ```cpp
   using Generator = BasinGenerator<>;
   Generator generator(2, 8, 16);
   generator.set_seed(42);
   generator.integrate(func, {.neval = 20000, .niter = 5, .adapt = true});
   generator.set_options({.frozen = true});
   generator.initialize_envelope(func, 10000); // Estimate the absolute integral
   generator.save("generator.khs");
   ```

2. Run N workers with distinct seeds and output files. For worker 101:

   ```cpp
   Generator worker(2);
   worker.load("generator.khs");
   worker.set_options({.frozen = true}); // Options are not saved in .khs
   worker.set_seed(101);
   worker.collect_envelope(func, 50000); // Collect without updating the envelope
   worker.save_envelope("worker_101.khe");
   // Worker 102 repeats this with seed 102 and worker_102.khe, etc.
   ```

3. Load the common checkpoint and merge each worker file once, in a fixed order:

   ```cpp
   Generator combined(2);
   combined.load("generator.khs");
   combined.set_options({.frozen = true});
   combined.merge_envelope("worker_101.khe");
   combined.merge_envelope("worker_102.khe");
   combined.save("generator_trained.khs");
   ```

   Each merge takes the maximum of corresponding envelope factors, then replays
   the incoming observations and adds their batch statistics. The envelope is
   ready for generation immediately; no additional `adapt_envelope()` call is
   needed. Files must match the generator type, numeric types, and sampling map
   (including BASIN's sampling order). All workers must use the same integrand
   and parameters; these are not checked by the file format.

4. Load the trained checkpoint in a production worker and generate events:

   ```cpp
   Generator production(2);
   production.load("generator_trained.khs");
   production.set_options({.frozen = true});
   production.set_seed(1001);
   double weight_sum = 0.0;
   auto events = production.generate_trials(
       func, 100000, [&](const Point<>& /*point*/, double weight) {
         // Store event coordinates and this weight, or fill histograms here.
         weight_sum += weight;
       });
   std::cout << "Events: " << events.n_events() << "\n";
   std::cout << "Integral: " << events.normalization() * weight_sum << "\n";
   ```

`generate_trials()` uses a fixed number of trials, so the accepted event count
varies. Keep the callback weights: they are normally `+1` or `-1`, with larger
magnitudes when the envelope is exceeded. The returned `normalization()` converts
their sum into an integral estimate.

Saving a `.khe` file does **not** clear the pending batch. Before collecting a new
batch for a separate export, call `clear_envelope_data()` to avoid exporting the
same observations and statistics again. Alternatively, `adapt_envelope()` applies
and clears a local batch. Files with observations must be merged once in a
consistent order: repeated imports count their batches again, and order can
affect the trained envelope. Files containing only factors can also be saved
after `raise_envelope()` or `optimize_envelope()`; merging those is order
independent and idempotent.

Collection stores violating observations in memory. An optional third argument,
`max_records`, caps the total pending record count; inspect the returned
`status()` and `n_evaluations()` if collection stops at that limit.

See [`examples/example_distributed_generator.cpp`](examples/example_distributed_generator.cpp)
for a complete example that simulates four workers, merges their files, and
generates events. It writes its checkpoints and envelopes under
`distributed_generator/` in the current directory:

```bash
cmake -S . -B build -DKAKUHEN_BUILD_EXAMPLES=ON
cmake --build build --target example_distributed_generator
./build/examples/example_distributed_generator
```

## Development

### Requirements

-   C++20 compiler and standard library with `std::format` support
-   CMake 3.18+

### Building Tests

```bash
cmake -S . -B build -DKAKUHEN_BUILD_TESTING=ON
cmake --build build
cd build && ctest --output-on-failure
```

### Building Documentation

Requires Doxygen and Sphinx.

```bash
# Install dependencies
pip install sphinx breathe furo

# Configure & Build
cmake -S . -B build -DKAKUHEN_BUILD_DOCS=ON
cmake --build build --target build_sphinx_html
```
Docs will be generated in `docs/sphinx/_build/html`.

## CMake Options

- `KAKUHEN_BUILD_TESTING`: Build tests.
- `KAKUHEN_BUILD_CLI`: Build the CLI tool.
- `KAKUHEN_BUILD_EXAMPLES`: Build examples.
- `KAKUHEN_BUILD_DOCS`: Build documentation.
- `KAKUHEN_ENABLE_COVERAGE`: Enable coverage (GCC/Clang only).

## Support

Use GitHub Issues for bugs and feature requests.

## License

Distributed under the Apache-2.0 License. See `LICENSE` for more information.

---

[^1]: **kakuhen** (確変), short for *kakuritsu hendō* (確率変動), means "probability change" and describes a system within the Japanese Pachinko (パチンコ) gambling game to enter "fever mode".
