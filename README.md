# kakuhen[^1]

> A modern C++20 header-only Monte Carlo integration library.

![Standard](https://img.shields.io/badge/standard-C%2B%2B20-blue.svg?logo=c%2B%2B)
![License](https://img.shields.io/badge/License-Apache_2.0-green)

**kakuhen** provides efficient, type-safe implementations of adaptive Monte Carlo integration algorithms for high-dimensional functions. Designed for performance and ease of use, it allows seamless state serialization, enabling checkpointing and parallelization of integration tasks.

## Features

- **Header-only**: No compilation required, just include and go.
- **Modern C++**: Built with C++20 concepts and features.
- **Algorithms**:
  - **Plain**: Naive Monte Carlo sampling.
  - **VEGAS**: Classic adaptive importance sampling algorithm.
  - **BASIN**: Blockwise Adaptive Sampling with Interdimensional Nesting (for complex correlations).
- **Serialization**: Save/load integrator state and accumulated data to/from disk (great for long-running jobs or distributed computing).
- **Type-safe**: Strongly typed units and checking to prevent configuration errors.

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

### Manual

Since it is header-only, you can simply copy the `include/kakuhen` directory to your project's include path or install it globally:

```bash
cmake -S . -B build
cmake --install build
```

## Quick Start

Here is a simple example integrating a 2D function using the VEGAS algorithm:

```cpp
#include "kakuhen/kakuhen.h"
#include <iostream>
#include <cmath>

int main() {
  using namespace kakuhen::integrator;

  // 1. Define the integrand
  // Input: Point struct containing coordinates (x) and weight
  // Output: double (function value)
  auto func = [](const Point<>& p) {
    const auto& x = p.x; 
    // Example: Integrate x^2 + y^2 over [0, 1]^2
    return x[0]*x[0] + x[1]*x[1];
  };

  // 2. Initialize Integrator (2 Dimensions)
  // Vegas(ndim, n_grid_bins)
  auto vegas = Vegas(2, 32); 
  vegas.set_seed(42); // Reproducibility

  // 3. Warmup (Adapt grid without recording final result)
  std::cout << "Warming up...\n";
  vegas.integrate(func, {
      .neval = 1000, 
      .niter = 5, 
      .adapt = true  // Update grid
  });

  // 4. Production Run (Fix grid, accumulate results)
  std::cout << "Running integration...\n";
  vegas.set_options({.adapt = false}); // Freeze grid
  
  auto result = vegas.integrate(func, {
      .neval = 10000, 
      .niter = 10
  });

  // 5. Output results
  std::cout << "Result: " << result.value() << " +/- " << result.error() << "\n";
  std::cout << "Chi2/dof: " << result.chi2dof() << "\n";

  return 0;
}
```

## Advanced Usage

### Serialization & Checkpointing

`kakuhen` allows you to save the full state of an integrator (grid, accumulated stats) to a file and resume later.

```cpp
// Save state
vegas.save("checkpoint.khs");

// ... application restart ...

// Load state
auto vegas_resumed = Vegas("checkpoint.khs");
vegas_resumed.integrate(func, {.neval = 5000, .niter = 5});
```

### Distributed Data Collection

You can run identical integrators in parallel (with different seeds), save their data to disk, and merge them later.

1.  Run N instances, each saving data: `vegas.save_data("run_1.khd");`
2.  Merge in a master process:
    ```cpp
    vegas.append_data("run_1.khd");
    vegas.append_data("run_2.khd");
    vegas.adapt(); // Refine grid based on combined data
    ```

## Development

### Requirements

-   C++20 compliant compiler (GCC 10+, Clang 10+, MSVC 19.29+)
-   CMake 3.18+

### Building Tests

```bash
cmake -S . -B build -DKAKUHEN_BUILD_TESTING=ON
cmake --build build
cd build/tests && ctest --output-on-failure
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
Docs will be generated in `build/docs/sphinx/_build/html`.

## License

Distributed under the Apache-2.0 License. See `LICENSE` for more information.

---

[^1]: **kakuhen** (確変), short for *kakuritsu hendō* (確率変動), means "probability change" and describes a system within the Japanese Pachinko (パチンコ) gambling game to enter "fever mode".
