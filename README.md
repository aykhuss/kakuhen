# kakuhen[^1]
> A Monte Carlo integrator library that provides different algorithms for high-dimensional integration.

## Install

We use [CMake](https://cmake.org/) for our project.
On macOS using [homebrew](https://brew.sh/), install it like so: `brew install cmake`.


## Building Documentation

To build the project's documentation, ensure you have Doxygen and Python (with Sphinx and Breathe installed via `pip install sphinx breathe`) available on your system.

Then, from the project root directory, follow these steps:

1.  **Configure CMake with Documentation Enabled**:
    ```bash
    cmake -S . -B build -DKAKUHEN_BUILD_DOCS=ON
    ```
    Replace 'build' with your preferred build directory if different.

2.  **Build the Documentation Target**:
    ```bash
    cmake --build build --target build_sphinx_html
    ```

After a successful build, the HTML documentation will be located in `build/docs/sphinx/_build/html` (or your chosen build directory). You can open `index.html` in your web browser to view it.


[^1]: kakuhen (確変), short for kakuritsu hendō (確率変動), means probability change and describes a system within the Japanese Pachinko (パチンコ) gambling game to enter "fever mode".