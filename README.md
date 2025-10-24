# kakuhen[^1]
> A Monte Carlo integrator library for high dimensions supporting multiple algorithms.

## Clone the repository

External libraries are included in this repository through submodules. In addition to cloning the repository, you therefore also need to initialise and update those modules:
```bash
git clone git@github.com:aykhuss/kakuhen.git
cd kakuhen
git submodule init
git submodule update
```

or just:
```bash
git clone --recurse-submodules git@github.com:aykhuss/kakuhen.git
```

## Install

We use [CMake](https://cmake.org/) to build our project.
On macOS using [homebrew](https://brew.sh/), install it like so: `brew install cmake`.

To build and install the project, run:
```bash
mkdir build
cd build
cmake ..
make
make install
```


[^1]: kakuhen (確変), short for kakuritsu hendō (確率変動), means probability change and describes a system within the Japanese Pachinko (パチンコ) gambling game to enter "fever mode".
