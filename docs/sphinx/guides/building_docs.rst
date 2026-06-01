Building Documentation
======================

This guide explains how to build the Doxygen XML and Sphinx site locally.

Requirements
------------

* Doxygen, for XML generation
* Sphinx with the ``breathe`` and ``furo`` Python packages
* CMake 3.18 or newer

On Ubuntu, for example:

.. code-block:: bash

    sudo apt-get update
    sudo apt-get install -y doxygen
    python -m pip install --upgrade pip
    pip install sphinx breathe furo

Build the docs
--------------

The docs are generated via CMake targets:

.. code-block:: bash

    cmake -S . -B build -DKAKUHEN_BUILD_DOCS=ON
    cmake --build build --target build_sphinx_html

Output is written to ``docs/sphinx/_build/html``.

If the target succeeds, open ``docs/sphinx/_build/html/index.html`` in a
browser.

Troubleshooting
---------------

If Breathe cannot find the Doxygen XML output, make sure you configured with
``-DKAKUHEN_BUILD_DOCS=ON`` and that the build directory is the same one you
used to generate Doxygen.
