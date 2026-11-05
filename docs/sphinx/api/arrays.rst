Arrays
======

.. Explicit types avoid rendering NDView deduction guides as functions before
   the class, which causes a symbol collision in Sphinx's C++ domain.

.. doxygenclass:: kakuhen::ndarray::NDArray
   :members:

.. doxygenclass:: kakuhen::ndarray::NDView
   :members:

``NDView`` deduces its element and index types from the array, preserving
element constness:

.. code-block:: cpp

   kakuhen::ndarray::NDArray<double> array({2, 3});
   kakuhen::ndarray::NDView view(array);
   const auto& const_array = array;
   kakuhen::ndarray::NDView const_view(const_array);

.. doxygenstruct:: kakuhen::ndarray::Slice
   :members:

.. doxygenvariable:: kakuhen::ndarray::_
