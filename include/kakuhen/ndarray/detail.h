#pragma once

#include <cassert>
#include <cstddef>

namespace kakuhen::ndarray::detail {

template <typename S>
inline S flat_index(const S* idx, const S* strides, const S* shape, S ndim) noexcept {
  S offset = 0;
  for (S i = 0; i < ndim; ++i) {
    assert(idx[i] >= 0);
    assert(idx[i] < shape[i]);
    offset += idx[i] * strides[i];
  }
  return offset;
}

}  // namespace kakuhen::ndarray::detail
