#pragma once

#include "kakuhen/util/serialize.h"
#include <iostream>
#include <stdexcept>

namespace kakuhen::util {

/// write custom user data to an output stream
template <typename UD>
void write_user_data_stream(std::ostream& out, const UD& user_data,
                            const std::string_view header = "USERDATA") {
  using namespace kakuhen::util::serialize;
  write_bytes(out, header.data(), header.size());
  serialize_one<UD>(out, user_data);
}

/// read custom user data from an input stream
template <typename UD>
void read_user_data_stream(std::istream& in, UD& user_data,
                           const std::string_view header = "USERDATA") {
  using namespace kakuhen::util::serialize;
  /// check that the header matches
  std::vector<char> buf(header.size());
  read_bytes(in, buf.data(), header.size());
  if (std::string_view(buf.data(), buf.size()) != header) {
    throw std::runtime_error("Incompatible user data header");
  }
  /// read in the user data
  deserialize_one<UD>(in, user_data);
}

}  // namespace kakuhen::util
