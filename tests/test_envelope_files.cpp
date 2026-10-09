#include "kakuhen/integrator/basin_generator.h"
#include "kakuhen/integrator/vegas_generator.h"
#include "test_support.h"
#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace kakuhen::integrator;
using test_support::read_file;
using test_support::TempDir;
using test_support::write_file;
using Catch::Approx;

namespace {
/// stream buffer that can only be read forward, like a pipe
struct ForwardOnlyBuffer : std::streambuf {
  explicit ForwardOnlyBuffer(std::string bytes) : data(std::move(bytes)) {
    setg(data.data(), data.data(), data.data() + data.size());
  }
  std::string data;
};

/// run in a scratch directory so default-prefix filenames never touch the cwd
struct ScopedCwd {
  std::filesystem::path previous = std::filesystem::current_path();
  explicit ScopedCwd(const std::filesystem::path& dir) {
    std::filesystem::current_path(dir);
  }
  ~ScopedCwd() {
    std::filesystem::current_path(previous);
  }
};

template <typename G>
struct EnvelopeFixture : G {
  using G::env_grid_hash;
  using G::env_propose;
  using G::envelope_;
  using G::G;
  using G::grid_;
  using G::write_header;
  using S = typename G::size_type;
  using IntBase = typename G::IntBase;

  void write_candidate(const std::filesystem::path& path, uint8_t version = 1) const {
    std::ofstream out(path, std::ios::binary);
    this->write_header(out, detail::FileType::ENVELOPE);
    kakuhen::util::serialize::serialize_one(out, version);
    kakuhen::util::serialize::serialize_one(out, this->env_grid_hash());
    envelope_.serialize(out);
    kakuhen::util::serialize::serialize_one<uint8_t>(out, 0);  // no pending batch
  }

  /// state file as the plain integrator writes it, i.e. without an envelope block
  void write_state_without_envelope(const std::filesystem::path& path) const {
    std::ofstream out(path, std::ios::binary);
    this->write_header(out, detail::FileType::STATE);
    IntBase::write_state_stream(out);
  }

  /// state file whose envelope block carries an unexpected tag
  void write_state_with_foreign_block(const std::filesystem::path& path,
                                      std::string_view tag) const {
    using namespace kakuhen::util::serialize;
    std::ofstream out(path, std::ios::binary);
    this->write_header(out, detail::FileType::STATE);
    IntBase::write_state_stream(out);
    write_bytes(out, tag.data(), tag.size());
  }
};

template <typename G>
auto make_generator() {
  auto gen = [] {
    if constexpr (std::same_as<G, BasinGenerator<>>) {
      return EnvelopeFixture<G>(2, 2, 2);
    } else {
      return EnvelopeFixture<G>(2, 4);
    }
  }();
  gen.set_options({.frozen = true, .verbosity = 0});
  return gen;
}

std::string snapshot(const auto& gen) {
  std::ostringstream out;
  gen.write_state_stream(out);
  gen.write_rng_state_stream(out);
  gen.write_data_stream(out);
  return out.str();
}

std::vector<double> proposals(auto& gen) {
  Point<> point(gen.ndim());
  std::vector<double> result;
  for (unsigned i = 0; i < 16; ++i) {
    result.push_back(gen.env_propose(point));
    result.push_back(point.weight);
    result.insert(result.end(), point.x.begin(), point.x.end());
  }
  return result;
}

}  // namespace

TEMPLATE_TEST_CASE("Envelope files merge factors without importing worker diagnostics",
                   "[generator][envelope_files]", VegasGenerator<>, BasinGenerator<>) {
  TempDir files("kakuhen-envelope");
  auto first = make_generator<TestType>();
  auto second = make_generator<TestType>();
  first.initialize_envelope([](const Point<>&) { return 2.0; }, 8ULL);
  second.initialize_envelope(4.0);
  second.raise_envelope([](const Point<>&) { return 3.0; }, 16ULL);
  for (decltype(first.ndim()) i = 0; i < first.envelope_.size(); ++i) {
    first.envelope_[i] = 1.0 + i;
    second.envelope_[i] = 8.0 - i;
  }
  first.scale_envelope(1.0);
  second.scale_envelope(1.0);
  first.save_envelope(files.dir / "first.khe");
  second.save_envelope(files.dir / "second.khe");

  auto merged = make_generator<TestType>();
  merged.initialize_envelope([](const Point<>&) { return 1.0; }, 12ULL);
  // the coordinator contributes a factor larger than either worker's
  merged.envelope_[0] = 10.0;
  merged.scale_envelope(1.0);
  merged.merge_envelope(files.dir / "first.khe");
  merged.merge_envelope(files.dir / "second.khe");
  REQUIRE(merged.envelope_ready());
  REQUIRE(merged.envelope_seed() == 1.0);
  REQUIRE(merged.abs_integral_estimate().count() == 12);
  REQUIRE(merged.abs_integral_estimate().value() == 1.0);
  REQUIRE(merged.envelope_[0] == 10.0);
  REQUIRE(merged.envelope_volume() == Approx(45.5));
  for (decltype(first.ndim()) i = 1; i < first.envelope_.size(); ++i)
    REQUIRE(merged.envelope_[i] == std::max(first.envelope_[i], second.envelope_[i]));
  REQUIRE(merged.predicted_efficiency() == Approx(1.0 / merged.envelope_volume()));

  auto reversed = make_generator<TestType>();
  reversed.initialize_envelope([](const Point<>&) { return 1.0; }, 12ULL);
  reversed.envelope_[0] = 10.0;
  reversed.scale_envelope(1.0);
  reversed.merge_envelope(files.dir / "second.khe");
  reversed.merge_envelope(files.dir / "first.khe");
  REQUIRE(snapshot(reversed) == snapshot(merged));
  const auto before_duplicate = snapshot(merged);
  merged.merge_envelope(files.dir / "first.khe");
  REQUIRE(snapshot(merged) == before_duplicate);
  REQUIRE(proposals(merged) == proposals(reversed));
}

TEMPLATE_TEST_CASE("An imported envelope works without a local seed or estimate",
                   "[generator][envelope_files]", VegasGenerator<>, BasinGenerator<>) {
  TempDir files("kakuhen-envelope");
  auto source = make_generator<TestType>();
  source.initialize_envelope([](const Point<>&) { return 1.0; }, 10ULL);
  source.save_envelope(files.dir / "worker.khe");
  auto target = make_generator<TestType>();
  SECTION("fresh generator") {}
  SECTION("uninitialized checkpoint") {
    target.save(files.dir / "empty.khs");
    target.load(files.dir / "empty.khs");
  }
  target.merge_envelope(files.dir / "worker.khe");
  REQUIRE(target.envelope_ready());
  REQUIRE(target.envelope_seed() == 0.0);
  REQUIRE(target.abs_integral_estimate().count() == 0);
  REQUIRE_THROWS_AS(target.predicted_efficiency(), std::runtime_error);
  target.save(files.dir / "combined.khs");
  auto restored = make_generator<TestType>();
  restored.load(files.dir / "combined.khs");
  REQUIRE(restored.envelope_ready());
  REQUIRE(restored.envelope_seed() == 0.0);
  REQUIRE_THROWS_AS(restored.predicted_efficiency(), std::runtime_error);
  auto result =
      restored.generate_trials([](const Point<>&) { return 1.0; }, 32ULL,
                               [](const Point<>&, double weight) { REQUIRE(weight == 1.0); });
  REQUIRE(result.n_events() == 32);
  REQUIRE(result.value() == 1.0);
  restored.raise_envelope([](const Point<>&) { return 0.0; }, 4ULL);
  REQUIRE_THROWS_AS(restored.predicted_efficiency(), std::runtime_error);
  restored.raise_envelope([](const Point<>&) { return 1.0; }, 4ULL);
  REQUIRE(restored.predicted_efficiency() == Approx(0.5));
  REQUIRE(restored.envelope_seed() == 0.0);
}

TEMPLATE_TEST_CASE("Envelope file failures preserve factors diagnostics RNG and proposals",
                   "[generator][envelope_files]", VegasGenerator<>, BasinGenerator<>) {
  TempDir files("kakuhen-envelope");
  const auto path = files.dir / "worker.khe";
  auto source = make_generator<TestType>();
  auto target = make_generator<TestType>();
  source.initialize_envelope(1.0);
  target.initialize_envelope([](const Point<>&) { return 2.0; }, 8ULL);
  target.set_seed(19);
  const auto expected_proposals = proposals(target);
  target.set_seed(19);
  const auto before = snapshot(target);
  const auto volume_before = target.envelope_volume();
  source.save_envelope(path);

  SECTION("missing file") {
    std::filesystem::remove(path);
  }
  SECTION("wrong file kind") {
    source.save(path);
  }
  SECTION("wrong generator kind") {
    if constexpr (std::same_as<TestType, VegasGenerator<>>) {
      auto other = make_generator<BasinGenerator<>>();
      other.initialize_envelope(1.0);
      other.save_envelope(path);
    } else {
      auto other = make_generator<VegasGenerator<>>();
      other.initialize_envelope(1.0);
      other.save_envelope(path);
    }
  }
  SECTION("wrong numeric representation") {
    auto bytes = read_file(path);
    bytes[detail::file_signature_size + sizeof(IntegratorId) + sizeof(detail::FileType)] ^= 1;
    write_file(path, bytes);
  }
  SECTION("unsupported version") {
    source.write_candidate(path, 255);
  }
  SECTION("changed grid") {
    source.grid_[0] *= 0.5;
    source.write_candidate(path);
  }
  SECTION("wrong table shape") {
    source.envelope_ = kakuhen::ndarray::NDArray<double, typename TestType::size_type>({1});
    source.envelope_.fill(1.0);
    source.write_candidate(path);
  }
  SECTION("invalid factors are not masked by the coordinator's maximum") {
    for (const double value : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
      source.envelope_[0] = value;
      source.write_candidate(path);
      REQUIRE_THROWS(target.merge_envelope(path));
      REQUIRE(snapshot(target) == before);
      REQUIRE(target.envelope_volume() == volume_before);
    }
  }
  SECTION("truncated header and payload") {
    const auto bytes = read_file(path);
    for (const auto size : {std::size_t(3), bytes.size() / 2, bytes.size() - 1}) {
      write_file(path, bytes.substr(0, size));
      REQUIRE_THROWS(target.merge_envelope(path));
      REQUIRE(snapshot(target) == before);
      REQUIRE(target.envelope_volume() == volume_before);
    }
  }
  SECTION("CDF overflow") {
    source.envelope_.fill(std::numeric_limits<double>::max());
    source.write_candidate(path);
  }
  SECTION("finite CDFs but overflowing total volume") {
    source.envelope_.fill(1e200);
    source.write_candidate(path);
  }
  REQUIRE_THROWS(target.merge_envelope(path));
  REQUIRE(snapshot(target) == before);
  REQUIRE(target.envelope_ready());
  REQUIRE(target.envelope_volume() == volume_before);
  REQUIRE(proposals(target) == expected_proposals);
}

TEMPLATE_TEST_CASE("Envelope file operations require a frozen grid and replace unusable envelopes",
                   "[generator][envelope_files]", VegasGenerator<>, BasinGenerator<>) {
  TempDir files("kakuhen-envelope");
  const auto path = files.dir / "worker.khe";
  auto source = make_generator<TestType>();
  source.initialize_envelope(1.0);
  source.raise_envelope([](const Point<>& p) { return 1.0 + p.x[0]; }, 64ULL);
  source.save_envelope(path);
  auto target = make_generator<TestType>();
  REQUIRE_THROWS(target.save_envelope(files.dir / "uninitialized.khe"));
  REQUIRE_FALSE(std::filesystem::exists(files.dir / "uninitialized.khe"));
  target.set_options({.frozen = false});
  const auto unfrozen = snapshot(target);
  REQUIRE_THROWS_AS(target.merge_envelope(path), std::invalid_argument);
  REQUIRE(snapshot(target) == unfrozen);
  target.set_options({.frozen = true});
  target.initialize_envelope(2.0);
  SECTION("stale envelope") {
    target.grid_[0] *= 0.5;
    // the incoming file matches the new grid, the existing envelope doesn't
    source.grid_[0] *= 0.5;
    source.initialize_envelope(1.0);
    source.raise_envelope([](const Point<>& p) { return 1.0 + p.x[0]; }, 64ULL);
    source.save_envelope(path);
  }
  REQUIRE_THROWS(target.save_envelope(files.dir / "invalid.khe"));
  REQUIRE_FALSE(std::filesystem::exists(files.dir / "invalid.khe"));
  // an unusable envelope adds nothing to the maximum: the import replaces it
  target.merge_envelope(path);
  REQUIRE(target.envelope_ready());
  REQUIRE(std::ranges::equal(target.envelope_, source.envelope_));
  REQUIRE(target.envelope_volume() == source.envelope_volume());
  REQUIRE(target.envelope_seed() == 0.0);
}

TEMPLATE_TEST_CASE("Envelope default filenames follow data filenames",
                   "[generator][envelope_files]", VegasGenerator<>, BasinGenerator<>) {
  TempDir files("kakuhen-envelope");
  ScopedCwd cwd(files.dir);
  auto gen = make_generator<TestType>();
  gen.set_seed(37);
  gen.initialize_envelope(1.0);
  std::filesystem::path expected;
  SECTION("file_path override") {
    gen.set_options({.file_path = files.dir / "campaign.khs"});
    expected = files.dir / "campaign.s37.khe";
  }
  SECTION("default grid and seed prefix") {
    expected = gen.prefix(true) + ".s37.khe";
  }
  const auto saved = gen.save_envelope();
  REQUIRE(saved == expected);
  REQUIRE(std::filesystem::exists(saved));
  REQUIRE(gen.file_envelope() == saved);
  REQUIRE(gen.merge_envelope().volume() == gen.envelope_volume());
}

TEST_CASE("BASIN envelope identity includes the sampling forest", "[generator][envelope_files]") {
  struct OrderedBasin : EnvelopeFixture<BasinGenerator<>> {
    OrderedBasin() : EnvelopeFixture(3, 2, 2) {
      set_options({.frozen = true, .verbosity = 0});
    }
    void chain() {
      order_(1, 0) = 0;
      order_(2, 0) = 1;
      nblocks_ = 1;
      for (S i = 0; i < ndim_; ++i)
        std::copy_n(&grid_(order_(i, 0), order_(i, 1), 0, 0), ndiv0_, &ordered_grid_(i, 0, 0));
    }
  } source, different, receiver;
  TempDir files("kakuhen-envelope");
  source.chain();
  receiver.chain();
  source.initialize_envelope(1.0);
  source.raise_envelope([](const Point<>& p) { return 1.0 + p.x[0] + p.x[1] * p.x[2]; }, 64ULL);
  source.save_envelope(files.dir / "chain.khe");
  REQUIRE(source.hash().value() == different.hash().value());
  REQUIRE_THROWS(different.merge_envelope(files.dir / "chain.khe"));
  receiver.merge_envelope(files.dir / "chain.khe");
  receiver.set_seed(5);
  source.set_seed(5);
  REQUIRE(proposals(receiver) == proposals(source));

  // rollback must restore conditional CDFs and their table views, not just roots
  receiver.set_seed(9);
  const auto expected = proposals(receiver);
  receiver.set_seed(9);
  const auto before = snapshot(receiver);
  const auto volume_before = receiver.envelope_volume();
  source.envelope_.fill(std::numeric_limits<double>::max());
  source.write_candidate(files.dir / "overflow.khe");
  REQUIRE_THROWS(receiver.merge_envelope(files.dir / "overflow.khe"));
  REQUIRE(snapshot(receiver) == before);
  REQUIRE(receiver.envelope_volume() == volume_before);
  REQUIRE(proposals(receiver) == expected);

  // a forest change without a grid-edge change invalidates a local envelope too
  different.initialize_envelope(1.0);
  different.chain();
  REQUIRE_FALSE(different.envelope_ready());
  REQUIRE_THROWS(different.save_envelope(files.dir / "stale.khe"));
  different.save(files.dir / "stale.khs");
  receiver.load(files.dir / "stale.khs");
  REQUIRE_FALSE(receiver.envelope_ready());
  // ...and an import on the new forest replaces it
  different.merge_envelope(files.dir / "chain.khe");
  REQUIRE(different.envelope_ready());
}

TEMPLATE_TEST_CASE("Maximum merging can overflow even when both input envelopes are usable",
                   "[generator][envelope_files]", VegasGenerator<>, BasinGenerator<>) {
  TempDir files("kakuhen-envelope");
  auto source = make_generator<TestType>();
  auto target = make_generator<TestType>();
  source.initialize_envelope(1.0);
  target.initialize_envelope(1.0);
  for (typename TestType::size_type i = 0; i < 8; ++i) {
    source.envelope_[i] = i < 4 ? 1e160 : 1e-160;
    target.envelope_[i] = i < 4 ? 1e-160 : 1e160;
  }
  source.scale_envelope(1.0);
  target.scale_envelope(1.0);
  REQUIRE(source.envelope_volume() == Approx(1.0));
  REQUIRE(target.envelope_volume() == Approx(1.0));
  source.save_envelope(files.dir / "finite.khe");
  target.set_seed(11);
  const auto expected = proposals(target);
  target.set_seed(11);
  const auto before = snapshot(target);
  REQUIRE_THROWS(target.merge_envelope(files.dir / "finite.khe"));
  REQUIRE(snapshot(target) == before);
  REQUIRE(proposals(target) == expected);
}

TEMPLATE_TEST_CASE("Appended user data does not masquerade as an envelope block",
                   "[generator][envelope_files]", VegasGenerator<>, BasinGenerator<>) {
  TempDir files("kakuhen-envelope");
  struct Payload {
    int id;
    double value;
  };
  const Payload written{42, 3.25};

  SECTION("state file without an envelope block") {
    // user records are appended, so they land exactly where the envelope block
    // would sit; the load must recognize them instead of reading them as a tag
    const auto path = files.dir / "no_envelope.khs";
    auto source = make_generator<TestType>();
    source.write_state_without_envelope(path);
    kakuhen::util::write_user_data(path, written, "MYKEY");

    auto target = make_generator<TestType>();
    REQUIRE_NOTHROW(target.load(path));
    REQUIRE_FALSE(target.envelope_ready());

    Payload read{};
    REQUIRE_NOTHROW(kakuhen::util::read_user_data(path, read, "MYKEY"));
    REQUIRE(read.id == written.id);
    REQUIRE(read.value == written.value);
  }

  SECTION("reading a plain state leaves the stream at the user-data record") {
    auto source = make_generator<TestType>();
    std::stringstream stream;
    static_cast<const typename TestType::IntBase&>(source).write_state_stream(stream);
    const auto record_start = stream.tellp();
    kakuhen::util::write_user_data_stream(stream, written, "MYKEY");

    auto target = make_generator<TestType>();
    REQUIRE_NOTHROW(target.read_state_stream(stream));
    REQUIRE_FALSE(target.envelope_ready());
    REQUIRE(stream.tellg() == record_start);
  }

  SECTION("state file with an envelope block") {
    const auto path = files.dir / "with_envelope.khs";
    auto source = make_generator<TestType>();
    source.initialize_envelope([](const Point<>&) { return 2.0; }, 8ULL);
    source.save(path);
    kakuhen::util::write_user_data(path, written, "MYKEY");

    auto target = make_generator<TestType>();
    REQUIRE_NOTHROW(target.load(path));
    REQUIRE(target.envelope_ready());
    REQUIRE(target.envelope_seed() == 2.0);
    REQUIRE(target.abs_integral_estimate().count() == 8);
    REQUIRE(target.envelope_volume() == Approx(source.envelope_volume()));
    source.set_seed(7);
    target.set_seed(7);
    REQUIRE(proposals(target) == proposals(source));

    Payload read{};
    REQUIRE_NOTHROW(kakuhen::util::read_user_data(path, read, "MYKEY"));
    REQUIRE(read.id == written.id);
    REQUIRE(read.value == written.value);
  }

  SECTION("streams that cannot rewind") {
    auto source = make_generator<TestType>();
    source.initialize_envelope([](const Point<>&) { return 2.0; }, 8ULL);
    std::stringstream with_envelope;
    source.write_state_stream(with_envelope);
    ForwardOnlyBuffer envelope_buffer(with_envelope.str());
    std::istream envelope_stream(&envelope_buffer);
    auto target = make_generator<TestType>();
    REQUIRE(envelope_stream.tellg() == std::streampos(-1));
    REQUIRE_NOTHROW(target.read_state_stream(envelope_stream));
    REQUIRE(target.envelope_ready());

    std::stringstream plain;
    static_cast<const typename TestType::IntBase&>(source).write_state_stream(plain);
    kakuhen::util::write_user_data_stream(plain, written, "MYKEY");
    ForwardOnlyBuffer plain_buffer(plain.str());
    std::istream plain_stream(&plain_buffer);
    REQUIRE_NOTHROW(target.read_state_stream(plain_stream));
    REQUIRE_FALSE(target.envelope_ready());
    REQUIRE(plain_stream.peek() == 'U');  // the record is left for the user-data reader
  }

  SECTION("an unrecognized block is still corruption") {
    const auto path = files.dir / "foreign.khs";
    auto source = make_generator<TestType>();
    for (const auto tag : {"XYZ", "USE", "USERDAT", "USEgarbage", "USERDATX"}) {
      CAPTURE(tag);
      source.write_state_with_foreign_block(path, tag);
      auto target = make_generator<TestType>();
      REQUIRE_THROWS_AS(target.load(path), std::runtime_error);
    }
  }
}

TEMPLATE_TEST_CASE("Failed first envelope imports can be retried", "[generator][envelope_files]",
                   VegasGenerator<>, BasinGenerator<>) {
  TempDir files("kakuhen-envelope");
  auto source = make_generator<TestType>();
  source.initialize_envelope(1.0);
  source.save_envelope(files.dir / "valid.khe");
  source.envelope_.fill(1e200);
  source.write_candidate(files.dir / "overflow.khe");
  auto target = make_generator<TestType>();
  const auto before = snapshot(target);
  REQUIRE_THROWS(target.merge_envelope(files.dir / "overflow.khe"));
  REQUIRE(snapshot(target) == before);
  REQUIRE_FALSE(target.envelope_ready());
  target.merge_envelope(files.dir / "valid.khe");
  REQUIRE(target.envelope_ready());
  REQUIRE(target.envelope_volume() == 1.0);
  REQUIRE(target.envelope_seed() == 0.0);
}
