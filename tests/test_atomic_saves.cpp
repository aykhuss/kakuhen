#include "kakuhen/integrator/vegas.h"
#include "test_support.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

using namespace kakuhen::integrator;
using test_support::read_file;
using test_support::TempDir;
using test_support::write_file;

namespace {
struct FileWriter : Vegas<> {
  FileWriter() : Vegas<>(1, 4) {}
  using Vegas<>::write_file;
  using Vegas<>::write_header;

  std::string expected(detail::FileType kind, std::string_view payload) const {
    std::ostringstream out;
    write_header(out, kind);
    out << payload;
    return out.str();
  }
};

std::size_t entry_count(const std::filesystem::path& dir) {
  return static_cast<std::size_t>(std::distance(std::filesystem::directory_iterator(dir),
                                                std::filesystem::directory_iterator{}));
}
}  // namespace

TEST_CASE("Atomic saves preserve destinations and unrelated temporary files", "[persistence]") {
  // GENERATE, not a loop: a SECTION opens at most once per run, so a loop
  // would only run every scenario for the first kind
  const auto kind =
      GENERATE(detail::FileType::STATE, detail::FileType::DATA, detail::FileType::ENVELOPE);
  CAPTURE(static_cast<unsigned>(kind));
  TempDir files("kakuhen-save");
  FileWriter writer;
  const auto path = files.dir / "target";
  const auto unrelated = files.dir / "target.tmp";
  write_file(path, "original");
  write_file(unrelated, "unrelated");
  SECTION("successful save publishes only when complete") {
    writer.write_file(path, kind, [&](std::ostream& out) {
      out << "complete";
      out.flush();
      REQUIRE(read_file(path) == "original");
    });
    REQUIRE(read_file(path) == writer.expected(kind, "complete"));
  }
  SECTION("throwing serializer") {
    REQUIRE_THROWS_AS(writer.write_file(path, kind,
                                        [](std::ostream& out) {
                                          out << "partial";
                                          throw std::runtime_error("serializer failed");
                                        }),
                      std::runtime_error);
    REQUIRE(read_file(path) == "original");
  }
  SECTION("stream failure") {
    REQUIRE_THROWS_AS(writer.write_file(path, kind,
                                        [](std::ostream& out) {
                                          out << "partial";
                                          out.setstate(std::ios::badbit);
                                        }),
                      std::ios_base::failure);
    REQUIRE(read_file(path) == "original");
  }
  SECTION("rename failure") {
    std::filesystem::remove(path);
    std::filesystem::create_directory(path);
    write_file(path / "keep", "original");
    REQUIRE_THROWS_AS(writer.write_file(path, kind, [](std::ostream& out) { out << "complete"; }),
                      std::filesystem::filesystem_error);
    REQUIRE(read_file(path / "keep") == "original");
  }
  REQUIRE(read_file(unrelated) == "unrelated");
  REQUIRE(entry_count(files.dir) == 2);  // all owned staging directories were removed
}

TEST_CASE("Overlapping saves publish independent complete payloads", "[persistence]") {
  TempDir files("kakuhen-save");
  FileWriter outer, inner;
  const auto path = files.dir / "shared.khs";
  const auto kind = detail::FileType::STATE;
  write_file(path, "original");
  bool fail_outer = false;
  SECTION("both writers succeed") {}
  SECTION("outer writer fails after inner publishes") {
    fail_outer = true;
  }
  const auto save = [&] {
    outer.write_file(path, kind, [&](std::ostream& out) {
      out << "outer-before";
      out.flush();
      // nested callbacks force overlapping writes without timing races
      inner.write_file(path, kind, [&](std::ostream& inner_out) {
        REQUIRE(read_file(path) == "original");
        inner_out << "inner-complete";
      });
      REQUIRE(read_file(path) == inner.expected(kind, "inner-complete"));
      out << "-after";
      out.flush();
      REQUIRE(read_file(path) == inner.expected(kind, "inner-complete"));
      if (fail_outer) throw std::runtime_error("outer failed");
    });
  };
  if (fail_outer) {
    REQUIRE_THROWS_AS(save(), std::runtime_error);
    REQUIRE(read_file(path) == inner.expected(kind, "inner-complete"));
  } else {
    REQUIRE_NOTHROW(save());
    REQUIRE(read_file(path) == outer.expected(kind, "outer-before-after"));
  }
  REQUIRE(entry_count(files.dir) == 1);
}

TEST_CASE("A failed overlapping writer does not remove another writer's staging file",
          "[persistence]") {
  TempDir files("kakuhen-save");
  FileWriter outer, inner;
  const auto path = files.dir / "shared.khd";
  const auto kind = detail::FileType::DATA;
  write_file(path, "original");
  outer.write_file(path, kind, [&](std::ostream& out) {
    out << "outer-before";
    out.flush();
    REQUIRE_THROWS_AS(inner.write_file(path, kind,
                                       [](std::ostream& inner_out) {
                                         inner_out << "inner-partial";
                                         throw std::runtime_error("inner failed");
                                       }),
                      std::runtime_error);
    REQUIRE(read_file(path) == "original");
    out << "-after";
  });
  REQUIRE(read_file(path) == outer.expected(kind, "outer-before-after"));
  REQUIRE(entry_count(files.dir) == 1);
}
