#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "kakuhen/histogram/histogram_registry.h"
#include "kakuhen/histogram/axis_data.h"
#include "kakuhen/util/hash.h"
#include "kakuhen/integrator/integrator_base.h"
#include "kakuhen/integrator/plain.h"
#include "kakuhen/integrator/vegas.h"
#include "kakuhen/integrator/basin.h"
#include "kakuhen/util/printer.h"
#include <sstream>

using namespace kakuhen;
using namespace kakuhen::histogram;
using Catch::Approx;

TEST_CASE("Coverage: HistogramRegistry and Axis misc", "[coverage]") {
    HistogramRegistry<> registry;

    SECTION("Axis from initializer list") {
        auto v_id = registry.create_axis<VariableAxis<double, uint32_t>>({0.0, 10.0, 100.0});
        auto h_id = registry.book("v_axis", v_id);
        REQUIRE(registry.get_name(h_id) == "v_axis");
    }

    SECTION("Invalid AxisId booking") {
        REQUIRE_THROWS_AS(registry.book("invalid", AxisId<uint32_t>{999}), std::out_of_range);
    }

    SECTION("Duplicate name booking") {
        auto id = registry.book("hist", 10);
        (void)id;
        REQUIRE_THROWS_AS(registry.book("hist", 5), std::invalid_argument);
    }

    SECTION("Registry type mismatch") {
        std::stringstream ss;
        registry.serialize(ss, true);

        using FloatTraits = util::NumericTraits<float, uint32_t, uint64_t>;
        HistogramRegistry<FloatTraits> wrong_registry;
        REQUIRE_THROWS_AS(wrong_registry.deserialize(ss, true), std::runtime_error);
    }
    
    SECTION("AxisData capacity overflow") {
        AxisData<double, uint16_t> axis_data;
        // Max uint16 is 65535
        std::vector<double> large(65535, 1.0);
        (void)axis_data.add_data(large);
        std::vector<double> one(1, 1.0);
        REQUIRE_THROWS_AS(axis_data.add_data(one), std::length_error);
    }

    SECTION("VariableAxis unsorted edges") {
        AxisData<float, uint32_t> data;
        REQUIRE_THROWS_AS((VariableAxis<float, uint32_t>(data, {10.0f, 0.0f, 100.0f})), std::invalid_argument);
    }

    SECTION("Booking with None axis") {
        // Use a unique type to ensure coverage is recorded for this translation unit
        using UniqueTraits = util::NumericTraits<double, uint16_t, uint32_t>;
        HistogramRegistry<UniqueTraits> registry2;
        registry2.book("none", 10);
        REQUIRE_THROWS_AS(registry2.book("none_err", AxisId<uint16_t>{0}), std::invalid_argument);
    }
}

TEST_CASE("Coverage: Serialization misc", "[coverage]") {
    SECTION("BinAccumulator type verification") {
        BinAccumulator<double> bin;
        std::stringstream ss;
        bin.serialize(ss, true);

        BinAccumulator<float> wrong_bin;
        REQUIRE_THROWS_AS(wrong_bin.deserialize(ss, true), std::runtime_error);
    }

    SECTION("AxisMetadata type verification") {
        AxisMetadata<double, uint32_t> meta{AxisType::Uniform, 0, 10, 12};
        std::stringstream ss;
        meta.serialize(ss, true);

        AxisMetadata<float, uint32_t> wrong_meta;
        REQUIRE_THROWS_AS(wrong_meta.deserialize(ss, true), std::runtime_error);
        
        ss.str("");
        meta.serialize(ss, true);
        AxisMetadata<double, uint16_t> wrong_meta_s;
        REQUIRE_THROWS_AS(wrong_meta_s.deserialize(ss, true), std::runtime_error);
    }

    SECTION("HistogramData overflow and type verification") {
        HistogramData<util::num_traits_t<double, uint16_t>> data;
        data.allocate(65535);
        REQUIRE_THROWS_AS(data.allocate(1), std::length_error);

        HistogramData<> data2;
        std::stringstream ss;
        data2.serialize(ss, true);
        HistogramData<util::num_traits_t<float>> wrong_data;
        REQUIRE_THROWS_AS(wrong_data.deserialize(ss, true), std::runtime_error);
    }
}

TEST_CASE("Coverage: Integrator and Utils misc", "[coverage]") {
    SECTION("IntegratorId to string") {
        REQUIRE(integrator::to_string(integrator::IntegratorId::PLAIN) == "Plain");
        REQUIRE(integrator::to_string(integrator::IntegratorId::VEGAS) == "Vegas");
        REQUIRE(integrator::to_string(integrator::IntegratorId::BASIN) == "Basin");
        REQUIRE(integrator::to_string(static_cast<integrator::IntegratorId>(99)) == "Unknown");
    }

    SECTION("Grid adaption error on Unsupported") {
        integrator::Plain<> plain(1);
        using opts_t = integrator::Plain<>::options_type;
        opts_t opts;
        opts.adapt = true;
        REQUIRE_THROWS_AS(plain.set_options(opts), std::invalid_argument);
    }

    SECTION("Hash encode hex") {
        util::Hash h;
        h.add(0x123456789abcdef0ull);
        REQUIRE(!h.encode_hex().empty());
    }

    SECTION("Basin and Vegas prefixes") {
        integrator::Basin<> basin(2);
        REQUIRE(basin.prefix(false) == "basin_2d");
        REQUIRE(basin.prefix(true).find("basin_2d_") == 0);

        integrator::Vegas<> vegas(3);
        REQUIRE(vegas.prefix(false) == "vegas_3d");
        REQUIRE(vegas.prefix(true).find("vegas_3d_") == 0);
    }

    SECTION("JSONPrinter special chars") {
        std::stringstream ss;
        util::printer::JSONPrinter printer(ss);
        printer.print_one("", "backspace\b formfeed\f carriage\r tab\t control\x01");
        std::string out = ss.str();
        REQUIRE(out.find("\\b") != std::string::npos);
        REQUIRE(out.find("\\f") != std::string::npos);
        REQUIRE(out.find("\\r") != std::string::npos);
        REQUIRE(out.find("\\t") != std::string::npos);
        REQUIRE(out.find("\\u0001") != std::string::npos);
    }

    SECTION("Result average and fallback") {
        // 1. Fallback (all variances zero)
        {
            integrator::Result<double, uint64_t> res;
            integrator::IntegralAccumulator<double, uint64_t> acc1, acc2;
            acc1.accumulate(10.0, 100.0); // n=1, var=0
            acc2.accumulate(20.0, 400.0); // n=1, var=0
            res.accumulate(acc1);
            res.accumulate(acc2);
            REQUIRE(res.value() == Approx(15.0));
        }
        // 2. Normal (variances > 0)
        {
            integrator::Result<double, uint64_t> res;
            integrator::IntegralAccumulator<double, uint64_t> acc1, acc2;
            acc1.accumulate(10.0, 100.0);
            acc1.accumulate(12.0, 144.0); // n=2, var > 0
            acc2.accumulate(20.0, 400.0);
            acc2.accumulate(22.0, 484.0); // n=2, var > 0
            res.accumulate(acc1);
            res.accumulate(acc2);
            REQUIRE(res.value() > 0.0);
            REQUIRE(res.variance() > 0.0);
            REQUIRE(res.error() > 0.0);
            REQUIRE(res.chi2() >= 0.0);
            REQUIRE(res.chi2dof() >= 0.0);
        }
    }
}

TEST_CASE("Coverage: NDArray misc", "[coverage]") {
    SECTION("NDArray type verification") {
        ndarray::NDArray<double, uint32_t> arr({2, 3});
        std::stringstream ss;
        arr.serialize(ss, true);

        ndarray::NDArray<float, uint32_t> wrong_arr;
        REQUIRE_THROWS_AS(wrong_arr.deserialize(ss, true), std::runtime_error);
    }

    SECTION("NDView default constructor") {
        ndarray::NDView<double, uint32_t> view;
        REQUIRE(view.ndim() == 0);
        REQUIRE(view.data() == nullptr);
    }

    SECTION("NDArray padding") {
        // T=double (8), S=uint8_t (1), ndim=1 -> metadata=2. 2 % 8 = 2. Padding = 6.
        ndarray::NDArray<double, uint8_t> arr({10});
        REQUIRE(arr.size() == 10);
    }
}

TEST_CASE("Coverage: HistogramBuffer reset", "[coverage]") {
    using SmallTraits = util::NumericTraits<double, uint8_t, uint64_t>;
    HistogramBuffer<SmallTraits> buffer;
    buffer.init(10, 10);
    
    HistogramData<SmallTraits> data;
    data.allocate(10);

    for (int i = 0; i < 16; ++i) {
        buffer.fill(0, 1.0);
        buffer.flush(data);
    }
    
    buffer.fill(0, 1.0);
    buffer.flush(data);
    REQUIRE(data.bins()[0].weight() == Approx(17.0));
}