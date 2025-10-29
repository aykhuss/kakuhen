#pragma once

#include "kakuhen/integrator/integral_accumulator.h"
#include "kakuhen/integrator/integrator_feature.h"
#include "kakuhen/integrator/numeric_traits.h"
#include "kakuhen/integrator/options.h"
#include "kakuhen/integrator/point.h"
#include "kakuhen/integrator/result.h"
#include "kakuhen/util/printer.h"
#include "kakuhen/util/serialize.h"
#include "kakuhen/util/type.h"
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string_view>
#include <system_error>

namespace kakuhen::integrator {

//> id's for the different integrators
enum class IntegratorId : uint8_t {
  PLAIN = 0,
  VEGAS = 1,
  BASIN = 2,
};
constexpr std::string_view to_string(IntegratorId id) noexcept {
  switch (id) {
    case IntegratorId::PLAIN:
      return "Plain";
    case IntegratorId::VEGAS:
      return "Vegas";
    case IntegratorId::BASIN:
      return "Basin";
  }
  return "Unknown";
}

//> Default integrator parameters
template <typename NT = num_traits_t<>>
struct IntegratorDefaults {
  using rng_type = std::mt19937_64;
  using dist_type = std::uniform_real_distribution<typename NT::value_type>;
};

template <typename Derived, typename NT = num_traits_t<>,
          typename RNG = typename IntegratorDefaults<NT>::rng_type,
          typename DIST = typename IntegratorDefaults<NT>::dist_type>
class IntegratorBase {
 public:
  using num_traits = NT;
  using value_type = typename num_traits::value_type;
  using size_type = typename num_traits::size_type;
  using count_type = typename num_traits::count_type;
  using seed_type = typename RNG::result_type;
  using point_type = Point<num_traits>;
  using int_acc_type = IntegralAccumulator<value_type, count_type>;
  using result_type = Result<value_type, count_type>;
  using options_type = Options<value_type, count_type, seed_type>;

  explicit IntegratorBase(size_type ndim = 0, const options_type& opts = {})
      : ndim_{ndim}, random_generator_{}, uniform_distribution_{0, 1}, opts_{opts} {
    /// if the integrator supports adation, set default to true
    if constexpr (Derived::has_feature(IntegratorFeature::ADAPT)) {
      if (!opts_.adapt) opts_.adapt = true;
    }
    /// set some default values for the options
    if (!opts_.niter) opts_.niter = 1;
    if (!opts_.verbosity) opts_.verbosity = 2;
    //@todo:  do static asserts here?
  }

  constexpr IntegratorId id() const noexcept {
    return Derived::id;
  }

  static constexpr bool has_feature(IntegratorFeature flag) noexcept {
    return detail::has_flag(Derived::features, flag);
  }

  // this is the way to set persistent options
  // optional argument to integrate are per-call overrides
  inline void set_options(const options_type& opts) {
    opts_.set(opts);
    if (opts.seed) random_generator_.seed(opts_.seed.value());
    if (opts.adapt && !has_feature(IntegratorFeature::ADAPT)) {
      throw std::invalid_argument(std::string(to_string(id())) + " does not support grid adaption");
    }
  }

  // seed
  inline void set_seed(seed_type seed) noexcept {
    set_options({.seed = seed});
  }
  inline void set_seed() {
    // set a new seed number (just pick next)
    //@todo:  go back in history and pick next seed number, otherwise use 1
    if (!opts_.seed) {
      set_options({.seed = 1});
    } else {
      set_options({.seed = opts_.seed.value() + 1});
    }
  }
  seed_type seed() const noexcept {
    if (!opts_.seed) set_seed();
    return opts_.seed.value();
  }

  // user_data
  inline void set_user_data(void* user_data = nullptr) noexcept {
    set_options({.user_data = user_data});
  }
  void* user_data() const noexcept {
    return opts_.user_data.value_or(nullptr);
  }

  // the main integration routine
  // using keys as options
  template <typename I, typename... Keys>
  result_type integrate(I&& integrand, const Keys&... keys) {
    options_type opts{};
    (keys.apply(opts), ...);
    return integrate(std::forward<I>(integrand), opts);
  }

  // the main integration routine
  // options are temporary overrides; persistent settings through set_options
  template <typename I>
  result_type integrate(I&& integrand, const options_type& opts) {
    // std::cout << "integrate: " << opts << std::endl;
    // local lvalue reference to make it callable multiple times
    auto& func = integrand;
    // set up local options & check settings
    Options orig_opts = opts_;
    set_options(opts);
    if (!opts_.neval) {
      throw std::invalid_argument("number of evaluations (neval) not set");
    }
    if (!opts_.niter) {
      throw std::invalid_argument("number of iterations (niter) not set");
    }
    // call the integration implementation for each iteration & accumulate
    result_type result;
    for (count_type iter = 0; iter < *opts_.niter; ++iter) {
      // if (opts_.verbosity && *opts_.verbosity > 0) {
      //   std::cout << "=== Iteration " << (iter + 1) << " / " << *opts_.niter << " [" <<
      //   *opts_.neval
      //             << "] ===\n";
      // }
      int_acc_type res_it = derived().integrate_impl(func, *opts_.neval);
      result.accumulate(res_it);
      if (opts_.verbosity && *opts_.verbosity > 0) {
        std::cout << "\n***** Integration by " << to_string(id());
        std::cout << " (Iteration " << iter + 1 << " / " << *opts_.niter << " ) *****\n";
        std::cout << "  integral(iter) = " << res_it.value() << " +/- " << res_it.error();
        std::cout << " (n=" << res_it.count() << ")\n";
        std::cout << "  integral(acc.) = " << result.value() << " +/- " << result.error();
        std::cout << " (n=" << result.count() << ")\n";
        std::cout << "***** chi^2/dof = " << result.chi2dof() << " *****\n";
      }

      // // check for convergence
      // bool converged = false;
      // if (opts_.rel_tol) {
      //   converged =
      //       converged || res.error() <= std::abs(res.value()) *
      //       *opts_.rel_tol;
      // }
      // if (opts_.abs_tol) {
      //   converged = converged || res.error() <= *opts_.abs_tol;
      // }
      // if (converged) {
      //   if (opts_.verbosity && *opts_.verbosity > 0) {
      //     std::cout << "Converged.\n";
      //   }
      //   break;
      // }

      //> adapt the grid if requested
      if constexpr (has_feature(IntegratorFeature::ADAPT)) {
        if (opts_.adapt && *opts_.adapt) {
          derived().adapt();
        }
      }

      //> save state/data if requested
      if constexpr (has_feature(IntegratorFeature::STATE)) {
        if (opts_.file_path) {
          derived().save();
        }
      }

    }  // for iter

    // int_acc_type res =
    //     derived().integrate_impl(std::forward<I>(integrand), *opts_.neval);
    // restore original options
    opts_ = orig_opts;
    // return all iterations
    return result;
  }

  // helper integration routines for convenience
  // @ todo

  /// implementation of the Printer interface
  template <typename P>
  void print(P& prt) const {
    using C = kakuhen::util::printer::Context;
    using namespace kakuhen::util::type;
    prt.reset();
    prt.template begin<C::OBJECT>();
    {
      prt.print_one("name", to_string(id()));
      prt.print_one("id", static_cast<std::underlying_type_t<IntegratorId>>(id()));
      prt.print_one("value_type", get_type_name<value_type>());
      prt.print_one("size_type", get_type_name<size_type>());
      prt.print_one("count_type", get_type_name<count_type>());
      prt.print_one("ndim", ndim_);
      if constexpr (has_feature(IntegratorFeature::STATE)) {
        //prt.template begin<C::OBJECT>("state");
        derived().print_state(prt);
        //prt.template end<C::OBJECT>(true);
      }
    }
    prt.template end<C::OBJECT>(true);
  }

  // save state of the integrator to a file
  template <typename D = Derived, typename = std::enable_if_t<detail::has_state_stream<D>::value>>
  void save(const std::filesystem::path& filepath) const {
    if (!has_feature(IntegratorFeature::STATE)) {
      throw std::runtime_error(std::string(to_string(id())) + " does not support saving state");
    }
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
      throw std::ios_base::failure("Failed to open state file: " + filepath.string());
    }
    write_header(ofs, FileType::STATE);
    derived().write_state_stream(ofs);
    if (!ofs) {
      throw std::ios_base::failure("Error writing state file: " + filepath.string());
    }
  }
  template <typename D = Derived, typename = std::enable_if_t<detail::has_state_stream<D>::value>>
  std::filesystem::path save() const {
    std::filesystem::path fstate = file_state();
    save(fstate);
    return fstate;
  }

  // load state of the integrator from a file
  template <typename D = Derived, typename = std::enable_if_t<detail::has_state_stream<D>::value>>
  void load(const std::filesystem::path& filepath) {
    if (!has_feature(IntegratorFeature::STATE)) {
      throw std::runtime_error(std::string(to_string(id())) + " does not support saving state");
    }
    std::error_code ec;
    if (std::filesystem::exists(filepath, ec)) {
      if (ec) {
        throw std::system_error(ec, "Failed to check if file exists");
      }
      std::ifstream ifs(filepath, std::ios::binary);
      if (!ifs.is_open()) {
        throw std::ios_base::failure("Failed to open state file: " + filepath.string());
      }
      read_header(ifs, FileType::STATE);
      derived().read_state_stream(ifs);
      if (!ifs) {
        throw std::ios_base::failure("Error reading state file: " + filepath.string());
      }
    } else {
      std::cout << "state file " << filepath.string() << " not found; skip loading\n";
    }
  }
  template <typename D = Derived, typename = std::enable_if_t<detail::has_state_stream<D>::value>>
  std::filesystem::path load() {
    std::filesystem::path fstate = file_state();
    load(fstate);
    return fstate;
  }

  // save accumulated data of the integrator to a file
  template <typename D = Derived, typename = std::enable_if_t<detail::has_data_stream<D>::value>>
  void save_data(const std::filesystem::path& filepath) const {
    if (!has_feature(IntegratorFeature::DATA)) {
      throw std::runtime_error(std::string(to_string(id())) +
                               " does not support data accumulation");
    }
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
      throw std::ios_base::failure("Failed to open data file: " + filepath.string());
    }
    write_header(ofs, FileType::DATA);
    derived().write_data_stream(ofs);
    if (!ofs) {
      throw std::ios_base::failure("Error writing data file: " + filepath.string());
    }
  }
  template <typename D = Derived, typename = std::enable_if_t<detail::has_data_stream<D>::value>>
  std::filesystem::path save_data() const {
    std::filesystem::path fdata = file_data();
    save_data(fdata);
    return fdata;
  }

  // append accumulated data of the integrator from a file
  template <typename D = Derived, typename = std::enable_if_t<detail::has_data_stream<D>::value>>
  void append_data(const std::filesystem::path& filepath) {
    if (!has_feature(IntegratorFeature::DATA)) {
      throw std::runtime_error(std::string(to_string(id())) +
                               " does not support data accumulation");
    }
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
      throw std::ios_base::failure("Failed to open data file: " + filepath.string());
    }
    read_header(ifs, FileType::DATA);
    derived().accumulate_data_stream(ifs);
    if (!ifs) {
      throw std::ios_base::failure("Error reading data file: " + filepath.string());
    }
  }
  template <typename D = Derived, typename = std::enable_if_t<detail::has_data_stream<D>::value>>
  std::filesystem::path append_data() {
    std::filesystem::path fdata = file_data();
    append_data(fdata);
    return fdata;
  }

  void write_rng_state_stream(std::ostream& out) const {
    out << random_generator_;
  }
  void read_rng_state_stream(std::istream& in) {
    in >> random_generator_;
  }
  void save_rng_state(const std::filesystem::path& filepath) const {
    std::ofstream ofs(filepath, std::ios::binary);
    write_rng_state_stream(ofs);
  }
  void load_rng_state(const std::filesystem::path& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
      throw std::ios_base::failure("Failed to open RNG state file: " + filepath.string());
    }
    read_rng_state_stream(ifs);
    if (!ifs) {
      throw std::ios_base::failure("Error reading RNG state file: " + filepath.string());
    }
  }

 protected:
  size_type ndim_;
  RNG random_generator_;
  DIST uniform_distribution_;
  options_type opts_;

  inline value_type ran() noexcept {
    return uniform_distribution_(random_generator_);
  }

  inline Derived& derived() {
    return static_cast<Derived&>(*this);
  }

  inline const Derived& derived() const {
    return static_cast<const Derived&>(*this);
  }

 private:
  static constexpr std::string_view file_signature_ = "KAKUHEN\0";  // +'\0' => 8 bytes
  static constexpr std::size_t file_signature_size_ = file_signature_.size();
  enum class FileType : uint8_t { STATE = 0, DATA = 1 };
  static constexpr std::string_view suffix_state_ = ".khs";
  static constexpr std::string_view suffix_data_ = ".khd";

  template <typename D = Derived, typename = std::enable_if_t<detail::has_prefix<D>::value>>
  inline std::filesystem::path file_state() const noexcept {
    std::filesystem::path fstate = derived().prefix() + std::string(suffix_state_);
    if (opts_.file_path) {
      fstate = *opts_.file_path;
      fstate.replace_extension(suffix_state_);
    }
    return fstate;
  }

  template <typename D = Derived, typename = std::enable_if_t<detail::has_prefix<D>::value>>
  inline std::filesystem::path file_data() const noexcept {
    std::string seed_suffix = ".s" + std::to_string(opts_.seed.value_or(0));
    std::filesystem::path fdata = derived().prefix(true) + seed_suffix + std::string(suffix_data_);
    if (opts_.file_path) {
      fdata = *opts_.file_path;
      fdata.replace_extension(seed_suffix + std::string(suffix_data_));
    }
    return fdata;
  }

  void write_header(std::ostream& out, FileType ftype) const {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    write_bytes(out, file_signature_.data(), file_signature_size_);
    serialize_one<IntegratorId>(out, id());
    serialize_one<FileType>(out, ftype);
    int16_t T_tos = get_type_or_size<value_type>();
    int16_t S_tos = get_type_or_size<size_type>();
    int16_t U_tos = get_type_or_size<count_type>();
    serialize_one<int16_t>(out, T_tos);
    serialize_one<int16_t>(out, S_tos);
    serialize_one<int16_t>(out, U_tos);
  }

  void read_header(std::istream& in, FileType expected_ftype) const {
    using namespace kakuhen::util::serialize;
    using namespace kakuhen::util::type;
    //> check the file signature
    std::array<char, file_signature_size_> buf{};
    read_bytes(in, buf.data(), file_signature_size_);
    if (std::string_view(buf.data(), buf.size()) != file_signature_) {
      throw std::runtime_error("Invalid kakuhen file signature");
    }
    //> integrator id check
    IntegratorId id_chk;
    deserialize_one<IntegratorId>(in, id_chk);
    if (id_chk != id()) {
      throw std::runtime_error("Integrator id mismatch");
    }
    //> file type check
    FileType ftype_chk;
    deserialize_one<FileType>(in, ftype_chk);
    if (ftype_chk != expected_ftype) {
      throw std::runtime_error("File type mismatch " +
                               std::to_string(static_cast<uint8_t>(ftype_chk)) +
                               " != " + std::to_string(static_cast<uint8_t>(expected_ftype)));
    }
    //> type checks
    int16_t T_tos;
    deserialize_one<int16_t>(in, T_tos);
    if (T_tos != get_type_or_size<value_type>()) {
      throw std::runtime_error("type or size mismatch for typename T");
    }
    int16_t S_tos;
    deserialize_one<int16_t>(in, S_tos);
    if (S_tos != get_type_or_size<size_type>()) {
      throw std::runtime_error("type or size mismatch for typename S");
    }
    int16_t U_tos;
    deserialize_one<int16_t>(in, U_tos);
    if (U_tos != get_type_or_size<count_type>()) {
      throw std::runtime_error("type or size mismatch for typename U");
    }
  }

};  // class IntegratorBase

}  // namespace kakuhen::integrator
