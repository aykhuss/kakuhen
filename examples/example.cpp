#include "kakuhen/kakuhen.h"
#include <iostream>
#include <vector>

int main() {
  // load the namespace for convenience
  using namespace kakuhen::integrator;

  // define the function we want to integrate;
  // Point is what is passed to the function with members
  // {.x[], .weight, .ndim, .sample_index, .user_data}
  auto func = [](const Point<>& point) {
    const auto& x = point.x;  // shorthand
    return (5 * x[0] * x[0] * x[0] * x[0] + 3 * x[0] * x[1] * x[1] + 2 * x[1]);
  };

  // setup vegas as our integrator for 2 dimension & 32 divisions in the grid
  auto vegas_int = Vegas(2, 32);
  // set some options
  vegas_int.set_seed(42);

  // let's do some warmup adaption for the grid and save it to file
  vegas_int.integrate(func, {.neval = 500, .niter = 5, .adapt = true});
  /// alternative where order of keys does not matter.
  // using keys = decltype(vegas_int)::options_type::keys;
  // vegas_int.integrate(func, keys::neval = 500, keys::niter = 5, keys::adapt = true);
  auto veg_file = "vegas_grid.khs";
  vegas_int.save(veg_file);
  std::cout << "worte vegas state to " << veg_file << "\n";

  // we can also parallelize the warmup accross independent runs
  std::vector<std::filesystem::path> data_files{};
  for (size_t i = 100; i < 110; ++i) {
    std::cout << "warmup run " << i << "...\n";
    auto vegas_int_i = Vegas(veg_file);
    vegas_int_i.set_seed(i);
    // skip adaption and instead dump the accumulated data to file
    vegas_int_i.integrate(func, {.neval = 1000, .adapt = false});
    auto data_file = "vegas_data_" + std::to_string(i) + ".khd";
    vegas_int_i.save_data(data_file);
    std::cout << " ... saved data to " << data_file << "\n";
    data_files.push_back(data_file);
  }

  // we can read in all the data files and run the adaption on the total dataset
  std::cout << "\nadapting grid from " << data_files.size() << " data files\n";
  for (const auto& df : data_files) {
    std::cout << "appending " << df << "\n";
    vegas_int.append_data(df);
  }
  vegas_int.adapt();

  // let's freeze the grid and do a production run
  vegas_int.set_options({.adapt = false});
  vegas_int.integrate(func, {.neval = 10000, .niter = 5});
  vegas_int.print_grid();

  // load the state from file: should print out the same grid
  auto vegas_int_2 = Vegas(veg_file);
  vegas_int.print_grid();

  // let's compare with the plain integrator
  auto plain_int = Plain(2);
  auto result = plain_int.integrate(func, {.neval = 10000, .niter = 5, .verbosity = 0});
  std::cout << "plain integral = " << result.value() << " +/- " << result.error();
  std::cout << " (ntotal=" << result.count() << ", chi2/dof=" << result.chi2dof() << ")\n";

  return 0;
}
