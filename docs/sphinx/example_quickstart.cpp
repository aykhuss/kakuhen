// [example-start]
#include <kakuhen/integrator/plain.h>

#include <iostream>
#include <vector>

int main() {
  // Define the function to integrate: f(x, y) = x^2 + y^2
  auto func = [](const kakuhen::integrator::Point<>& p) {
    return p.x[0] * p.x[0] + p.x[1] * p.x[1];
  };

  // Create a Plain Monte Carlo integrator for 2 dimensions
  kakuhen::integrator::Plain plain_integrator(2);

  // Set integration parameters
  int num_evaluations = 100000;
  int num_iterations = 10;  // For statistical analysis

  // Perform the integration
  auto result =
      plain_integrator.integrate(func, {.neval = num_evaluations, .niter = num_iterations});

  // Print the result
  std::cout << "Estimated Integral: " << result.value() << std::endl;
  std::cout << "Error: " << result.error() << std::endl;
  std::cout << "Chi2/dof: " << result.chi2dof() << std::endl;

  return 0;
}
// [example-end]
