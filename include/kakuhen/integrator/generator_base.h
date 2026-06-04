#pragma once

#include "kakuhen/integrator/integral_accumulator.h"
#include "kakuhen/integrator/point.h"
#include "kakuhen/util/numeric_traits.h"
#include "result.h"
#include <cassert>
#include <utility>

namespace kakuhen::integrator {

template <typename Derived, typename NT = util::num_traits_t<>>
class GeneratorBase {
 public:
  using num_traits = NT;
  using value_type = typename num_traits::value_type;
  using size_type = typename num_traits::size_type;
  using count_type = typename num_traits::count_type;
  using point_type = Point<num_traits>;
  using int_acc_type = IntegralAccumulator<value_type, count_type>;
  using result_type = Result<value_type, count_type>;

  // shorthands to save typing (taken from NT, not Derived: Derived is an
  // incomplete type while this base is instantiated in its base-clause).
  using S = size_type;
  using T = value_type;
  using U = count_type;

  template <typename I>
  result_type optimize_bound(I&& integrand, U neval /* todo: , T step_increment = ...*/) {
    assert(derived().is_frozen());
    result_type result;
    // if we ever want to do multiple iterations?
    int_acc_type res_it = derived().optimize_bound_impl(std::forward<I>(integrand), neval);
    result.accumulate(res_it);

    return result;
  }

  inline void bound_init(T abs_integral) {
    derived().bound_init_impl(abs_integral);
  }

  template <typename I, typename ECB>
  result_type generate_events(I&& integrand, U neval, ECB&& event_call_back, T event_norm = T(1)) {
    result_type result;
    int_acc_type res_it = derived().generate_events_impl(
        std::forward<I>(integrand), neval, std::forward<ECB>(event_call_back), event_norm);
    result.accumulate(res_it);
    return result;
  }

 protected:
  /*!
   * @brief Provides access to the derived class instance.
   *
   * This is part of the CRTP pattern.
   *
   * @return A reference to the derived class.
   */
  inline Derived& derived() {
    return static_cast<Derived&>(*this);
  }

  /*!
   * @brief Provides const access to the derived class instance.
   *
   * This is part of the CRTP pattern.
   *
   * @return A const reference to the derived class.
   */
  inline const Derived& derived() const {
    return static_cast<const Derived&>(*this);
  }
};

}  // namespace kakuhen::integrator
