#pragma once

#include <cassert>
#include <iostream>
#include <stack>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace kakuhen::util::printer {

/// @brief Context for the printer (Object or Array).
enum class Context { OBJECT, ARRAY };

/*!
 * @brief Base class for printers, using CRTP.
 *
 * This class provides a generic interface for printing various types of data
 * to an output stream. It uses the Curiously Recurring Template Pattern (CRTP)
 * to allow derived classes to implement specific printing formats (e.g., JSON,
 * XML) while sharing common functionalities.
 *
 * @tparam Derived The derived printer class.
 */
template <typename Derived>
class PrinterBase {
 public:
  /*!
   * @brief Constructs a PrinterBase.
   *
   * @param os The output stream to print to.
   */
  explicit PrinterBase(std::ostream& os) : os_(os) {}

  /*!
   * @brief Stream operator overload.
   *
   * @tparam T The type of the value to print.
   * @param value The value to print.
   * @return A reference to the derived printer object.
   */
  template <typename T>
  PrinterBase& operator<<(const T& value) {
    os_ << value;
    return *this;
  }

  /*!
   * @brief Resets the internal state of the printer.
   *
   * This typically clears any accumulated context or formatting states.
   */
  void reset() {
    os_.clear();
  }

  /*!
   * @brief Inserts a line break or formatting specific to the printer.
   *
   * For some printers (e.g., JSON), this might involve adding indentation.
   */
  void break_line() { /* noop in base */ }

  /*!
   * @brief Begins a new context (e.g., object or array).
   *
   * @tparam C The context type (Context::OBJECT or Context::ARRAY).
   * @param key An optional key associated with the context (e.g., for JSON object members).
   */
  template <Context C>
  void begin([[maybe_unused]] std::string_view key = {}) { /* noop in base */ }

  /*!
   * @brief Ends the current context.
   *
   * @tparam C The context type.
   * @param do_break Whether to force a line break after ending the context.
   */
  template <Context C>
  void end([[maybe_unused]] bool do_break = false) { /* noop in base */ }

  /*!
   * @brief Prints a single key-value pair or a standalone value.
   *
   * The actual printing logic is implemented in the derived class.
   *
   * @tparam T The type of the value to print.
   * @param key The key associated with the value (can be empty for standalone values).
   * @param value The value to print.
   */
  template <typename T>
  void print_one(std::string_view key, T&& value) {
    derived().template print_one_impl<T>(key, std::forward<T>(value));
  }

  /*!
   * @brief Prints a range of elements as an array.
   *
   * This function iterates through a range defined by iterators and prints
   * each element. It can optionally prepend and append elements.
   *
   * @tparam Iterator The iterator type for the range.
   * @tparam PreRange Type of initializer list for prepended elements.
   * @tparam PostRange Type of initializer list for appended elements.
   * @param key An optional key for the array.
   * @param first An iterator to the beginning of the range.
   * @param last An iterator to the end of the range.
   * @param prepend An initializer list of elements to print before the range.
   * @param append An initializer list of elements to print after the range.
   */
  template <typename Iterator,
            typename PreRange =
                std::initializer_list<typename std::iterator_traits<Iterator>::value_type>,
            typename PostRange =
                std::initializer_list<typename std::iterator_traits<Iterator>::value_type>>
  void print_array(std::string_view key, Iterator first, Iterator last,
                   const PreRange& prepend = {}, const PostRange& append = {}) {
    auto print_elem = [&](auto&& elem) {
      if constexpr (is_printable_v<decltype(elem)>)
        print_one({}, elem);
      else
        elem.print(*this);
    };

    derived().template begin<Context::ARRAY>(key);

    // Prepend
    for (auto&& e : prepend)
      print_elem(e);
    // Main range
    for (auto it = first; it != last; ++it)
      print_elem(*it);
    // Append
    for (auto&& e : append)
      print_elem(e);

    derived().template end<Context::ARRAY>();
  }

  /*!
   * @brief Prints an array-like container.
   *
   * Overload that takes an array-like container (e.g., `std::vector`, `NDArray`).
   *
   * @tparam Array The array-like container type.
   * @tparam PreRange Type of initializer list for prepended elements.
   * @tparam PostRange Type of initializer list for appended elements.
   * @param key An optional key for the array.
   * @param a The array-like container.
   * @param prepend An initializer list of elements to print before the array.
   * @param append An initializer list of elements to print after the array.
   */
  template <typename Array, typename PreRange = std::initializer_list<typename Array::value_type>,
            typename PostRange = std::initializer_list<typename Array::value_type>>
  void print_array(std::string_view key, const Array& a, const PreRange& prepend = {},
                   const PostRange& append = {}) {
    print_array(key, std::begin(a), std::end(a), prepend, append);
  }

  /*!
   * @brief Prints a C-style array (pointer and count).
   *
   * Overload that takes a pointer to the first element and a count.
   *
   * @tparam T The type of elements in the array.
   * @tparam PreRange Type of initializer list for prepended elements.
   * @tparam PostRange Type of initializer list for appended elements.
   * @param key An optional key for the array.
   * @param ptr Pointer to the first element.
   * @param count The number of elements to print.
   * @param prepend An initializer list of elements to print before the array.
   * @param append An initializer list of elements to print after the array.
   */
  template <typename T, typename PreRange = std::initializer_list<T>,
            typename PostRange = std::initializer_list<T>>
  void print_array(std::string_view key, const T* ptr, std::size_t count,
                   const PreRange& prepend = {}, const PostRange& append = {}) {
    print_array(key, ptr, ptr + count, prepend, append);
  }

 protected:
  std::ostream& os_;

  inline Derived& derived() {
    return static_cast<Derived&>(*this);
  }

  inline const Derived& derived() const {
    return static_cast<const Derived&>(*this);
  }

  template <typename U>
  static inline constexpr bool is_printable_v =
      std::is_arithmetic_v<std::decay_t<U>> || std::is_same_v<std::decay_t<U>, std::string> ||
      std::is_same_v<std::decay_t<U>, std::string_view> ||
      std::is_constructible_v<std::string_view, U>;
};

/*!
 * @brief A concrete printer class that generates JSON output.
 *
 * This class inherits from `PrinterBase` and implements the `print_one_impl`
 * method to format output as JSON. It supports objects, arrays, and basic
 * data types, handling JSON-specific escaping and formatting (indentation).
 */
class JSONPrinter : public PrinterBase<JSONPrinter> {
 public:
  // dependent class: need to explicitly load things from the Base
  using Base = PrinterBase<JSONPrinter>;

  explicit JSONPrinter(std::ostream& os, uint8_t indent = 0)
      : Base(os), indent_{indent}, context_stack_{}, first_(true) {
    reset();
  }

  /*!
   * @brief Resets the JSON printer to its initial state.
   */
  void reset() {
    Base::reset();
    context_stack_ = {};
    first_ = true;
  }

  /*!
   * @brief Inserts a line break and indentation if `indent_` is greater than 0.
   */
  void break_line() {
    if (indent_ <= 0) return;  // no indent: single line
    os_ << '\n' << std::string(context_stack_.size() * indent_, ' ');
  }

  /*!
   * @brief Begins a new JSON object or array.
   *
   * @tparam C The context type (Context::OBJECT or Context::ARRAY).
   * @param key An optional key for the object member or array.
   */
  template <Context C>
  void begin(std::string_view key = {}) {
    prefix(key);
    if constexpr (C == Context::OBJECT)
      os_ << "{";
    else if constexpr (C == Context::ARRAY)
      os_ << "[";
    else
      std::cerr << "Invalid context\n";
    context_stack_.push(C);
    first_ = true;
  }

  /*!
   * @brief Ends the current JSON object or array.
   *
   * @tparam C The context type.
   * @param do_break Whether to force a line break after ending the context.
   */
  template <Context C>
  void end(bool do_break = false) {
    assert(context_stack_.top() == C);
    context_stack_.pop();
    if (do_break && indent_ > 0) break_line();
    if constexpr (C == Context::OBJECT)
      os_ << "}";
    else if constexpr (C == Context::ARRAY)
      os_ << "]";
    else
      std::cerr << "Invalid context\n";
    first_ = false;
  }

 private:
  friend class PrinterBase<JSONPrinter>;

  using Base::os_;
  uint8_t indent_ = 0;
  std::stack<Context> context_stack_;
  bool first_;

  template <typename T>
  void print_one_impl(std::string_view key, T&& value) {
    prefix(key);
    if constexpr (std::is_arithmetic_v<std::decay_t<T>>) {
      /// the `+` trick here forces that uint8_t are not printed as char
      os_ << +std::forward<T>(value);
    } else if constexpr (std::is_same_v<std::decay_t<T>, std::string_view> ||
                         std::is_same_v<std::decay_t<T>, std::string> ||
                         std::is_constructible_v<std::string_view, T>) {
      print_escaped(std::forward<T>(value));
    } else {
      static_assert(sizeof(T) == 0, "Unsupported type for print_one");
    }
    first_ = false;
  }

  void prefix(std::string_view key) {
    if (!first_) os_ << (indent_ > 0 ? ", " : ",");  // no indent: no padding
    if (!key.empty()) break_line();
    if (!context_stack_.empty() && context_stack_.top() == Context::OBJECT && !key.empty()) {
      print_escaped(key);
      os_ << (indent_ > 0 ? ": " : ":");
    }
  }

  // Optimized escape that writes directly to stream
  void print_escaped(std::string_view s) {
    os_ << '"';
    for (char c_signed : s) {
      const auto c = static_cast<unsigned char>(c_signed);
      switch (c) {
        case '"':
          os_ << "\\\"";
          break;
        case '\\':
          os_ << "\\\\";
          break;
        case '\b':
          os_ << "\\b";
          break;
        case '\f':
          os_ << "\\f";
          break;
        case '\n':
          os_ << "\\n";
          break;
        case '\r':
          os_ << "\\r";
          break;
        case '\t':
          os_ << "\\t";
          break;
        default:
          if (c < 0x20) {
            // non-printable control character -> use \u00XX
            constexpr char hex[] = "0123456789abcdef";
            os_ << "\\u00" << hex[(c >> 4) & 0xF] << hex[c & 0xF];
          } else {
            os_ << static_cast<char>(c);
          }
          break;
      }
    }
    os_ << '"';
  }
};

}  // namespace kakuhen::util::printer
