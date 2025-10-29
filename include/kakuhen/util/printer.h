#pragma once

#include <cassert>
#include <iostream>
#include <stack>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace kakuhen::util::printer {

enum class Context { OBJECT, ARRAY };

template <typename Derived>
class PrinterBase {
 public:
  explicit PrinterBase(std::ostream& os) : os_(os) {}

  template <typename T>
  PrinterBase& operator<<(const T& value) {
    os_ << value;
    return *this;
  }

  void reset() {
    os_.clear();
  }

  void break_line() { /* noop in base */ }

  template <Context C>
  void begin(std::string_view key = {}) { /* noop in base */ }

  template <Context C>
  void end(bool do_break = false) { /* noop in base */ }

  template <typename T>
  void print_one(std::string_view key, T&& value) {
    derived().template print_one_impl<T>(key, std::forward<T>(value));
  }

  // print a range [first, last) as an array
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

  template <typename Array, typename PreRange = std::initializer_list<typename Array::value_type>,
            typename PostRange = std::initializer_list<typename Array::value_type>>
  void print_array(std::string_view key, const Array& a, const PreRange& prepend = {},
                   const PostRange& append = {}) {
    print_array(key, std::begin(a), std::end(a), prepend, append);
  }

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
      std::is_same_v<std::decay_t<U>, std::string_view>;

};  // class PrinterBase

class JSONPrinter : public PrinterBase<JSONPrinter> {
 public:
  // dependent class: need to explicitly load things from the Base
  using Base = PrinterBase<JSONPrinter>;

  explicit JSONPrinter(std::ostream& os, uint8_t indent = 0)
      : Base(os), indent_{indent}, context_stack_{}, first_(true) {
    reset();
  }

  void reset() {
    Base::reset();
    context_stack_ = {};
    first_ = true;
  }

  void break_line() {
    if (indent_ <= 0) return;  // no indent: single line
    os_ << '\n' << std::string(context_stack_.size() * indent_, ' ');
  }

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

  template <typename T>
  void print_one_impl(std::string_view key, T&& value) {
    prefix(key);
    if constexpr (std::is_arithmetic_v<std::decay_t<T>>) {
      /// the `+` trick here forces that uint8_t are not printed as char
      os_ << +std::forward<T>(value);
    } else if constexpr (std::is_same_v<std::decay_t<T>, std::string_view> ||
                         std::is_same_v<std::decay_t<T>, std::string>) {
      os_ << escape(std::forward<T>(value));
    } else {
      static_assert(sizeof(T) == 0, "Unsupported type for print_one");
    }
    first_ = false;
  }

 private:
  using Base::os_;
  uint8_t indent_ = 0;
  std::stack<Context> context_stack_;
  bool first_;

  void prefix(std::string_view key) {
    if (!first_) os_ << (indent_ > 0 ? ", " : ",");  // no indent: no padding
    if (!key.empty()) break_line();
    if (!context_stack_.empty() && context_stack_.top() == Context::OBJECT && !key.empty())
      os_ << escape(key) << (indent_ > 0 ? ": " : ":");
  }

  static std::string escape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 10);  // reserve extra to reduce reallocs
    out += '"';                  // JSON strings start with a quote
    for (unsigned char c : s) {
      switch (c) {
        case '"':
          out += "\\\"";
          break;
        case '\\':
          out += "\\\\";
          break;
        case '\b':
          out += "\\b";
          break;
        case '\f':
          out += "\\f";
          break;
        case '\n':
          out += "\\n";
          break;
        case '\r':
          out += "\\r";
          break;
        case '\t':
          out += "\\t";
          break;
        default:
          if (c < 0x20) {
            // non-printable control character -> use \u00XX
            out += "\\u";
            constexpr char hex[] = "0123456789abcdef";
            out += hex[(c >> 4) & 0xF];
            out += hex[c & 0xF];
          } else {
            out += c;
          }
          break;
      }
    }
    out += '"';  // closing quote
    return out;
  }
};

}  // namespace kakuhen::util::printer
