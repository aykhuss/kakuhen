#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace kakuhen::util {

/**
 * @brief A small-capacity optimized vector for trivially copyable types.
 *
 * SmallVector provides a vector-like interface but stores its first N elements
 * in-place (on the stack or within the object itself). This avoids heap allocations
 * for small collections. It is restricted to trivially copyable types to allow
 * highly efficient memory operations like std::memcpy.
 *
 * @tparam T The type of elements. Must be trivially copyable.
 * @tparam N The number of elements to store inline. Must be greater than 0.
 */
template <typename T, std::size_t N>
class SmallVector {
  static_assert(std::is_trivially_copyable_v<T>,
                "SmallVector requires trivially copyable T");
  static_assert(N > 0, "SmallVector requires N > 0");

 public:
  using value_type = T;
  using size_type = std::size_t;
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using iterator = T*;
  using const_iterator = const T*;

  /**
   * @brief Constructs an empty SmallVector with inline capacity N.
   */
  SmallVector() noexcept : size_(0), capacity_(static_cast<std::uint32_t>(N)), heap_(nullptr) {}

  /**
   * @brief Constructs a SmallVector with elements from an initializer list.
   * @param init Initializer list of elements.
   */
  SmallVector(std::initializer_list<T> init) : SmallVector() {
    reserve(init.size());
    std::memcpy(data(), init.begin(), init.size() * sizeof(T));
    size_ = static_cast<std::uint32_t>(init.size());
  }

  /**
   * @brief Destructor. Releases heap memory if it was allocated.
   */
  ~SmallVector() {
    if (heap_) {
      ::operator delete(heap_);
    }
  }

  /**
   * @brief Copy constructor.
   * @param other The SmallVector to copy from.
   */
  SmallVector(const SmallVector& other) : SmallVector() {
    assign_from(other);
  }

  /**
   * @brief Copy assignment operator.
   * @param other The SmallVector to copy from.
   * @return Reference to this SmallVector.
   */
  SmallVector& operator=(const SmallVector& other) {
    if (this != &other) {
      assign_from(other);
    }
    return *this;
  }

  /**
   * @brief Move constructor.
   * @param other The SmallVector to move from.
   */
  SmallVector(SmallVector&& other) noexcept : SmallVector() {
    move_from(std::move(other));
  }

  /**
   * @brief Move assignment operator.
   * @param other The SmallVector to move from.
   * @return Reference to this SmallVector.
   */
  SmallVector& operator=(SmallVector&& other) noexcept {
    if (this != &other) {
      move_from(std::move(other));
    }
    return *this;
  }

  // --- Element Access ---

  /** @brief Returns a reference to the element at index i. */
  [[nodiscard]] reference operator[](size_type i) noexcept {
    assert(i < size_);
    return data()[i];
  }

  /** @brief Returns a const reference to the element at index i. */
  [[nodiscard]] const_reference operator[](size_type i) const noexcept {
    assert(i < size_);
    return data()[i];
  }

  /** @brief Returns a reference to the element at index i with bounds checking. */
  [[nodiscard]] reference at(size_type i) {
    if (i >= size_) {
      throw std::out_of_range("SmallVector::at index out of range");
    }
    return data()[i];
  }

  /** @brief Returns a const reference to the element at index i with bounds checking. */
  [[nodiscard]] const_reference at(size_type i) const {
    if (i >= size_) {
      throw std::out_of_range("SmallVector::at index out of range");
    }
    return data()[i];
  }

  /** @brief Returns a reference to the first element. */
  [[nodiscard]] reference front() noexcept {
    assert(size_ > 0);
    return data()[0];
  }

  /** @brief Returns a const reference to the first element. */
  [[nodiscard]] const_reference front() const noexcept {
    assert(size_ > 0);
    return data()[0];
  }

  /** @brief Returns a reference to the last element. */
  [[nodiscard]] reference back() noexcept {
    assert(size_ > 0);
    return data()[size_ - 1];
  }

  /** @brief Returns a const reference to the last element. */
  [[nodiscard]] const_reference back() const noexcept {
    assert(size_ > 0);
    return data()[size_ - 1];
  }

  /** @brief Returns a pointer to the underlying data array. */
  [[nodiscard]] pointer data() noexcept {
    return heap_ ? heap_ : inline_ptr();
  }

  /** @brief Returns a const pointer to the underlying data array. */
  [[nodiscard]] const_pointer data() const noexcept {
    return heap_ ? heap_ : inline_ptr();
  }

  // --- Iterators ---

  [[nodiscard]] iterator begin() noexcept { return data(); }
  [[nodiscard]] const_iterator begin() const noexcept { return data(); }
  [[nodiscard]] const_iterator cbegin() const noexcept { return data(); }
  [[nodiscard]] iterator end() noexcept { return data() + size_; }
  [[nodiscard]] const_iterator end() const noexcept { return data() + size_; }
  [[nodiscard]] const_iterator cend() const noexcept { return data() + size_; }

  // --- Capacity ---

  /** @brief Returns true if the vector is empty. */
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

  /** @brief Returns the number of elements in the vector. */
  [[nodiscard]] size_type size() const noexcept { return size_; }

  /** @brief Returns the current capacity of the vector. */
  [[nodiscard]] size_type capacity() const noexcept { return capacity_; }

  /** @brief Returns the fixed inline capacity. */
  [[nodiscard]] static constexpr size_type inline_capacity() noexcept { return N; }

  /**
   * @brief Ensures that the capacity is at least new_cap.
   * @param new_cap The desired minimum capacity.
   */
  void reserve(size_type new_cap) {
    if (new_cap > capacity_) {
      reallocate(new_cap);
    }
  }

  /**
   * @brief Reduces capacity to match size, moving back to inline storage if possible.
   */
  void shrink_to_fit() {
    if (!heap_ || size_ > N) return;
    T* old_heap = heap_;
    if (size_ > 0) {
      std::memcpy(inline_ptr(), old_heap, size_ * sizeof(T));
    }
    heap_ = nullptr;
    capacity_ = static_cast<std::uint32_t>(N);
    ::operator delete(old_heap);
  }

  // --- Modifiers ---

  /** @brief Removes all elements. Heap memory is retained. */
  void clear() noexcept { size_ = 0; }

  /** @brief Appends an element to the end. */
  void push_back(const T& value) {
    if (size_ == capacity_) {
      grow();
    }
    data()[size_++] = value;
  }

  /** @brief Constructs an element in-place at the end. */
  template <typename... Args>
  reference emplace_back(Args&&... args) {
    if (size_ == capacity_) {
      grow();
    }
    pointer p = &data()[size_++];
    new (p) T(std::forward<Args>(args)...);
    return *p;
  }

  /** @brief Removes the last element. */
  void pop_back() noexcept {
    assert(size_ > 0);
    --size_;
  }

  /**
   * @brief Resizes the vector to new_size.
   * @param new_size The new number of elements.
   */
  void resize(size_type new_size) {
    if (new_size > capacity_) {
      reserve(new_size);
    }
    size_ = static_cast<std::uint32_t>(new_size);
  }

  /**
   * @brief Resizes the vector to new_size, filling new elements with value.
   * @param new_size The new number of elements.
   * @param value The value to fill new elements with.
   */
  void resize(size_type new_size, const T& value) {
    size_type old_size = size_;
    resize(new_size);
    if (new_size > old_size) {
      std::fill(data() + old_size, data() + new_size, value);
    }
  }

  /** @brief Compares two SmallVectors for equality. */
  bool operator==(const SmallVector& other) const noexcept {
    if (size_ != other.size_) return false;
    if (size_ == 0) return true;
    return std::memcmp(data(), other.data(), size_ * sizeof(T)) == 0;
  }

  /** @brief Compares two SmallVectors for inequality. */
  bool operator!=(const SmallVector& other) const noexcept {
    return !(*this == other);
  }

 private:
  void grow() {
    reallocate(capacity_ == 0 ? N : capacity_ * 2);
  }

  void reallocate(size_type new_cap) {
    assert(new_cap >= size_);
    T* new_heap = static_cast<T*>(::operator new(new_cap * sizeof(T)));
    if (size_ > 0) {
      std::memcpy(new_heap, data(), size_ * sizeof(T));
    }
    if (heap_) {
      ::operator delete(heap_);
    }
    heap_ = new_heap;
    capacity_ = static_cast<std::uint32_t>(new_cap);
  }

  pointer inline_ptr() noexcept {
    return reinterpret_cast<pointer>(inline_storage_);
  }

  const_pointer inline_ptr() const noexcept {
    return reinterpret_cast<const_pointer>(inline_storage_);
  }

  void assign_from(const SmallVector& other) {
    if (other.size_ <= N) {
      if (heap_) {
        ::operator delete(heap_);
        heap_ = nullptr;
      }
      capacity_ = static_cast<std::uint32_t>(N);
      size_ = other.size_;
      if (size_ > 0) {
        std::memcpy(inline_ptr(), other.data(), size_ * sizeof(T));
      }
    } else {
      if (!heap_ || capacity_ < other.size_) {
        if (heap_) ::operator delete(heap_);
        heap_ = static_cast<T*>(::operator new(other.size_ * sizeof(T)));
        capacity_ = other.size_;
      }
      size_ = other.size_;
      std::memcpy(heap_, other.data(), size_ * sizeof(T));
    }
  }

  void move_from(SmallVector&& other) noexcept {
    if (other.heap_) {
      if (heap_) ::operator delete(heap_);
      heap_ = other.heap_;
      capacity_ = other.capacity_;
      size_ = other.size_;
      other.heap_ = nullptr;
      other.capacity_ = static_cast<std::uint32_t>(N);
      other.size_ = 0;
    } else {
      if (heap_) {
        ::operator delete(heap_);
        heap_ = nullptr;
      }
      capacity_ = static_cast<std::uint32_t>(N);
      size_ = other.size_;
      if (size_ > 0) {
        std::memcpy(inline_ptr(), other.inline_ptr(), size_ * sizeof(T));
      }
      other.size_ = 0;
    }
  }

 private:
  std::uint32_t size_;
  std::uint32_t capacity_;
  T* heap_;
  alignas(T) std::byte inline_storage_[N * sizeof(T)];
};

}  // namespace kakuhen::util