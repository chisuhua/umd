#pragma once

#include <iostream>
#include <string>
#include <algorithm>
#include "stdint.h"
#include <cstdint>
#include <cassert>

#define __forceinline __inline__ __attribute__((always_inline))

/** uint64_t constant */
#define ULL(N)          ((uint64_t)N##ULL)
/** int64_t constant */
#define LL(N)           ((int64_t)N##LL)

/// @brief: Finds out the min one of two inputs, input must support ">"
/// operator.
/// @param: a(Input), a reference to type T.
/// @param: b(Input), a reference to type T.
/// @return: T.
template <class T>
static __forceinline T Min(const T& a, const T& b) {
  return (a > b) ? b : a;
}

template <class T, class... Arg>
static __forceinline T Min(const T& a, const T& b, Arg... args) {
  return Min(a, Min(b, args...));
}

/// @brief: Find out the max one of two inputs, input must support ">" operator.
/// @param: a(Input), a reference to type T.
/// @param: b(Input), a reference to type T.
/// @return: T.
template <class T>
static __forceinline T Max(const T& a, const T& b) {
  return (b > a) ? b : a;
}

template <class T, class... Arg>
static __forceinline T Max(const T& a, const T& b, Arg... args) {
  return Max(a, Max(b, args...));
}

/// @brief: Checks if a value is power of two, if it is, return true. Be careful
/// when passing 0.
/// @param: val(Input), the data to be checked.
/// @return: bool.
template <typename T>
static __forceinline bool isPowerOf2(T val) {
  return (val & (val - 1)) == 0;
    // return n && !(n & (n - 1));
}

/// @brief: Calculates the floor value aligned based on parameter of alignment.
/// If value is at the boundary of alignment, it is unchanged.
/// @param: value(Input), value to be calculated.
/// @param: alignment(Input), alignment value.
/// @return: T.
template <typename T>
static __forceinline T alignDown(T value, size_t alignment) {
  assert(isPowerOf2(alignment));
  return (T)(value & ~(alignment - 1));
}

/// @brief: Same as previous one, but first parameter becomes pointer, for more
/// info, see the previous desciption.
/// @param: value(Input), pointer to type T.
/// @param: alignment(Input), alignment value.
/// @return: T*, pointer to type T.
template <typename T>
static __forceinline T* alignDown(T* value, size_t alignment) {
  return (T*)alignDown((intptr_t)value, alignment);
}

/// @brief: Calculates the ceiling value aligned based on parameter of
/// alignment.
/// If value is at the boundary of alignment, it is unchanged.
/// @param: value(Input), value to be calculated.
/// @param: alignment(Input), alignment value.
/// @param: T.
template <typename T>
static __forceinline T alignUp(T value, size_t alignment) {
  return alignDown((T)(value + alignment - 1), alignment);
}

/// @brief: Same as previous one, but first parameter becomes pointer, for more
/// info, see the previous desciption.
/// @param: value(Input), pointer to type T.
/// @param: alignment(Input), alignment value.
/// @return: T*, pointer to type T.
template <typename T>
static __forceinline T* alignUp(T* value, size_t alignment) {
  return (T*)alignDown((intptr_t)((uint8_t*)value + alignment - 1), alignment);
}

/// @brief: Checks if the input value is at the boundary of alignment, if it is,
/// @return true.
/// @param: value(Input), value to be checked.
/// @param: alignment(Input), alignment value.
/// @return: bool.
template <typename T>
static __forceinline bool isMultipleOf(T value, size_t alignment) {
  return (alignUp(value, alignment) == value);
}

/// @brief: Same as previous one, but first parameter becomes pointer, for more
/// info, see the previous desciption.
/// @param: value(Input), pointer to type T.
/// @param: alignment(Input), alignment value.
/// @return: bool.
template <typename T>
static __forceinline bool isMultipleOf(T* value, size_t alignment) {
  return (alignUp(value, alignment) == value);
}


static __forceinline uint32_t NextPow2(uint32_t value) {
  if (value == 0) return 1;
  uint32_t v = value - 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

static __forceinline uint64_t NextPow2(uint64_t value) {
  if (value == 0) return 1;
  uint64_t v = value - 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v |= v >> 32;
  return v + 1;
}

template <uint32_t lowBit, uint32_t highBit, typename T>
static __forceinline uint32_t BitSelect(T p) {
  static_assert(sizeof(T) <= sizeof(uintptr_t), "Type out of range.");
  static_assert(highBit < sizeof(uintptr_t) * 8, "Bit index out of range.");

  uintptr_t ptr = p;
  if (highBit != (sizeof(uintptr_t) * 8 - 1))
    return (uint32_t)((ptr & ((1ull << (highBit + 1)) - 1)) >> lowBit);
  else
    return (uint32_t)(ptr >> lowBit);
}

inline uint32_t PtrLow16Shift8(const void* p) {
  uintptr_t ptr = reinterpret_cast<uintptr_t>(p);
  return (uint32_t)((ptr & 0xFFFFULL) >> 8);
}

inline uint32_t PtrHigh64Shift16(const void* p) {
  uintptr_t ptr = reinterpret_cast<uintptr_t>(p);
  return (uint32_t)((ptr & 0xFFFFFFFFFFFF0000ULL) >> 16);
}

inline uint32_t PtrLow40Shift8(const void* p) {
  uintptr_t ptr = reinterpret_cast<uintptr_t>(p);
  return (uint32_t)((ptr & 0xFFFFFFFFFFULL) >> 8);
}

inline uint32_t PtrHigh64Shift40(const void* p) {
  uintptr_t ptr = reinterpret_cast<uintptr_t>(p);
  return (uint32_t)((ptr & 0xFFFFFF0000000000ULL) >> 40);
}

inline uint32_t PtrLow32(const void* p) {
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
}

inline uint32_t PtrHigh32(const void* p) {
  uint32_t ptr = 0;
#ifdef HSA_LARGE_MODEL
  ptr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p) >> 32);
#endif
  return ptr;
}

inline uint64_t
power(uint32_t n, uint32_t e)
{
    uint64_t result = 1;
    uint64_t component = n;
    while (e) {
        uint64_t last = result;
        if (e & 0x1)
            result *= component;
        // warn_if(result < last, "power() overflowed!");
        e >>= 1;
        component *= component;
    }
    return result;
}

template <class T>
inline typename std::enable_if<std::is_integral<T>::value, int>::type
floorLog2(T x)
{
    assert(x > 0);

    // A guaranteed unsigned version of x.
    uint64_t ux = (typename std::make_unsigned<T>::type)x;

    int y = 0;
    constexpr auto ts = sizeof(T);

    if (ts >= 8 && (ux & ULL(0xffffffff00000000))) { y += 32; ux >>= 32; }
    if (ts >= 4 && (ux & ULL(0x00000000ffff0000))) { y += 16; ux >>= 16; }
    if (ts >= 2 && (ux & ULL(0x000000000000ff00))) { y +=  8; ux >>=  8; }
    if (ux & ULL(0x00000000000000f0)) { y +=  4; ux >>=  4; }
    if (ux & ULL(0x000000000000000c)) { y +=  2; ux >>=  2; }
    if (ux & ULL(0x0000000000000002)) { y +=  1; }

    return y;
}

template <class T>
inline int
ceilLog2(const T& n)
{
    assert(n > 0);
    if (n == 1)
        return 0;

    return floorLog2(n - (T)1) + 1;
}

template <class T, class U>
inline T
divCeil(const T& a, const U& b)
{
    return (a + b - 1) / b;
}

/**
 * This function is used to align addresses in memory.
 *
 * @param val is the address to be aligned.
 * @param align is the alignment. Can only be a power of 2.
 * @return The aligned address. The smallest number divisible
 * by @param align which is greater than or equal to @param val.
*/
template <class T, class U>
inline T
roundUp(const T& val, const U& align)
{
    assert(isPowerOf2(align));
    T mask = (T)align - 1;
    return (val + mask) & ~mask;
}

/**
 * This function is used to align addresses in memory.
 *
 * @param val is the address to be aligned.
 * @param align is the alignment. Can only be a power of 2.
 * @return The aligned address. The biggest number divisible
 * by @param align which is less than or equal to @param val.
*/
template <class T, class U>
inline T
roundDown(const T& val, const U& align)
{
    assert(isPowerOf2(align));
    T mask = (T)align - 1;
    return val & ~mask;
}

/* Void pointer arithmetic (or remove -Wpointer-arith to allow void pointers arithmetic) */
#define VOID_PTR_ADD32(ptr,n) (void*)((uint32_t*)(ptr) + n)/*ptr + offset*/
#define VOID_PTR_ADD(ptr,n) (void*)((uint8_t*)(ptr) + n)/*ptr + offset*/
#define VOID_PTR_SUB(ptr,n) (void*)((uint8_t*)(ptr) - n)/*ptr - offset*/
#define VOID_PTRS_SUB(ptr1,ptr2) (uint64_t)((uint8_t*)(ptr1) - (uint8_t*)(ptr2)) /*ptr1 - ptr2*/

