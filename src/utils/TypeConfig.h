#ifndef dftefeTypeConfig_h
#define dftefeTypeConfig_h

#include <complex>
#include <cstdint>

namespace dftefe
{
  using size_type        = unsigned int;
  using global_size_type = unsigned long int;
  using uInt = std::uint64_t; // only in matrix free. check later.

  /* define RealType of a ValueType */

  template <typename T>
  struct RealType
  {
    typedef void Type;
  };

  template <>
  struct RealType<int>
  {
    typedef int Type;
  };

  template <>
  struct RealType<float>
  {
    typedef float Type;
  };

  template <>
  struct RealType<double>
  {
    typedef double Type;
  };

  template <>
  struct RealType<std::complex<float>>
  {
    typedef float Type;
  };

  template <>
  struct RealType<std::complex<double>>
  {
    typedef double Type;
  };

} // namespace dftefe
#endif
