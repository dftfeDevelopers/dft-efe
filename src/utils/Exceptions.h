#ifndef dftefeExceptions_h
#define dftefeExceptions_h

#include <string>
#include <stdexcept>
/**
@brief provides an interface for exception handling.
It two overrides on the assert(expr) function in C/C++ and two wrappers on
std::exception.

The overrides on assert(expr) are useful for debug mode testing. That is,
you want the assert to be executed *only in debug mode* and not in release
mode (i.e., if NDEBUG is defined). The two assert overrides are
1. DFTEFE_Assert(expr): same as std::assert(expr). Throws an assert
if expr is false
2. DFTEFE_AssertWithMsg(expr,msg): same as above but takes an
additional message in the form of string to display if expr is false.

It also provides two preprocessor flags, DFTEFE_DISABLE_ASSERT and
DFTEFE_ENABLE_ASSERT, that you can set to override the NDEBUG flag in a
particular source file. This is provided to allow selective enabling or
disabling of Assert and AssertWithMsg without any relation to whether NDEBUG
is defined or not (NDEBUG is typically defined globally for all files
through compiler options).
For example, if in a file you have
#define DFTEFE_DISABLE_ASSERT
#include "Exceptions.h"
then it would disable all any calls to Assert or AssertWithMsg in that file,
regardless of whether NDEBUG is defined. Also, it has no bearing on
std::assert (i.e., any calls to std::assert in that file will still be
governed by NDEBUG) Similarly, if in a file you have #define
DFTEFE_ENABLE_ASSERT #include "Exceptions.h" then it would enable all calls
to Assert or AssertWithMsg regardless in that file, regardless of whether
NDEBUG is defined. Also, it has no bearning on std::assert (i.e., any calls
to std::assert in that file will still be governed by NDEBUG)

It also provides two wrappers on std::exception and its derived classes
(e.g., std::runtime_error, std::domain_error, etc.) The two wrappers are:
1. dftefe::utils::throwException(expr,msg): a generic exception handler
which throws an optional message (msg) if expr evaluates to false. It
combines std::exception with an  additional messgae. (Note: the
std::exception has no easy way of taking in a message).
2. dftefe::utils::throwException<T>(expr, msg): similar to the above, but
takes a specific derived class of std::exception handler as a template
parameter. The derived std::exception must have a constructor that takes in
a string. For the ease of the user, we have typedef-ed some commonly used
derived classes of std::exception. A user can use the typedefs as the
template parameter instead. Available typedefs LogicError - std::logic_error
   InvalidArgument - std::invalid_argument
   DomainError - std::domain_error
   LengthError - std::length_error
   OutOfRangeError - std::out_of_range
   FutureError - std::future_error
   RuntimeError	- std::runtime_error
   OverflowError - std::overflow_error
   UnderflowError - std::underflow_error
*/

#undef DFTEFE_Assert
#undef DFTEFE_AssertWithMsg

#if defined(DFTEFE_DISABLE_ASSERT) || \
  (!defined(DFTEFE_ENABLE_ASSERT) && defined(NDEBUG))
#  include <assert.h> // .h to support old libraries w/o <cassert> - effect is the same
#  define DFTEFE_Assert(expr) ((void)0)
#  define DFTEFE_AssertWithMsg(expr, msg) ((void)0)

#elif defined(DFTEFE_ENABLE_ASSERT) && defined(NDEBUG)
#  undef NDEBUG // disabling NDEBUG to forcibly enable assert for sources that
                // set DFTEFE_ENABLE_ASSERT even when in release mode (with
                // NDEBUG)
#  include <assert.h> // .h to support old libraries w/o <cassert> - effect is the same
#  define DFTEFE_Assert(expr) assert(expr)
#  define DFTEFE_AssertWithMsg(expr, msg) assert((expr) && (msg))

#else
#  include <assert.h> // .h to support old libraries w/o <cassert> - effect is the same
#  define DFTEFE_Assert(expr) assert(expr)
#  define DFTEFE_AssertWithMsg(expr, msg) assert((expr) && (msg))

#endif

#ifdef DFTEFE_WITH_DEVICE_CUDA
#  include <utils/DeviceExceptions.cuh>
#endif

namespace dftefe
{
  namespace utils
  {
    typedef std::logic_error      LogicError;
    typedef std::invalid_argument InvalidArgument;
    typedef std::domain_error     DomainError;
    typedef std::length_error     LengthError;
    typedef std::out_of_range     OutOfRangeError;
    typedef std::runtime_error    RuntimeError;
    typedef std::overflow_error   OverflowError;
    typedef std::underflow_error  UnderflowError;

    void
    throwException(bool condition, std::string msg = "");

    template <class T>
    void
    throwException(bool condition, std::string msg = "");

  } // namespace utils
} // namespace dftefe
#include "Exceptions.t.cpp"
#endif // dftefeExceptions_h
