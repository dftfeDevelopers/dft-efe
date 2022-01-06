#ifndef dftefeFunction_h
#define dftefeFunction_h
#include <vector>
namespace dftefe
{
  namespace utils
  {
    template <typename T, typename Q>
    class Function
    {
    public:
      virtual ~Function() = default;
      virtual Q
      operator()(const T &t) const = 0;
      virtual std::vector<Q>
      operator()(const std::vector<T> &t) const = 0;
    };
  } // end of namespace utils
} // end of namespace dftefe
#endif // dftefeFunction_h
