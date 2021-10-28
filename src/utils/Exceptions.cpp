#include "Exceptions.h"

#include <exception>

namespace dftefe
{
  namespace utils
  {
    namespace
    {
      /**
       * @brief A class that derives from the std::exception to throw a custom message
       */
      class ExceptionWithMsg : public std::exception
      {
      public:
        ExceptionWithMsg(std::string const &msg)
          : d_msg(msg)
        {}
        virtual char const *
        what() const noexcept override
        {
          return d_msg.c_str();
        }

      private:
        std::string d_msg;
      };

    } // end of unnamed namespace


    void
    throwException(bool condition, std::string msg)
    {
      if (!condition)
        {
          throw ExceptionWithMsg(msg);
        }
    }



  } // end of namespace utils

} // end of namespace dftefe
