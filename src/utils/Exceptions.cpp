/******************************************************************************
 * Copyright (c) 2021.                                                        *
 * The Regents of the University of Michigan and DFT-EFE developers.          *
 *                                                                            *
 * This file is part of the DFT-EFE code.                                     *
 *                                                                            *
 * DFT-EFE is free software: you can redistribute it and/or modify            *
 *   it under the terms of the Lesser GNU General Public License as           *
 *   published by the Free Software Foundation, either version 3 of           *
 *   the License, or (at your option) any later version.                      *
 *                                                                            *
 * DFT-EFE is distributed in the hope that it will be useful, but             *
 *   WITHOUT ANY WARRANTY; without even the implied warranty                  *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                     *
 *   See the Lesser GNU General Public License for more details.              *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public           *
 *   License at the top level of DFT-EFE distribution.  If not, see           *
 *   <https://www.gnu.org/licenses/>.                                         *
 ******************************************************************************/

/*
 * @author Bikash Kanungo
 */
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
