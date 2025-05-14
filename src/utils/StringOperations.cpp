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
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <utils/StringOperations.h>
#include <utils/Exceptions.h>
namespace dftefe
{
  namespace utils
  {
    namespace stringOps
    {
      bool
      strToInt(const std::string s, int &i)
      {
        try
          {
            i = boost::lexical_cast<int>(s);
          }
        catch (const boost::bad_lexical_cast &e)
          {
            return false;
          }
        return true;
      }

      bool
      strToDouble(const std::string s, double &x)
      {
        try
          {
            x = boost::lexical_cast<double>(s);
          }
        catch (const boost::bad_lexical_cast &e)
          {
            return false;
          }
        return true;
      }

      bool
      strToBool(const std::string s, bool &x)
      {
        if(s == "t" || s == "T")
        {
          x = true;
          return true;
        }
        else if(s == "f" || s == "F")
        {
          x = false;
          return true;
        }
        else
          return false;
      }

      void
      trim(std::string &s)
      {
        boost::algorithm::trim(s);
      }

      std::string
      trimCopy(const std::string &s)
      {
        return boost::algorithm::trim_copy(s);
      }

      bool
      splitStringToInts(const std::string s,
                        std::vector<int> &vals,
                        size_type         reserveSize)
      {
        std::istringstream ss(s);
        std::string        word;
        size_type          wordCount = 0;
        vals.resize(0);
        vals.reserve(reserveSize);
        bool convSuccess = false;
        int  x;
        while (ss >> word)
          {
            convSuccess = utils::stringOps::strToInt(word, x);
            if (!convSuccess)
              break;
            vals.push_back(x);
          }
        return convSuccess;
      }

      bool
      splitStringToDoubles(const std::string    s,
                           std::vector<double> &vals,
                           size_type            reserveSize)
      {
        std::istringstream ss(s);
        std::string        word;
        size_type          wordCount = 0;
        vals.resize(0);
        vals.reserve(reserveSize);
        bool   convSuccess = false;
        double x;
        while (ss >> word)
          {
            convSuccess = utils::stringOps::strToDouble(word, x);
            if (!convSuccess)
              break;
            vals.push_back(x);
          }
        return convSuccess;
      }
    } // end of namespace stringOps
  }   // end of namespace utils
} // end of namespace dftefe
