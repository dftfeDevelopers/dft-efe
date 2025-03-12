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
#ifndef dftefeConditionalOstream_h
#define dftefeConditionalOstream_h

#include <ostream>
#include <utils/Defaults.h>
#include <iomanip>

//
// ACKNOWLEDGEMENT: The implementation of this class is borrowed from deal.II
//
namespace dftefe
{
  namespace utils
  {
    /**
     * @brief Provides an interface to print based on whether a certain condition is
     * met or not. Typical use cases include:
     * (a) printing based on different verbosity level
     * (b) printing only from a certain processor while running in parallel
     *
     */
    class ConditionalOStream
    {
    public:
      /**
       * @brief Constructor
       *
       * @param[in] stream Reference to the ostream object onto which the output
       * is to be written (e.g., std::cout, a file)
       * @param[in] active Boolean to define whether printing to ostream be
       * allowed or not
       */
      ConditionalOStream(
        std::ostream &  stream,
        const bool      active    = true,
        const size_type precision = ConditionalOStreamDefaults::PRECISION);

      /**
       * @brief Function to set the condition for printing to the output stream
       * associated with this object
       *
       * @param[in] active Booleam to define whether printing to ostream be
       * allowed or not
       */
      void
      setCondition(const bool active);

      /**
       * @brief Function that returns true if printing to the output stream associated
       * with this object is allowed
       *
       * @return True if printing to the output stream associated with this
       * object is allowed, else returns false
       */
      bool
      isActive() const;

      /**
       * @brief Function that returns the underlying ostream object associated with this object
       *
       * @return ostream object associated with this object
       */
      std::ostream &
      getOStream() const;

      /**
       * @brief Overload the insertion or << operator
       *
       * @tparam T Template parameter which can be any of the C++ standard datatypes
       *  (e.g., bool, int, double, char, string, ...)
       * @param[in] t Value to be printed
       */
      template <typename T>
      const ConditionalOStream &
      operator<<(const T &t) const;

      /**
       * @brief Overload the insertion or << operator which takes an input
       * function pointer to a manipulator (e.g., std::endl, std::ends,
       * std::flush, etc). See
       * https://cplusplus.com/reference/ostream/ostream/operator%3C%3C/ for a
       * list of manipulators.
       *
       * @param[in] p Pointer to a function that takes and returns a stream
       * object. It generally is a manipulator function.
       */
      const ConditionalOStream &
      operator<<(std::ostream &(*p)(std::ostream &)) const;

    private:
      std::ostream &d_outputStream;
      bool          d_activeFlag;
    }; // end of class ConditionalOStream

  } // end of namespace utils
} // end of namespace dftefe

#include <utils/ConditionalOStream.t.cpp>
#endif // dftefeConditionalOstream_h
