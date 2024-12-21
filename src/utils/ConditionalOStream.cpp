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
#include <utils/ConditionalOStream.h>

//
// ACKNOWLEDGEMENT: The implementation of this class is borrowed from deal.II
//
namespace dftefe
{
  namespace utils
  {
    ConditionalOStream::ConditionalOStream(std::ostream &stream,
                                           const bool    active)
      : d_outputStream(stream)
      , d_activeFlag(active)
    {}

    void
    ConditionalOStream::setCondition(bool active)
    {
      d_activeFlag = active;
    }

    bool
    ConditionalOStream::isActive() const
    {
      return d_activeFlag;
    }

    std::ostream &
    ConditionalOStream::getOStream() const
    {
      return d_outputStream;
    }

  } // end of namespace utils
} // end of namespace dftefe
