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
* @author Vishal Subramanian
*/

#ifndef dftefeFEConstraintsBase_h
#define dftefeFEConstraintsBase_h

#include "FEBasisManager.h"
#include <utils/TypeConfig.h>
namespace dftefe
{
  namespace basis
  {
    /**
     *
     * An abstract class to handle the Constraints related
     * to hanging nodes and boundary consitions
     */
    template <typename ValueType>
    class FEConstraintsBase
    {
    public:
      virtual void clear () = 0 ;

      virtual void makeHangingNodeConstraint(
        std::shared_ptr<FEBasisManager> feBasis) = 0;

      virtual void addLine(size_type lineId) = 0;

      virtual void setInhomogeneity(size_type lineId,
                                     ValueType constraintValue) = 0;
      virtual void close() = 0;

      virtual bool isClosed() =0;

      virtual void setHomogeneousDirichletBC() = 0 ;

      virtual bool isConstrained( size_type nodeId) = 0;



    };

  }
}

#endif // dftefeFEConstraintsBase_h
