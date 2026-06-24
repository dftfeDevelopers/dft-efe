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

#ifndef dftefe_RDM1_h
#define dftefe_RDM1_h

// C++ standard libraries
#include <vector>
#include <set>
#include <utility>
#include <unordered_map>
#include <functional>
#include <memory>

#include <utils/MemoryStorage.h>
#include <utils/TypeConfig.h>
#include <ksdft/KSAttributes.h>
#include <linearAlgebra/MultiVector.h>
#include <quadrature/QuadratureValuesContainer.h>

namespace dftefe
{
  namespace ksdft
  {
    /**
     * @brief Abstract class for the one-particle reduced density matrix
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class RDM1
    {
      /*
       * @brief typdefs
       */
    public:
      using AttrStorage = std::vector<
        dftefe::quadrature::
          QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST>>;
      using AttrStorageRef      = std::reference_wrapper<AttrStorage>;
      using AttrStorageConstRef = std::reference_wrapper<const AttrStorage>;

    public:
      /**
        @brief Default destructor
        */
      virtual ~RDM1() = default;

      virtual void
      setEvalDescrFlag(const bool evalFlag) = 0;

      virtual void
      getDescriptors(
        const std::set<DensityDescrAttr> &                 densityAttrs,
        const std::set<WfcDescrAttr> &                     wfcAttrs,
        std::unordered_map<DensityDescrAttr, AttrStorage> &densityAttrVals,
        std::unordered_map<WfcDescrAttr, AttrStorage> &    wfcAttrVals) = 0;

      virtual void
      setDescriptors(
        const std::unordered_map<DensityDescrAttr, AttrStorage>
          &                                                  densityAttrVals,
        const std::unordered_map<WfcDescrAttr, AttrStorage> &wfcAttrVals) = 0;

      virtual void
      getDensityObs(
        const std::set<DensityObsAttr> &densityObsAttrs,
        std::unordered_map<DensityObsAttr, std::vector<std::vector<double>>>
          &densityObsAttrVals) = 0;

      virtual bool
      isSpinPolarized() const = 0;

      virtual bool
      isNonCollinear() const = 0;

      virtual bool
      isSOC() const = 0;

      virtual size_type
      getnKSOrbs() const = 0;

      virtual std::vector<double>
      getkPointCoords() const = 0;

      virtual std::vector<double>
      getkPointWeights() const = 0;

      virtual std::unique_ptr<RDM1<ValueType, memorySpace>>
      clone() const = 0;
    };

  } // namespace ksdft
} // namespace dftefe
#endif // dftefe_RDM1_h
