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
 * @author Avirup Sircar
 */

#include <ksdft/DensityCalculatorKernels.h>
#include <utils/DataTypeOverloads.h>

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueType,
              typename RealType,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    DensityCalculatorKernels<ValueType, RealType, memorySpace, dim>::
      computeRhoInBatch(
        const size_type                       batchSize,
        const std::pair<size_type, size_type> cellRange,
        const RealType *                      occupationInBatch,
        ValueType *                           psiBatchQuad,
        RealType *                            modPsiSqBatchQuad,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
                                                     quadRuleContainer,
        RealType *                                   rhoBatch,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
        const SpinMode                               spinMode)
    {
      size_type numQuadInBlock = 0;
      for (size_type iCell = cellRange.first; iCell < cellRange.second; iCell++)
        numQuadInBlock += quadRuleContainer->nCellQuadraturePoints(iCell);

      const size_type ncomp = (spinMode == SpinMode::Unpolarized) ? 1 :
                              (spinMode == SpinMode::Collinear)   ? 2 :
                                                                    4;
      const size_type batchN =
        (spinMode == SpinMode::Unpolarized) ? batchSize : batchSize / 2;

      for (size_type ic = 0; ic < ncomp; ++ic)
        for (size_type q = 0; q < numQuadInBlock; ++q)
          rhoBatch[ic * numQuadInBlock + q] = (RealType)0;

      size_type cumulativeQuadInCell = 0, cumulativeQuadPsiInCell = 0;
      for (size_type iCell = cellRange.first; iCell < cellRange.second; iCell++)
        {
          const size_type numQuadInCell =
            quadRuleContainer->nCellQuadraturePoints(iCell);
          for (size_type iQuad = 0; iQuad < numQuadInCell; iQuad++)
            {
              const size_type q = cumulativeQuadInCell + iQuad;

              if (spinMode == SpinMode::Unpolarized)
                {
                  RealType b = 0;
                  for (size_type i = 0; i < batchSize; i++)
                    {
                      const ValueType psi =
                        psiBatchQuad[cumulativeQuadPsiInCell +
                                     batchSize * iQuad + i];
                      const RealType absSqPsi = utils::absSq(psi);
                      modPsiSqBatchQuad[cumulativeQuadPsiInCell +
                                        batchSize * iQuad + i] = absSqPsi;
                      b += 2.0 * absSqPsi * occupationInBatch[i];
                    }
                  rhoBatch[q] = b;
                }
              else if (spinMode == SpinMode::Collinear)
                {
                  RealType b0 = 0, b1 = 0;
                  for (size_type n = 0; n < batchN; n++)
                    {
                      const ValueType psi_up =
                        psiBatchQuad[cumulativeQuadPsiInCell +
                                     batchSize * iQuad + n];
                      const ValueType psi_dn =
                        psiBatchQuad[cumulativeQuadPsiInCell +
                                     batchSize * iQuad + batchN + n];
                      const RealType sq_up = utils::absSq(psi_up);
                      const RealType sq_dn = utils::absSq(psi_dn);
                      modPsiSqBatchQuad[cumulativeQuadPsiInCell +
                                        batchSize * iQuad + n]          = sq_up;
                      modPsiSqBatchQuad[cumulativeQuadPsiInCell +
                                        batchSize * iQuad + batchN + n] = sq_dn;
                      const RealType val_up = occupationInBatch[n] * sq_up;
                      const RealType val_dn =
                        occupationInBatch[batchN + n] * sq_dn;
                      b0 += val_up + val_dn;
                      b1 += val_up - val_dn;
                    }
                  rhoBatch[0 * numQuadInBlock + q] = b0;
                  rhoBatch[1 * numQuadInBlock + q] = b1;
                }
              else // NonCollinear
                {
                  RealType b0 = 0, b1 = 0, b2 = 0, b3 = 0;
                  for (size_type n = 0; n < batchN; n++)
                    {
                      const ValueType psi_up =
                        psiBatchQuad[cumulativeQuadPsiInCell +
                                     batchSize * iQuad + n];
                      const ValueType psi_dn =
                        psiBatchQuad[cumulativeQuadPsiInCell +
                                     batchSize * iQuad + batchN + n];
                      const RealType sq_up = utils::absSq(psi_up);
                      const RealType sq_dn = utils::absSq(psi_dn);
                      const RealType cross_re =
                        utils::realPart(utils::conjugate(psi_up) * psi_dn);
                      const RealType cross_im =
                        utils::imagPart(utils::conjugate(psi_up) * psi_dn);
                      const RealType occ = occupationInBatch[n];
                      b0 += occ * (sq_up + sq_dn);
                      b1 += occ * (sq_up - sq_dn);
                      b2 += occ * 2 * cross_im;
                      b3 += occ * 2 * cross_re;
                    }
                  rhoBatch[0 * numQuadInBlock + q] = b0;
                  rhoBatch[1 * numQuadInBlock + q] = b1;
                  rhoBatch[2 * numQuadInBlock + q] = b2;
                  rhoBatch[3 * numQuadInBlock + q] = b3;
                }
            }
          cumulativeQuadPsiInCell += numQuadInCell * batchSize;
          cumulativeQuadInCell += numQuadInCell;
        }
    }

    template <typename ValueType,
              typename RealType,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    DensityCalculatorKernels<ValueType, RealType, memorySpace, dim>::
      computeGradRhoInBatch(
        const size_type                       batchSize,
        const std::pair<size_type, size_type> cellRange,
        const RealType *                      occupationInBatch,
        const ValueType *                     psiBatchQuad,
        const ValueType *                     gradPsiBatchQuad,
        RealType *                            psiGradPsiBatch,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
                                                     quadRuleContainer,
        RealType *                                   gradRhoBatch,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
        const SpinMode                               spinMode)
    {
      size_type numQuadInBlock = 0;
      for (size_type iCell = cellRange.first; iCell < cellRange.second; iCell++)
        numQuadInBlock += quadRuleContainer->nCellQuadraturePoints(iCell);

      const size_type ncomp = (spinMode == SpinMode::Unpolarized) ? 1 :
                              (spinMode == SpinMode::Collinear)   ? 2 :
                                                                    4;
      const size_type batchN =
        (spinMode == SpinMode::Unpolarized) ? batchSize : batchSize / 2;

      for (size_type ic = 0; ic < ncomp; ++ic)
        for (size_type qd = 0; qd < numQuadInBlock * dim; ++qd)
          gradRhoBatch[ic * numQuadInBlock * dim + qd] = (RealType)0;

      size_type cumulativeQuadInCell    = 0;
      size_type cumulativeQuadPsiInCell = 0;
      size_type cumulativeGradPsiInCell = 0;
      for (size_type iCell = cellRange.first; iCell < cellRange.second; iCell++)
        {
          const size_type numQuadInCell =
            quadRuleContainer->nCellQuadraturePoints(iCell);
          for (size_type iQuad = 0; iQuad < numQuadInCell; iQuad++)
            {
              const size_type q = cumulativeQuadInCell + iQuad;
              for (size_type iDim = 0; iDim < dim; iDim++)
                {
                  if (spinMode == SpinMode::Unpolarized)
                    {
                      RealType b = 0;
                      for (size_type i = 0; i < batchSize; i++)
                        {
                          const ValueType psi =
                            psiBatchQuad[cumulativeQuadPsiInCell +
                                         batchSize * iQuad + i];
                          const ValueType gradPsi =
                            gradPsiBatchQuad[cumulativeGradPsiInCell + i +
                                             batchSize * (iQuad * dim + iDim)];
                          b += 4.0 * occupationInBatch[i] *
                               utils::realPart(utils::conjugate(psi) * gradPsi);
                        }
                      gradRhoBatch[q * dim + iDim] = b;
                    }
                  else if (spinMode == SpinMode::Collinear)
                    {
                      RealType b0 = 0, b1 = 0;
                      for (size_type n = 0; n < batchN; n++)
                        {
                          const ValueType psi_up =
                            psiBatchQuad[cumulativeQuadPsiInCell +
                                         batchSize * iQuad + n];
                          const ValueType gPsi_up =
                            gradPsiBatchQuad[cumulativeGradPsiInCell + n +
                                             batchSize * (iQuad * dim + iDim)];
                          const ValueType psi_dn =
                            psiBatchQuad[cumulativeQuadPsiInCell +
                                         batchSize * iQuad + batchN + n];
                          const ValueType gPsi_dn =
                            gradPsiBatchQuad[cumulativeGradPsiInCell + batchN +
                                             n +
                                             batchSize * (iQuad * dim + iDim)];
                          const RealType contrib_up =
                            2.0 * occupationInBatch[n] *
                            utils::realPart(utils::conjugate(psi_up) * gPsi_up);
                          const RealType contrib_dn =
                            2.0 * occupationInBatch[batchN + n] *
                            utils::realPart(utils::conjugate(psi_dn) * gPsi_dn);
                          b0 += contrib_up + contrib_dn;
                          b1 += contrib_up - contrib_dn;
                        }
                      gradRhoBatch[0 * numQuadInBlock * dim + q * dim + iDim] =
                        b0;
                      gradRhoBatch[1 * numQuadInBlock * dim + q * dim + iDim] =
                        b1;
                    }
                  else // NonCollinear
                    {
                      RealType b0 = 0, b1 = 0, b2 = 0, b3 = 0;
                      for (size_type n = 0; n < batchN; n++)
                        {
                          const ValueType psi_up =
                            psiBatchQuad[cumulativeQuadPsiInCell +
                                         batchSize * iQuad + n];
                          const ValueType gPsi_up =
                            gradPsiBatchQuad[cumulativeGradPsiInCell + n +
                                             batchSize * (iQuad * dim + iDim)];
                          const ValueType psi_dn =
                            psiBatchQuad[cumulativeQuadPsiInCell +
                                         batchSize * iQuad + batchN + n];
                          const ValueType gPsi_dn =
                            gradPsiBatchQuad[cumulativeGradPsiInCell + batchN +
                                             n +
                                             batchSize * (iQuad * dim + iDim)];
                          const RealType occ = occupationInBatch[n];
                          const RealType re_up_gup =
                            utils::realPart(utils::conjugate(psi_up) * gPsi_up);
                          const RealType re_dn_gdn =
                            utils::realPart(utils::conjugate(psi_dn) * gPsi_dn);
                          const RealType re_up_gdn =
                            utils::realPart(utils::conjugate(psi_up) * gPsi_dn);
                          const RealType im_up_gdn =
                            utils::imagPart(utils::conjugate(psi_up) * gPsi_dn);
                          b0 += 2.0 * occ * (re_up_gup + re_dn_gdn);
                          b1 += 2.0 * occ * (re_up_gup - re_dn_gdn);
                          b2 += 2.0 * occ * im_up_gdn;
                          b3 += 2.0 * occ * re_up_gdn;
                        }
                      gradRhoBatch[0 * numQuadInBlock * dim + q * dim + iDim] =
                        b0;
                      gradRhoBatch[1 * numQuadInBlock * dim + q * dim + iDim] =
                        b1;
                      gradRhoBatch[2 * numQuadInBlock * dim + q * dim + iDim] =
                        b2;
                      gradRhoBatch[3 * numQuadInBlock * dim + q * dim + iDim] =
                        b3;
                    }
                }
            }
          cumulativeQuadPsiInCell += numQuadInCell * batchSize;
          cumulativeGradPsiInCell += numQuadInCell * batchSize * dim;
          cumulativeQuadInCell += numQuadInCell;
        }
    }

    template class DensityCalculatorKernels<double,
                                            double,
                                            dftefe::utils::MemorySpace::HOST,
                                            3>;
    template class DensityCalculatorKernels<float,
                                            double,
                                            dftefe::utils::MemorySpace::HOST,
                                            3>;
    template class DensityCalculatorKernels<std::complex<double>,
                                            double,
                                            dftefe::utils::MemorySpace::HOST,
                                            3>;
    template class DensityCalculatorKernels<std::complex<float>,
                                            double,
                                            dftefe::utils::MemorySpace::HOST,
                                            3>;

#ifdef DFTEFE_WITH_DEVICE
    template class DensityCalculatorKernels<
      double,
      double,
      dftefe::utils::MemorySpace::HOST_PINNED,
      3>;
    template class DensityCalculatorKernels<
      float,
      double,
      dftefe::utils::MemorySpace::HOST_PINNED,
      3>;
    template class DensityCalculatorKernels<
      std::complex<double>,
      double,
      dftefe::utils::MemorySpace::HOST_PINNED,
      3>;
    template class DensityCalculatorKernels<
      std::complex<float>,
      double,
      dftefe::utils::MemorySpace::HOST_PINNED,
      3>;
#endif

  } // end of namespace ksdft
} // end of namespace dftefe
