// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//

/**
 * @author Gourab Panigrahi
 *
 */

#ifndef matrixFreeDevice_H_
#define matrixFreeDevice_H_
#include <cstdint>
#include <stdexcept>
#include <utils/TypeConfig.h>

namespace dftefe
{
  // List of operators
  enum operatorList
  {
    Laplace   = 0,
    Helmholtz = 1,
    LDA       = 2,
    GGA       = 3,
    Count     = 4
  };

  /**
   * @brief MatrixFreeDevice class template. template parameter nDofsPerDim
   * is the finite element polynomial order. nQuadPointsPerDim is the order of
   * the Gauss quadrature rule. batchSize is the size of batch tuned to hardware
   *
   * @author Gourab Panigrahi
   *
   */
  template <typename T,
            dftefe::operatorList operatorID,
            std::uint32_t        nDofsPerDim,
            std::uint32_t        nQuadPointsPerDim,
            std::uint32_t        batchSize>
  struct MatrixFreeDevice
  {
    static void
    init(T *constMemDataHost, std::size_t constMemDataSize);

    static void
    computeLaplaceX(T *           dst,
                    T *           src,
                    T *           jacobianFactor,
                    dftefe::uInt *map,
                    T *           shapeBuffer,
                    dftefe::uInt  nCells,
                    dftefe::uInt  nBatch);

    static void
    computeHelmholtzX(T *           dst,
                      T *           src,
                      T *           jacobianFactor,
                      dftefe::uInt *map,
                      T *           shapeBuffer,
                      T             coeffHelmholtz,
                      dftefe::uInt  nCells,
                      dftefe::uInt  nBatch);

    static void
    constraintsDistribute(T *                 src,
                          const dftefe::uInt *constrainingNodeBuckets,
                          const dftefe::uInt *constrainingNodeOffset,
                          const dftefe::uInt *constrainedNodeBuckets,
                          const dftefe::uInt *constrainedNodeOffset,
                          const T *           weightMatrixList,
                          const dftefe::uInt *weightMatrixOffset,
                          const T *           inhomogenityList,
                          const dftefe::uInt *ghostMap,
                          const dftefe::uInt  inhomogenityListSize,
                          const dftefe::uInt  nBatch,
                          const dftefe::uInt  nOwnedDofs,
                          const dftefe::uInt  nGhostDofs);

    static void
    constraintsDistributeTranspose(T *                 dst,
                                   T *                 src,
                                   const dftefe::uInt *constrainingNodeBuckets,
                                   const dftefe::uInt *constrainingNodeOffset,
                                   const dftefe::uInt *constrainedNodeBuckets,
                                   const dftefe::uInt *constrainedNodeOffset,
                                   const T *           weightMatrixList,
                                   const dftefe::uInt *weightMatrixOffset,
                                   const dftefe::uInt *ghostMap,
                                   const dftefe::uInt  inhomogenityListSize,
                                   const dftefe::uInt  nBatch,
                                   const dftefe::uInt  nOwnedDofs,
                                   const dftefe::uInt  nGhostDofs);
  };

} // namespace dftefe
#endif // matrixFreeDevice_H_
