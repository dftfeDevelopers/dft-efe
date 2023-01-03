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

#ifndef dftefeLinearAlgebraProfiler_h
#define dftefeLinearAlgebraProfiler_h

#include <utils/TypeConfig.h>
#include <utils/MPITypes.h>
#include <linearAlgebra/LinearAlgebraTypes.h>
#include <linearAlgebra/Defaults.h>
#include <iostream>
#include <string>
#include <vector>
#include <time.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    class LinearAlgebraProfiler
    {
    public:
      using Time = time_t;

    public:
      LinearAlgebraProfiler(
        std::ostream &  stream       = std::cout,
        const size_type printFreq    = PrintControlDefaults::PRINT_FREQ,
        const size_type wallTimeFreq = PrintControlDefaults::WALL_TIME_FREQ,
        const ParallelPrintType printType =
          PrintControlDefaults::PARALLEL_PRINT_TYPE,
        const bool printFinal = PrintControlDefaults::PRINT_FINAL,
        const bool printTotalWallTime =
          PrintControlDefaults::PRINT_TOTAL_WALL_TIME,
        const size_type   precision = PrintControlDefaults::PRECISION,
        const std::string delimiter = PrintControlDefaults::DELIMITER);

      void
      registerStart(const utils::mpi::MPIComm &mpiComm);
      void
      registerEnd(const std::string &s);
      void
      registerIterStart(const size_type iter);
      void
      registerIterEnd(const std::string &s);

      size_type
      getPrintFrequency() const;
      size_type
      getWallTimeFrequency() const;
      ParallelPrintType
      getParallelPrintType() const;
      bool
      getPrintFinal() const;
      bool
      getPrintTotalWallTime() const;
      size_type
      getPrecision() const;
      std::string
      getDelimiter() const;

    private:
      std::ostream &      d_stream;
      size_type           d_printFreq;
      size_type           d_wallTimeFreq;
      ParallelPrintType   d_printType;
      bool                d_printFinal;
      bool                d_printTotalWallTime;
      Time                d_tStart;
      Time                d_tEnd;
      Time                d_tIterStart;
      Time                d_tIterEnd;
      size_type           d_precision;
      std::string         d_delimiter;
      utils::mpi::MPIComm d_mpiComm;
      int                 d_myRank;
      size_type           d_iter;
      bool                d_myRankPrintFlag;
      std::string         d_myRankStringPrefix;
    }; // end of class LinearAlgebraProfiler
  }    // end of namespace linearAlgebra
} // end of namespace dftefe
#endif // dftefeLinearAlgebraProfiler_h
