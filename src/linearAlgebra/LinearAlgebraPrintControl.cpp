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

#include <utils/MPIWrapper.h>
#include <linearAlgebra/LinearAlgebraPrintControl.h>
#include <time.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    namespace
    {
      inline void
      getTime(LinearAlgebraPrintControl::Time &t)
      {
        time(&t);
      }

      double
      getTimeDiff(const LinearAlgebraPrintControl::Time &t1,
                  const LinearAlgebraPrintControl::Time &t2)
      {
        return (double)(t2 - t1);
      }

      inline std::string
      getTimeUnit()
      {
        return "secs";
      }

    } // namespace


    //
    // Constructor
    //
    LinearAlgebraPrintControl::LinearAlgebraPrintControl(
      std::ostream &  stream /*= std::cout*/,
      const size_type printFreq /*= PrintControlDefaults::PRINT_FREQ*/,
      const size_type wallTimeFreq /*= PrintControlDefaults::WALL_TIME_FREQ*/,
      const ParallelPrintType
                 printType /*= PrintControlDefaults::PARALLEL_PRINT_TYPE*/,
      const bool printFinal /*= PrintControlDefaults::PRINT_FINAL*/,
      const bool
        printTotalWallTime /*= PrintControlDefaults::PRINT_TOTAL_WALL_TIME*/,
      const size_type   precision /*= PrintControlDefaults::PRECISION*/,
      const std::string delimiter /*= PrintControlDefaults::DELIMITER*/)
      : d_stream(stream)
      , d_printFreq(printFreq)
      , d_wallTimeFreq(wallTimeFreq)
      , d_printType(printType)
      , d_printFinal(printFinal)
      , d_printTotalWallTime(printTotalWallTime)
      , d_precision(precision)
      , d_delimiter(delimiter)
      , d_tStart(0.0)
      , d_tEnd(0.0)
      , d_tIterStart(0.0)
      , d_tIterEnd(0.0)
      , d_mpiComm(utils::mpi::MPICommNull)
      , d_myRank(0)
      , d_iter(0)
      , d_myRankPrintFlag(false)
      , d_myRankStringPrefix("")
    {}

    void
    LinearAlgebraPrintControl::registerStart(const utils::mpi::MPIComm &mpiComm)
    {
      d_mpiComm = mpiComm;
      int err   = utils::mpi::MPICommRank(d_mpiComm, &d_myRank);
      const std::pair<bool, std::string> isSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(isSuccessAndMsg.first, isSuccessAndMsg.second);
      if (d_printType == ParallelPrintType::NONE)
        d_myRankPrintFlag = false;
      else if (d_printType == ParallelPrintType::ALL)
        {
          d_myRankPrintFlag    = true;
          d_myRankStringPrefix = "[" + std::to_string(d_myRank) + "] ";
        }
      else if (d_printType == ParallelPrintType::ROOT_ONLY && d_myRank == 0)
        d_myRankPrintFlag = true;
      else
        utils::throwException<utils::InvalidArgument>(
          false,
          "Invalid argument passed for ParallelPrintType to LinearAlgebraPrintControl");

      if (d_printTotalWallTime)
        {
          utils::mpi::MPIBarrier(d_mpiComm);
          getTime(d_tStart);
        }
    }

    void
    LinearAlgebraPrintControl::registerEnd(const std::string &s)
    {
      std::string msg = "";
      if (d_printFinal)
        {
          msg = s;
        }

      if (d_printTotalWallTime)
        {
          utils::mpi::MPIBarrier(d_mpiComm);
          getTime(d_tEnd);
          msg += " Total wall time: " +
                 std::to_string(getTimeDiff(d_tStart, d_tEnd)) + " " +
                 getTimeUnit();
        }

      if (d_myRankPrintFlag && (d_printFinal || d_printTotalWallTime))
        {
          d_stream << d_myRankStringPrefix << msg << std::endl;
        }
    }

    void
    LinearAlgebraPrintControl::registerIterStart(const size_type iter)
    {
      d_iter = iter;
      if (d_iter % d_wallTimeFreq == 0 && d_wallTimeFreq > 0)
        {
          utils::mpi::MPIBarrier(d_mpiComm);
          getTime(d_tIterStart);
        }
    }

    void
    LinearAlgebraPrintControl::registerIterEnd(const std::string &s)
    {
      std::string msg = "";
      if (d_iter % d_printFreq == 0 && d_printFreq > 0)
        {
          msg = s;
        }

      if (d_iter % d_wallTimeFreq == 0 && d_wallTimeFreq > 0)
        {
          utils::mpi::MPIBarrier(d_mpiComm);
          getTime(d_tIterEnd);
          msg += " Total wall time: " +
                 std::to_string(getTimeDiff(d_tIterStart, d_tIterEnd)) + " " +
                 getTimeUnit();
        }

      if (d_myRankPrintFlag && (d_printFreq > 0 || d_wallTimeFreq > 0))
        d_stream << d_myRankStringPrefix << msg << std::endl;
    }

    size_type
    LinearAlgebraPrintControl::getPrintFrequency() const
    {
      return d_printFreq;
    }

    size_type
    LinearAlgebraPrintControl::getWallTimeFrequency() const
    {
      return d_wallTimeFreq;
    }

    ParallelPrintType
    LinearAlgebraPrintControl::getParallelPrintType() const
    {
      return d_printType;
    }

    bool
    LinearAlgebraPrintControl::getPrintFinal() const
    {
      return d_printFinal;
    }

    bool
    LinearAlgebraPrintControl::getPrintTotalWallTime() const
    {
      return d_printTotalWallTime;
    }

    size_type
    LinearAlgebraPrintControl::getPrecision() const
    {
      return d_precision;
    }

    std::string
    LinearAlgebraPrintControl::getDelimiter() const
    {
      return d_delimiter;
    }


  } // end of namespace linearAlgebra
} // end of namespace dftefe
