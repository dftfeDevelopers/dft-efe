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

#ifndef dftefeTimer_h
#define dftefeTimer_h

#include <utils/TypeConfig.h>
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>
#include <utils/ConditionalOStream.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
namespace dftefe
{
  namespace utils
  {
    class Timer
    {
    public:
      using wallClockType = std::chrono::high_resolution_clock;

    public:
      Timer(const utils::mpi::MPIComm mpiComm      = utils::mpi::MPICommSelf,
            const bool                syncLapTimes = false);

      const utils::mpi::MinMaxAvg<double> &
      getLastLapWallTimeData() const;

      const utils::mpi::MinMaxAvg<double> &
      getAccumulatedWallTimeData() const;

      void
      printLastLapWallTimeData(
        const ConditionalOStream &stream = ConditionalOStream(std::cout)) const;

      void
      printAccumulatedWallTimeData(
        const ConditionalOStream &stream = ConditionalOStream(std::cout)) const;

      void
      start();

      void
      stop();

      void
      reset();

      void
      restart();

      double
      wallTime() const;

      double
      lastWallTime() const;

    private:
      template <class clockType>
      struct ClockMeasurements
      {
        using timePointType = typename clockType::time_point;

        using durationType = typename clockType::duration;

        timePointType currentLapStartTime;

        durationType accumulatedTime;

        durationType lastLapTime;

        ClockMeasurements();

        void
        reset();
      };

      ClockMeasurements<wallClockType> d_wallTimes;

      bool d_running;

      utils::mpi::MPIComm d_mpiComm;

      bool d_syncLapTimes;

      utils::mpi::MinMaxAvg<double> d_lastLapWallTimeData;

      utils::mpi::MinMaxAvg<double> d_accumulatedWallTimeData;

    }; // end of class Timer
  }    // end of namespace utils
} // end of namespace dftefe
#endif // dftefeTimer_h
