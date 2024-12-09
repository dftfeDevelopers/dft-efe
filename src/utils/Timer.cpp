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
#include <utils/Timer.h>
#include <utils/Exceptions.h>
namespace dftefe
{
  namespace utils
  {
    namespace TimerInternal
    {
      template <typename T>
      struct isDuration : std::false_type
      {};

      template <typename Rep, typename Period>
      struct isDuration<std::chrono::duration<Rep, Period>> : std::true_type
      {};

      template <typename T>
      T
      fromSeconds(const double time)
      {
        DFTEFE_AssertWithMsg(isDuration<T>::value,
                             "The template type should be a duration type.");
        return T(std::lround(T::period::den * (time / T::period::num)));
      }

      template <typename Rep, typename Period>
      double
      toSeconds(const std::chrono::duration<Rep, Period> duration)
      {
        return Period::num * double(duration.count()) / Period::den;
      }

      void
      clearTimingData(utils::mpi::MinMaxAvg<double> &data)
      {
        double snan = std::numeric_limits<double>::signaling_NaN();
        data.min    = snan;
        data.max    = snan;
        data.avg    = snan;
      }

    } // namespace TimerInternal

    template <class clockType>
    Timer::ClockMeasurements<clockType>::ClockMeasurements()
      : currentLapStartTime(clockType::now())
      , accumulatedTime(durationType::zero())
      , lastLapTime(durationType::zero())
    {}

    template <class clockType>
    void
    Timer::ClockMeasurements<clockType>::reset()
    {
      currentLapStartTime = clockType::now();
      accumulatedTime     = durationType::zero();
      lastLapTime         = durationType::zero();
    }

    Timer::Timer(const utils::mpi::MPIComm mpiComm, const bool syncLapTimes)
      : d_running(false)
      , d_mpiComm(mpiComm)
      , d_syncLapTimes(syncLapTimes)
    {
      reset();
      start();
    }

    const utils::mpi::MinMaxAvg<double> &
    Timer::getLastLapWallTimeData() const
    {
      return d_lastLapWallTimeData;
    }

    const utils::mpi::MinMaxAvg<double> &
    Timer::getAccumulatedWallTimeData() const
    {
      return d_accumulatedWallTimeData;
    }

    void
    Timer::printLastLapWallTimeData(const ConditionalOStream &stream) const
    {
      stream << " Wall time:"
             << ", min=" << d_lastLapWallTimeData.min
             << " max=" << d_lastLapWallTimeData.max
             << ", avg=" << d_lastLapWallTimeData.avg << std::endl;
    }

    void
    Timer::printAccumulatedWallTimeData(const ConditionalOStream &stream) const
    {
      stream << " Wall time:"
             << ", min=" << d_accumulatedWallTimeData.min
             << " max=" << d_accumulatedWallTimeData.max
             << ", avg=" << d_accumulatedWallTimeData.avg << std::endl;
    }

    void
    Timer::start()
    {
      d_running = true;
      if (d_syncLapTimes)
        {
          utils::mpi::MPIBarrier(d_mpiComm);
        }
      d_wallTimes.currentLapStartTime = wallClockType::now();
    }

    void
    Timer::stop()
    {
      if (d_running)
        {
          d_running = false;

          d_wallTimes.lastLapTime =
            wallClockType::now() - d_wallTimes.currentLapStartTime;

          d_lastLapWallTimeData =
            utils::mpi::MPIAllreduceMinMaxAvg<double, utils::MemorySpace::HOST>(
              TimerInternal::toSeconds(d_wallTimes.lastLapTime), d_mpiComm);

          if (d_syncLapTimes)
            {
              d_wallTimes.lastLapTime =
                TimerInternal::fromSeconds<decltype(d_wallTimes)::durationType>(
                  d_lastLapWallTimeData.max);
            }
          d_wallTimes.accumulatedTime += d_wallTimes.lastLapTime;
          d_accumulatedWallTimeData =
            utils::mpi::MPIAllreduceMinMaxAvg<double, utils::MemorySpace::HOST>(
              TimerInternal::toSeconds(d_wallTimes.accumulatedTime), d_mpiComm);
        }
    }

    double
    Timer::wallTime() const
    {
      wallClockType::duration currentElapsedWallTime;
      if (d_running)
        currentElapsedWallTime = wallClockType::now() -
                                 d_wallTimes.currentLapStartTime +
                                 d_wallTimes.accumulatedTime;
      else
        currentElapsedWallTime = d_wallTimes.accumulatedTime;

      return TimerInternal::toSeconds(currentElapsedWallTime);
    }

    double
    Timer::lastWallTime() const
    {
      return TimerInternal::toSeconds(d_wallTimes.lastLapTime);
    }

    void
    Timer::reset()
    {
      d_wallTimes.reset();
      d_running = false;
      TimerInternal::clearTimingData(d_lastLapWallTimeData);
      TimerInternal::clearTimingData(d_accumulatedWallTimeData);
    }

  } // end of namespace utils
} // end of namespace dftefe
