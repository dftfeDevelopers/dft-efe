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

#ifndef dftefeProfiler_h
#define dftefeProfiler_h

#include <utils/TypeConfig.h>
#include <utils/MPITypes.h>
#include <list>
#include <map>
#include <iomanip>
#include <utils/Timer.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include "sys/types.h"
#include "sys/sysinfo.h"
namespace dftefe
{
  namespace utils
  {
    template <dftefe::utils::MemorySpace memorySpace>
    class Profiler
    {
    public:
      struct Section
      {
        Timer        timer;
        double       totalWallTime;
        size_type nCalls;
      };

      Profiler(const std::string &profileName = "");

      Profiler(const mpi::MPIComm &mpiComm,
               const std::string & profileName = "");

      Profiler(const mpi::MPIComm &      mpiComm,
               const ConditionalOStream &stream,
               const std::string &       profileName = "");

      ~Profiler();

      void
      registerStart(const std::string &sectionName);
      void
      registerEnd(const std::string &sectionName = "");
      void
      print() const;
      Timer
      getSectionTimer(const std::string &sectionName) const;
      double
      getSectionTotalWallTime(const std::string &sectionName) const;
      size_type
      getSectionCalls(const std::string &sectionName) const;
      void
      reset();

    private:
      std::map<std::string, Section> d_SectionsMap;
      std::vector<std::string>       d_insertionOrder;
      std::list<std::string>         d_activeSections;
      ConditionalOStream             d_stream;
      const mpi::MPIComm             d_mpiComm;
      Timer                          d_totalTime;
      std::string                    d_profileName;

    }; // end of class Profiler

    //
    // helper function
    //
    static inline void
    printCurrentMemoryUsage(const utils::mpi::MPIComm &mpiComm,
                            const std::string          message)
    {
      int rank;
      mpi::MPICommRank(mpiComm, &rank);
      ConditionalOStream cout(ConditionalOStream(std::cout));
      cout.setCondition(rank == 0);
      mpi::MPIBarrier(mpiComm);
      struct sysinfo memInfo;
      sysinfo(&memInfo);
      double totalVirtualMem = memInfo.totalram;
      totalVirtualMem += memInfo.totalswap;
      totalVirtualMem *= memInfo.mem_unit;
      double virtualMemUsed = memInfo.totalram - memInfo.freeram;
      virtualMemUsed += memInfo.totalswap - memInfo.freeswap;
      virtualMemUsed *= memInfo.mem_unit;
      auto minMaxAvg =
        mpi::MPIAllreduceMinMaxAvg<double, utils::MemorySpace::HOST>(
          virtualMemUsed, mpiComm);
      const double maxBytes = minMaxAvg.max;
      cout << std::endl
           << message + ", Current maximum memory usage across all processors: "
           << maxBytes / 1024.0 / 1024.0 / 1024.0 << " GB out of "
           << totalVirtualMem / 1024.0 / 1024.0 / 1024.0 << std::endl
           << std::endl;
      mpi::MPIBarrier(mpiComm);
    }

  } // end of namespace utils
} // end of namespace dftefe
#include "Profiler.t.cpp"
#endif // dftefeProfiler_h
