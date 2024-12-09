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
namespace dftefe
{
  namespace utils
  {
    class Profiler
    {
    public:
      struct Section
      {
        Timer        timer;
        double       totalWallTime;
        unsigned int nCalls;
      };

      Profiler(const ConditionalOStream &stream = ConditionalOStream(std::cout),
               const utils::mpi::MPIComm &mpiComm = utils::mpi::MPICommSelf);

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
      unsigned int
      getSectionCalls(const std::string &sectionName) const;

    private:
      std::map<std::string, Section> d_SectionsMap;
      std::list<std::string>         d_activeSections;
      const ConditionalOStream       d_stream;
      const utils::mpi::MPIComm      d_mpiComm;
      Timer                          d_totalTime;

    }; // end of class Profiler
  }    // end of namespace utils
} // end of namespace dftefe
#endif // dftefeProfiler_h
