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

#include <utils/MPIWrapper.h>
#include <mutex>
#include <utils/Profiler.h>
#include <iomanip>

namespace dftefe
{
  namespace utils
  {
    template <dftefe::utils::MemorySpace memorySpace>
    Profiler<memorySpace>::Profiler(const std::string &profileName)
      : d_stream(ConditionalOStream(std::cout))
      , d_mpiComm(mpi::MPICommSelf)
      , d_profileName(profileName)
    {}

    template <dftefe::utils::MemorySpace memorySpace>
    Profiler<memorySpace>::Profiler(const mpi::MPIComm &mpiComm,
                       const std::string & profileName)
      : d_mpiComm(mpiComm)
      , d_stream(ConditionalOStream(std::cout))
      , d_profileName(profileName)
    {
      // Get the rank of the process
      int rank;
      mpi::MPICommRank(d_mpiComm, &rank);
      d_stream.setCondition(rank == 0);
    }

    template <dftefe::utils::MemorySpace memorySpace>
    Profiler<memorySpace>::Profiler(const mpi::MPIComm &      mpiComm,
                       const ConditionalOStream &stream,
                       const std::string &       profileName)
      : d_stream(stream)
      , d_mpiComm(mpiComm)
      , d_profileName(profileName)
    {}

    template <dftefe::utils::MemorySpace memorySpace>
    Profiler<memorySpace>::~Profiler()
    {
      while (d_activeSections.size() > 0)
        {
          registerEnd();
          DFTEFE_Assert("Profile of some Active Sections are not printed out.");
        }
    }

    template <dftefe::utils::MemorySpace memorySpace>
    void
    Profiler<memorySpace>::registerStart(const std::string &sectionName)
    {
#if defined(DFTEFE_WITH_DEVICE)
      if constexpr (memorySpace == dftefe::utils::MemorySpace::DEVICE)
        {
          deviceError_t err = utils::deviceSynchronize();
          DEVICE_API_CHECK(err);
        }
#endif
      // add device synchronize for gpu
      std::mutex                  mutex;
      std::lock_guard<std::mutex> lock(std::mutex);

      DFTEFE_AssertWithMsg(sectionName.empty() == false,
                           "Section string is empty.");

      DFTEFE_AssertWithMsg(std::find(d_activeSections.begin(),
                                     d_activeSections.end(),
                                     sectionName) == d_activeSections.end(),
                           (std::string(
                              "Cannot enter the already active section <") +
                            sectionName + ">.")
                             .c_str());

      if (d_SectionsMap.find(sectionName) == d_SectionsMap.end())
        {
          if (d_mpiComm != mpi::MPICommSelf)
            {
              // create a new timer for this section. the second argument
              // will ensure that we have an MPI barrier before starting
              // and stopping a timer, and this ensures that we get the
              // maximum run time for this section over all processors.
              // The d_mpiComm from Profiler is passed to the
              // Timer here, so this Timer will collect timing information
              // among all processes inside d_mpiComm.
              d_SectionsMap[sectionName].timer = Timer(d_mpiComm, true);
            }

          d_SectionsMap[sectionName].totalWallTime = 0;
          d_SectionsMap[sectionName].nCalls        = 0;
          d_insertionOrder.push_back(sectionName);
        }

      d_SectionsMap[sectionName].timer.reset();
      d_SectionsMap[sectionName].timer.start();
      ++d_SectionsMap[sectionName].nCalls;

      d_activeSections.push_back(sectionName);
    }

    template <dftefe::utils::MemorySpace memorySpace>
    void
    Profiler<memorySpace>::registerEnd(const std::string &sectionName)
    {
#if defined(DFTEFE_WITH_DEVICE)
      if constexpr (memorySpace == dftefe::utils::MemorySpace::DEVICE)
        {
          deviceError_t err = utils::deviceSynchronize();
          DEVICE_API_CHECK(err);
        }
#endif
      DFTEFE_AssertWithMsg(
        !d_activeSections.empty(),
        "Cannot exit any section because none has been entered!");

      std::mutex                  mutex;
      std::lock_guard<std::mutex> lock(mutex);

      if (!sectionName.empty())
        {
          DFTEFE_AssertWithMsg(
            d_SectionsMap.find(sectionName) != d_SectionsMap.end(),
            "Cannot delete a section that was never created.");
          DFTEFE_AssertWithMsg(
            std::find(d_activeSections.begin(),
                      d_activeSections.end(),
                      sectionName) != d_activeSections.end(),
            "Cannot delete a section that has not been entered.");
        }

      // if no string is given, exit the last
      // active section.
      const std::string actualSectionName =
        (sectionName.empty() ? d_activeSections.back() : sectionName);

      d_SectionsMap[actualSectionName].timer.stop();
      d_SectionsMap[actualSectionName].totalWallTime +=
        d_SectionsMap[actualSectionName].timer.lastWallTime();

      // delete the index from the list of
      // active ones
      d_activeSections.erase(std::find(d_activeSections.begin(),
                                       d_activeSections.end(),
                                       actualSectionName));
    }

    template <dftefe::utils::MemorySpace memorySpace>
    void
    Profiler<memorySpace>::print() const
    {
      // we are going to change the precision and width of output below. store
      // the old values so the get restored when exiting this function
      std::ostream &os = d_stream.getOStream();
      std::ios_base::fmtflags oldFlags = os.flags();
      std::streamsize oldPrecision = os.precision();
      std::streamsize oldWidth = os.width();

      // get the maximum width among all d_SectionsMap
      size_type maxWidth = 0;
      for (const auto &i : d_SectionsMap)
        maxWidth =
          std::max(maxWidth, static_cast<size_type>(i.first.size()));

      // 32 is the default width until | character
      maxWidth = std::max(maxWidth + 1, static_cast<size_type>(32));
      const std::string extraDash  = std::string(maxWidth - 32, '-');
      const std::string extraSpace = std::string(maxWidth - 32, ' ');

      double totalWallTime = d_totalTime.wallTime();

      // now generate a nice table
      d_stream << "\n\n";
      if (d_profileName != "")
        {
          d_stream << "+---------------------------------------------"
                   << extraDash << "+------------"
                   << "+------------+\n";
          d_stream << std::string(maxWidth - 16, ' ') << d_profileName
                   << std::string(maxWidth - 16, ' ') << "\n";
        }
      d_stream << ""
               << "+---------------------------------------------" << extraDash
               << "+------------"
               << "+------------+\n"
               << "| Total wallclock time elapsed since start    " << extraSpace
               << "|";
      d_stream << std::setw(10) << std::setprecision(3) << std::right;
      d_stream << totalWallTime << "s |            |\n";
      d_stream << "|                                             " << extraSpace
               << "|            "
               << "|            |\n";
      d_stream << "| Section                         " << extraSpace
               << "| no. calls |";
      d_stream << std::setw(10);
      d_stream << std::setprecision(3);
      d_stream << "  wall time | % of total |\n";
      d_stream << "+---------------------------------" << extraDash
               << "+-----------+------------"
               << "+------------+";
      for (const auto &name : d_insertionOrder)
        {
          const auto &sec     = d_SectionsMap.at(name);
          std::string nameOut = name;

          // resize the array so that it is always of the same size
          size_type posNonSpace = nameOut.find_first_not_of(' ');
          nameOut.erase(0, posNonSpace);
          nameOut.resize(maxWidth, ' ');
          d_stream << std::endl;
          d_stream << "| " << nameOut;
          d_stream << "| ";
          d_stream << std::setw(9);
          d_stream << sec.nCalls << " |";
          d_stream << std::setw(10);
          d_stream << std::setprecision(3);
          d_stream << sec.totalWallTime << "s |";
          d_stream << std::setw(10);

          if (totalWallTime != 0)
            {
              // if run time was less than 0.1%, just print a zero to avoid
              // printing silly things such as "2.45e-6%". otherwise print
              // the actual percentage
              const double fraction = sec.totalWallTime / totalWallTime;
              if (fraction > 0.001)
                {
                  d_stream << std::setprecision(3);
                  d_stream << fraction * 100;
                }
              else
                d_stream << 0.0;

              d_stream << "% |";
            }
          else
            d_stream << 0.0 << "% |";
        }
      d_stream << std::endl
               << "+---------------------------------" << extraDash
               << "+-----------+"
               << "------------+------------+\n"
               << std::endl;

      os.flags(oldFlags);
      os.precision(oldPrecision);
      os.width(oldWidth);
    }

    template <dftefe::utils::MemorySpace memorySpace>
    Timer
    Profiler<memorySpace>::getSectionTimer(const std::string &sectionName) const
    {
      auto it = d_SectionsMap.find(sectionName);
      if (!sectionName.empty())
        {
          DFTEFE_AssertWithMsg(
            it != d_SectionsMap.end(),
            "Cannot get timer of a section that was never created.");
          DFTEFE_AssertWithMsg(
            std::find(d_activeSections.begin(),
                      d_activeSections.end(),
                      sectionName) != d_activeSections.end(),
            "Cannot get timer of a section that has not been entered.");
        }
      return it->second.timer;
    }

    template <dftefe::utils::MemorySpace memorySpace>
    double
    Profiler<memorySpace>::getSectionTotalWallTime(const std::string &sectionName) const
    {
      auto it = d_SectionsMap.find(sectionName);
      if (!sectionName.empty())
        {
          DFTEFE_AssertWithMsg(
            it != d_SectionsMap.end(),
            "Cannot get TotalWallTime of a section that was never created.");
          DFTEFE_AssertWithMsg(
            std::find(d_activeSections.begin(),
                      d_activeSections.end(),
                      sectionName) != d_activeSections.end(),
            "Cannot get TotalWallTime of a section that has not been entered.");
        }
      return it->second.totalWallTime;
    }

    template <dftefe::utils::MemorySpace memorySpace>
    size_type
    Profiler<memorySpace>::getSectionCalls(const std::string &sectionName) const
    {
      auto it = d_SectionsMap.find(sectionName);
      if (!sectionName.empty())
        {
          DFTEFE_AssertWithMsg(
            it != d_SectionsMap.end(),
            "Cannot get Calls of a section that was never created.");
          DFTEFE_AssertWithMsg(
            std::find(d_activeSections.begin(),
                      d_activeSections.end(),
                      sectionName) != d_activeSections.end(),
            "Cannot get Calls of a section that has not been entered.");
        }
      return it->second.nCalls;
    }

    template <dftefe::utils::MemorySpace memorySpace>
    void
    Profiler<memorySpace>::reset()
    {
      // add device synchronize for gpu
      std::mutex                  mutex;
      std::lock_guard<std::mutex> lock(mutex);
      d_SectionsMap.clear();
      d_insertionOrder.clear();
      d_activeSections.clear();
      d_totalTime.restart();
    }
  } // namespace utils
} // end of namespace dftefe
