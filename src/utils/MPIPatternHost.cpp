#include <utils/MPIPatternHost.h>

namespace dftefe
{
  namespace utils
  {
    MPIPatternHost::MPIPatternHost(
      const std::pair<global_size_type, global_size_type> locallyOwnedRange,
      const std::vector<dftefe::global_size_type> &       ghostIndices,
      MPI_Comm &                                          d_mpiComm)
    {}

  } // end of namespace utils

} // end of namespace dftefe
