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
#include <utils/MPIErrorCodeHandler.h>
#include <utils/Exceptions.h>
#include <map>
namespace dftefe
{
  namespace utils
  {
    namespace
    {
#ifdef DFTEFE_WITH_MPI
      const int Success                 = MPI_SUCCESS;
      const int ErrBuffer               = MPI_ERR_BUFFER;
      const int ErrCount                = MPI_ERR_COUNT;
      const int ErrType                 = MPI_ERR_TYPE;
      const int ErrTag                  = MPI_ERR_TAG;
      const int ErrComm                 = MPI_ERR_COMM;
      const int ErrRank                 = MPI_ERR_RANK;
      const int ErrRequest              = MPI_ERR_REQUEST;
      const int ErrRoot                 = MPI_ERR_ROOT;
      const int ErrGroup                = MPI_ERR_GROUP;
      const int ErrOp                   = MPI_ERR_OP;
      const int ErrTopology             = MPI_ERR_TOPOLOGY;
      const int ErrDims                 = MPI_ERR_DIMS;
      const int ErrArg                  = MPI_ERR_ARG;
      const int ErrUnknown              = MPI_ERR_UNKNOWN;
      const int ErrTruncate             = MPI_ERR_TRUNCATE;
      const int ErrOther                = MPI_ERR_OTHER;
      const int ErrIntern               = MPI_ERR_INTERN;
      const int ErrInStatus             = MPI_ERR_IN_STATUS;
      const int ErrPending              = MPI_ERR_PENDING;
      const int ErrKeyVal               = MPI_ERR_KEYVAL;
      const int ErrNoMem                = MPI_ERR_NO_MEM;
      const int ErrBase                 = MPI_ERR_BASE;
      const int ErrInfoKey              = MPI_ERR_INFO_KEY;
      const int ErrInfoValue            = MPI_ERR_INFO_VALUE;
      const int ErrInfoNoKey            = MPI_ERR_INFO_NOKEY;
      const int ErrSpawn                = MPI_ERR_SPAWN;
      const int ErrPort                 = MPI_ERR_PORT;
      const int ErrService              = MPI_ERR_SERVICE;
      const int ErrName                 = MPI_ERR_NAME;
      const int ErrWin                  = MPI_ERR_WIN;
      const int ErrSize                 = MPI_ERR_SIZE;
      const int ErrDisp                 = MPI_ERR_DISP;
      const int ErrInfo                 = MPI_ERR_INFO;
      const int ErrLockType             = MPI_ERR_LOCKTYPE;
      const int ErrAssert               = MPI_ERR_ASSERT;
      const int ErrRMAConflict          = MPI_ERR_RMA_CONFLICT;
      const int ErrRMASync              = MPI_ERR_RMA_SYNC;
      const int ErrFile                 = MPI_ERR_FILE;
      const int ErrNotSame              = MPI_ERR_NOT_SAME;
      const int ErrAMode                = MPI_ERR_AMODE;
      const int ErrUnsupportedDataRep   = MPI_ERR_UNSUPPORTED_DATAREP;
      const int ErrUnsupportedOperation = MPI_ERR_UNSUPPORTED_OPERATION;
      const int ErrNoSuchFile           = MPI_ERR_NO_SUCH_FILE;
      const int ErrFileExists           = MPI_ERR_FILE_EXISTS;
      const int ErrBadFile              = MPI_ERR_BAD_FILE;
      const int ErrAccess               = MPI_ERR_ACCESS;
      const int ErrNoSpace              = MPI_ERR_NO_SPACE;
      const int ErrQuota                = MPI_ERR_QUOTA;
      const int ErrReadOnly             = MPI_ERR_READ_ONLY;
      const int ErrFileInUse            = MPI_ERR_FILE_IN_USE;
      const int ErrDupDataRep           = MPI_ERR_DUP_DATAREP;
      const int ErrConversion           = MPI_ERR_CONVERSION;
      const int ErrIO                   = MPI_ERR_IO;
      const int ErrLastCode             = MPI_ERR_LASTCODE;

      const std::map<int, std::string> mpiErrorCodeToMsgMap = {
        {Success, "MPI_SUCCESS: No error"},
        {ErrBuffer, "MPI_ERR_BUFFER: Invalid buffer pointer"},
        {ErrCount, "MPI_ERR_COUNT count argument"},
        {ErrType, "MPI_ERR_TYPE: Invalid datatype argument"},
        {ErrTag, "MPI_ERR_TAG: Invalid tag argument"},
        {ErrComm, "MPI_ERR_COMM: Invalid communicator"},
        {ErrRank, "MPI_ERR_RANK: Invalid rank"},
        {ErrRequest, "MPI_ERR_REQUEST: Invalid request (handle)"},
        {ErrRoot, "MPI_ERR_ROOT: Invalid root"},
        {ErrGroup, "MPI_ERR_GROUP: Invalid group"},
        {ErrOp, "MPI_ERR_OP: Invalid operation"},
        {ErrTopology, "MPI_ERR_TOPOLOGY: Invalid topology"},
        {ErrDims, "MPI_ERR_DIMS: Invalid dimension argument"},
        {ErrArg, "MPI_ERR_ARG: Invalid argument of some other kind"},
        {ErrUnknown, "MPI_ERR_UNKNOWN: Unknown error"},
        {ErrTruncate, "MPI_ERR_TRUNCATE: Message truncated on receive"},
        {ErrOther, "MPI_ERR_OTHER: Known error not in this list"},
        {ErrIntern, "MPI_ERR_INTERN: Internal MPI (implementation) error"},
        {ErrInStatus, "MPI_ERR_IN_STATUS: Error code is in status"},
        {ErrPending, "MPI_ERR_PENDING: Pending request"},
        {ErrKeyVal, "MPI_ERR_KEYVAL: Invalid keyval has been passed "},
        {ErrNoMem,
         "MPI_ERR_NO_MEM: MPI_ALLOC_MEM failed because memory "
         "is exhausted"},
        {ErrBase, "MPI_ERR_BASE	Invalid base passed to MPI_FREE_MEM"},
        {ErrInfoKey,
         "MPI_ERR_INFO_KEY:	Key longer than "
         "MPI_MAX_INFO_KEY"},
        {ErrInfoValue,
         "MPI_ERR_INFO_VALUE: Value longer than "
         "MPI_MAX_INFO_VAL"},
        {ErrInfoNoKey,
         "MPI_ERR_INFO_NOKEY: Invalid key passed to "
         "MPI_INFO_DELETE"},
        {ErrSpawn, "MPI_ERR_SPAWN: Error in spawning processes"},
        {ErrPort, "MPI_ERR_PORT: Invalid port name passed to MPI_COMM_CONNECT"},
        {ErrService,
         "MPI_ERR_SERVICE: Invalid service name passed to "
         "MPI_UNPUBLISH_NAME"},
        {ErrName,
         "MPI_ERR_NAME: Invalid service name passed to "
         "MPI_LOOKUP_NAME"},
        {ErrWin, "MPI_ERR_WIN: Invalid win argument"},
        {ErrSize, "MPI_ERR_SIZE: Invalid size argument"},
        {ErrDisp, "MPI_ERR_DISP: Invalid disp argument"},
        {ErrInfo, "MPI_ERR_INFO: Invalid info argument"},
        {ErrLockType, "MPI_ERR_LOCKTYPE: Invalid locktype argument"},
        {ErrAssert, "MPI_ERR_ASSERT: Invalid assert argument"},
        {ErrRMAConflict,
         "MPI_ERR_RMA_CONFLICT: Conflicting accesses to "
         "window"},
        {ErrRMASync, "MPI_ERR_RMA_SYNC: Wrong synchronization of RMA calls"},
        {ErrFile, "MPI_ERR_FILE: Invalid file handle'"},
        {ErrNotSame,
         "MPI_ERR_NOT_SAME: Collective argument not identical on "
         "all processes, or collective routines called in a different order "
         "by different processes"},
        {ErrAMode,
         "MPI_ERR_AMODE: Error related to the amode passed to "
         "MPI_FILE_OPEN"},
        {ErrUnsupportedDataRep,
         "MPI_ERR_UNSUPPORTED_DATAREP: Unsupported "
         "datarep passed to MPI_FILE_SET_VIEW"},
        {ErrUnsupportedOperation,
         "MPI_ERR_UNSUPPORTED_OPERATION: Unsupported "
         " operation, such as seeking on a file which supports sequential "
         "access only"},
        {ErrNoSuchFile, "MPI_ERR_NO_SUCH_FILE: File does not exist"},
        {ErrFileExists, "MPI_ERR_FILE_EXISTS: File exists"},
        {ErrBadFile,
         "MPI_ERR_BAD_FILE: Invalid file name "
         "(e.g., path name too long)"},
        {ErrAccess, "MPI_ERR_ACCESS: Permission denied"},
        {ErrNoSpace, "MPI_ERR_NO_SPACE: Not enough space"},
        {ErrQuota, "MPI_ERR_QUOTA: Quota exceeded"},
        {ErrReadOnly, "MPI_ERR_READ_ONLY: Read-only file or file system"},
        {ErrFileInUse,
         "MPI_ERR_FILE_IN_USE: File operation could not be "
         " completed, as the file is currently open by some process"},
        {ErrDupDataRep,
         "MPI_ERR_DUP_DATAREP: Conversion "
         "functions could not be registered because a data representation "
         "identifier that was already defined was passed to "
         "MPI_REGISTER_DATAREP"},
        {ErrConversion,
         "MPI_ERR_CONVERSION: An error occurred "
         "in a user supplied data conversion function."},
        {ErrIO, "MPI_ERR_IO	Other I/O error"},
        {ErrLastCode, "MPI_ERR_LASTCODE	Last error code"}};
#else
      const int                        Success              = 0;
      const std::map<int, std::string> mpiErrorCodeToMsgMap = {
        {Success, "MPI Success"}};
#endif // DFTEFE_WITH_MPI
    }  // namespace
    bool
    MPIErrorCodeHandler::isSuccess(const int errCode)
    {
      return (errCode == Success);
    }

    std::string
    MPIErrorCodeHandler::getErrMsg(const int errCode)
    {
      std::string returnValue = "";
      auto        it          = mpiErrorCodeToMsgMap.find(errCode);
      if (it != mpiErrorCodeToMsgMap.end())
        returnValue = it->second;
      else
        throwException<InvalidArgument>(false,
                                        "Invalid error code " +
                                          std::to_string(errCode) +
                                          " provided.");
      return returnValue;
    }

    std::pair<bool, std::string>
    MPIErrorCodeHandler::getIsSuccessAndMessage(const int errCode)
    {
      const bool        successFlag = isSuccess(errCode);
      const std::string msg         = getErrMsg(errCode);
      return std::make_pair(successFlag, msg);
    }
  } // end of namespace utils
} // end of namespace dftefe
