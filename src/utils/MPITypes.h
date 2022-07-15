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

#ifndef dftefeMPITypes_h
#define dftefeMPITypes_h

#ifdef DFTEFE_WITH_MPI
#  include <mpi.h>
#endif // DFTEFE_WITH_MPI

namespace dftefe
{
  namespace utils
  {
    namespace mpi
    {
      /**
       * @brief The following provides a unified framework to use
       * typedefs that will be valid both with and without the use
       * of an MPI library. This allows the user to write application
       * code in terms of the following typedefs, and as a result,
       * compile the code successfully both with and without MPI.
       * The idea is that while not using an MPI library, we uses
       * aliases (typedefs) for the MPI datatypes (e.g.m MPIFloat, MPIByte,
       * etc.) On the other hand, while not using an MPI library, the same
       * aliases are equated to int (integer). The choice of int is arbitrary,
       * one can use any other low-memory datatype (e.e., char, unsigned cha,
       * etc)
       *
       * There are several useful and widely used macros that are defined in
       * mpi.h (e.g., MPIAnyTag, MPIStatusIgnore, etc.). Thus, for the case
       * where an MPI library is not used, we define those macros with default
       * integral values. As as result, one can use those macros in an
       * application code, irrespective of the use of an MPI library.
       */
      using ErrorCode = int;
#ifdef DFTEFE_WITH_MPI
      using MPIInfo         = MPI_Info;
      using MPIDatatype     = MPI_Datatype;
      using MPIComm         = MPI_Comm;
      using MPIRequest      = MPI_Request;
      using MPIStatus       = MPI_Status;
      using MPIGroup        = MPI_Group;
      using MPIOp           = MPI_Op;
      using MPIErrhandler   = MPI_Errhandler;
      using MPIUserFunction = MPI_User_function;
#else  // DFTEFE_WITH_MPI
      using MPI_Status = struct
      {
        int count      = 0;
        int cancelled  = 0;
        int MPI_SOURCE = 0;
        int MPI_TAG    = 0;
        int MPI_ERROR  = 0;
      };

      using MPIInfo         = int;
      using MPIDatatype     = int;
      using MPIComm         = int;
      using MPIRequest      = int;
      using MPIStatus       = int;
      using MPIGroup        = int;
      using MPIOp           = int;
      using MPIErrhandler   = int;
      using MPIUserFunction = void(void *, void *, int *, MPIDatatype *);
#endif // DFTEFE_WITH_MPI
      /**
       * @note There are special MPI related macros that are usually
       * defined in mpi.h. But since this part pertains to
       * the case where an MPI library is not used, we will have to
       * define those macros with some default values, so
       * that the user side code can seamlessly use these
       * macros both the cases (i.e., with and without MPI)
       */
      extern int        MPISuccess;
      extern int        MPIAnyTag;
      extern int        MPIAnySource;
      extern MPIStatus *MPIStatusIgnore;
      extern MPIStatus *MPIStatusesIgnore;
      extern int *      MPIErrCodesIgnore;

      extern MPIDatatype MPIChar;
      extern MPIDatatype MPISignedChar;
      extern MPIDatatype MPIUnsignedChar;
      extern MPIDatatype MPIByte;
      extern MPIDatatype MPIWChar;
      extern MPIDatatype MPIShort;
      extern MPIDatatype MPIUnsignedShort;
      extern MPIDatatype MPIInt;
      extern MPIDatatype MPIUnsigned;
      extern MPIDatatype MPILong;
      extern MPIDatatype MPIUnsignedLong;
      extern MPIDatatype MPIFloat;
      extern MPIDatatype MPIDouble;
      extern MPIDatatype MPILongDouble;
      extern MPIDatatype MPILongLongInt;
      extern MPIDatatype MPIUnsignedLongLong;
      extern MPIDatatype MPILongLong;
      extern MPIDatatype MPIComplex;
      extern MPIDatatype MPIDoubleComplex;

      extern MPIComm MPICommWorld;
      extern MPIComm MPICommSelf;

      extern MPIComm       MPICommNull;
      extern MPIGroup      MPIGroupNull;
      extern MPIDatatype   MPIDatatypeNull;
      extern MPIRequest    MPIRequestNull;
      extern MPIErrhandler MPIErrHandlerNull;
      extern MPIOp         MPIOpNull;
      extern MPIOp         MPIMax;
      extern MPIOp         MPIMin;
      extern MPIOp         MPISum;
      extern MPIOp         MPIProd;
      extern MPIOp         MPILAnd;
      extern MPIOp         MPIBAnd;
      extern MPIOp         MPILOr;
      extern MPIOp         MPIBOr;
      extern MPIOp         MPILXOr;
      extern MPIOp         MPIBXOr;
      extern MPIOp         MPIMinLoc;
      extern MPIOp         MPIMaxLoc;
      extern MPIOp         MPIReplace;

      extern int MPIThreadSingle;
      extern int MPIThreadFunneled;
      extern int MPIThreadMultiple;
      extern int MPIThreadSerialized;
    } // end of namespace mpi
  }   // end of namespace utils
} // end of namespace dftefe
#endif // dftefeMPITypes_h
