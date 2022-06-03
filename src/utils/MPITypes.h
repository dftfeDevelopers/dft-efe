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
       * aliases (typedefs) for the MPI datatypes (e.g.m MPI_FLOAT, MPI_BYTE,
       * etc.) On the other hand, while not using an MPI library, the same
       * aliases are equated to int (integer). The choice of int is arbitrary,
       * one can use any other low-memory datatype (e.e., char, unsigned cha,
       * etc)
       *
       * There are several useful and widely used macros that are defined in
       * mpi.h (e.g., MPI_ANY_TAG, MPI_STATUS_IGNORE, etc.). Thus, for the case
       * where an MPI library is not used, we define those macros with default
       * integral values. As as result, one can use those macros in an
       * application code, irrespective of the use of an MPI library.
       */
      using ErrorCode = int;
#ifdef DFTEFE_WITH_MPI
      using MPI_Info       = MPI_Info;
      using MPI_Datatype   = MPI_Datatype;
      using MPI_Comm       = MPI_Comm;
      using MPI_Request    = MPI_Request;
      using MPI_Status     = MPI_Status;
      using MPI_Group      = MPI_Group;
      using MPI_Op         = MPI_Op;
      using MPI_Fint       = MPI_Fint;
      using MPI_Tag        = MPI_Tag;
      using MPI_Errhandler = MPI_Errhandler;
#else  // DFTEFE_WITH_MPI
      using MPI_Status = struct
      {
        int count      = 0;
        int cancelled  = 0;
        int MPI_SOURCE = 0;
        int MPI_TAG    = 0;
        int MPI_ERROR  = 0;
      };

      using MPI_Info       = int;
      using MPI_Datatype   = int;
      using MPI_Comm       = int;
      using MPI_Request    = int;
      using MPI_Status     = int;
      using MPI_Group      = int;
      using MPI_Op         = int;
      using MPI_Fint       = int;
      using MPI_Tag        = int;
      using MPI_Errhandler = int;
#endif // DFTEFE_WITH_MPI
      /**
       * @note There are special MPI related macros that are usually
       * defined in mpi.h. But since this part pertains to
       * the case where an MPI library is not used, we will have to
       * define those macros with some default values, so
       * that the user side code can seamlessly use these
       * macros both the cases (i.e., with and without MPI)
       */
      extern int         MPI_SUCCESS;
      extern int         MPI_ANY_TAG;
      extern int         MPI_ANY_SOURCE;
      extern MPI_Status *MPI_STATUS_IGNORE;
      extern MPI_Status *MPI_STATUSES_IGNORE;
      extern int *       MPI_ERRCODES_IGNORE;

      extern MPI_Datatype MPI_CHAR;
      extern MPI_Datatype MPI_SIGNED_CHAR;
      extern MPI_Datatype MPI_UNSIGNED_CHAR;
      extern MPI_Datatype MPI_BYTE;
      extern MPI_Datatype MPI_WCHAR;
      extern MPI_Datatype MPI_SHORT;
      extern MPI_Datatype MPI_UNSIGNED_SHORT;
      extern MPI_Datatype MPI_INT;
      extern MPI_Datatype MPI_UNSIGNED;
      extern MPI_Datatype MPI_LONG;
      extern MPI_Datatype MPI_UNSIGNED_LONG;
      extern MPI_Datatype MPI_FLOAT;
      extern MPI_Datatype MPI_DOUBLE;
      extern MPI_Datatype MPI_LONG_DOUBLE;
      extern MPI_Datatype MPI_LONG_LONG_INT;
      extern MPI_Datatype MPI_UNSIGNED_LONG_LONG;
      extern MPI_Datatype MPI_LONG_LONG;
      extern MPI_Datatype MPI_COMPLEX;
      extern MPI_Datatype MPI_DOUBLE_COMPLEX;

      extern MPI_Comm MPI_COMM_WORLD;
      extern MPI_Comm MPI_COMM_SELF;

      extern MPI_Comm       MPI_COMM_NULL;
      extern MPI_Group      MPI_GROUP_NULL;
      extern MPI_Datatype   MPI_DATATYPE_NULL;
      extern MPI_Request    MPI_REQUEST_NULL;
      extern MPI_Errhandler MPI_ERRHANDLER_NULL;
      extern MPI_Op         MPI_OP_NULL;
      extern MPI_Op         MPI_MAX;
      extern MPI_Op         MPI_MIN;
      extern MPI_Op         MPI_SUM;
      extern MPI_Op         MPI_PROD;
      extern MPI_Op         MPI_LAND;
      extern MPI_Op         MPI_BAND;
      extern MPI_Op         MPI_LOR;
      extern MPI_Op         MPI_BOR;
      extern MPI_Op         MPI_LXOR;
      extern MPI_Op         MPI_BXOR;
      extern MPI_Op         MPI_MINLOC;
      extern MPI_Op         MPI_MAXLOC;
      extern MPI_Op         MPI_REPLACE;

      extern int MPI_THREAD_SINGLE;
      extern int MPI_THREAD_FUNNELED;
      extern int MPI_THREAD_MULTIPLE;
      extern int MPI_THREAD_SERIALIZED;
    } // end of namespace mpi
  }   // end of namespace utils
} // end of namespace dftefe
#endif // dftefeMPITypes_h
