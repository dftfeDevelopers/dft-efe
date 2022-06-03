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

#include <utils/MPITypes.h>
namespace dftefe
{
  namespace utils
  {
#ifdef DFTEFE_WITH_MPI
    //
    // ANY or IGNORE objects
    //
    int              mpi::MPI_SUCCESS         = MPI_SUCCESS;
    int              mpi::MPI_ANY_TAG         = MPI_ANY_TAG;
    int              mpi::MPI_ANY_SOURCE      = MPI_ANY_SOURCE;
    mpi::MPI_Status *mpi::MPI_STATUS_IGNORE   = MPI_STATUS_IGNORE;
    mpi::MPI_Status *mpi::MPI_STATUSES_IGNORE = MPI_STATUSES_IGNORE;
    int *            mpi::MPI_ERRCODES_IGNORE = MPI_ERRCODES_IGNORE;

    //
    // Datatype objects
    //
    mpi::MPI_Datatype mpi::MPI_CHAR               = MPI_CHAR;
    mpi::MPI_Datatype mpi::MPI_SIGNED_CHAR        = MPI_SIGNED_CHAR;
    mpi::MPI_Datatype mpi::MPI_UNSIGNED_CHAR      = MPI_UNSIGNED_CHAR;
    mpi::MPI_Datatype mpi::MPI_BYTE               = MPI_BYTE;
    mpi::MPI_Datatype mpi::MPI_WCHAR              = MPI_WCHAR;
    mpi::MPI_Datatype mpi::MPI_SHORT              = MPI_SHORT;
    mpi::MPI_Datatype mpi::MPI_UNSIGNED_SHORT     = MPI_UNSIGNED_SHORT;
    mpi::MPI_Datatype mpi::MPI_INT                = MPI_INT;
    mpi::MPI_Datatype mpi::MPI_UNSIGNED           = MPI_UNSIGNED;
    mpi::MPI_Datatype mpi::MPI_LONG               = MPI_LONG;
    mpi::MPI_Datatype mpi::MPI_UNSIGNED_LONG      = MPI_UNSIGNED_LONG;
    mpi::MPI_Datatype mpi::MPI_FLOAT              = MPI_FLOAT;
    mpi::MPI_Datatype mpi::MPI_DOUBLE             = MPI_DOUBLE;
    mpi::MPI_Datatype mpi::MPI_LONG_DOUBLE        = MPI_LONG_DOUBLE;
    mpi::MPI_Datatype mpi::MPI_LONG_LONG_INT      = MPI_LONG_LONG_INT;
    mpi::MPI_Datatype mpi::MPI_UNSIGNED_LONG_LONG = MPI_UNSIGNED_LONG_LONG;
    mpi::MPI_Datatype mpi::MPI_LONG_LONG          = MPI_LONG_LONG;
    mpi::MPI_Datatype mpi::MPI_COMPLEX            = MPI_COMPLEX;
    mpi::MPI_Datatype mpi::MPI_DOUBLE_COMPLEX     = MPI_DOUBLE_COMPLEX;

    //
    // Null objects
    //
    mpi::MPI_Comm       mpi::MPI_COMM_NULL       = MPI_COMM_NULL;
    mpi::MPI_Op         mpi::MPI_OP_NULL         = MPI_OP_NULL;
    mpi::MPI_Group      mpi::MPI_GROUP_NULL      = MPI_GROUP_NULL;
    mpi::MPI_Datatype   mpi::MPI_DATATYPE_NULL   = MPI_DATATYPE_NULL;
    mpi::MPI_Request    mpi::MPI_REQUEST_NULL    = MPI_REQUEST_NULL;
    mpi::MPI_Errhandler mpi::MPI_ERRHANDLER_NULL = MPI_ERRHANDLER_NULL;

    //
    // World and Self communicator
    //
    mpi::MPI_Comm mpi::MPI_COMM_WORLD = MPI_COMM_WORLD;
    mpi::MPI_Comm mpi::MPI_COMM_SELF  = MPI_COMM_SELF;

    //
    // MPI_Op objects
    //
    mpi::MPI_Op mpi::MPI_MAX     = MPI_MAX;
    mpi::MPI_Op mpi::MPI_MIN     = MPI_MIN;
    mpi::MPI_Op mpi::MPI_SUM     = MPI_SUM;
    mpi::MPI_Op mpi::MPI_PROD    = MPI_PROD;
    mpi::MPI_Op mpi::MPI_LAND    = MPI_LAND;
    mpi::MPI_Op mpi::MPI_BAND    = MPI_BAND;
    mpi::MPI_Op mpi::MPI_LOR     = MPI_LOR;
    mpi::MPI_Op mpi::MPI_BOR     = MPI_BOR;
    mpi::MPI_Op mpi::MPI_LXOR    = MPI_LXOR;
    mpi::MPI_Op mpi::MPI_BXOR    = MPI_BXOR;
    mpi::MPI_Op mpi::MPI_MINLOC  = MPI_MINLOC;
    mpi::MPI_Op mpi::MPI_MAXLOC  = MPI_MAXLOC;
    mpi::MPI_Op mpi::MPI_REPLACE = MPI_REPLACE;

    int mpi::MPI_THREAD_SINGLE     = MPI_THREAD_SINGLE;
    int mpi::MPI_THREAD_FUNNELED   = MPI_THREAD_FUNNELED;
    int mpi::MPI_THREAD_MULTIPLE   = MPI_THREAD_MULTIPLE;
    int mpi::MPI_THREAD_SERIALIZED = MPI_THREAD_SERIALIZED;

#else  // DFTEF_WITH_MPI

    //
    // ANY or IGNORE objects
    //
    int              mpi::MPI_SUCCESS         = 0;
    int              mpi::MPI_ANY_TAG         = -1;
    int              mpi::MPI_ANY_SOURCE      = -1;
    mpi::MPI_Status *mpi::MPI_STATUS_IGNORE   = nullptr;
    mpi::MPI_Status *mpi::MPI_STATUSES_IGNORE = nullptr;
    int *            mpi::MPI_ERRCODES_IGNORE = nullptr;

    //
    // Datatype objects
    //
    mpi::MPI_Datatype mpi::MPI_CHAR           = ((mpi::MPI_Datatype)0x4c000101);
    mpi::MPI_Datatype mpi::MPI_SIGNED_CHAR    = ((mpi::MPI_Datatype)0x4c000118);
    mpi::MPI_Datatype mpi::MPI_UNSIGNED_CHAR  = ((mpi::MPI_Datatype)0x4c000102);
    mpi::MPI_Datatype mpi::MPI_BYTE           = ((mpi::MPI_Datatype)0x4c00010d);
    mpi::MPI_Datatype mpi::MPI_WCHAR          = ((mpi::MPI_Datatype)0x4c00040e);
    mpi::MPI_Datatype mpi::MPI_SHORT          = ((mpi::MPI_Datatype)0x4c000203);
    mpi::MPI_Datatype mpi::MPI_UNSIGNED_SHORT = ((mpi::MPI_Datatype)0x4c000204);
    mpi::MPI_Datatype mpi::MPI_INT            = ((mpi::MPI_Datatype)0x4c000405);
    mpi::MPI_Datatype mpi::MPI_UNSIGNED       = ((mpi::MPI_Datatype)0x4c000406);
    mpi::MPI_Datatype mpi::MPI_LONG           = ((mpi::MPI_Datatype)0x4c000407);
    mpi::MPI_Datatype mpi::MPI_UNSIGNED_LONG  = ((mpi::MPI_Datatype)0x4c000408);
    mpi::MPI_Datatype mpi::MPI_FLOAT          = ((mpi::MPI_Datatype)0x4c00040a);
    mpi::MPI_Datatype mpi::MPI_DOUBLE         = ((mpi::MPI_Datatype)0x4c00080b);
    mpi::MPI_Datatype mpi::MPI_LONG_DOUBLE    = ((mpi::MPI_Datatype)0x4c00080c);
    mpi::MPI_Datatype mpi::MPI_LONG_LONG_INT  = ((mpi::MPI_Datatype)0x4c000809);
    mpi::MPI_Datatype mpi::MPI_UNSIGNED_LONG_LONG =
      ((mpi::MPI_Datatype)0x4c000819);
    mpi::MPI_Datatype mpi::MPI_LONG_LONG      = ((mpi::MPI_Datatype)0x4c000809);
    mpi::MPI_Datatype mpi::MPI_COMPLEX        = ((mpi::MPI_Datatype)1275070494);
    mpi::MPI_Datatype mpi::MPI_DOUBLE_COMPLEX = ((mpi::MPI_Datatype)1275072546);

    //
    // Null objects
    //
    mpi::MPI_Comm     mpi::MPI_COMM_NULL     = ((mpi::MPI_Comm)0x04000000);
    mpi::MPI_Op       mpi::MPI_OP_NULL       = ((mpi::MPI_Op)0x18000000);
    mpi::MPI_Group    mpi::MPI_GROUP_NULL    = ((mpi::MPI_Group)0x08000000);
    mpi::MPI_Datatype mpi::MPI_DATATYPE_NULL = ((mpi::MPI_Datatype)0x0c000000);
    mpi::MPI_Request  mpi::MPI_REQUEST_NULL  = ((mpi::MPI_Request)0x2c000000);
    mpi::MPI_Errhandler mpi::MPI_ERRHANDLER_NULL =
      ((mpi::MPI_Errhandler)0x14000000);

    //
    // World and Self communicator
    //
    mpi::MPI_Comm mpi::MPI_COMM_WORLD = ((mpi::MPI_Comm)0x44000000);
    mpi::MPI_Comm mpi::MPI_COMM_SELF  = ((mpi::MPI_Comm)0x44000001);

    //
    // MPI_Op objects
    //
    mpi::MPI_Op mpi::MPI_MAX     = (mpi::MPI_Op)(0x58000001);
    mpi::MPI_Op mpi::MPI_MIN     = (mpi::MPI_Op)(0x58000002);
    mpi::MPI_Op mpi::MPI_SUM     = (mpi::MPI_Op)(0x58000003);
    mpi::MPI_Op mpi::MPI_PROD    = (mpi::MPI_Op)(0x58000004);
    mpi::MPI_Op mpi::MPI_LAND    = (mpi::MPI_Op)(0x58000005);
    mpi::MPI_Op mpi::MPI_BAND    = (mpi::MPI_Op)(0x58000006);
    mpi::MPI_Op mpi::MPI_LOR     = (mpi::MPI_Op)(0x58000007);
    mpi::MPI_Op mpi::MPI_BOR     = (mpi::MPI_Op)(0x58000008);
    mpi::MPI_Op mpi::MPI_LXOR    = (mpi::MPI_Op)(0x58000009);
    mpi::MPI_Op mpi::MPI_BXOR    = (mpi::MPI_Op)(0x5800000a);
    mpi::MPI_Op mpi::MPI_MINLOC  = (mpi::MPI_Op)(0x5800000b);
    mpi::MPI_Op mpi::MPI_MAXLOC  = (mpi::MPI_Op)(0x5800000c);
    mpi::MPI_Op mpi::MPI_REPLACE = (mpi::MPI_Op)(0x5800000d);

    int mpi::MPI_THREAD_SINGLE     = 0;
    int mpi::MPI_THREAD_FUNNELED   = 1;
    int mpi::MPI_THREAD_MULTIPLE   = 2;
    int mpi::MPI_THREAD_SERIALIZED = 3;
#endif // DFTEFE_WITH_MPI
  }    // end of namespace utils
} // end of namespace dftefe
