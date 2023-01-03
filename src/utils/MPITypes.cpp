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
    int             mpi::MPISuccess        = MPI_SUCCESS;
    int             mpi::MPIAnyTag         = MPI_ANY_TAG;
    int             mpi::MPIAnySource      = MPI_ANY_SOURCE;
    mpi::MPIStatus *mpi::MPIStatusIgnore   = MPI_STATUS_IGNORE;
    mpi::MPIStatus *mpi::MPIStatusesIgnore = MPI_STATUSES_IGNORE;
    int *           mpi::MPIErrCodesIgnore = MPI_ERRCODES_IGNORE;

    //
    // Datatype objects
    //
    mpi::MPIDatatype mpi::MPIChar             = MPI_CHAR;
    mpi::MPIDatatype mpi::MPISignedChar       = MPI_SIGNED_CHAR;
    mpi::MPIDatatype mpi::MPIUnsignedChar     = MPI_UNSIGNED_CHAR;
    mpi::MPIDatatype mpi::MPIByte             = MPI_BYTE;
    mpi::MPIDatatype mpi::MPIWChar            = MPI_WCHAR;
    mpi::MPIDatatype mpi::MPIShort            = MPI_SHORT;
    mpi::MPIDatatype mpi::MPIUnsignedShort    = MPI_UNSIGNED_SHORT;
    mpi::MPIDatatype mpi::MPIInt              = MPI_INT;
    mpi::MPIDatatype mpi::MPIUnsigned         = MPI_UNSIGNED;
    mpi::MPIDatatype mpi::MPILong             = MPI_LONG;
    mpi::MPIDatatype mpi::MPIUnsignedLong     = MPI_UNSIGNED_LONG;
    mpi::MPIDatatype mpi::MPIFloat            = MPI_FLOAT;
    mpi::MPIDatatype mpi::MPIDouble           = MPI_DOUBLE;
    mpi::MPIDatatype mpi::MPILongDouble       = MPI_LONG_DOUBLE;
    mpi::MPIDatatype mpi::MPILongLongInt      = MPI_LONG_LONG_INT;
    mpi::MPIDatatype mpi::MPIUnsignedLongLong = MPI_UNSIGNED_LONG_LONG;
    mpi::MPIDatatype mpi::MPILongLong         = MPI_LONG_LONG;
    mpi::MPIDatatype mpi::MPIComplex          = MPI_COMPLEX;
    mpi::MPIDatatype mpi::MPIDoubleComplex    = MPI_DOUBLE_COMPLEX;

    //
    // Null objects
    //
    mpi::MPIComm       mpi::MPICommNull       = MPI_COMM_NULL;
    mpi::MPIOp         mpi::MPIOpNull         = MPI_OP_NULL;
    mpi::MPIGroup      mpi::MPIGroupNull      = MPI_GROUP_NULL;
    mpi::MPIDatatype   mpi::MPIDatatypeNull   = MPI_DATATYPE_NULL;
    mpi::MPIRequest    mpi::MPIRequestNull    = MPI_REQUEST_NULL;
    mpi::MPIErrhandler mpi::MPIErrHandlerNull = MPI_ERRHANDLER_NULL;

    //
    // World and Self communicator
    //
    mpi::MPIComm mpi::MPICommWorld = MPI_COMM_WORLD;
    mpi::MPIComm mpi::MPICommSelf  = MPI_COMM_SELF;

    //
    // MPI_Op objects
    //
    mpi::MPIOp mpi::MPIMax     = MPI_MAX;
    mpi::MPIOp mpi::MPIMin     = MPI_MIN;
    mpi::MPIOp mpi::MPISum     = MPI_SUM;
    mpi::MPIOp mpi::MPIProd    = MPI_PROD;
    mpi::MPIOp mpi::MPILAnd    = MPI_LAND;
    mpi::MPIOp mpi::MPIBAnd    = MPI_BAND;
    mpi::MPIOp mpi::MPILOr     = MPI_LOR;
    mpi::MPIOp mpi::MPIBOr     = MPI_BOR;
    mpi::MPIOp mpi::MPILXOr    = MPI_LXOR;
    mpi::MPIOp mpi::MPIBXOr    = MPI_BXOR;
    mpi::MPIOp mpi::MPIMinLoc  = MPI_MINLOC;
    mpi::MPIOp mpi::MPIMaxLoc  = MPI_MAXLOC;
    mpi::MPIOp mpi::MPIReplace = MPI_REPLACE;

    int mpi::MPIThreadSingle     = MPI_THREAD_SINGLE;
    int mpi::MPIThreadFunneled   = MPI_THREAD_FUNNELED;
    int mpi::MPIThreadMultiple   = MPI_THREAD_MULTIPLE;
    int mpi::MPIThreadSerialized = MPI_THREAD_SERIALIZED;

#else  // DFTEFE_WITH_MPI

    //
    // ANY or IGNORE objects
    //
    int             mpi::MPISuccess        = 0;
    int             mpi::MPIAnyTag         = -1;
    int             mpi::MPIAnySource      = -1;
    mpi::MPIStatus *mpi::MPIStatusIgnore   = nullptr;
    mpi::MPIStatus *mpi::MPIStatusesIgnore = nullptr;
    int *           mpi::MPIErrCodesIgnore = nullptr;

    //
    // Datatype objects
    //
    mpi::MPIDatatype mpi::MPIChar             = ((mpi::MPIDatatype)0x4c000101);
    mpi::MPIDatatype mpi::MPISignedChar       = ((mpi::MPIDatatype)0x4c000118);
    mpi::MPIDatatype mpi::MPIUnsignedChar     = ((mpi::MPIDatatype)0x4c000102);
    mpi::MPIDatatype mpi::MPIByte             = ((mpi::MPIDatatype)0x4c00010d);
    mpi::MPIDatatype mpi::MPIWChar            = ((mpi::MPIDatatype)0x4c00040e);
    mpi::MPIDatatype mpi::MPIShort            = ((mpi::MPIDatatype)0x4c000203);
    mpi::MPIDatatype mpi::MPIUnsignedShort    = ((mpi::MPIDatatype)0x4c000204);
    mpi::MPIDatatype mpi::MPIInt              = ((mpi::MPIDatatype)0x4c000405);
    mpi::MPIDatatype mpi::MPIUnsigned         = ((mpi::MPIDatatype)0x4c000406);
    mpi::MPIDatatype mpi::MPILong             = ((mpi::MPIDatatype)0x4c000407);
    mpi::MPIDatatype mpi::MPIUnsignedLong     = ((mpi::MPIDatatype)0x4c000408);
    mpi::MPIDatatype mpi::MPIFloat            = ((mpi::MPIDatatype)0x4c00040a);
    mpi::MPIDatatype mpi::MPIDouble           = ((mpi::MPIDatatype)0x4c00080b);
    mpi::MPIDatatype mpi::MPILongDouble       = ((mpi::MPIDatatype)0x4c00080c);
    mpi::MPIDatatype mpi::MPILongLongInt      = ((mpi::MPIDatatype)0x4c000809);
    mpi::MPIDatatype mpi::MPIUnsignedLongLong = ((mpi::MPIDatatype)0x4c000819);
    mpi::MPIDatatype mpi::MPILongLong         = ((mpi::MPIDatatype)0x4c000809);
    mpi::MPIDatatype mpi::MPIComplex          = ((mpi::MPIDatatype)1275070494);
    mpi::MPIDatatype mpi::MPIDoubleComplex    = ((mpi::MPIDatatype)1275072546);

    //
    // Null objects
    //
    mpi::MPIComm       mpi::MPICommNull     = ((mpi::MPIComm)0x04000000);
    mpi::MPIOp         mpi::MPIOpNull       = ((mpi::MPIOp)0x18000000);
    mpi::MPIGroup      mpi::MPIGroupNull    = ((mpi::MPIGroup)0x08000000);
    mpi::MPIDatatype   mpi::MPIDatatypeNull = ((mpi::MPIDatatype)0x0c000000);
    mpi::MPIRequest    mpi::MPIRequestNull  = ((mpi::MPIRequest)0x2c000000);
    mpi::MPIErrhandler mpi::MPIErrHandlerNull =
      ((mpi::MPIErrhandler)0x14000000);

    //
    // World and Self communicator
    //
    mpi::MPIComm mpi::MPICommWorld = ((mpi::MPIComm)0x44000000);
    mpi::MPIComm mpi::MPICommSelf  = ((mpi::MPIComm)0x44000001);

    //
    // MPI_Op objects
    //
    mpi::MPIOp mpi::MPIMax     = (mpi::MPIOp)(0x58000001);
    mpi::MPIOp mpi::MPIMin     = (mpi::MPIOp)(0x58000002);
    mpi::MPIOp mpi::MPISum     = (mpi::MPIOp)(0x58000003);
    mpi::MPIOp mpi::MPIProd    = (mpi::MPIOp)(0x58000004);
    mpi::MPIOp mpi::MPILAnd    = (mpi::MPIOp)(0x58000005);
    mpi::MPIOp mpi::MPIBAnd    = (mpi::MPIOp)(0x58000006);
    mpi::MPIOp mpi::MPILOr     = (mpi::MPIOp)(0x58000007);
    mpi::MPIOp mpi::MPIBOr     = (mpi::MPIOp)(0x58000008);
    mpi::MPIOp mpi::MPILXOr    = (mpi::MPIOp)(0x58000009);
    mpi::MPIOp mpi::MPIBXOr    = (mpi::MPIOp)(0x5800000a);
    mpi::MPIOp mpi::MPIMinLoc  = (mpi::MPIOp)(0x5800000b);
    mpi::MPIOp mpi::MPIMaxLoc  = (mpi::MPIOp)(0x5800000c);
    mpi::MPIOp mpi::MPIReplace = (mpi::MPIOp)(0x5800000d);

    int mpi::MPIThreadSingle     = 0;
    int mpi::MPIThreadFunneled   = 1;
    int mpi::MPIThreadMultiple   = 2;
    int mpi::MPIThreadSerialized = 3;
#endif // DFTEFE_WITH_MPI

    //
    // define the getMPIDatatype() static function for various
    // specializations of Types<T>
    //
    mpi::MPIDatatype
    mpi::Types<char>::getMPIDatatype()
    {
      return mpi::MPIChar;
    }

    mpi::MPIDatatype
    mpi::Types<signed char>::getMPIDatatype()
    {
      return mpi::MPISignedChar;
    }

    mpi::MPIDatatype
    mpi::Types<unsigned char>::getMPIDatatype()
    {
      return mpi::MPIUnsignedChar;
    }

    mpi::MPIDatatype
    mpi::Types<wchar_t>::getMPIDatatype()
    {
      return mpi::MPIWChar;
    }

    mpi::MPIDatatype
    mpi::Types<short>::getMPIDatatype()
    {
      return mpi::MPIShort;
    }

    mpi::MPIDatatype
    mpi::Types<unsigned short>::getMPIDatatype()
    {
      return mpi::MPIUnsignedShort;
    }

    mpi::MPIDatatype
    mpi::Types<int>::getMPIDatatype()
    {
      return mpi::MPIInt;
    }

    mpi::MPIDatatype
    mpi::Types<unsigned int>::getMPIDatatype()
    {
      return mpi::MPIUnsigned;
    }

    mpi::MPIDatatype
    mpi::Types<long>::getMPIDatatype()
    {
      return mpi::MPILong;
    }

    mpi::MPIDatatype
    mpi::Types<unsigned long>::getMPIDatatype()
    {
      return mpi::MPIUnsignedLong;
    }

    mpi::MPIDatatype
    mpi::Types<float>::getMPIDatatype()
    {
      return mpi::MPIFloat;
    }

    mpi::MPIDatatype
    mpi::Types<double>::getMPIDatatype()
    {
      return mpi::MPIDouble;
    }

    mpi::MPIDatatype
    mpi::Types<long double>::getMPIDatatype()
    {
      return mpi::MPILongDouble;
    }

    mpi::MPIDatatype
    mpi::Types<long long int>::getMPIDatatype()
    {
      return mpi::MPILongLongInt;
    }

    mpi::MPIDatatype
    mpi::Types<unsigned long long int>::getMPIDatatype()
    {
      return mpi::MPIUnsignedLongLong;
    }

    mpi::MPIDatatype
    mpi::Types<std::complex<float>>::getMPIDatatype()
    {
      return mpi::MPIComplex;
    }

    mpi::MPIDatatype
    mpi::Types<std::complex<double>>::getMPIDatatype()
    {
      return mpi::MPIDoubleComplex;
    }

  } // end of namespace utils
} // end of namespace dftefe
