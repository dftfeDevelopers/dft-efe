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
#include <utils/MPIWrapper.h>
#include <utils/Exceptions.h>
#include <utils/MemoryTransfer.h>
#include <complex>
namespace dftefe
{
  namespace utils
  {
    namespace mpi
    {
      namespace
      {
        template <MemorySpace memorySpace>
        void
        copyBuffer(const void *sendbuf,
                   void *      recvbuf,
                   int         count,
                   MPIDatatype datatype)
        {
          if (datatype == MPIChar)
            {
              char *      r = static_cast<char *>(recvbuf);
              const char *s = static_cast<const char *>(sendbuf);
              MemoryTransfer<memorySpace, memorySpace>::copy(count, r, s);
            }

          else if (datatype == MPISignedChar)
            {
              signed char *      r = static_cast<signed char *>(recvbuf);
              const signed char *s = static_cast<const signed char *>(sendbuf);
              MemoryTransfer<memorySpace, memorySpace>::copy(count, r, s);
            }

          else if (datatype == MPIUnsignedChar)
            {
              unsigned char *      r = static_cast<unsigned char *>(recvbuf);
              const unsigned char *s =
                static_cast<const unsigned char *>(sendbuf);
              MemoryTransfer<memorySpace, memorySpace>::copy(count, r, s);
            }

          else if (datatype == MPIWChar)
            {
              wchar_t *      r = static_cast<wchar_t *>(recvbuf);
              const wchar_t *s = static_cast<const wchar_t *>(sendbuf);
              MemoryTransfer<memorySpace, memorySpace>::copy(count, r, s);
            }

          else if (datatype == MPIShort)
            {
              short int *      r = static_cast<short int *>(recvbuf);
              const short int *s = static_cast<const short int *>(sendbuf);
              MemoryTransfer<memorySpace, memorySpace>::copy(count, r, s);
            }

          else if (datatype == MPIUnsignedShort)
            {
              short unsigned int *r =
                static_cast<unsigned short int *>(recvbuf);
              const unsigned short int *s =
                static_cast<const unsigned short int *>(sendbuf);
              MemoryTransfer<memorySpace, memorySpace>::copy(count, r, s);
            }

          else if (datatype == MPIInt)
            {
              int *      r = static_cast<int *>(recvbuf);
              const int *s = static_cast<const int *>(sendbuf);
              MemoryTransfer<memorySpace, memorySpace>::copy(count, r, s);
            }

          else if (datatype == MPIUnsigned)
            {
              unsigned int *      r = static_cast<unsigned int *>(recvbuf);
              const unsigned int *s =
                static_cast<const unsigned int *>(sendbuf);
              MemoryTransfer<memorySpace, memorySpace>::copy(count, r, s);
            }

          else if (datatype == MPILong)
            {
              long int *      r = static_cast<long int *>(recvbuf);
              const long int *s = static_cast<const long int *>(sendbuf);
              MemoryTransfer<memorySpace, memorySpace>::copy(count, r, s);
            }

          else if (datatype == MPIUnsignedLong)
            {
              unsigned long int *r = static_cast<unsigned long int *>(recvbuf);
              const unsigned long int *s =
                static_cast<const unsigned long int *>(sendbuf);
              MemoryTransfer<memorySpace, memorySpace>::copy(count, r, s);
            }

          else if (datatype == MPIFloat)
            {
              float *      r = static_cast<float *>(recvbuf);
              const float *s = static_cast<const float *>(sendbuf);
              MemoryTransfer<memorySpace, memorySpace>::copy(count, r, s);
            }

          else if (datatype == MPIDouble)
            {
              double *      r = static_cast<double *>(recvbuf);
              const double *s = static_cast<const double *>(sendbuf);
              MemoryTransfer<memorySpace, memorySpace>::copy(count, r, s);
            }

          else if (datatype == MPILongDouble)
            {
              long double *      r = static_cast<long double *>(recvbuf);
              const long double *s = static_cast<const long double *>(sendbuf);
              MemoryTransfer<memorySpace, memorySpace>::copy(count, r, s);
            }

          else if (datatype == MPILongLongInt || datatype == MPILongLong)
            {
              long long int *      r = static_cast<long long int *>(recvbuf);
              const long long int *s =
                static_cast<const long long int *>(sendbuf);
              MemoryTransfer<memorySpace, memorySpace>::copy(count, r, s);
            }

          else if (datatype == MPIUnsignedLongLong)
            {
              unsigned long long int *r =
                static_cast<unsigned long long int *>(recvbuf);
              const unsigned long long int *s =
                static_cast<const unsigned long long int *>(sendbuf);
              MemoryTransfer<memorySpace, memorySpace>::copy(count, r, s);
            }

          else if (datatype == MPIComplex)
            {
              std::complex<float> *r =
                static_cast<std::complex<float> *>(recvbuf);
              const std::complex<float> *s =
                static_cast<const std::complex<float> *>(sendbuf);
              MemoryTransfer<memorySpace, memorySpace>::copy(count, r, s);
            }
          else if (datatype == MPIDoubleComplex)
            {
              std::complex<double> *r =
                static_cast<std::complex<double> *>(recvbuf);
              const std::complex<double> *s =
                static_cast<const std::complex<double> *>(sendbuf);
              MemoryTransfer<memorySpace, memorySpace>::copy(count, r, s);
            }

          else
            {
              DFTEFE_AssertWithMsg(
                false,
                "Copying from send to receive buffer for the given dftefe::utils::mpi::MPIDatatype is not"
                " implemented when not linking with an MPI library");
            }
        }

      } // namespace

#ifdef DFTEFE_WITH_MPI
      template <MemorySpace memorySpace>
      int
      MPIAllreduce(const void *sendbuf,
                   void *      recvbuf,
                   int         count,
                   MPIDatatype datatype,
                   MPIOp       op,
                   MPIComm     comm)
      {
        return ::MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
      }

      template <MemorySpace memorySpace>
      int
      MPIAllgather(const void *sendbuf,
                   int         sendcount,
                   MPIDatatype sendtype,
                   void *      recvbuf,
                   int         recvcount,
                   MPIDatatype recvtype,
                   MPIComm     comm)
      {
        return ::MPI_Allgather(
          sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
      }

      template <MemorySpace memorySpace>
      int
      MPIAllgatherv(const void *sendbuf,
                    int         sendcount,
                    MPIDatatype sendtype,
                    void *      recvbuf,
                    const int * recvcounts,
                    const int * displs,
                    MPIDatatype recvtype,
                    MPIComm     comm)
      {
        return ::MPI_Allgatherv(sendbuf,
                                sendcount,
                                sendtype,
                                recvbuf,
                                recvcounts,
                                displs,
                                recvtype,
                                comm);
      }

      template <MemorySpace memorySpace>
      int
      MPIBcast(void *      buffer,
               int         count,
               MPIDatatype datatype,
               int         root,
               MPIComm     comm)
      {
        return ::MPI_Bcast(buffer, count, datatype, root, comm);
      }

      template <MemorySpace memorySpace>
      int
      MPIIrecv(void *      buf,
               int         count,
               MPIDatatype datatype,
               int         source,
               int         tag,
               MPIComm     comm,
               MPIRequest *request)
      {
        return ::MPI_Irecv(buf, count, datatype, source, tag, comm, request);
      }

      template <MemorySpace memorySpace>
      int
      MPIRecv(void *      buf,
              int         count,
              MPIDatatype datatype,
              int         source,
              int         tag,
              MPIComm     comm,
              MPIStatus * status)
      {
        return ::MPI_Recv(buf, count, datatype, source, tag, comm, status);
      }


      template <MemorySpace memorySpace>
      int
      MPIReduce(void *      sendbuf,
                void *      recvbuf,
                int         count,
                MPIDatatype datatype,
                MPIOp       op,
                int         root,
                MPIComm     comm)
      {
        return ::MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
      }


      template <MemorySpace memorySpace>
      int
      MPISend(const void *buf,
              int         count,
              MPIDatatype datatype,
              int         dest,
              int         tag,
              MPIComm     comm)
      {
        return ::MPI_Send(buf, count, datatype, dest, tag, comm);
      }

      template <MemorySpace memorySpace>
      int
      MPISendrecv(const void *sendbuf,
                  int         sendcount,
                  MPIDatatype sendtype,
                  int         dest,
                  int         sendtag,
                  void *      recvbuf,
                  int         recvcount,
                  MPIDatatype recvtype,
                  int         source,
                  int         recvtag,
                  MPIComm     comm,
                  MPIStatus * status)
      {
        return ::MPI_Sendrecv(sendbuf,
                              sendcount,
                              sendtype,
                              dest,
                              sendtag,
                              recvbuf,
                              recvcount,
                              recvtype,
                              source,
                              recvtag,
                              comm,
                              status);
      }

      template <MemorySpace memorySpace>
      int
      MPIIsend(const void *buf,
               int         count,
               MPIDatatype datatype,
               int         dest,
               int         tag,
               MPIComm     comm,
               MPIRequest *request)
      {
        return ::MPI_Isend(buf, count, datatype, dest, tag, comm, request);
      }

      template <MemorySpace memorySpace>
      int
      MPISsend(const void *buf,
               int         count,
               MPIDatatype datatype,
               int         dest,
               int         tag,
               MPIComm     comm)
      {
        return ::MPI_Ssend(buf, count, datatype, dest, tag, comm);
      }

      template <MemorySpace memorySpace>
      int
      MPIIssend(const void *buf,
                int         count,
                MPIDatatype datatype,
                int         dest,
                int         tag,
                MPIComm     comm,
                MPIRequest *request)
      {
        return ::MPI_Issend(buf, count, datatype, dest, tag, comm, request);
      }


      template <typename T, MemorySpace memorySpace>
      MinMaxAvg<T>
      MPIAllreduceMinMaxAvg(const T &data, MPIComm comm)
      {
        MinMaxAvg<T> retVal;
        utils::mpi::MPIAllreduce<memorySpace>(
          &data,
          &retVal.min,
          1,
          utils::mpi::Types<T>::getMPIDatatype(),
          utils::mpi::MPIMin,
          comm);
        utils::mpi::MPIAllreduce<memorySpace>(
          &data,
          &retVal.max,
          1,
          utils::mpi::Types<T>::getMPIDatatype(),
          utils::mpi::MPIMax,
          comm);
        utils::mpi::MPIAllreduce<memorySpace>(
          &data,
          &retVal.avg,
          1,
          utils::mpi::Types<T>::getMPIDatatype(),
          utils::mpi::MPISum,
          comm);
        int numProcs;
        utils::mpi::MPICommSize(comm, &numProcs);
        retVal.avg /= numProcs;
        return retVal;
      }

#else  // DFTEFE_WITH_MPI

      template <MemorySpace memorySpace>
      int
      MPIAllreduce(const void *sendbuf,
                   void *      recvbuf,
                   int         count,
                   MPIDatatype datatype,
                   MPIOp       op,
                   MPIComm     comm)
      {
        DFTEFE_AssertWithMsg(
          op == MPISum || op == MPIMax || op == MPIMin,
          "Use of MPIOp other than MPISum, MPIMax, and MPIMin in MPIAllreduce is "
          "not implemented when not using an MPI library");
        copyBuffer<memorySpace>(sendbuf, recvbuf, count, datatype);
        return MPISuccess;
      }

      template <MemorySpace memorySpace>
      int
      MPIAllgather(const void *sendbuf,
                   int         sendcount,
                   MPIDatatype sendtype,
                   void *      recvbuf,
                   int         recvcount,
                   MPIDatatype recvtype,
                   MPIComm     comm)
      {
        DFTEFE_AssertWithMsg(
          sendcount == recvcount,
          "Use MPIAllgather with different send and receive count is not allowed when not linking to an MPI library");
        DFTEFE_AssertWithMsg(
          sendtype == recvtype,
          "Use MPIAllgather with different send and receive datatypes is not allowed when not linking to an MPI library");
        copyBuffer<memorySpace>(sendbuf, recvbuf, sendcount, sendtype);
        return MPISuccess;
      }

      template <MemorySpace memorySpace>
      int
      MPIAllgatherv(const void *sendbuf,
                    int         sendcount,
                    MPIDatatype sendtype,
                    void *      recvbuf,
                    const int * recvcounts,
                    const int * displs,
                    MPIDatatype recvtype,
                    MPIComm     comm)
      {
        DFTEFE_AssertWithMsg(
          sendcount == recvcounts[0],
          "Use MPIAllgatherv with different send and receive count is not allowed when not linking to an MPI library");
        DFTEFE_AssertWithMsg(
          sendtype == recvtype,
          "Use MPIAllgatherv with different send and receive datatypes is not allowed when not linking to an MPI library");
        copyBuffer<memorySpace>(sendbuf, recvbuf, sendcount, sendtype);
        return MPISuccess;
      }

      template <MemorySpace memorySpace>
      int
      MPIBcast(void *      buffer,
               int         count,
               MPIDatatype datatype,
               int         root,
               MPIComm     comm)
      {
        return MPISuccess;
      }

      template <MemorySpace memorySpace>
      int
      MPIIrecv(void *      buf,
               int         count,
               MPIDatatype datatype,
               int         source,
               int         tag,
               MPIComm     comm,
               MPIRequest *request)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIIrecv is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      template <MemorySpace memorySpace>
      int
      MPIRecv(void *      buf,
              int         count,
              MPIDatatype datatype,
              int         source,
              int         tag,
              MPIComm     comm,
              MPIStatus * status)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIRecv is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }


      template <MemorySpace memorySpace>
      int
      MPIReduce(void *      sendbuf,
                void *      recvbuf,
                int         count,
                MPIDatatype datatype,
                MPIOp       op,
                int         root,
                MPIComm     comm)
      {
        DFTEFE_AssertWithMsg(
          op == MPISum || op == MPIMax || op == MPIMin,
          "Use of MPIOp other than MPISum, MPIMax, and MPIMin is MPIReduce is "
          "not implemented when not using an MPI library");
        copyBuffer<memorySpace>(sendbuf, recvbuf, count, datatype);
        return MPISuccess;
      }

      template <MemorySpace memorySpace>
      int
      MPISend(const void *buf,
              int         count,
              MPIDatatype datatype,
              int         dest,
              int         tag,
              MPIComm     comm)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPISend is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      template <MemorySpace memorySpace>
      int
      MPISendrecv(const void *sendbuf,
                  int         sendcount,
                  MPIDatatype sendtype,
                  int         dest,
                  int         sendtag,
                  void *      recvbuf,
                  int         recvcount,
                  MPIDatatype recvtype,
                  int         source,
                  int         recvtag,
                  MPIComm     comm,
                  MPIStatus * status)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPISendrecv is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      template <MemorySpace memorySpace>
      int
      MPIIsend(const void *buf,
               int         count,
               MPIDatatype datatype,
               int         dest,
               int         tag,
               MPIComm     comm,
               MPIRequest *request)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIIsend is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      template <MemorySpace memorySpace>
      int
      MPISsend(const void *buf,
               int         count,
               MPIDatatype datatype,
               int         dest,
               int         tag,
               MPIComm     comm)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPISsend is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      template <MemorySpace memorySpace>
      int
      MPIIssend(const void *buf,
                int         count,
                MPIDatatype datatype,
                int         dest,
                int         tag,
                MPIComm     comm,
                MPIRequest *request)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIIssend is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      template <typename T, MemorySpace memorySpace>
      MinMaxAvg<T>
      MPIAllreduceMinMaxAvg(const T &data, MPIComm comm)
      {
        MinMaxAvg<T> retVal;
        retVal.min = data;
        retVal.max = data;
        retVal.avg = data;
        return retVal;
      }
#endif // DFTEFE_WITH_MPI

      // template <typename T>
      // MPIDatatype
      // MPIGetDatatype()
      //{
      //  MPIDatatype returnValue = MPIByte;
      //  if (std::is_same<T, double>::value)
      //    returnValue = MPIDouble;
      //  else if (std::is_same<T, std::complex<double>>::value)
      //    returnValue = MPIDoubleComplex;
      //  else if (std::is_same<T, float>::value)
      //    returnValue = MPIFloat;
      //  else if (std::is_same<T, std::complex<float>>::value)
      //    returnValue = MPIComplex;
      //  else if (std::is_same<T, int>::value)
      //    returnValue = MPIInt;
      //  else if (std::is_same<T, unsigned int>::value)
      //    returnValue = MPIUnsigned;
      //  else if (std::is_same<T, long int>::value)
      //    returnValue = MPILong;
      //  else if (std::is_same<T, unsigned long int>::value)
      //    returnValue = MPIUnsignedLong;
      //  else if (std::is_same<T, char>::value)
      //    returnValue = MPIChar;
      //  else if (std::is_same<T, signed char>::value)
      //    returnValue = MPISignedChar;
      //  else if (std::is_same<T, unsigned char>::value)
      //    returnValue = MPIUnsignedChar;
      //  else if (std::is_same<T, wchar_t>::value)
      //    returnValue = MPIWChar;
      //  else if (std::is_same<T, short int>::value)
      //    returnValue = MPIShort;
      //  else if (std::is_same<T, unsigned short int>::value)
      //    returnValue = MPIUnsignedShort;
      //  else if (std::is_same<T, long long int>::value)
      //    returnValue = MPILongLongInt;
      //  else if (std::is_same<T, long long>::value)
      //    returnValue = MPILongLong;
      //  else if (std::is_same<T, unsigned long long int>::value)
      //    returnValue = MPIUnsignedLongLong;
      //  else if (std::is_same<T, long double>::value)
      //    returnValue = MPILongDouble;
      //  else
      //    throwException<InvalidArgument>(
      //      false, "Invalid typename/datatpe passed to
      //      mpi::MPIGetDatatype()");

      //  return returnValue;
      //}
    } // end of namespace mpi
  }   // end of namespace utils
} // end of namespace dftefe
