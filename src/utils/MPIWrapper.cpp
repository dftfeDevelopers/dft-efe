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
#include <complex>
namespace dftefe
{
  namespace utils
  {
    namespace mpi
    {
      namespace
      {
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
              std::copy(s, s + count, r);
            }

          else if (datatype == MPISignedChar)
            {
              signed char *      r = static_cast<signed char *>(recvbuf);
              const signed char *s = static_cast<const signed char *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPIUnsignedChar)
            {
              unsigned char *      r = static_cast<unsigned char *>(recvbuf);
              const unsigned char *s =
                static_cast<const unsigned char *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPIWChar)
            {
              wchar_t *      r = static_cast<wchar_t *>(recvbuf);
              const wchar_t *s = static_cast<const wchar_t *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPIShort)
            {
              short int *      r = static_cast<short int *>(recvbuf);
              const short int *s = static_cast<const short int *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPIUnsignedShort)
            {
              short unsigned int *r =
                static_cast<unsigned short int *>(recvbuf);
              const unsigned short int *s =
                static_cast<const unsigned short int *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPIInt)
            {
              int *      r = static_cast<int *>(recvbuf);
              const int *s = static_cast<const int *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPIUnsigned)
            {
              unsigned int *      r = static_cast<unsigned int *>(recvbuf);
              const unsigned int *s =
                static_cast<const unsigned int *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPILong)
            {
              long int *      r = static_cast<long int *>(recvbuf);
              const long int *s = static_cast<const long int *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPIUnsignedLong)
            {
              unsigned long int *r = static_cast<unsigned long int *>(recvbuf);
              const unsigned long int *s =
                static_cast<const unsigned long int *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPIFloat)
            {
              float *      r = static_cast<float *>(recvbuf);
              const float *s = static_cast<const float *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPIDouble)
            {
              double *      r = static_cast<double *>(recvbuf);
              const double *s = static_cast<const double *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPILongDouble)
            {
              long double *      r = static_cast<long double *>(recvbuf);
              const long double *s = static_cast<const long double *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPILongLongInt || datatype == MPILongLong)
            {
              long long int *      r = static_cast<long long int *>(recvbuf);
              const long long int *s =
                static_cast<const long long int *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPIUnsignedLongLong)
            {
              unsigned long long int *r =
                static_cast<unsigned long long int *>(recvbuf);
              const unsigned long long int *s =
                static_cast<const unsigned long long int *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPIComplex)
            {
              std::complex<float> *r =
                static_cast<std::complex<float> *>(recvbuf);
              const std::complex<float> *s =
                static_cast<const std::complex<float> *>(sendbuf);
              std::copy(s, s + count, r);
            }
          else if (datatype == MPIDoubleComplex)
            {
              std::complex<double> *r =
                static_cast<std::complex<double> *>(recvbuf);
              const std::complex<double> *s =
                static_cast<const std::complex<double> *>(sendbuf);
              std::copy(s, s + count, r);
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
      int
      MPITypeContiguous(int count, MPIDatatype oldtype, MPIDatatype *newtype)
      {
        return ::MPI_Type_contiguous(count, oldtype, newtype);
      }

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

      int
      MPIBarrier(MPIComm comm)
      {
        return ::MPI_Barrier(comm);
      }

      int
      MPIIbarrier(MPIComm comm, MPIRequest *request)
      {
        return ::MPI_Ibarrier(comm, request);
      }

      int
      MPIBcast(void *      buffer,
               int         count,
               MPIDatatype datatype,
               int         root,
               MPIComm     comm)
      {
        return ::MPI_Bcast(buffer, count, datatype, root, comm);
      }

      int
      MPICommCreateGroup(MPIComm  comm,
                         MPIGroup group,
                         int      tag,
                         MPIComm *newcomm)
      {
        return ::MPI_Comm_create_group(comm, group, tag, newcomm);
      }

      int
      MPICommFree(MPIComm *comm)
      {
        return ::MPI_Comm_free(comm);
      }

      int
      MPICommGroup(MPIComm comm, MPIGroup *group)
      {
        return ::MPI_Comm_group(comm, group);
      }

      int
      MPICommRank(MPIComm comm, int *rank)
      {
        return ::MPI_Comm_rank(comm, rank);
      }

      int
      MPICommSize(MPIComm comm, int *size)
      {
        return ::MPI_Comm_size(comm, size);
      }

      int
      MPIGroupFree(MPIGroup *group)
      {
        return ::MPI_Group_free(group);
      }

      int
      MPIGroupIncl(MPIGroup group, int n, const int ranks[], MPIGroup *newgroup)
      {
        return ::MPI_Group_incl(group, n, ranks, newgroup);
      }

      int
      MPIGroupTranslateRanks(MPIGroup  group1,
                             int       n,
                             const int ranks1[],
                             MPIGroup  group2,
                             int       ranks2[])
      {
        return ::MPI_Group_translate_ranks(group1, n, ranks1, group2, ranks2);
      }

      int
      MPIInit(int *argc, char ***argv)
      {
        return ::MPI_Init(argc, argv);
      }

      int
      MPIInitialized(int *flag)
      {
        return ::MPI_Initialized(flag);
      }

      int
      MPIIprobe(int source, int tag, MPIComm comm, int *flag, MPIStatus *status)
      {
        return ::MPI_Iprobe(source, tag, comm, flag, status);
      }

      int
      MPITest(MPIRequest *request, int *flag, MPIStatus *status)
      {
        return ::MPI_Test(request, flag, status);
      }

      int
      MPITestall(int         count,
                 MPIRequest *requests,
                 int *       flag,
                 MPIStatus * statuses)
      {
        return ::MPI_Testall(count, requests, flag, statuses);
      }

      int
      MPIInitThread(int *argc, char ***argv, int required, int *provided)
      {
        return ::MPI_Init_thread(argc, argv, required, provided);
      }

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

      int
      MPIOpCreate(MPIUserFunction *user_fn, int commute, MPI_Op *op)
      {
        return ::MPI_Op_create(user_fn, commute, op);
      }

      int
      MPIOpFree(MPIOp *op)
      {
        return ::MPI_Op_free(op);
      }

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

      int
      MPIRequestFree(MPIRequest *request)
      {
        return ::MPI_Request_free(request);
      }

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

      int
      MPITypecommit(MPIDatatype *datatype)
      {
        return ::MPI_Type_commit(datatype);
      }

      int
      MPITypeFree(MPIDatatype *datatype)
      {
        return ::MPI_Type_free(datatype);
      }

      int
      MPITypeVector(int          count,
                    int          blocklength,
                    int          stride,
                    MPIDatatype  oldtype,
                    MPIDatatype *newtype)
      {
        return ::MPI_Type_vector(count, blocklength, stride, oldtype, newtype);
      }

      int
      MPIWait(MPIRequest *request, MPIStatus *status)
      {
        return ::MPI_Wait(request, status);
      }

      int
      MPIWaitall(int count, MPIRequest requests[], MPIStatus statuses[])
      {
        return ::MPI_Waitall(count, requests, statuses);
      }

      int
      MPIErrorString(int errorcode, char *string, int *resultlen)
      {
        return ::MPI_Error_string(errorcode, string, resultlen);
      }

      int
      MPIFinalize(void)
      {
        return ::MPI_Finalize();
      }

#else  // DFTEFE_WITH_MPI
      int
      MPITypeContiguous(int count, MPIDatatype oldtype, MPIDatatype *newtype)
      {
        DFTEFE_AssertWithMsg(
          false,
          "Use of MPITypeContiguous() is not allowed when not linking to an MPI library");
      }

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
        copyBuffer(sendbuf, recvbuf, count, datatype);
        return MPISuccess;
      }

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
        copyBuffer(sendbuf, recvbuf, sendcount, sendtype);
        return MPISuccess;
      }

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
        copyBuffer(sendbuf, recvbuf, sendcount, sendtype);
        return MPISuccess;
      }

      int
      MPIBarrier(MPIComm comm)
      {
        return MPISuccess;
      }

      int
      MPIIbarrier(MPIComm comm, MPIRequest *request)
      {
        return MPISuccess;
      }

      int
      MPIBcast(void *      buffer,
               int         count,
               MPIDatatype datatype,
               int         root,
               MPIComm     comm)
      {
        return MPISuccess;
      }

      int
      MPICommCreateGroup(MPIComm  comm,
                         MPIGroup group,
                         int      tag,
                         MPIComm *newcomm)
      {
        return MPISuccess;
      }

      int
      MPICommFree(MPIComm *comm)
      {
        return MPISuccess;
      }

      int
      MPICommGroup(MPIComm comm, MPIGroup *group)
      {
        return MPISuccess;
      }

      int
      MPICommRank(MPIComm comm, int *rank)
      {
        *rank = 0;
        return MPISuccess;
      }

      int
      MPICommSize(MPIComm comm, int *size)
      {
        *size = 1;
        return MPISuccess;
      }

      int
      MPIGroupFree(MPIGroup *group)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIGroup_free is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPIGroupIncl(MPIGroup group, int n, const int ranks[], MPIGroup *newgroup)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIGroupIncl is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPIGroupTranslateRanks(MPIGroup  group1,
                             int       n,
                             const int ranks1[],
                             MPIGroup  group2,
                             int       ranks2[])
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIGroup_translate_ranks is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPIInit(int *argc, char ***argv)
      {
        return MPISuccess;
      }

      int
      MPIInitialized(int *flag)
      {
        *flag = 1;
        return MPISuccess;
      }

      int
      MPIIprobe(int source, int tag, MPIComm comm, int *flag, MPIStatus *status)
      {
        *flag = 1;
        return MPISuccess;
      }

      int
      MPITest(MPIRequest *request, int *flag, MPIStatus *status)
      {
        *flag = 1;
        return MPISuccess;
      }

      int
      MPITestall(int         count,
                 MPIRequest *requests,
                 int *       flag,
                 MPIStatus * statuses)
      {
        *flag = 1;
        return MPISuccess;
      }

      int
      MPIInitThread(int *argc, char ***argv, int required, int *provided)
      {
        *provided = MPITHREAD_MULTIPLE;
        return MPISuccess;
      }

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

      int
      MPIOpCreate(MPIUserFunction *user_fn, int commute, MPIOp *op)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIOp_create is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPIOpFree(MPIOp *op)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIOp_free is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

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
        copyBuffer(sendbuf, recvbuf, count, datatype);
        return MPISuccess;
      }

      int
      MPIRequestFree(MPIRequest *request)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIRequest_free is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

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

      int
      MPITypeCommit(MPIDatatype *datatype)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIType_commit is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPITypeFree(MPIDatatype *datatype)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIType_free is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPITypeVector(int          count,
                    int          blocklength,
                    int          stride,
                    MPIDatatype  oldtype,
                    MPIDatatype *newtype)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIType_vector is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPIWait(MPIRequest *request, MPIStatus *status)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIWait is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPIWaitall(int count, MPIRequest requests[], MPIStatus statuses[])
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIWait_All is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPIErrorString(int errorcode, char *string, int *resultlen)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIError_string is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPIFinalize(void)
      {
        return MPISuccess;
      }
#endif // DFTEFE_WITH_MPI
    }  // end of namespace mpi
  }    // end of namespace utils
} // end of namespace dftefe
