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
        copyBuffer(const void * sendbuf,
                   void *       recvbuf,
                   int          count,
                   MPI_Datatype datatype)
        {
          if (datatype == MPI_CHAR)
            {
              char *      r = static_cast<char *>(recvbuf);
              const char *s = static_cast<const char *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPI_SIGNED_CHAR)
            {
              signed char *      r = static_cast<signed char *>(recvbuf);
              const signed char *s = static_cast<const signed char *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPI_UNSIGNED_CHAR)
            {
              unsigned char *      r = static_cast<unsigned char *>(recvbuf);
              const unsigned char *s =
                static_cast<const unsigned char *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPI_WCHAR)
            {
              wchar_t *      r = static_cast<wchar_t *>(recvbuf);
              const wchar_t *s = static_cast<const wchar_t *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPI_SHORT)
            {
              short int *      r = static_cast<short int *>(recvbuf);
              const short int *s = static_cast<const short int *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPI_UNSIGNED_SHORT)
            {
              short unsigned int *r =
                static_cast<unsigned short int *>(recvbuf);
              const unsigned short int *s =
                static_cast<const unsigned short int *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPI_INT)
            {
              int *      r = static_cast<int *>(recvbuf);
              const int *s = static_cast<const int *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPI_UNSIGNED)
            {
              unsigned int *      r = static_cast<unsigned int *>(recvbuf);
              const unsigned int *s =
                static_cast<const unsigned int *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPI_LONG)
            {
              long int *      r = static_cast<long int *>(recvbuf);
              const long int *s = static_cast<const long int *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPI_UNSIGNED_LONG)
            {
              unsigned long int *r = static_cast<unsigned long int *>(recvbuf);
              const unsigned long int *s =
                static_cast<const unsigned long int *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPI_FLOAT)
            {
              float *      r = static_cast<float *>(recvbuf);
              const float *s = static_cast<const float *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPI_DOUBLE)
            {
              double *      r = static_cast<double *>(recvbuf);
              const double *s = static_cast<const double *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPI_LONG_DOUBLE)
            {
              long double *      r = static_cast<long double *>(recvbuf);
              const long double *s = static_cast<const long double *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPI_LONG_DOUBLE)
            {
              long double *      r = static_cast<long double *>(recvbuf);
              const long double *s = static_cast<const long double *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPI_LONG_LONG_INT || datatype == MPI_LONG_LONG)
            {
              long long int *      r = static_cast<long long int *>(recvbuf);
              const long long int *s =
                static_cast<const long long int *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPI_UNSIGNED_LONG_LONG)
            {
              unsigned long long int *r =
                static_cast<unsigned long long int *>(recvbuf);
              const unsigned long long int *s =
                static_cast<const unsigned long long int *>(sendbuf);
              std::copy(s, s + count, r);
            }

          else if (datatype == MPI_COMPLEX)
            {
              std::complex<float> *r =
                static_cast<std::complex<float> *>(recvbuf);
              const std::complex<float> *s =
                static_cast<const std::complex<float> *>(sendbuf);
              std::copy(s, s + count, r);
            }
          else if (datatype == MPI_DOUBLE_COMPLEX)
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
                "Copying from send to receive buffer for the given MPI_Datatype is not"
                " implemented while not linking with an MPI library");
            }
        }

      } // namespace

#ifdef DFTEFE_WITH_MPI
      int
      MPI_Type_contiguous(int           count,
                          MPI_Datatype  oldtype,
                          MPI_Datatype *newtype)
      {
        return MPI_Type_contiguous(count, oldtype, newtype);
      }

      int
      MPI_Allreduce(const void * sendbuf,
                    void *       recvbuf,
                    int          count,
                    MPI_Datatype datatype,
                    MPI_Op       op,
                    MPI_Comm     comm)
      {
        return ::MPI_Allreduce();
      }

      int
      MPI_Allgather(const void * sendbuf,
                    int          sendcount,
                    MPI_Datatype sendtype,
                    void *       recvbuf,
                    int          recvcount,
                    MPI_Datatype recvtype,
                    MPI_Comm     comm)
      {
        return ::MPI_Allgather(
          sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
      }

      int
      MPI_Allgatherv(const void * sendbuf,
                     int          sendcount,
                     MPI_Datatype sendtype,
                     void *       recvbuf,
                     const int *  recvcounts,
                     const int *  displs,
                     MPI_Datatype recvtype,
                     MPI_Comm     comm)
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
      MPI_Barrier(MPI_Comm comm)
      {
        return ::MPI_Barrier(comm);
        ;
      }

      int
      MPI_Ibarrier(MPI_Comm comm, MPI_Request *request)
      {
        return ::MPI_Barrier(comm, request);
      }

      int
      MPI_Bcast(void *       buffer,
                int          count,
                MPI_Datatype datatype,
                int          root,
                MPI_Comm     comm)
      {
        return ::MPI_Bcast(buffer, count, datatype, root, comm);
      }

      int
      MPI_Comm_create_group(MPI_Comm  comm,
                            MPI_Group group,
                            int       tag,
                            MPI_Comm *newcomm)
      {
        return ::MPI_Comm_create_group(comm, group, tag, newcomm);
      }

      int
      MPI_Comm_free(MPI_Comm *comm)
      {
        return ::MPI_Comm_free(comm);
      }

      int
      MPI_Comm_group(MPI_Comm comm, MPI_Group *group)
      {
        return ::MPI_Comm_group(comm, group);
      }

      int
      MPI_Comm_rank(MPI_Comm comm, int *rank)
      {
        return ::MPI_Comm_rank(comm, rank);
      }

      int
      MPI_Comm_size(MPI_Comm comm, int *size)
      {
        return ::MPI_Comm_size(comm, size);
      }

      MPI_Fint
      MPI_Comm_f2c(MPI_Comm comm)
      {
        return ::MPI_Comm_f2c(comm);
      }

      int
      MPI_Group_free(MPI_Group *group)
      {
        return ::MPI_Group_free(group);
      }

      int
      MPI_Group_incl(MPI_Group  group,
                     int        n,
                     const int  ranks[],
                     MPI_Group *newgroup)
      {
        return ::MPI_Group_incl(group, n, ranks, newgroup);
      }

      int
      MPI_Group_translate_ranks(MPI_Group group1,
                                int       n,
                                const int ranks1[],
                                MPI_Group group2,
                                int       ranks2[])
      {
        return ::MPI_Group_translate_ranks(group1, n, ranks1, group2, ranks2);
      }

      int
      MPI_Init(int *argc, char ***argv)
      {
        return ::MPI_int(argc, argv);
      }

      int
      MPI_Initialized(int *flag)
      {
        return ::MPI_Initialized(flag);
      }

      int
      MPI_Iprobe(int         source,
                 int         tag,
                 MPI_Comm    comm,
                 int *       flag,
                 MPI_Status *status)
      {
        return ::MPI_Iprobe(source, tag, comm, flag, status);
      }

      int
      MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)
      {
        return ::MPI_Test(request, flag, status);
      }

      int
      MPI_Testall(int          count,
                  MPI_Request *requests,
                  int *        flag,
                  MPI_Status * statuses)
      {
        return ::MPI_Testall(count, requests, flag, statuses);
      }

      int
      MPI_Init_thread(int *argc, char ***argv, int required, int *provided)
      {
        return ::MPI_init_thread(argc, argv, required, provided);
      }

      int
      MPI_Irecv(void *       buf,
                int          count,
                MPI_Datatype datatype,
                int          source,
                int          tag,
                MPI_Comm     comm,
                MPI_Request *request)
      {
        return ::MPI_Irecv(buf, count, datatype, source, tag, comm, request);
      }

      int
      MPI_Recv(void *       buf,
               int          count,
               MPI_Datatype datatype,
               int          source,
               int          tag,
               MPI_Comm     comm,
               MPI_Status * status)
      {
        return ::MPI_Recv(buf, count, datatype, source, tag, comm, status);
      }

      int
      MPI_Op_create(MPI_User_function *user_fn, int commute, MPI_Op *op)
      {
        return ::MPI_Op_create(user_fn, commute, op);
      }

      int
      MPI_Op_free(MPI_Op *op)
      {
        return ::MPI_Op(op);
      }

      int
      MPI_Reduce(void *       sendbuf,
                 void *       recvbuf,
                 int          count,
                 MPI_Datatype datatype,
                 MPI_Op       op,
                 int          root,
                 MPI_Comm     comm)
      {
        return ::MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
      }

      int
      MPI_Request_free(MPI_Request *request)
      {
        return ::MPI_Request_free(request);
      }

      int
      MPI_Send(const void * buf,
               int          count,
               MPI_Datatype datatype,
               int          dest,
               int          tag,
               MPI_Comm     comm)
      {
        return ::MPI_Send(buf, count, datatype, dest, tag, comm);
      }

      int
      MPI_Sendrecv(const void * sendbuf,
                   int          sendcount,
                   MPI_Datatype sendtype,
                   int          dest,
                   int          sendtag,
                   void *       recvbuf,
                   int          recvcount,
                   MPI_Datatype recvtype,
                   int          source,
                   int          recvtag,
                   MPI_Comm     comm,
                   MPI_Status * status)
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
      MPI_Isend(const void * buf,
                int          count,
                MPI_Datatype datatype,
                int          dest,
                int          tag,
                MPI_Comm     comm,
                MPI_Request *request)
      {
        return ::MPI_Isend(buf, count, datatype, dest, tag, comm, request);
      }

      int
      MPI_Ssend(const void * buf,
                int          count,
                MPI_Datatype datatype,
                int          dest,
                int          tag,
                MPI_Comm     comm)
      {
        return ::MPI_Ssend(buf, count, datatype, dest, tag, comm);
      }

      int
      MPI_Issend(const void * buf,
                 int          count,
                 MPI_Datatype datatype,
                 int          dest,
                 int          tag,
                 MPI_Comm     comm,
                 MPI_Request *request)
      {
        return ::MPI_Issend(buf, count, datatype, dest, tag, comm, request);
      }

      int
      MPI_Type_commit(MPI_Datatype *datatype)
      {
        return ::MPI_Type_commit(datatype);
      }

      int
      MPI_Type_free(MPI_Datatype *datatype)
      {
        return ::MPI_Type_free(datatype);
      }

      int
      MPI_Type_vector(int           count,
                      int           blocklength,
                      int           stride,
                      MPI_Datatype  oldtype,
                      MPI_Datatype *newtype)
      {
        return ::MPI_Type_vector(count, blocklength, stride, oldtype, newtype);
      }

      int
      MPI_Wait(MPI_Request *request, MPI_Status *status)
      {
        return ::MPI_Wait(request, status);
      }

      int
      MPI_Waitall(int count, MPI_Request requests[], MPI_Status statuses[])
      {
        return ::MPI_Waitall(count, requests, statuses);
      }

      int
      MPI_Error_string(int errorcode, char *string, int *resultlen)
      {
        return ::MPI_Error_string(errorcode, string, resultlen);
      }

      int
      MPI_Finalize(void)
      {
        return ::MPI_Finalize(void);
      }

#else  // DFTEFE_WITH_MPI
      int
      MPI_Type_contiguous(int           count,
                          MPI_Datatype  oldtype,
                          MPI_Datatype *newtype)
      {
        DFTEFE_AssertWithMsg(
          false,
          "Use of MPI_Type_contiguous() is not allowed when not linking to an MPI library");
      }

      int
      MPI_Allreduce(const void * sendbuf,
                    void *       recvbuf,
                    int          count,
                    MPI_Datatype datatype,
                    MPI_Op       op,
                    MPI_Comm     comm)
      {
        DFTEFE_AssertWithMsg(
          op == MPI_SUM || op == MPI_MAX || op == MPI_MIN,
          "Use of MPI_Op other than MPI_SUM, MPI_MX, and MPI_MIN is MPI_Allreduce is "
          "not implemented when not using an MPI library");
        copyBuffer(sendbuf, recvbuf, count, datatype);
        return MPI_SUCCESS;
      }

      int
      MPI_Allgather(const void * sendbuf,
                    int          sendcount,
                    MPI_Datatype sendtype,
                    void *       recvbuf,
                    int          recvcount,
                    MPI_Datatype recvtype,
                    MPI_Comm     comm)
      {
        DFTEFE_AssertWithMsg(
          sendcount == recvcount,
          "Use MPI_Allgather with different send and receive count is not allowed when not linking to an MPI library");
        DFTEFE_AssertWithMsg(
          sendtype == recvtype,
          "Use MPI_Allgather with different send and receive datatypes is not allowed when not linking to an MPI library");
        copyBuffer(sendbuf, recvbuf, sendcount, sendtype);
        return MPI_SUCCESS;
      }

      int
      MPI_Allgatherv(const void * sendbuf,
                     int          sendcount,
                     MPI_Datatype sendtype,
                     void *       recvbuf,
                     const int *  recvcounts,
                     const int *  displs,
                     MPI_Datatype recvtype,
                     MPI_Comm     comm)
      {
        DFTEFE_AssertWithMsg(
          sendcount == recvcounts[0],
          "Use MPI_Allgatherv with different send and receive count is not allowed when not linking to an MPI library");
        DFTEFE_AssertWithMsg(
          sendtype == recvtype,
          "Use MPI_Allgatherv with different send and receive datatypes is not allowed when not linking to an MPI library");
        copyBuffer(sendbuf, recvbuf, sendcount, sendtype);
        return MPI_SUCCESS;
      }

      int
      MPI_Barrier(MPI_Comm comm)
      {
        return MPI_SUCCESS;
      }

      int
      MPI_Ibarrier(MPI_Comm comm, MPI_Request *request)
      {
        return MPI_SUCCESS;
      }

      int
      MPI_Bcast(void *       buffer,
                int          count,
                MPI_Datatype datatype,
                int          root,
                MPI_Comm     comm)
      {
        return MPI_SUCCESS;
      }

      int
      MPI_Comm_create_group(MPI_Comm  comm,
                            MPI_Group group,
                            int       tag,
                            MPI_Comm *newcomm)
      {
        return MPI_SUCCESS;
      }

      int
      MPI_Comm_free(MPI_Comm *comm)
      {
        return MPI_SUCCESS;
      }

      int
      MPI_Comm_group(MPI_Comm comm, MPI_Group *group)
      {
        return MPI_SUCCESS;
      }

      int
      MPI_Comm_rank(MPI_Comm comm, int *rank)
      {
        *rank = 0;
        return MPI_SUCCESS;
      }

      int
      MPI_Comm_size(MPI_Comm comm, int *size)
      {
        *size = 1;
        return MPI_SUCCESS;
      }

      MPI_Fint
      MPI_Comm_f2c(MPI_Comm comm)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Comm_f2c is not implemented when not linking with an MPI library.");
        return 0;
      }

      int
      MPI_Group_free(MPI_Group *group)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Group_free is not implemented when not linking with an MPI library.");
        return MPI_SUCCESS;
      }

      int
      MPI_Group_incl(MPI_Group  group,
                     int        n,
                     const int  ranks[],
                     MPI_Group *newgroup)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Group_incl is not implemented when not linking with an MPI library.");
        return MPI_SUCCESS;
      }

      int
      MPI_Group_translate_ranks(MPI_Group group1,
                                int       n,
                                const int ranks1[],
                                MPI_Group group2,
                                int       ranks2[])
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Group_translate_ranks is not implemented when not linking with an MPI library.");
        return MPI_SUCCESS;
      }

      int
      MPI_Init(int *argc, char ***argv)
      {
        return MPI_SUCCESS;
      }

      int
      MPI_Initialized(int *flag)
      {
        *flag = 1;
        return MPI_SUCCESS;
      }

      int
      MPI_Iprobe(int         source,
                 int         tag,
                 MPI_Comm    comm,
                 int *       flag,
                 MPI_Status *status)
      {
        *flag = 1;
        return MPI_SUCCESS;
      }

      int
      MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)
      {
        *flag = 1;
        return MPI_SUCCESS;
      }

      int
      MPI_Testall(int          count,
                  MPI_Request *requests,
                  int *        flag,
                  MPI_Status * statuses)
      {
        *flag = 1;
        return MPI_SUCCESS;
      }

      int
      MPI_Init_thread(int *argc, char ***argv, int required, int *provided)
      {
        *provided = MPI_THREAD_MULTIPLE;
        return MPI_SUCCESS;
      }

      int
      MPI_Irecv(void *       buf,
                int          count,
                MPI_Datatype datatype,
                int          source,
                int          tag,
                MPI_Comm     comm,
                MPI_Request *request)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Irecv is not implemented when not linking with an MPI library.");
        return MPI_SUCCESS;
      }


      int
      MPI_Recv(void *       buf,
               int          count,
               MPI_Datatype datatype,
               int          source,
               int          tag,
               MPI_Comm     comm,
               MPI_Status * status)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Recv is not implemented when not linking with an MPI library.");
        return MPI_SUCCESS;
      }

      int
      MPI_Op_create(MPI_User_function *user_fn, int commute, MPI_Op *op)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Op_create is not implemented when not linking with an MPI library.");
        return MPI_SUCCESS;
      }

      int
      MPI_Op_free(MPI_Op *op)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Op_free is not implemented when not linking with an MPI library.");
        return MPI_SUCCESS;
      }

      int
      MPI_Reduce(void *       sendbuf,
                 void *       recvbuf,
                 int          count,
                 MPI_Datatype datatype,
                 MPI_Op       op,
                 int          root,
                 MPI_Comm     comm)
      {
        DFTEFE_AssertWithMsg(
          op == MPI_SUM || op == MPI_MAX || op == MPI_MIN,
          "Use of MPI_Op other than MPI_SUM, MPI_MX, and MPI_MIN is MPI_Reduce is "
          "not implemented when not using an MPI library");
        copyBuffer(sendbuf, recvbuf, count, datatype);
        return MPI_SUCCESS;
      }

      int
      MPI_Request_free(MPI_Request *request)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Request_free is not implemented when not linking with an MPI library.");
        return MPI_SUCCESS;
      }

      int
      MPI_Send(const void * buf,
               int          count,
               MPI_Datatype datatype,
               int          dest,
               int          tag,
               MPI_Comm     comm)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Send is not implemented when not linking with an MPI library.");
        return MPI_SUCCESS;
      }

      int
      MPI_Sendrecv(const void * sendbuf,
                   int          sendcount,
                   MPI_Datatype sendtype,
                   int          dest,
                   int          sendtag,
                   void *       recvbuf,
                   int          recvcount,
                   MPI_Datatype recvtype,
                   int          source,
                   int          recvtag,
                   MPI_Comm     comm,
                   MPI_Status * status)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Sendrecv is not implemented when not linking with an MPI library.");
        return MPI_SUCCESS;
      }

      int
      MPI_Isend(const void * buf,
                int          count,
                MPI_Datatype datatype,
                int          dest,
                int          tag,
                MPI_Comm     comm,
                MPI_Request *request)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Isend is not implemented when not linking with an MPI library.");
        return MPI_SUCCESS;
      }

      int
      MPI_Ssend(const void * buf,
                int          count,
                MPI_Datatype datatype,
                int          dest,
                int          tag,
                MPI_Comm     comm)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Ssend is not implemented when not linking with an MPI library.");
        return MPI_SUCCESS;
      }

      int
      MPI_Issend(const void * buf,
                 int          count,
                 MPI_Datatype datatype,
                 int          dest,
                 int          tag,
                 MPI_Comm     comm,
                 MPI_Request *request)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Issend is not implemented when not linking with an MPI library.");
        return MPI_SUCCESS;
      }

      int
      MPI_Type_commit(MPI_Datatype *datatype)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Type_commit is not implemented when not linking with an MPI library.");
        return MPI_SUCCESS;
      }

      int
      MPI_Type_free(MPI_Datatype *datatype)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Type_free is not implemented when not linking with an MPI library.");
        return MPI_SUCCESS;
      }

      int
      MPI_Type_vector(int           count,
                      int           blocklength,
                      int           stride,
                      MPI_Datatype  oldtype,
                      MPI_Datatype *newtype)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Type_vector is not implemented when not linking with an MPI library.");
        return MPI_SUCCESS;
      }

      int
      MPI_Wait(MPI_Request *request, MPI_Status *status)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Wait is not implemented when not linking with an MPI library.");
        return MPI_SUCCESS;
      }

      int
      MPI_Waitall(int count, MPI_Request requests[], MPI_Status statuses[])
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Wait_All is not implemented when not linking with an MPI library.");
        return MPI_SUCCESS;
      }

      int
      MPI_Error_string(int errorcode, char *string, int *resultlen)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPI_Error_string is not implemented when not linking with an MPI library.");
        return MPI_SUCCESS;
      }

      int
      MPI_Finalize(void)
      {
        return MPI_SUCCESS;
      }
#endif // DFTEFE_WITH_MPI
    }  // end of namespace mpi
  }    // end of namespace utils
} // end of namespace dftefe
