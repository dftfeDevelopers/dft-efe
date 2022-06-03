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

#ifndef dftefeMPIWrapper_h
#define dftefeMPIWrapper_h

#include <utils/MPITypes.h>
#include <utils/TypeConfig.h>
namespace dftefe
{
  namespace utils
  {
    namespace mpi
    {
      int
      MPI_Type_contiguous(int           count,
                          MPI_Datatype  oldtype,
                          MPI_Datatype *newtype);

      int
      MPI_Allreduce(const void * sendbuf,
                    void *       recvbuf,
                    int          count,
                    MPI_Datatype datatype,
                    MPI_Op       op,
                    MPI_Comm     comm);

      int
      MPI_Allgather(const void * sendbuf,
                    int          sendcount,
                    MPI_Datatype sendtype,
                    void *       recvbuf,
                    int          recvcount,
                    MPI_Datatype recvtype,
                    MPI_Comm     comm);

      int
      MPI_Allgatherv(const void * sendbuf,
                     int          sendcount,
                     MPI_Datatype sendtype,
                     void *       recvbuf,
                     const int *  recvcounts,
                     const int *  displs,
                     MPI_Datatype recvtype,
                     MPI_Comm     comm);

      int
      MPI_Barrier(MPI_Comm comm);
      int
      MPI_Ibarrier(MPI_Comm comm, MPI_Request *request);

      int
      MPI_Bcast(void *       buffer,
                int          count,
                MPI_Datatype datatype,
                int          root,
                MPI_Comm     comm);

      int
      MPI_Comm_create_group(MPI_Comm  comm,
                            MPI_Group group,
                            int       tag,
                            MPI_Comm *newcomm);

      int
      MPI_Comm_free(MPI_Comm *comm);
      int
      MPI_Comm_group(MPI_Comm comm, MPI_Group *group);
      int
      MPI_Comm_rank(MPI_Comm comm, int *rank);
      int
      MPI_Comm_size(MPI_Comm comm, int *size);
      MPI_Fint
      MPI_Comm_f2c(MPI_Comm comm);

      int
      MPI_Group_free(MPI_Group *group);

      int
      MPI_Group_incl(MPI_Group  group,
                     int        n,
                     const int  ranks[],
                     MPI_Group *newgroup);

      int
      MPI_Group_translate_ranks(MPI_Group group1,
                                int       n,
                                const int ranks1[],
                                MPI_Group group2,
                                int       ranks2[]);

      int
      MPI_Init(int *argc, char ***argv);

      int
      MPI_Init_thread(int *argc, char ***argv, int required, int *provided);

      int
      MPI_Initialized(int *flag);

      int
      MPI_Iprobe(int         source,
                 int         tag,
                 MPI_Comm    comm,
                 int *       flag,
                 MPI_Status *status);

      int
      MPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
      int
      MPI_Testall(int          count,
                  MPI_Request *requests,
                  int *        flag,
                  MPI_Status * statuses);

      int
      MPI_Irecv(void *       buf,
                int          count,
                MPI_Datatype datatype,
                int          source,
                int          tag,
                MPI_Comm     comm,
                MPI_Request *request);

      int
      MPI_Isend(const void * buf,
                int          count,
                MPI_Datatype datatype,
                int          dest,
                int          tag,
                MPI_Comm     comm,
                MPI_Request *request);

      int
      MPI_Recv(void *       buf,
               int          count,
               MPI_Datatype datatype,
               int          source,
               int          tag,
               MPI_Comm     comm,
               MPI_Status * status);

      int
      MPI_Op_create(MPI_User_function *user_fn, int commute, MPI_Op *op);

      int
      MPI_Op_free(MPI_Op *op);

      int
      MPI_Reduce(void *       sendbuf,
                 void *       recvbuf,
                 int          count,
                 MPI_Datatype datatype,
                 MPI_Op       op,
                 int          root,
                 MPI_Comm     comm);

      int
      MPI_Request_free(MPI_Request *request);

      int
      MPI_Send(const void * buf,
               int          count,
               MPI_Datatype datatype,
               int          dest,
               int          tag,
               MPI_Comm     comm);

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
                   MPI_Status * status);

      int
      MPI_Issend(const void * buf,
                 int          count,
                 MPI_Datatype datatype,
                 int          dest,
                 int          tag,
                 MPI_Comm     comm,
                 MPI_Request *request);

      int
      MPI_Ssend(const void * buf,
                int          count,
                MPI_Datatype datatype,
                int          dest,
                int          tag,
                MPI_Comm     comm);

      int
      MPI_Type_commit(MPI_Datatype *datatype);

      int
      MPI_Type_free(MPI_Datatype *datatype);

      int
      MPI_Type_vector(int           count,
                      int           blocklength,
                      int           stride,
                      MPI_Datatype  oldtype,
                      MPI_Datatype *newtype);

      int
      MPI_Wait(MPI_Request *request, MPI_Status *status);

      int
      MPI_Waitall(int count, MPI_Request requests[], MPI_Status statuses[]);

      int
      MPI_Error_string(int errorcode, char *string, int *resultlen);

    } // end of namespace mpi
  }   // end of namespace utils
} // end of namespace dftefe
#endif // dftefeMPIWrapper_h
