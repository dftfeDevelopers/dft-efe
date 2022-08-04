
//
// deal.II header
//
#include <deal.II/base/data_out_base.h>

//
//
//
#include <iostream>
#include <stdio.h>

#include <slate/slate.hh>
#include <slate/print.hh>
#ifdef DFTEFE_WITH_MPI
#  include <mpi.h>
#endif

#ifdef DFTEFE_WITH_DEVICE
#  include <utils/DeviceUtils.h>
#endif

#include <linearAlgebra/AbstractMatrix.h>

int
main(int argc, char **argv)
{
  // Initialize the MPI environment via dealii, as dealii internally takes care
  // of p4est MPI intialization
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int  name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // Print off a hello world message
  printf("Hello world from processor %s, rank %d out of %d processors\n",
         processor_name,
         world_rank,
         world_size);

#ifdef DFTEFE_WITH_DEVICE
  const bool useDevice = false;
  if (useDevice)
    {
      dftefe::utils::DeviceUtils::initialize(world_rank);
    }
  printf("This is gpu code");
#endif

//  slate::Matrix<double> A(5, 5, 2, 1, 1, MPI_COMM_WORLD);
//  A.insertLocalTiles();
//  int cnt = 0;
//  for (int64_t j = 0; j < A.nt(); ++j)
//    for (int64_t i = 0; i < A.mt(); ++i)
//      if (A.tileIsLocal(i, j))
//        {
//          slate::Tile<double> T = A(i, j);
//          for (int64_t jj = 0; jj < T.nb(); ++jj)
//            for (int64_t ii = 0; ii < T.mb(); ++ii)
//              T.at( ii, jj ) = cnt++;
//        }
//  slate::HermitianMatrix<double> B(slate::Uplo::Upper, 5, 2, 1, 1, MPI_COMM_WORLD);
//  std::vector<double> b(5,0);
////  slate::heev(B, b, A);
//  slate::print("matrix", A);
}