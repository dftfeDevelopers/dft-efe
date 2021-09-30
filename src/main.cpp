
//
// deal.II header
//
#include <deal.II/base/data_out_base.h>

//
//
//
#include <iostream>
#include <mpi.h>
#include <stdio.h>

#include <Vector.h>

#ifdef DFTEFE_WITH_CUDA
#  include <CUDAUtils.h>
#endif

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

#ifdef DFTEFE_WITH_CUDA
  const bool useGPU = false;
  if (useGPU)
    {
      dftefe::CUDAUtils::initialize(world_rank);
    }
#endif

  dftefe::Vector<double, dftefe::MemorySpace::HOST> v1;
  v1.testDgemv();
}
