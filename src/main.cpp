
//
// deal.II header
//
#include <deal.II/base/data_out_base.h>

//
//
//
#include <mpi.h>
#include <stdio.h>

#include <MemoryManager.h>
#include <iostream>

 extern "C"
  {
    void
    dgemv_(const char *        TRANS,
           const unsigned int *M,
           const unsigned int *N,
           const double *      alpha,
           const double *      A,
           const unsigned int *LDA,
           const double *      X,
           const unsigned int *INCX,
           const double *      beta,
           double *            C,
           const unsigned int *INCY);
}

int main(int argc, char** argv) {
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
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d out of %d processors\n",
           processor_name, world_rank, world_size);

    double * data  = dftefe::MemoryManager<double, dftefe::MemorySpace::HOST>::allocate(100);
    std::cout << data[0] << " " << data[99] << std::endl;
}

