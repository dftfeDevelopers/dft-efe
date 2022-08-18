
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

#include <linearAlgebra/GeneralMatrix.h>
#include <linearAlgebra/MatrixOperations.h>

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

  std::vector<double> dA(25, 0.0), dB(15, 0.0);
  for (int i = 0; i < 25; ++i) dA[i] = i+1;
  for (int i = 0; i < 15; ++i) dB[i] = i+25;

//    dftefe::linearAlgebra::HermitianMatrix<double, dftefe::utils::MemorySpace::HOST>
//      A(dftefe::linearAlgebra::Uplo::Lower, 5, MPI_COMM_SELF, 1, 1);

  dftefe::linearAlgebra::GeneralMatrix<double, dftefe::utils::MemorySpace::HOST>
    A(5, 5, MPI_COMM_SELF, 1, 1, 5, 5);
//    B(5, 3, MPI_COMM_SELF, 1, 1, 3, 5),
//    C(5, 3, MPI_COMM_SELF, 1, 1, 3, 5);
  //A.setValues(dA.data());
//  A.setValues(1, 4, 2, 3, dA.data());
//  B.setValues(dB.data());
  slate::print("matrix A", A.getSlateMatrix());
//  slate::print("matrix B", B.getSlateMatrix());
//  dftefe::linearAlgebra::MatrixOperations::multiply<double, dftefe::utils::MemorySpace::HOST>(1.0, A, B, 0.0, C);

//  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, 5, 3, 5, 1.0, A.getSlateMatrix()(0,0).data(), 5, B.getSlateMatrix()(0,0).data(), 5, 0.0, C.getSlateMatrix()(0,0).data(), 5);
//  slate::print("matrix C", C.getSlateMatrix());

//  std::vector<double> dD = {1.148258296912840, 1.001372859769502, 0.994403249881076, 1.638906883577435, 1.316661477930142, 1.001372859769502, 2.019228542698278, 1.612894116737191, 1.932571381399861, 2.047167619624847, 0.994403249881076, 1.612894116737191, 2.790965744547513, 2.622748446448737, 2.828958221480860, 1.638906883577435, 1.932571381399861, 2.622748446448737, 3.486600154718786, 3.283956672039348, 1.316661477930142, 2.047167619624847, 2.828958221480860, 3.283956672039348, 3.352644057275798};
//    dftefe::linearAlgebra::GeneralMatrix<double, dftefe::utils::MemorySpace::HOST>
//      D(5, 5, MPI_COMM_SELF, 1, 1, 5, 5), Z(5, 5, MPI_COMM_SELF, 1, 1, 5, 5);
//    D.setValues(dD.data());
//    slate::print("matrix", D.getSlateMatrix());
//
//    slate::SymmetricMatrix<double> Dh(slate::Uplo::Lower, D.getSlateMatrix());
//    slate::print("matrix", Dh);
//    lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, 5, Dh(0,0).data(), 5, Z.getSlateMatrix()(0,0).data());
//    slate::print("matrix", Z.getSlateMatrix());
//    slate::print("matrix", Dh);

}