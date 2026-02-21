/******************************************************************************
 * Copyright (c) 2022-2022.                                                   *
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
* @author Ian C. Lin.
*/

#include <linearAlgebra/BlasLapack.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <cassert>

#define CHECK_CUDA(call) \
    do { \
        if ((call) != cudaSuccess) { \
            std::cerr << "CUDA error at line " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUBLAS(call) \
    do { \
        if ((call) != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at line " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// int
// main(int argc, char **argv)
// {const dftefe::utils::MemorySpace Host = dftefe::utils::MemorySpace::HOST;
//  const dftefe::utils::MemorySpace Device = dftefe::utils::MemorySpace::DEVICE;
//  using namespace dftefe::linearAlgebra::blasLapack;

//  double tol = 1e-12;

//  dftefe::size_type Am = 10, An = 5, Bm = An, Bn = 3, Cm = Am, Cn = Bn;

//  std::vector<double> colMajA = {
//    -4.47950, 3.59405,  3.10196,  -6.74777, -7.62005, -0.03272, 9.19488,
//    -3.19229, 1.70536,  -5.52376, 5.02534,  -4.89810, 0.11914,  3.98153,
//    7.81807,  9.18583,  0.94431,  -7.22751, -7.01412, -4.84983, 6.81435,
//    -4.91436, 6.28570,  -5.12950, 8.58527,  -3.00032, -6.06809, -4.97832,
//    2.32089,  -0.53422, -2.96681, 6.61657,  1.70528,  0.99447,  8.34387,
//    -4.28322, 5.14400,  5.07458,  -2.39108, 1.35643,  -8.48291, -8.92100,
//    0.61595,  5.58334,  8.68021,  -7.40188, 1.37647,  -0.61219, -9.76196,
//    -3.25755};
//  std::vector<double> colMajB    = {-6.75635,
//                                 5.88569,
//                                 -3.77570,
//                                 0.57066,
//                                 -6.68703,
//                                 2.03964,
//                                 -4.74057,
//                                 3.08158,
//                                 3.78429,
//                                 4.96303,
//                                 -0.98917,
//                                 -8.32357,
//                                 -5.42046,
//                                 8.26675,
//                                 -6.95244};
//  std::vector<double> colMajCRef = {
//    89.14615577730,  28.87459761170,  -47.13536487310, 51.62334246520,
//    11.79974547470,  112.66665505990, -39.92364529520, 4.81558075210,
//    2.34609631420,   33.35030931760,  -65.28878498300, -3.82981037450,
//    34.64222997030,  -16.97106108840, 48.50769588330,  -105.80323235970,
//    21.87622226840,  28.57566837200,  -13.61618494700, -0.95595880270,
//    -39.88364761690, 180.57289087160, -28.31665330300, -29.25840391830,
//    -95.44479256550, -44.11060117130, 48.89084619830,  136.50768579580,
//    92.21831704020,  82.58874393800};

//  std::vector<double> C(Cm * Cn, 0.0);

//  int lda = Am, ldb = Bm, ldc = Cm;

//  double alpha = 1.0, beta = 0.0;

//  int device;
//  //BlasQueue<Device> queue(device, 0);

//   dftefe::linearAlgebra::LinAlgOpContext<Device> laoc;

//  dftefe::utils::MemoryStorage<double, Device> dA(colMajA.size(), 0);
//  dftefe::utils::MemoryStorage<double, Device> dB(colMajB.size(), 0);
//  dftefe::utils::MemoryStorage<double, Device> dC(C.size(), 0);

//  dftefe::utils::MemoryTransfer<Device, Host>::copy(colMajA.size(), dA.data(), colMajA.data());
//  dftefe::utils::MemoryTransfer<Device, Host>::copy(colMajB.size(), dB.data(), colMajB.data());

//  gemm<double, double, Device>(
//       'N',
//       'N',
//       Am,
//       Cn,
//       An,
//       alpha,
//       dA.data(),
//       lda,
//       dB.data(),
//       ldb,
//       beta,
//       dC.data(),
//       ldc,
//       laoc);

//  dftefe::utils::MemoryTransfer<Host, Device>::copy(C.size(), C.data(), dC.data());

//  for(dftefe::size_type i = 0; i < C.size(); ++i)
//    {
//      if(std::fabs(C[i]-colMajCRef[i]) > tol)
//        {
//          std::string msg = "At index " + std::to_string(i) +
//                            " mismatch of entries after doing column major dftefe::linearAlgebra::blasWrapper::Gemm. "
//                            " dftefe::linearAlgebra::blasWrapper::Gemm value: " + std::to_string(C[i]) +
//                            " reference value: " + std::to_string(colMajCRef[i]);
//          throw std::runtime_error(msg);
//        }
//    }
// }
// }

int main(int argc, char** argv)
{
    using namespace dftefe::linearAlgebra::blasLapack;
    const dftefe::utils::MemorySpace  Device = dftefe::utils::MemorySpace::DEVICE;
    const dftefe::utils::MemorySpace  Host   = dftefe::utils::MemorySpace::HOST;
    if(argc < 9){
        std::cout<<"Usage:\n";
        std::cout<<"batch M N K factorx factory factorz streams\n";
        return 0;
    }

    int batch_size = std::atoi(argv[1]);
    int MSize      = std::atoi(argv[2]);
    int NSize      = std::atoi(argv[3]);
    int KSize      = std::atoi(argv[4]);
    int factorx    = std::atoi(argv[5]);
    int factory    = std::atoi(argv[6]);
    int factorz    = std::atoi(argv[7]);
    int streams    = std::atoi(argv[8]);

    // =============================
    // Variable sizes
    // =============================
    std::vector<int> M(batch_size),
                     N(batch_size),
                     K(batch_size);

    for(int i=0;i<batch_size;++i){
        M[i] = MSize + (MSize * (factorx - 1) * i) / (batch_size - 1);
        N[i] = NSize + (NSize * (factory - 1) * i) / (batch_size - 1);
        K[i] = KSize + (KSize * (factorz - 1) * i) / (batch_size - 1);
    }

    // Shuffle the batch randomly
    std::mt19937 gen(42); // fixed seed for reproducibility
    std::vector<int> indices(batch_size);
    for(int i = 0; i < batch_size; ++i) indices[i] = i;
    std::shuffle(indices.begin(), indices.end(), gen);

    // Apply the shuffle
    std::vector<int> M_shuffled(batch_size), N_shuffled(batch_size), K_shuffled(batch_size);
    for(int i = 0; i < batch_size; ++i) {
        M_shuffled[i] = M[indices[i]];
        N_shuffled[i] = N[indices[i]];
        K_shuffled[i] = K[indices[i]];
        std::cout << M_shuffled[i] << "\n";
    }

    // Replace original vectors
    M = std::move(M_shuffled);
    N = std::move(N_shuffled);
    K = std::move(K_shuffled);

    // =============================
    // Offsets
    // =============================

    long long totalA=0,totalB=0,totalC=0;

    for(int i=0;i<batch_size;++i){
        totalA+=(long long)M[i]*K[i];
        totalB+=(long long)K[i]*N[i];
        totalC+=(long long)M[i]*N[i];
    }

    // =============================
    // Host buffers (column-major)
    // =============================
    std::vector<double> colMajA(totalA),
                        colMajB(totalB),
                        C(totalC,0.0),
                        Cref(totalC,0.0);

    std::uniform_real_distribution<double> dist(0.0,1.0);

    for(long long i=0;i<totalA;++i) colMajA[i]=dist(gen);
    for(long long i=0;i<totalB;++i) colMajB[i]=dist(gen);

    // =============================
    // Device storage using framework
    // =============================

    dftefe::utils::MemoryStorage<double, Device> dA(colMajA.size(), 0);
    dftefe::utils::MemoryStorage<double, Device> dB(colMajB.size(), 0);
    dftefe::utils::MemoryStorage<double, Device> dC(C.size(), 0);

    dftefe::utils::MemoryTransfer<Device, Host>
        ::copy(colMajA.size(), dA.data(), colMajA.data());

    dftefe::utils::MemoryTransfer<Device, Host>
        ::copy(colMajB.size(), dB.data(), colMajB.data());

    // =============================
    // Strides & dims
    // =============================
    std::vector<dftefe::size_type> strideA(batch_size),
                           strideB(batch_size),
                           strideC(batch_size),
                           m(batch_size),
                           n(batch_size),
                           k(batch_size),
                           ldda(batch_size),
                           lddb(batch_size),
                           lddc(batch_size);

    for(int i=0;i<batch_size;++i){
        strideA[i]=M[i]*K[i];
        strideB[i]=K[i]*N[i];
        strideC[i]=M[i]*N[i];
        m[i]=M[i];
        n[i]=N[i];
        k[i]=K[i];
        ldda[i]=M[i];
        lddb[i]=K[i];
        lddc[i]=M[i];
    }

    double alpha=1.0, beta=0.0;
    std::vector<char> transA(batch_size, 'N');
    std::vector<char> transB(batch_size, 'N');

    // =============================
    // CPU reference
    // =============================
    dftefe::linearAlgebra::LinAlgOpContext<Host> host_context;

    gemmStridedVarBatched<double,double,Host>(
        batch_size,transA.data(), transB.data(),
        strideA.data(), strideB.data(), strideC.data(),
        m.data(), n.data(), k.data(),
        alpha,
        colMajA.data(), ldda.data(),
        colMajB.data(), lddb.data(),
        beta,
        Cref.data(), lddc.data(),
        host_context
    );

    double cpu_ms = 0.0;

    int NUM_ITERS = 10;

    for(int iter=0; iter<NUM_ITERS; ++iter)
    {
        std::fill(Cref.begin(), Cref.end(), 0.0);

        auto start = std::chrono::high_resolution_clock::now();

        gemmStridedVarBatched<double,double,Host>(
            batch_size,transA.data(), transB.data(),
            strideA.data(), strideB.data(), strideC.data(),
            m.data(), n.data(), k.data(),
            alpha,
            colMajA.data(), ldda.data(),
            colMajB.data(), lddb.data(),
            beta,
            Cref.data(), lddc.data(),
            host_context
        );

        auto end = std::chrono::high_resolution_clock::now();

        cpu_ms += std::chrono::duration<double,std::milli>(end-start).count();
    }

    cpu_ms /= NUM_ITERS;

    std::cout<<"CPU avg time: "<<cpu_ms<<" ms\n";

    // =============================
    // GPU run
    // =============================
    dftefe::linearAlgebra::LinAlgOpContext<Device> device_context(streams);

    gemmStridedVarBatched<double,double,Device>(
        batch_size,transA.data(), transB.data(),
        strideA.data(), strideB.data(), strideC.data(),
        m.data(), n.data(), k.data(),
        alpha,
        dA.data(), ldda.data(),
        dB.data(), lddb.data(),
        beta,
        dC.data(), lddc.data(),
        device_context
    );

    double gpu_compute_ms = 0.0;

    for(int iter=0; iter<NUM_ITERS; ++iter)
    {
        dC.setValue(0);   // or your equivalent zeroing

        auto start = std::chrono::high_resolution_clock::now();

        gemmStridedVarBatched<double,double,Device>(
            batch_size,transA.data(), transB.data(),
            strideA.data(), strideB.data(), strideC.data(),
            m.data(), n.data(), k.data(),
            alpha,
            dA.data(), ldda.data(),
            dB.data(), lddb.data(),
            beta,
            dC.data(), lddc.data(),
            device_context
        );

        auto end = std::chrono::high_resolution_clock::now();

        gpu_compute_ms +=
            std::chrono::duration<double,std::milli>(end-start).count();
    }

    gpu_compute_ms /= NUM_ITERS;

    std::cout<<"GPU compute avg time: "
            << gpu_compute_ms<<" ms\n";

    // Copy back
    dftefe::utils::MemoryTransfer<Host, Device>
        ::copy(C.size(), C.data(), dC.data());

    // =============================
    // Error check
    // =============================
    double max_err=0.0;
    for(long long i=0;i<totalC;++i)
        max_err = std::max(max_err,
                           std::abs(C[i]-Cref[i]));

    std::cout<<"Max abs error: "<<max_err<<"\n";

// ============================================
// Padded Strided Batched GEMM (single large buffer)
// ============================================
int Mmax = *std::max_element(M.begin(), M.end());
int Nmax = *std::max_element(N.begin(), N.end());
int Kmax = *std::max_element(K.begin(), K.end());

long long strideApad = (long long)Mmax*Kmax;
long long strideBpad = (long long)Kmax*Nmax;
long long strideCpad = (long long)Mmax*Nmax;

std::cout << "Max. Elements: " <<  Mmax << " " << Nmax << " " << Kmax << " " << "\n" ;

double *dA_big, *dB_big, *dC_big;
CHECK_CUDA(cudaMalloc((void**)&dA_big, strideApad*batch_size*sizeof(double)));
CHECK_CUDA(cudaMalloc((void**)&dB_big, strideBpad*batch_size*sizeof(double)));
CHECK_CUDA(cudaMalloc((void**)&dC_big, strideCpad*batch_size*sizeof(double)));

// zero out padding
CHECK_CUDA(cudaMemset(dA_big, 0, strideApad*batch_size*sizeof(double)));
CHECK_CUDA(cudaMemset(dB_big, 0, strideBpad*batch_size*sizeof(double)));
CHECK_CUDA(cudaMemset(dC_big, 0, strideCpad*batch_size*sizeof(double)));


    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

// Warmup
CHECK_CUBLAS(
    cublasDgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        Mmax, Nmax, Kmax,
        &alpha,
        dA_big, Mmax, strideApad,
        dB_big, Kmax, strideBpad,
        &beta,
        dC_big, Mmax, strideCpad,
        batch_size
    )
);
CHECK_CUDA(cudaDeviceSynchronize());

// Timing loop
double total_batched_ms = 0.0;
for(int iter=0; iter<NUM_ITERS; ++iter) {
    auto start = std::chrono::high_resolution_clock::now();

    CHECK_CUBLAS(
        cublasDgemmStridedBatched(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            Mmax, Nmax, Kmax,
            &alpha,
            dA_big, Mmax, strideApad,
            dB_big, Kmax, strideBpad,
            &beta,
            dC_big, Mmax, strideCpad,
            batch_size
        )
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    total_batched_ms += std::chrono::duration<double,std::milli>(end-start).count();
}
double batched_ms = total_batched_ms / NUM_ITERS;
std::cout << "GPU padded strided batched avg time: " << batched_ms << " ms\n";

}