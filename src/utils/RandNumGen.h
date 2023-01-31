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

#ifndef dftefeRandNumGen_h
#define dftefeRandNumGen_h

#include <random>
#include <complex>
namespace dftefe
{
  namespace utils
  {
    template <typename T>
    class RandNumGen
    {
    public:
      RandNumGen(T min, T max);
      ~RandNumGen() = default;
      T
      generate();
    };

    template <>
    class RandNumGen<short>
    {
    public:
      RandNumGen(short min, short max);
      ~RandNumGen() = default;
      short
      generate();

    private:
      std::mt19937                         d_gen;
      std::uniform_int_distribution<short> d_dis;
    };

    template <>
    class RandNumGen<unsigned short>
    {
    public:
      RandNumGen(unsigned short min, unsigned short max);
      ~RandNumGen() = default;
      unsigned short
      generate();

    private:
      std::mt19937                                  d_gen;
      std::uniform_int_distribution<unsigned short> d_dis;
    };

    template <>
    class RandNumGen<int>
    {
    public:
      RandNumGen(int min, int max);
      ~RandNumGen() = default;
      int
      generate();

    private:
      std::mt19937                       d_gen;
      std::uniform_int_distribution<int> d_dis;
    };

    template <>
    class RandNumGen<unsigned int>
    {
    public:
      RandNumGen(unsigned int min, unsigned int max);
      ~RandNumGen() = default;
      unsigned int
      generate();

    private:
      std::mt19937                                d_gen;
      std::uniform_int_distribution<unsigned int> d_dis;
    };

    template <>
    class RandNumGen<long>
    {
    public:
      RandNumGen(long min, long max);
      ~RandNumGen() = default;
      long
      generate();

    private:
      std::mt19937                        d_gen;
      std::uniform_int_distribution<long> d_dis;
    };

    template <>
    class RandNumGen<unsigned long>
    {
    public:
      RandNumGen(unsigned long min, unsigned long max);
      ~RandNumGen() = default;
      unsigned long
      generate();

    private:
      std::mt19937                                 d_gen;
      std::uniform_int_distribution<unsigned long> d_dis;
    };

    template <>
    class RandNumGen<long long>
    {
    public:
      RandNumGen(long long min, long long max);
      ~RandNumGen() = default;
      long long
      generate();

    private:
      std::mt19937                             d_gen;
      std::uniform_int_distribution<long long> d_dis;
    };

    template <>
    class RandNumGen<unsigned long long>
    {
    public:
      RandNumGen(unsigned long long min, unsigned long long max);
      ~RandNumGen() = default;
      unsigned long long
      generate();

    private:
      std::mt19937                                      d_gen;
      std::uniform_int_distribution<unsigned long long> d_dis;
    };

    template <>
    class RandNumGen<float>
    {
    public:
      RandNumGen(float min, float max);
      ~RandNumGen() = default;
      float
      generate();

    private:
      std::mt19937                          d_gen;
      std::uniform_real_distribution<float> d_dis;
    };

    template <>
    class RandNumGen<double>
    {
    public:
      RandNumGen(double min, double max);
      ~RandNumGen() = default;
      double
      generate();

    private:
      std::mt19937                           d_gen;
      std::uniform_real_distribution<double> d_dis;
    };

    template <>
    class RandNumGen<long double>
    {
    public:
      RandNumGen(long double min, long double max);
      ~RandNumGen() = default;
      long double
      generate();

    private:
      std::mt19937                                d_gen;
      std::uniform_real_distribution<long double> d_dis;
    };

    template <>
    class RandNumGen<std::complex<float>>
    {
    public:
      RandNumGen(std::complex<float> min, std::complex<float> max);
      ~RandNumGen() = default;
      std::complex<float>
      generate();

    private:
      std::mt19937                          d_gen;
      std::uniform_real_distribution<float> d_dis;
    };

    template <>
    class RandNumGen<std::complex<double>>
    {
    public:
      RandNumGen(std::complex<double> min, std::complex<double> max);
      ~RandNumGen() = default;
      std::complex<double>
      generate();

    private:
      std::mt19937                           d_gen;
      std::uniform_real_distribution<double> d_dis;
    };

    template <>
    class RandNumGen<std::complex<long double>>
    {
    public:
      RandNumGen(std::complex<long double> min, std::complex<long double> max);
      ~RandNumGen() = default;
      std::complex<long double>
      generate();

    private:
      std::mt19937                                d_gen;
      std::uniform_real_distribution<long double> d_dis;
    };
  } // end of namespace utils
} // end of namespace dftefe
#include <utils/RandNumGen.t.cpp>
#endif // dftefeRandNumGen_h
