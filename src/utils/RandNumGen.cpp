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

#include <utils/RandNumGen.h>
#include <string>
#include <vector>
#include <algorithm>
namespace dftefe
{
  namespace utils
  {
    //
    // Random Number Generator for short datatype
    //
    RandNumGen<short>::RandNumGen(short        min,
                                  short        max,
                                  bool         isReproducible,
                                  unsigned int seed)
      : d_dis(min, max)
    {
      if (isReproducible)
        d_gen = std::mt19937(seed);
      else
        {
          std::random_device rd;
          d_gen = std::mt19937(rd());
        }
    }

    short
    RandNumGen<short>::generate()
    {
      return d_dis(d_gen);
    }


    //
    // Random Number Generator for unsigned short datatype
    //
    RandNumGen<unsigned short>::RandNumGen(unsigned short min,
                                           unsigned short max,
                                           bool           isReproducible,
                                           unsigned int   seed)
      : d_dis(min, max)
    {
      if (isReproducible)
        d_gen = std::mt19937(seed);
      else
        {
          std::random_device rd;
          d_gen = std::mt19937(rd());
        }
    }

    unsigned short
    RandNumGen<unsigned short>::generate()
    {
      return d_dis(d_gen);
    }

    //
    // Random Number Generator for int datatype
    //
    RandNumGen<int>::RandNumGen(int          min,
                                int          max,
                                bool         isReproducible,
                                unsigned int seed)
      : d_dis(min, max)
    {
      if (isReproducible)
        d_gen = std::mt19937(seed);
      else
        {
          std::random_device rd;
          d_gen = std::mt19937(rd());
        }
    }

    int
    RandNumGen<int>::generate()
    {
      return d_dis(d_gen);
    }

    //
    // Random Number Generator for unsigned int datatype
    //
    RandNumGen<unsigned int>::RandNumGen(unsigned int min,
                                         unsigned int max,
                                         bool         isReproducible,
                                         unsigned int seed)
      : d_dis(min, max)
    {
      if (isReproducible)
        d_gen = std::mt19937(seed);
      else
        {
          std::random_device rd;
          d_gen = std::mt19937(rd());
        }
    }

    unsigned int
    RandNumGen<unsigned int>::generate()
    {
      return d_dis(d_gen);
    }

    //
    // Random Number Generator for long datatype
    //
    RandNumGen<long>::RandNumGen(long         min,
                                 long         max,
                                 bool         isReproducible,
                                 unsigned int seed)
      : d_dis(min, max)
    {
      if (isReproducible)
        d_gen = std::mt19937(seed);
      else
        {
          std::random_device rd;
          d_gen = std::mt19937(rd());
        }
    }

    long
    RandNumGen<long>::generate()
    {
      return d_dis(d_gen);
    }

    //
    // Random Number Generator for unsigned long datatype
    //
    RandNumGen<unsigned long>::RandNumGen(unsigned long min,
                                          unsigned long max,
                                          bool          isReproducible,
                                          unsigned int  seed)
      : d_dis(min, max)
    {
      if (isReproducible)
        d_gen = std::mt19937(seed);
      else
        {
          std::random_device rd;
          d_gen = std::mt19937(rd());
        }
    }

    unsigned long
    RandNumGen<unsigned long>::generate()
    {
      return d_dis(d_gen);
    }

    //
    // Random Number Generator for long long datatype
    //
    RandNumGen<long long>::RandNumGen(long long    min,
                                      long long    max,
                                      bool         isReproducible,
                                      unsigned int seed)
      : d_dis(min, max)
    {
      if (isReproducible)
        d_gen = std::mt19937(seed);
      else
        {
          std::random_device rd;
          d_gen = std::mt19937(rd());
        }
    }

    long long
    RandNumGen<long long>::generate()
    {
      return d_dis(d_gen);
    }

    //
    // Random Number Generator for unsigned long long datatype
    //
    RandNumGen<unsigned long long>::RandNumGen(unsigned long long min,
                                               unsigned long long max,
                                               bool         isReproducible,
                                               unsigned int seed)
      : d_dis(min, max)
    {
      if (isReproducible)
        d_gen = std::mt19937(seed);
      else
        {
          std::random_device rd;
          d_gen = std::mt19937(rd());
        }
    }

    unsigned long long
    RandNumGen<unsigned long long>::generate()
    {
      return d_dis(d_gen);
    }

    //
    // Random Number Generator for float datatype
    //
    RandNumGen<float>::RandNumGen(float        min,
                                  float        max,
                                  bool         isReproducible,
                                  unsigned int seed)
      : d_dis(min, max)
    {
      if (isReproducible)
        d_gen = std::mt19937(seed);
      else
        {
          std::random_device rd;
          d_gen = std::mt19937(rd());
        }
    }

    float
    RandNumGen<float>::generate()
    {
      return d_dis(d_gen);
    }

    //
    // Random Number Generator for double datatype
    //
    RandNumGen<double>::RandNumGen(double       min,
                                   double       max,
                                   bool         isReproducible,
                                   unsigned int seed)
      : d_dis(min, max)
    {
      if (isReproducible)
        d_gen = std::mt19937(seed);
      else
        {
          std::random_device rd;
          d_gen = std::mt19937(rd());
        }
    }

    double
    RandNumGen<double>::generate()
    {
      return d_dis(d_gen);
    }

    //
    // Random Number Generator for long double datatype
    //
    RandNumGen<long double>::RandNumGen(long double  min,
                                        long double  max,
                                        bool         isReproducible,
                                        unsigned int seed)
      : d_dis(min, max)
    {
      if (isReproducible)
        d_gen = std::mt19937(seed);
      else
        {
          std::random_device rd;
          d_gen = std::mt19937(rd());
        }
    }

    long double
    RandNumGen<long double>::generate()
    {
      return d_dis(d_gen);
    }

    //
    // Random Number Generator for std::complex<float> datatype
    //
    RandNumGen<std::complex<float>>::RandNumGen(std::complex<float> min,
                                                std::complex<float> max,
                                                bool         isReproducible,
                                                unsigned int seed)
    {
      std::vector<float> vals   = {min.real(),
                                 min.imag(),
                                 max.real(),
                                 max.imag()};
      float              minAll = *(std::min_element(vals.begin(), vals.end()));
      float              maxAll = *(std::max_element(vals.begin(), vals.end()));
      d_dis = std::uniform_real_distribution<float>(minAll, maxAll);
      if (isReproducible)
        d_gen = std::mt19937(seed);
      else
        {
          std::random_device rd;
          d_gen = std::mt19937(rd());
        }
    }

    std::complex<float>
    RandNumGen<std::complex<float>>::generate()
    {
      return std::complex<float>(d_dis(d_gen), d_dis(d_gen));
    }

    //
    // Random Number Generator for std::complex<double> datatype
    //
    RandNumGen<std::complex<double>>::RandNumGen(std::complex<double> min,
                                                 std::complex<double> max,
                                                 bool         isReproducible,
                                                 unsigned int seed)
    {
      std::vector<double> vals = {min.real(),
                                  min.imag(),
                                  max.real(),
                                  max.imag()};
      double minAll            = *(std::min_element(vals.begin(), vals.end()));
      double maxAll            = *(std::max_element(vals.begin(), vals.end()));
      d_dis = std::uniform_real_distribution<double>(minAll, maxAll);
      if (isReproducible)
        d_gen = std::mt19937(seed);
      else
        {
          std::random_device rd;
          d_gen = std::mt19937(rd());
        }
    }

    std::complex<double>
    RandNumGen<std::complex<double>>::generate()
    {
      return std::complex<double>(d_dis(d_gen), d_dis(d_gen));
    }

    //
    // Random Number Generator for std::complex<long double> datatype
    //
    RandNumGen<std::complex<long double>>::RandNumGen(
      std::complex<long double> min,
      std::complex<long double> max,
      bool                      isReproducible,
      unsigned int              seed)
    {
      std::vector<long double> vals = {min.real(),
                                       min.imag(),
                                       max.real(),
                                       max.imag()};
      long double minAll = *(std::min_element(vals.begin(), vals.end()));
      long double maxAll = *(std::max_element(vals.begin(), vals.end()));
      d_dis = std::uniform_real_distribution<long double>(minAll, maxAll);
      if (isReproducible)
        d_gen = std::mt19937(seed);
      else
        {
          std::random_device rd;
          d_gen = std::mt19937(rd());
        }
    }

    std::complex<long double>
    RandNumGen<std::complex<long double>>::generate()
    {
      return std::complex<long double>(d_dis(d_gen), d_dis(d_gen));
    }
  } // end of namespace utils
} // end of namespace dftefe
