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

#ifndef dftefeUtilsDefaults_h
#define dftefeUtilsDefaults_h

#include <complex>
#include <string>
#include <utils/TypeConfig.h>
namespace dftefe
{
  namespace utils
  {
    template <typename T>
    class Types
    {
    public:
      static const T zero;
    };

    template <>
    class Types<int>
    {
    public:
      static const int zero;
    };

    template <>
    class Types<unsigned int>
    {
    public:
      static const unsigned int zero;
    };

    template <>
    class Types<short int>
    {
    public:
      static const short int zero;
    };

    template <>
    class Types<unsigned short int>
    {
    public:
      static const unsigned short int zero;
    };

    template <>
    class Types<long int>
    {
    public:
      static const long int zero;
    };

    template <>
    class Types<unsigned long int>
    {
    public:
      static const unsigned long int zero;
    };

    template <>
    class Types<double>
    {
    public:
      static const double zero;
    };

    template <>
    class Types<float>
    {
    public:
      static const float zero;
    };

    template <>
    class Types<std::complex<double>>
    {
    public:
      static const std::complex<double> zero;
    };

    template <>
    class Types<std::complex<float>>
    {
    public:
      static const std::complex<float> zero;
    };

    template <>
    class Types<char>
    {
    public:
      static const char zero;
    };

    template <>
    class Types<std::string>
    {
    public:
      static const std::string zero;
    };

    class ConditionalOStreamDefaults
    {
    public:
      //
      // floating point precision to which one should print
      //
      static const size_type PRECISION;
    };

  } // end of namespace utils
} // end of namespace dftefe
#include <utils/Defaults.t.cpp>
#endif // dftefeUtilsDefaults_h
