// ---------------------------------------------------------------------
//
// Copyright (C) 2005 - 2018 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#ifndef dftefeLapackSupport_h
#define dftefeLapackSupport_h

namespace dftefe
{
namespace linearAlgebra
{
  namespace types
  {
    using LapackInt = int64_t;
  } // namespace types

  namespace LAPACKSupport
  {
    /**
     * Most of the LAPACK functions one can apply to a matrix (e.g., by calling
     * the member functions of this class) change its content in some ways. For
     * example, they may invert the matrix, or may replace it by a matrix whose
     * columns represent the eigenvectors of the original content of the matrix.
     * The elements of this enumeration are therefore used to track what is
     * currently being stored by this object.
     */
    enum State
    {
      /// Contents is actually a matrix.
      matrix,
      /// Contents is the inverse of a matrix.
      inverse_matrix,
      /// Contents is an LU decomposition.
      lu,
      /// Contents is a Cholesky decomposition.
      cholesky,
      /// Eigenvalue vector is filled
      eigenvalues,
      /// Matrix contains singular value decomposition,
      svd,
      /// Matrix is the inverse of a singular value decomposition
      inverse_svd,
      /// Contents is something useless.
      unusable = 0x8000
    };

    /**
     * %Function printing the name of a State.
     */
    inline const char *
    state_name(State s)
    {
      switch (s)
        {
          case matrix:
            return "matrix";
          case inverse_matrix:
            return "inverse matrix";
          case lu:
            return "lu decomposition";
          case cholesky:
            return "cholesky decomposition";
          case eigenvalues:
            return "eigenvalues";
          case svd:
            return "svd";
          case inverse_svd:
            return "inverse_svd";
          case unusable:
            return "unusable";
          default:
            return "unknown";
        }
    }

    /**
     * A matrix can have certain features allowing for optimization, but hard to
     * test. These are listed here.
     */
    enum Property
    {
      /// No special properties
      general = 0,
      /// Matrix is symmetric
      hermitian = 1,
      /// Matrix is upper triangular
      upper_triangular = 2,
      /// Matrix is lower triangular
      lower_triangular = 4,
      /// Matrix is diagonal
      diagonal = 6,
      /// Matrix is in upper Hessenberg form
      hessenberg = 8
    };

    /**
     * %Function printing the name of a Property.
     */
    inline const char *
    property_name(const Property s)
    {
      switch (s)
        {
          case general:
            return "general";
          case hermitian:
            return "hermitian";
          case upper_triangular:
            return "upper triangular";
          case lower_triangular:
            return "lower triangular";
          case diagonal:
            return "diagonal";
          case hessenberg:
            return "Hessenberg";
        }

      DFTEFE_Assert(false);
      return "invalid";
    }

    /**
     * Character constant.
     */
    static const char A = 'A';
    /**
     * Character constant.
     */
    static const char N = 'N';
    /**
     * Character constant.
     */
    static const char O = 'O';
    /**
     * Character constant.
     */
    static const char T = 'T';
    /**
     * Character constant for conjugate transpose.
     */
    static const char C = 'C';
    /**
     * Character constant.
     */
    static const char U = 'U';
    /**
     * Character constant.
     */
    static const char L = 'L';
    /**
     * Character constant.
     */
    static const char V = 'V';
    /**
     * Integer constant.
     */
    static const types::LapackInt zero = 0;
    /**
     * Integer constant.
     */
    static const types::LapackInt one = 1;

  } // namespace LAPACKSupport
} // namespace linearAlgebra
} // namespace dftefe
#endif