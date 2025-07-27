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

// @author Sambit Das, Aviup Sircar
//

#include "ScalapackTemplates.h"
#include <mutex>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename NumberType>
    ScaLAPACKMatrix<NumberType>::ScaLAPACKMatrix(
      const size_type                           n_rows_,
      const size_type                           n_columns_,
      const std::shared_ptr<const ProcessGrid> &process_grid,
      const size_type                           row_block_size_,
      const size_type                           column_block_size_,
      const LAPACKSupport::Property             property_)
      : uplo('L')
      , // for non-hermitian matrices this is not needed
      first_process_row(0)
      , first_process_column(0)
      , submatrix_row(1)
      , submatrix_column(1)
    {
      reinit(n_rows_,
             n_columns_,
             process_grid,
             row_block_size_,
             column_block_size_,
             property_);
    }

    template <typename NumberType>
    ScaLAPACKMatrix<NumberType>::ScaLAPACKMatrix(
      const size_type                           size,
      const std::shared_ptr<const ProcessGrid> &process_grid,
      const size_type                           block_size,
      const LAPACKSupport::Property             property)
      : ScaLAPACKMatrix<NumberType>(size,
                                    size,
                                    process_grid,
                                    block_size,
                                    block_size,
                                    property)
    {}

    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::reinit(
      const size_type                           n_rows_,
      const size_type                           n_columns_,
      const std::shared_ptr<const ProcessGrid> &process_grid,
      const size_type                           row_block_size_,
      const size_type                           column_block_size_,
      const LAPACKSupport::Property             property_)
    {
      DFTEFE_AssertWithMsg(row_block_size_ > 0,
                           "Row block size has to be positive.");
      DFTEFE_AssertWithMsg(column_block_size_ > 0,
                           "Column block size has to be positive.");
      DFTEFE_AssertWithMsg(
        row_block_size_ <= n_rows_,
        "Row block size can not be greater than the number of rows of the matrix");
      DFTEFE_AssertWithMsg(
        column_block_size_ <= n_columns_,
        "Column block size can not be greater than the number of columns of the matrix");

      state             = LAPACKSupport::State::matrix;
      property          = property_;
      grid              = process_grid;
      n_rows            = n_rows_;
      n_columns         = n_columns_;
      row_block_size    = row_block_size_;
      column_block_size = column_block_size_;

      if (grid->is_process_active())
        {
          // Get local sizes:
          n_local_rows    = numroc_(&n_rows,
                                 &row_block_size,
                                 &(grid->get_this_process_row()),
                                 &first_process_row,
                                 &(grid->get_process_grid_rows()));
          n_local_columns = numroc_(&n_columns,
                                    &column_block_size,
                                    &(grid->get_this_process_column()),
                                    &first_process_column,
                                    &(grid->get_process_grid_columns()));

          // LLD_A = MAX(1,NUMROC(M_A, MB_A, MYROW, RSRC_A, NPROW)), different
          // between processes
          int lda = std::max(1, n_local_rows);

          int info = 0;
          descinit_(descriptor,
                    &n_rows,
                    &n_columns,
                    &row_block_size,
                    &column_block_size,
                    &first_process_row,
                    &first_process_column,
                    &(grid->get_blacs_context()),
                    &lda,
                    &info);
          utils::throwException(
            info == 0,
            "Cannot initialize the array descriptor for distributed matrix.");

          values.clear();
          values.resize(n_local_rows * n_local_columns, NumberType(0.0));
          // this->TransposeTable<NumberType>::reinit(n_local_rows,
          // n_local_columns);
        }
      else
        {
          // set process-local variables to something telling:
          n_local_rows    = -1;
          n_local_columns = -1;
          std::fill(std::begin(descriptor), std::end(descriptor), -1);
        }
    }



    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::reinit(
      const size_type                           size,
      const std::shared_ptr<const ProcessGrid> &process_grid,
      const size_type                           block_size,
      const LAPACKSupport::Property             property)
    {
      reinit(size, size, process_grid, block_size, block_size, property);
    }

    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::set_property(
      const LAPACKSupport::Property property_)
    {
      property = property_;
    }



    template <typename NumberType>
    LAPACKSupport::Property
    ScaLAPACKMatrix<NumberType>::get_property() const
    {
      return property;
    }


    template <typename NumberType>
    LAPACKSupport::State
    ScaLAPACKMatrix<NumberType>::get_state() const
    {
      return state;
    }


    template <typename NumberType>
    unsigned int
    ScaLAPACKMatrix<NumberType>::global_row(const unsigned int loc_row) const
    {
      DFTEFE_AssertWithMsg(n_local_rows >= 0 &&
                             loc_row < static_cast<unsigned int>(n_local_rows),
                           "Exceeding index id: " + loc_row + ", Range is: (" +
                             0 + "," + n_local_rows + ")");
      const int i = loc_row + 1;
      return indxl2g_(&i,
                      &row_block_size,
                      &(grid->get_this_process_row()),
                      &first_process_row,
                      &(grid->get_process_grid_rows())) -
             1;
    }



    template <typename NumberType>
    unsigned int
    ScaLAPACKMatrix<NumberType>::global_column(
      const unsigned int loc_column) const
    {
      DFTEFE_AssertWithMsg(n_local_columns >= 0 &&
                             loc_column <
                               static_cast<unsigned int>(n_local_columns),
                           "Exceeding index id: " + loc_column +
                             ", Range is: (" + 0 + "," + n_local_columns + ")");
      const int j = loc_column + 1;
      return indxl2g_(&j,
                      &column_block_size,
                      &(grid->get_this_process_column()),
                      &first_process_column,
                      &(grid->get_process_grid_columns())) -
             1;
    }

    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::conjugate()
    {
      if (std::is_same<NumberType, std::complex<double>>::value)
        {
          if (this->grid->is_process_active())
            {
              NumberType *A_loc =
                (this->values.size() > 0) ? this->values.data() : nullptr;
              const int totalSize = n_rows * n_columns;
              const int incx      = 1;
              pplacgv(&totalSize,
                      A_loc,
                      &submatrix_row,
                      &submatrix_column,
                      descriptor,
                      &incx);
            }
        }
    }


    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::copy_to(
      ScaLAPACKMatrix<NumberType> &dest) const
    {
      DFTEFE_AssertWithMsg(n_rows == dest.n_rows,
                           ("Dimension mismatch between " + n_rows + " and " +
                            dest.n_rows));
      DFTEFE_AssertWithMsg(n_columns == dest.n_columns,
                           ("Dimension mismatch between " + n_columns +
                            " and " + dest.n_columns));

      if (this->grid->is_process_active())
        DFTEFE_AssertWithMsg(
          this->descriptor[0] == 1,
          "Copying of ScaLAPACK matrices only implemented for dense matrices");
      if (dest.grid->is_process_active())
        DFTEFE_AssertWithMsg(
          dest.descriptor[0] == 1,
          "Copying of ScaLAPACK matrices only implemented for dense matrices");

      /*
       * just in case of different process grids or block-cyclic distributions
       * inter-process communication is necessary
       * if distributed matrices have the same process grid and block sizes,
       * local copying is enough
       */
      if ((this->grid != dest.grid) ||
          (row_block_size != dest.row_block_size) ||
          (column_block_size != dest.column_block_size))
        {
          /*
           * get the MPI communicator, which is the union of the source and
           * destination MPI communicator
           */
          int                  ierr = 0;
          utils::mpi::MPIGroup group_source, group_dest, group_union;
          ierr = utils::mpi::MPICommGroup(this->grid->get_mpi_communicator(),
                                          &group_source);
          std::pair<bool, std::string> mpiIsSuccessAndMsg =
            utils::mpi::MPIErrIsSuccessAndMsg(ierr);
          DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                               "MPI Error:" + mpiIsSuccessAndMsg.second);
          ierr = utils::mpi::MPICommGroup(dest.grid->get_mpi_communicator(),
                                          &group_dest);
          mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(ierr);
          DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                               "MPI Error:" + mpiIsSuccessAndMsg.second);
          ierr =
            utils::mpi::MPIGroupUnion(group_source, group_dest, &group_union);
          mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(ierr);
          DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                               "MPI Error:" + mpiIsSuccessAndMsg.second);
          utils::mpi::MPIComm mpi_communicator_union;

          // to create a communicator representing the union of the source
          // and destination MPI communicator we need a communicator that
          // is guaranteed to contain all desired processes -- i.e.,
          // MPI_COMM_WORLD. on the other hand, as documented in the MPI
          // standard, MPI_Comm_create_group is not collective on all
          // processes in the first argument, but instead is collective on
          // only those processes listed in the group. in other words,
          // there is really no harm in passing MPI_COMM_WORLD as the
          // first argument, even if the program we are currently running
          // and that is calling this function only works on a subset of
          // processes. the same holds for the wrapper/fallback we are using
          // here.

          const int mpi_tag =
            static_cast<int>(utils::mpi::MPITags::SCALAPACK_COPY_TO2);
          ierr = utils::mpi::MPICommCreateGroup(utils::mpi::MPICommWorld,
                                                group_union,
                                                mpi_tag,
                                                &mpi_communicator_union);
          mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(ierr);
          DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                               "MPI Error:" + mpiIsSuccessAndMsg.second);

          /*
           * The routine pgemr2d requires a BLACS context resembling at least
           * the union of process grids described by the BLACS contexts of
           * matrix A and
           * B
           */
          int union_blacs_context = Csys2blacs_handle(mpi_communicator_union);
          const char *order       = "Col";
          int         union_n_process_rows;
          utils::mpi::MPICommSize(mpi_communicator_union,
                                  &union_n_process_rows);
          int union_n_process_columns = 1;
          Cblacs_gridinit(&union_blacs_context,
                          order,
                          union_n_process_rows,
                          union_n_process_columns);

          const NumberType *loc_vals_source = nullptr;
          NumberType *      loc_vals_dest   = nullptr;

          if (this->grid->is_process_active() && (this->values.size() > 0))
            {
              DFTEFE_AssertWithMsg(
                this->values.size() > 0,
                "source: process is active but local matrix empty");
              loc_vals_source = this->values.data();
            }
          if (dest.grid->is_process_active() && (dest.values.size() > 0))
            {
              DFTEFE_AssertWithMsg(
                dest.values.size() > 0,
                "destination: process is active but local matrix empty");
              loc_vals_dest = dest.values.data();
            }
          pgemr2d(&n_rows,
                  &n_columns,
                  loc_vals_source,
                  &submatrix_row,
                  &submatrix_column,
                  descriptor,
                  loc_vals_dest,
                  &dest.submatrix_row,
                  &dest.submatrix_column,
                  dest.descriptor,
                  &union_blacs_context);

          Cblacs_gridexit(union_blacs_context);

          if (mpi_communicator_union != utils::mpi::MPICommNull)
            {
              ierr = utils::mpi::MPICommFree(&mpi_communicator_union);
              mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(ierr);
              DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                                   "MPI Error:" + mpiIsSuccessAndMsg.second);
            }
          ierr               = utils::mpi::MPIGroupFree(&group_source);
          mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(ierr);
          DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                               "MPI Error:" + mpiIsSuccessAndMsg.second);
          ierr               = utils::mpi::MPIGroupFree(&group_dest);
          mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(ierr);
          DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                               "MPI Error:" + mpiIsSuccessAndMsg.second);
          ierr               = utils::mpi::MPIGroupFree(&group_union);
          mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(ierr);
          DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                               "MPI Error:" + mpiIsSuccessAndMsg.second);
        }
      else
        // process is active in the process grid
        if (this->grid->is_process_active())
        dest.values = this->values;

      dest.state    = state;
      dest.property = property;
    }

    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::add(const ScaLAPACKMatrix<NumberType> &B,
                                     const NumberType                   alpha,
                                     const NumberType                   beta,
                                     const bool transpose_B)
    {
      if (transpose_B)
        {
          DFTEFE_AssertWithMsg(n_rows == B.n_columns,
                               ("Dimension mismatch between " + n_rows +
                                " and " + B.n_columns));
          DFTEFE_AssertWithMsg(n_columns == B.n_rows,
                               ("Dimension mismatch between " + n_columns +
                                " and " + B.n_rows));
          DFTEFE_AssertWithMsg(column_block_size == B.row_block_size,
                               ("Dimension mismatch between " +
                                column_block_size + " and " +
                                B.row_block_size));
          DFTEFE_AssertWithMsg(row_block_size == B.column_block_size,
                               ("Dimension mismatch between " + row_block_size +
                                " and " + B.column_block_size));
        }
      else
        {
          DFTEFE_AssertWithMsg(n_rows == B.n_rows,
                               ("Dimension mismatch between " + n_rows +
                                " and " + B.n_rows));
          DFTEFE_AssertWithMsg(n_columns == B.n_columns,
                               ("Dimension mismatch between " + n_columns +
                                " and " + B.n_columns));
          DFTEFE_AssertWithMsg(column_block_size == B.column_block_size,
                               ("Dimension mismatch between " +
                                column_block_size + " and " +
                                B.column_block_size));
          DFTEFE_AssertWithMsg(row_block_size == B.row_block_size,
                               ("Dimension mismatch between " + row_block_size +
                                " and " + B.row_block_size));
        }
      DFTEFE_AssertWithMsg(
        this->grid == B.grid,
        "The matrices A and B need to have the same process grid");

      if (this->grid->is_process_active())
        {
          char        trans_b = transpose_B ? 'T' : 'N';
          NumberType *A_loc =
            (this->values.size() > 0) ? this->values.data() : nullptr;
          const NumberType *B_loc =
            (B.values.size() > 0) ? B.values.data() : nullptr;

          pgeadd(&trans_b,
                 &n_rows,
                 &n_columns,
                 &beta,
                 B_loc,
                 &B.submatrix_row,
                 &B.submatrix_column,
                 B.descriptor,
                 &alpha,
                 A_loc,
                 &submatrix_row,
                 &submatrix_column,
                 descriptor);
        }
      state = LAPACKSupport::matrix;
    }

    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::zadd(const ScaLAPACKMatrix<NumberType> &B,
                                      const NumberType                   alpha,
                                      const NumberType                   beta,
                                      const bool conjugate_transpose_B)
    {
      if (conjugate_transpose_B)
        {
          DFTEFE_AssertWithMsg(n_rows == B.n_columns,
                               ("Dimension mismatch between " + n_rows +
                                " and " + B.n_columns));
          DFTEFE_AssertWithMsg(n_columns == B.n_rows,
                               ("Dimension mismatch between " + n_columns +
                                " and " + B.n_rows));
          DFTEFE_AssertWithMsg(column_block_size == B.row_block_size,
                               ("Dimension mismatch between " +
                                column_block_size + " and " +
                                B.row_block_size));
          DFTEFE_AssertWithMsg(row_block_size == B.column_block_size,
                               ("Dimension mismatch between " + row_block_size +
                                " and " + B.column_block_size));
        }
      else
        {
          DFTEFE_AssertWithMsg(n_rows == B.n_rows,
                               ("Dimension mismatch between " + n_rows +
                                " and " + B.n_rows));
          DFTEFE_AssertWithMsg(n_columns == B.n_columns,
                               ("Dimension mismatch between " + n_columns +
                                " and " + B.n_columns));
          DFTEFE_AssertWithMsg(column_block_size == B.column_block_size,
                               ("Dimension mismatch between " +
                                column_block_size + " and " +
                                B.column_block_size));
          DFTEFE_AssertWithMsg(row_block_size == B.row_block_size,
                               ("Dimension mismatch between " + row_block_size +
                                " and " + B.row_block_size));
        }
      DFTEFE_AssertWithMsg(
        this->grid == B.grid,
        "The matrices A and B need to have the same process grid");

      if (this->grid->is_process_active())
        {
          char trans_b =
            conjugate_transpose_B ?
              (std::is_same<NumberType, std::complex<double>>::value ? 'C' :
                                                                       'T') :
              'N';
          NumberType *A_loc =
            (this->values.size() > 0) ? this->values.data() : nullptr;
          const NumberType *B_loc =
            (B.values.size() > 0) ? B.values.data() : nullptr;

          pgeadd(&trans_b,
                 &n_rows,
                 &n_columns,
                 &beta,
                 B_loc,
                 &B.submatrix_row,
                 &B.submatrix_column,
                 B.descriptor,
                 &alpha,
                 A_loc,
                 &submatrix_row,
                 &submatrix_column,
                 descriptor);
        }
      state = LAPACKSupport::matrix;
    }

    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::copy_transposed(
      const ScaLAPACKMatrix<NumberType> &B)
    {
      add(B, 0, 1, true);
    }

    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::copy_conjugate_transposed(
      const ScaLAPACKMatrix<NumberType> &B)
    {
      zadd(B, 0, 1, true);
    }

    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::mult(const NumberType                   b,
                                      const ScaLAPACKMatrix<NumberType> &B,
                                      const NumberType                   c,
                                      ScaLAPACKMatrix<NumberType> &      C,
                                      const bool transpose_A,
                                      const bool transpose_B) const
    {
      DFTEFE_AssertWithMsg(
        this->grid == B.grid,
        "The matrices A and B need to have the same process grid");
      DFTEFE_AssertWithMsg(
        C.grid == B.grid,
        "The matrices B and C need to have the same process grid");

      // see for further info:
      // https://www.ibm.com/support/knowledgecenter/SSNR5K_4.2.0/com.ibm.cluster.pessl.v4r2.pssl100.doc/am6gr_lgemm.htm
      if (!transpose_A && !transpose_B)
        {
          DFTEFE_AssertWithMsg(this->n_columns == B.n_rows,
                               ("Dimension mismatch between " +
                                this->n_columns + " and " + B.n_rows));
          DFTEFE_AssertWithMsg(this->n_rows == C.n_rows,
                               ("Dimension mismatch between " + this->n_rows +
                                " and " + C.n_rows));
          DFTEFE_AssertWithMsg(B.n_columns == C.n_columns,
                               ("Dimension mismatch between " + B.n_columns +
                                " and " + C.n_columns));
          DFTEFE_AssertWithMsg(this->row_block_size == C.row_block_size,
                               ("Dimension mismatch between " +
                                this->row_block_size + " and " +
                                C.row_block_size));
          DFTEFE_AssertWithMsg(this->column_block_size == B.row_block_size,
                               ("Dimension mismatch between " +
                                this->column_block_size + " and " +
                                B.row_block_size));
          DFTEFE_AssertWithMsg(B.column_block_size == C.column_block_size,
                               ("Dimension mismatch between " +
                                B.column_block_size + " and " +
                                C.column_block_size));
        }
      else if (transpose_A && !transpose_B)
        {
          DFTEFE_AssertWithMsg(this->n_rows == B.n_rows,
                               ("Dimension mismatch between " + this->n_rows +
                                " and " + B.n_rows));
          DFTEFE_AssertWithMsg(this->n_columns == C.n_rows,
                               ("Dimension mismatch between " +
                                this->n_columns + " and " + C.n_rows));
          DFTEFE_AssertWithMsg(B.n_columns == C.n_columns,
                               ("Dimension mismatch between " + B.n_columns +
                                " and " + C.n_columns));
          DFTEFE_AssertWithMsg(this->column_block_size == C.row_block_size,
                               ("Dimension mismatch between " +
                                this->column_block_size + " and " +
                                C.row_block_size));
          DFTEFE_AssertWithMsg(this->row_block_size == B.row_block_size,
                               ("Dimension mismatch between " +
                                this->row_block_size + " and " +
                                B.row_block_size));
          DFTEFE_AssertWithMsg(B.column_block_size == C.column_block_size,
                               ("Dimension mismatch between " +
                                  B.column_block_size,
                                C.column_block_size));
        }
      else if (!transpose_A && transpose_B)
        {
          DFTEFE_AssertWithMsg(this->n_columns == B.n_columns,
                               ("Dimension mismatch between " +
                                this->n_columns + " and " + B.n_columns));
          DFTEFE_AssertWithMsg(this->n_rows == C.n_rows,
                               ("Dimension mismatch between " + this->n_rows +
                                " and " + C.n_rows));
          DFTEFE_AssertWithMsg(B.n_rows == C.n_columns,
                               ("Dimension mismatch between " + B.n_rows +
                                " and " + C.n_columns));
          DFTEFE_AssertWithMsg(this->row_block_size == C.row_block_size,
                               ("Dimension mismatch between " +
                                this->row_block_size + " and " +
                                C.row_block_size));
          DFTEFE_AssertWithMsg(this->column_block_size == B.column_block_size,
                               ("Dimension mismatch between " +
                                this->column_block_size + " and " +
                                B.column_block_size));
          DFTEFE_AssertWithMsg(B.row_block_size == C.column_block_size,
                               ("Dimension mismatch between " +
                                B.row_block_size + " and " +
                                C.column_block_size));
        }
      else // if (transpose_A && transpose_B)
        {
          DFTEFE_AssertWithMsg(this->n_rows == B.n_columns,
                               ("Dimension mismatch between " + this->n_rows +
                                " and " + B.n_columns));
          DFTEFE_AssertWithMsg(this->n_columns == C.n_rows,
                               ("Dimension mismatch between " +
                                this->n_columns + " and " + C.n_rows));
          DFTEFE_AssertWithMsg(B.n_rows == C.n_columns,
                               ("Dimension mismatch between " + B.n_rows +
                                " and " + C.n_columns));
          DFTEFE_AssertWithMsg(this->column_block_size == C.row_block_size,
                               ("Dimension mismatch between " +
                                this->row_block_size + " and " +
                                C.row_block_size));
          DFTEFE_AssertWithMsg(this->row_block_size == B.column_block_size,
                               ("Dimension mismatch between " +
                                this->column_block_size + " and " +
                                B.row_block_size));
          DFTEFE_AssertWithMsg(B.row_block_size == C.column_block_size,
                               ("Dimension mismatch between " +
                                B.column_block_size + " and " +
                                C.column_block_size));
        }

      if (this->grid->is_process_active())
        {
          char trans_a = transpose_A ? 'T' : 'N';
          char trans_b = transpose_B ? 'T' : 'N';

          const NumberType *A_loc =
            (this->values.size() > 0) ? this->values.data() : nullptr;
          const NumberType *B_loc =
            (B.values.size() > 0) ? B.values.data() : nullptr;
          NumberType *C_loc = (C.values.size() > 0) ? C.values.data() : nullptr;
          int         m     = C.n_rows;
          int         n     = C.n_columns;
          int         k     = transpose_A ? this->n_rows : this->n_columns;

          pgemm(&trans_a,
                &trans_b,
                &m,
                &n,
                &k,
                &b,
                A_loc,
                &(this->submatrix_row),
                &(this->submatrix_column),
                this->descriptor,
                B_loc,
                &B.submatrix_row,
                &B.submatrix_column,
                B.descriptor,
                &c,
                C_loc,
                &C.submatrix_row,
                &C.submatrix_column,
                C.descriptor);
        }
      C.state = LAPACKSupport::matrix;
    }

    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::zmult(const NumberType                   b,
                                       const ScaLAPACKMatrix<NumberType> &B,
                                       const NumberType                   c,
                                       ScaLAPACKMatrix<NumberType> &      C,
                                       const bool conjugate_transpose_A,
                                       const bool conjugate_transpose_B) const
    {
      DFTEFE_AssertWithMsg(
        this->grid == B.grid,
        "The matrices A and B need to have the same process grid");
      DFTEFE_AssertWithMsg(
        C.grid == B.grid,
        "The matrices B and C need to have the same process grid");

      // see for further info:
      // https://www.ibm.com/support/knowledgecenter/SSNR5K_4.2.0/com.ibm.cluster.pessl.v4r2.pssl100.doc/am6gr_lgemm.htm
      if (!conjugate_transpose_A && !conjugate_transpose_B)
        {
          DFTEFE_AssertWithMsg(this->n_columns == B.n_rows,
                               ("Dimension mismatch between " +
                                this->n_columns + " and " + B.n_rows));
          DFTEFE_AssertWithMsg(this->n_rows == C.n_rows,
                               ("Dimension mismatch between " + this->n_rows +
                                " and " + C.n_rows));
          DFTEFE_AssertWithMsg(B.n_columns == C.n_columns,
                               ("Dimension mismatch between " + B.n_columns +
                                " and " + C.n_columns));
          DFTEFE_AssertWithMsg(this->row_block_size == C.row_block_size,
                               ("Dimension mismatch between " +
                                this->row_block_size + " and " +
                                C.row_block_size));
          DFTEFE_AssertWithMsg(this->column_block_size == B.row_block_size,
                               ("Dimension mismatch between " +
                                this->column_block_size + " and " +
                                B.row_block_size));
          DFTEFE_AssertWithMsg(B.column_block_size == C.column_block_size,
                               ("Dimension mismatch between " +
                                B.column_block_size + " and " +
                                C.column_block_size));
        }
      else if (conjugate_transpose_A && !conjugate_transpose_B)
        {
          DFTEFE_AssertWithMsg(this->n_rows == B.n_rows,
                               ("Dimension mismatch between " + this->n_rows +
                                " and " + B.n_rows));
          DFTEFE_AssertWithMsg(this->n_columns == C.n_rows,
                               ("Dimension mismatch between " +
                                this->n_columns + " and " + C.n_rows));
          DFTEFE_AssertWithMsg(B.n_columns == C.n_columns,
                               ("Dimension mismatch between " + B.n_columns +
                                " and " + C.n_columns));
          DFTEFE_AssertWithMsg(this->column_block_size == C.row_block_size,
                               ("Dimension mismatch between " +
                                this->column_block_size + " and " +
                                C.row_block_size));
          DFTEFE_AssertWithMsg(this->row_block_size == B.row_block_size,
                               ("Dimension mismatch between " +
                                this->row_block_size + " and " +
                                B.row_block_size));
          DFTEFE_AssertWithMsg(B.column_block_size == C.column_block_size,
                               ("Dimension mismatch between " +
                                B.column_block_size + " and " +
                                C.column_block_size));
        }
      else if (!conjugate_transpose_A && conjugate_transpose_B)
        {
          DFTEFE_AssertWithMsg(this->n_columns == B.n_columns,
                               ("Dimension mismatch between " +
                                this->n_columns + " and " + B.n_columns));
          DFTEFE_AssertWithMsg(this->n_rows == C.n_rows,
                               ("Dimension mismatch between " + this->n_rows +
                                " and " + C.n_rows));
          DFTEFE_AssertWithMsg(B.n_rows == C.n_columns,
                               ("Dimension mismatch between " + B.n_rows +
                                " and " + C.n_columns));
          DFTEFE_AssertWithMsg(this->row_block_size == C.row_block_size,
                               ("Dimension mismatch between " +
                                this->row_block_size + " and " +
                                C.row_block_size));
          DFTEFE_AssertWithMsg(this->column_block_size == B.column_block_size,
                               ("Dimension mismatch between " +
                                this->column_block_size + " and " +
                                B.column_block_size));
          DFTEFE_AssertWithMsg(B.row_block_size == C.column_block_size,
                               ("Dimension mismatch between " +
                                B.row_block_size + " and " +
                                C.column_block_size));
        }
      else // if (transpose_A && transpose_B)
        {
          DFTEFE_AssertWithMsg(this->n_rows == B.n_columns,
                               ("Dimension mismatch between " + this->n_rows +
                                " and " + B.n_columns));
          DFTEFE_AssertWithMsg(this->n_columns == C.n_rows,
                               ("Dimension mismatch between " +
                                this->n_columns + " and " + C.n_rows));
          DFTEFE_AssertWithMsg(B.n_rows == C.n_columns,
                               ("Dimension mismatch between " + B.n_rows +
                                " and " + C.n_columns));
          DFTEFE_AssertWithMsg(this->column_block_size == C.row_block_size,
                               ("Dimension mismatch between " +
                                this->row_block_size + " and " +
                                C.row_block_size));
          DFTEFE_AssertWithMsg(this->row_block_size == B.column_block_size,
                               ("Dimension mismatch between " +
                                this->column_block_size + " and " +
                                B.row_block_size));
          DFTEFE_AssertWithMsg(B.row_block_size == C.column_block_size,
                               ("Dimension mismatch between " +
                                B.column_block_size + " and " +
                                C.column_block_size));
        }

      if (this->grid->is_process_active())
        {
          char trans_a =
            conjugate_transpose_A ?
              (std::is_same<NumberType, std::complex<double>>::value ? 'C' :
                                                                       'T') :
              'N';
          char trans_b =
            conjugate_transpose_B ?
              (std::is_same<NumberType, std::complex<double>>::value ? 'C' :
                                                                       'T') :
              'N';

          const NumberType *A_loc =
            (this->values.size() > 0) ? this->values.data() : nullptr;
          const NumberType *B_loc =
            (B.values.size() > 0) ? B.values.data() : nullptr;
          NumberType *C_loc = (C.values.size() > 0) ? C.values.data() : nullptr;
          int         m     = C.n_rows;
          int         n     = C.n_columns;
          int k = conjugate_transpose_A ? this->n_rows : this->n_columns;

          pgemm(&trans_a,
                &trans_b,
                &m,
                &n,
                &k,
                &b,
                A_loc,
                &(this->submatrix_row),
                &(this->submatrix_column),
                this->descriptor,
                B_loc,
                &B.submatrix_row,
                &B.submatrix_column,
                B.descriptor,
                &c,
                C_loc,
                &C.submatrix_row,
                &C.submatrix_column,
                C.descriptor);
        }
      C.state = LAPACKSupport::matrix;
    }

    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::mmult(ScaLAPACKMatrix<NumberType> &      C,
                                       const ScaLAPACKMatrix<NumberType> &B,
                                       const bool adding) const
    {
      if (adding)
        mult(1., B, 1., C, false, false);
      else
        mult(1., B, 0, C, false, false);
    }



    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::Tmmult(ScaLAPACKMatrix<NumberType> &      C,
                                        const ScaLAPACKMatrix<NumberType> &B,
                                        const bool adding) const
    {
      if (adding)
        mult(1., B, 1., C, true, false);
      else
        mult(1., B, 0, C, true, false);
    }



    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::mTmult(ScaLAPACKMatrix<NumberType> &      C,
                                        const ScaLAPACKMatrix<NumberType> &B,
                                        const bool adding) const
    {
      if (adding)
        mult(1., B, 1., C, false, true);
      else
        mult(1., B, 0, C, false, true);
    }



    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::TmTmult(ScaLAPACKMatrix<NumberType> &      C,
                                         const ScaLAPACKMatrix<NumberType> &B,
                                         const bool adding) const
    {
      if (adding)
        mult(1., B, 1., C, true, true);
      else
        mult(1., B, 0, C, true, true);
    }


    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::zmmult(ScaLAPACKMatrix<NumberType> &      C,
                                        const ScaLAPACKMatrix<NumberType> &B,
                                        const bool adding) const
    {
      if (adding)
        zmult(1., B, 1., C, false, false);
      else
        zmult(1., B, 0, C, false, false);
    }



    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::zCmmult(ScaLAPACKMatrix<NumberType> &      C,
                                         const ScaLAPACKMatrix<NumberType> &B,
                                         const bool adding) const
    {
      if (adding)
        zmult(1., B, 1., C, true, false);
      else
        zmult(1., B, 0, C, true, false);
    }



    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::zmCmult(ScaLAPACKMatrix<NumberType> &      C,
                                         const ScaLAPACKMatrix<NumberType> &B,
                                         const bool adding) const
    {
      if (adding)
        zmult(1., B, 1., C, false, true);
      else
        zmult(1., B, 0, C, false, true);
    }



    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::zCmCmult(ScaLAPACKMatrix<NumberType> &      C,
                                          const ScaLAPACKMatrix<NumberType> &B,
                                          const bool adding) const
    {
      if (adding)
        zmult(1., B, 1., C, true, true);
      else
        zmult(1., B, 0, C, true, true);
    }

    template <typename NumberType>
    ScalapackError
    ScaLAPACKMatrix<NumberType>::compute_cholesky_factorization()
    {
      ScalapackError returnVal;
      DFTEFE_AssertWithMsg(
        n_columns == n_rows && property == LAPACKSupport::Property::hermitian,
        "Cholesky factorization can be applied to hermitian matrices only.");
      DFTEFE_AssertWithMsg(
        state == LAPACKSupport::matrix,
        "Matrix has to be in Matrix state before calling this function.");

      if (grid->is_process_active())
        {
          int         info  = 0;
          NumberType *A_loc = this->values.data();
          // pdpotrf_(&uplo,&n_columns,A_loc,&submatrix_row,&submatrix_column,descriptor,&info);
          ppotrf(&uplo,
                 &n_columns,
                 A_loc,
                 &submatrix_row,
                 &submatrix_column,
                 descriptor,
                 &info);
          if (info != 0)
            {
              returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                ScalapackErrorCode::FAILED_CHOLESKY_FACTORIZATION);
              returnVal.msg += std::to_string(info) + " .";
            }
          else
            returnVal =
              ScalapackErrorMsg::isSuccessAndMsg(ScalapackErrorCode::SUCCESS);
        }
      state    = LAPACKSupport::cholesky;
      property = (uplo == 'L' ? LAPACKSupport::lower_triangular :
                                LAPACKSupport::upper_triangular);

      return returnVal;
    }

    template <typename NumberType>
    ScalapackError
    ScaLAPACKMatrix<NumberType>::compute_lu_factorization()
    {
      ScalapackError returnVal;
      DFTEFE_AssertWithMsg(
        state == LAPACKSupport::matrix,
        "Matrix has to be in Matrix state before calling this function.");

      if (grid->is_process_active())
        {
          int         info  = 0;
          NumberType *A_loc = this->values.data();

          const int iarow = indxg2p_(&submatrix_row,
                                     &row_block_size,
                                     &(grid->get_this_process_row()),
                                     &first_process_row,
                                     &(grid->get_process_grid_rows()));
          const int mp    = numroc_(&n_rows,
                                 &row_block_size,
                                 &(grid->get_this_process_row()),
                                 &iarow,
                                 &(grid->get_process_grid_rows()));
          ipiv.resize(mp + row_block_size);

          pgetrf(&n_rows,
                 &n_columns,
                 A_loc,
                 &submatrix_row,
                 &submatrix_column,
                 descriptor,
                 ipiv.data(),
                 &info);
          if (info != 0)
            {
              returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                ScalapackErrorCode::FAILED_LU_FACTORIZATION);
              returnVal.msg += std::to_string(info) + " .";
            }
          else
            returnVal =
              ScalapackErrorMsg::isSuccessAndMsg(ScalapackErrorCode::SUCCESS);
        }
      state    = LAPACKSupport::State::lu;
      property = LAPACKSupport::Property::general;
      return returnVal;
    }

    template <typename NumberType>
    ScalapackError
    ScaLAPACKMatrix<NumberType>::invert()
    {
      ScalapackError returnVal;
      // Check whether matrix is hermitian and save flag.
      // If a Cholesky factorization has been applied previously,
      // the original matrix was hermitian.
      const bool is_hermitian = (property == LAPACKSupport::hermitian ||
                                 state == LAPACKSupport::State::cholesky);

      // Check whether matrix is triangular and is in an unfactorized state.
      const bool is_triangular =
        (property == LAPACKSupport::upper_triangular ||
         property == LAPACKSupport::lower_triangular) &&
        (state == LAPACKSupport::State::matrix ||
         state == LAPACKSupport::State::inverse_matrix);

      if (is_triangular)
        {
          if (grid->is_process_active())
            {
              const char uploTriangular =
                property == LAPACKSupport::upper_triangular ? 'U' : 'L';
              const char  diag  = 'N';
              int         info  = 0;
              NumberType *A_loc = this->values.data();
              ptrtri(&uploTriangular,
                     &diag,
                     &n_columns,
                     A_loc,
                     &submatrix_row,
                     &submatrix_column,
                     descriptor,
                     &info);
              if (info != 0)
                {
                  returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                    ScalapackErrorCode::FAILED_MATRIX_INVERT);
                  returnVal.msg += std::to_string(info) + " .";
                }
              else
                returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                  ScalapackErrorCode::SUCCESS);
              // The inversion is stored in the same part as the triangular
              // matrix, so we don't need to re-set the property here.
            }
        }
      else
        {
          // Matrix is neither in Cholesky nor LU state.
          // Compute the required factorizations based on the property of the
          // matrix.
          if (!(state == LAPACKSupport::State::lu ||
                state == LAPACKSupport::State::cholesky))
            {
              if (is_hermitian)
                compute_cholesky_factorization();
              else
                compute_lu_factorization();
            }
          if (grid->is_process_active())
            {
              int         info  = 0;
              NumberType *A_loc = this->values.data();

              if (is_hermitian)
                {
                  ppotri(&uplo,
                         &n_columns,
                         A_loc,
                         &submatrix_row,
                         &submatrix_column,
                         descriptor,
                         &info);
                  if (info != 0)
                    {
                      returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                        ScalapackErrorCode::FAILED_MATRIX_INVERT);
                      returnVal.msg += std::to_string(info) + " .";
                    }
                  else
                    returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                      ScalapackErrorCode::SUCCESS);
                  property = LAPACKSupport::Property::hermitian;
                }
              else
                {
                  int lwork = -1, liwork = -1;
                  work.resize(1);
                  iwork.resize(1);

                  pgetri(&n_columns,
                         A_loc,
                         &submatrix_row,
                         &submatrix_column,
                         descriptor,
                         ipiv.data(),
                         work.data(),
                         &lwork,
                         iwork.data(),
                         &liwork,
                         &info);

                  if (info != 0)
                    {
                      returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                        ScalapackErrorCode::FAILED_MATRIX_INVERT);
                      returnVal.msg += std::to_string(info) + " .";
                    }
                  else
                    returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                      ScalapackErrorCode::SUCCESS);
                  lwork  = lworkFromWork(work);
                  liwork = iwork[0];
                  work.resize(lwork);
                  iwork.resize(liwork);

                  pgetri(&n_columns,
                         A_loc,
                         &submatrix_row,
                         &submatrix_column,
                         descriptor,
                         ipiv.data(),
                         work.data(),
                         &lwork,
                         iwork.data(),
                         &liwork,
                         &info);

                  if (info != 0)
                    {
                      returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                        ScalapackErrorCode::FAILED_MATRIX_INVERT);
                      returnVal.msg += std::to_string(info) + " .";
                    }
                  else
                    returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                      ScalapackErrorCode::SUCCESS);
                }
            }
        }
      state = LAPACKSupport::State::inverse_matrix;
      return returnVal;
    }

    template <typename NumberType>
    std::vector<double>
    ScaLAPACKMatrix<NumberType>::eigenpairs_hermitian_by_index(
      const std::pair<unsigned int, unsigned int> &index_limits,
      const bool                                   compute_eigenvectors)
    {
      // check validity of index limits
      DFTEFE_Assert(index_limits.first < n_rows);
      DFTEFE_Assert(index_limits.second < n_rows);

      std::pair<unsigned int, unsigned int> idx =
        std::make_pair(std::min(index_limits.first, index_limits.second),
                       std::max(index_limits.first, index_limits.second));

      // compute all eigenvalues/eigenvectors
      if (idx.first == 0 && idx.second == static_cast<unsigned int>(n_rows - 1))
        return eigenpairs_hermitian(compute_eigenvectors);
      else
        return eigenpairs_hermitian(compute_eigenvectors, idx);
    }


    template <typename NumberType>
    std::vector<double>
    ScaLAPACKMatrix<NumberType>::eigenpairs_hermitian(
      const bool                                   compute_eigenvectors,
      const std::pair<unsigned int, unsigned int> &eigenvalue_idx,
      const std::pair<double, double> &            eigenvalue_limits)
    {
      ScalapackError returnVal;
      DFTEFE_AssertWithMsg(
        state == LAPACKSupport::matrix,
        "Matrix has to be in Matrix state before calling this function.");
      DFTEFE_AssertWithMsg(property == LAPACKSupport::hermitian,
                           "Matrix has to be hermitian for this operation.");

      std::lock_guard<std::mutex> lock(std::mutex);

      const bool use_values = (std::isnan(eigenvalue_limits.first) ||
                               std::isnan(eigenvalue_limits.second)) ?
                                false :
                                true;
      const bool use_indices =
        ((eigenvalue_idx.first == std::numeric_limits<unsigned int>::max()) ||
         (eigenvalue_idx.second == std::numeric_limits<unsigned int>::max())) ?
          false :
          true;

      DFTEFE_AssertWithMsg(
        !(use_values && use_indices),
        "Prescribing both the index and value range for the eigenvalues is ambiguous");

      // if computation of eigenvectors is not required use a sufficiently small
      // distributed matrix
      std::unique_ptr<ScaLAPACKMatrix<NumberType>> eigenvectors =
        compute_eigenvectors ?
          std::make_unique<ScaLAPACKMatrix<NumberType>>(n_rows,
                                                        grid,
                                                        row_block_size) :
          std::make_unique<ScaLAPACKMatrix<NumberType>>(
            grid->get_process_grid_rows(),
            grid->get_process_grid_columns(),
            grid,
            1,
            1);

      eigenvectors->property = property;
      // number of eigenvalues to be returned from psyevx; upon successful exit
      // ev contains the m seclected eigenvalues in ascending order set to all
      // eigenvaleus in case we will be using psyev.
      int                 m = n_rows;
      std::vector<double> ev(n_rows);

      if (grid->is_process_active())
        {
          int info = 0;
          /*
           * for jobz==N only eigenvalues are computed, for jobz='V' also the
           * eigenvectors of the matrix are computed
           */
          char jobz  = compute_eigenvectors ? 'V' : 'N';
          char range = 'A';
          // default value is to compute all eigenvalues and optionally
          // eigenvectors
          bool   all_eigenpairs = true;
          double vl             = 0.0;
          double vu             = 0.0;
          int    il = 1, iu = 1;
          // number of eigenvectors to be returned;
          // upon successful exit the first m=nz columns contain the selected
          // eigenvectors (only if jobz=='V')
          int    nz     = 0;
          double abstol = 0.0;

          // orfac decides which eigenvectors should be reorthogonalized
          // see
          // http://www.netlib.org/scalapack/explore-html/df/d1a/pdsyevx_8f_source.html
          // for explanation to keeps simple no reorthogonalized will be done by
          // setting orfac to 0
          double orfac = 0;
          // contains the indices of eigenvectors that failed to converge
          std::vector<int> ifail;
          // This array contains indices of eigenvectors corresponding to
          // a cluster of eigenvalues that could not be reorthogonalized
          // due to insufficient workspace
          // see
          // http://www.netlib.org/scalapack/explore-html/df/d1a/pdsyevx_8f_source.html
          // for explanation
          std::vector<int> iclustr;
          // This array contains the gap between eigenvalues whose
          // eigenvectors could not be reorthogonalized.
          // see
          // http://www.netlib.org/scalapack/explore-html/df/d1a/pdsyevx_8f_source.html
          // for explanation
          std::vector<double> gap(n_local_rows * n_local_columns);

          // index range for eigenvalues is not specified
          if (!use_indices)
            {
              // interval for eigenvalues is not specified and consequently all
              // eigenvalues/eigenpairs will be computed
              if (!use_values)
                {
                  range          = 'A';
                  all_eigenpairs = true;
                }
              else
                {
                  range          = 'V';
                  all_eigenpairs = false;
                  vl =
                    std::min(eigenvalue_limits.first, eigenvalue_limits.second);
                  vu =
                    std::max(eigenvalue_limits.first, eigenvalue_limits.second);
                }
            }
          else
            {
              range          = 'I';
              all_eigenpairs = false;
              // as Fortran starts counting/indexing from 1 unlike C/C++, where
              // it starts from 0
              il = std::min(eigenvalue_idx.first, eigenvalue_idx.second) + 1;
              iu = std::max(eigenvalue_idx.first, eigenvalue_idx.second) + 1;
            }
          NumberType *A_loc = this->values.data();
          /*
           * by setting lwork to -1 a workspace query for optimal length of work
           * is performed
           */
          int         lwork  = -1;
          int         liwork = -1;
          NumberType *eigenvectors_loc =
            (compute_eigenvectors ? eigenvectors->values.data() : nullptr);
          work.resize(1);
          iwork.resize(1);

          if (all_eigenpairs)
            {
              psyev(&jobz,
                    &uplo,
                    &n_rows,
                    A_loc,
                    &submatrix_row,
                    &submatrix_column,
                    descriptor,
                    ev.data(),
                    eigenvectors_loc,
                    &eigenvectors->submatrix_row,
                    &eigenvectors->submatrix_column,
                    eigenvectors->descriptor,
                    work.data(),
                    &lwork,
                    &info);
              if (info != 0)
                {
                  returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                    ScalapackErrorCode::FAILED_STANDARD_HERMITIAN_EIGENPROBLEM);
                  returnVal.msg += std::to_string(info) + " .";
                }
              else
                returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                  ScalapackErrorCode::SUCCESS);
            }
          else
            {
              char cmach = compute_eigenvectors ? 'U' : 'S';
              plamch(&(this->grid->get_blacs_context()), &cmach, abstol);
              abstol *= 2;
              ifail.resize(n_rows);
              iclustr.resize(2 * grid->get_process_grid_rows() *
                             grid->get_process_grid_columns());
              gap.resize(grid->get_process_grid_rows() *
                         grid->get_process_grid_columns());

              psyevx(&jobz,
                     &range,
                     &uplo,
                     &n_rows,
                     A_loc,
                     &submatrix_row,
                     &submatrix_column,
                     descriptor,
                     &vl,
                     &vu,
                     &il,
                     &iu,
                     &abstol,
                     &m,
                     &nz,
                     ev.data(),
                     &orfac,
                     eigenvectors_loc,
                     &eigenvectors->submatrix_row,
                     &eigenvectors->submatrix_column,
                     eigenvectors->descriptor,
                     work.data(),
                     &lwork,
                     iwork.data(),
                     &liwork,
                     ifail.data(),
                     iclustr.data(),
                     gap.data(),
                     &info);
              if (info != 0)
                {
                  returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                    ScalapackErrorCode::FAILED_STANDARD_HERMITIAN_EIGENPROBLEM);
                  returnVal.msg += std::to_string(info) + " .";
                }
              else
                returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                  ScalapackErrorCode::SUCCESS);
            }
          lwork = lworkFromWork(work);
          work.resize(lwork);

          if (all_eigenpairs)
            {
              psyev(&jobz,
                    &uplo,
                    &n_rows,
                    A_loc,
                    &submatrix_row,
                    &submatrix_column,
                    descriptor,
                    ev.data(),
                    eigenvectors_loc,
                    &eigenvectors->submatrix_row,
                    &eigenvectors->submatrix_column,
                    eigenvectors->descriptor,
                    work.data(),
                    &lwork,
                    &info);

              if (info != 0)
                {
                  returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                    ScalapackErrorCode::FAILED_STANDARD_HERMITIAN_EIGENPROBLEM);
                  returnVal.msg += std::to_string(info) + " .";
                }
              else
                returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                  ScalapackErrorCode::SUCCESS);
            }
          else
            {
              liwork = iwork[0];
              DFTEFE_Assert(liwork > 0);
              iwork.resize(liwork);

              psyevx(&jobz,
                     &range,
                     &uplo,
                     &n_rows,
                     A_loc,
                     &submatrix_row,
                     &submatrix_column,
                     descriptor,
                     &vl,
                     &vu,
                     &il,
                     &iu,
                     &abstol,
                     &m,
                     &nz,
                     ev.data(),
                     &orfac,
                     eigenvectors_loc,
                     &eigenvectors->submatrix_row,
                     &eigenvectors->submatrix_column,
                     eigenvectors->descriptor,
                     work.data(),
                     &lwork,
                     iwork.data(),
                     &liwork,
                     ifail.data(),
                     iclustr.data(),
                     gap.data(),
                     &info);

              if (info != 0)
                {
                  returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                    ScalapackErrorCode::FAILED_STANDARD_HERMITIAN_EIGENPROBLEM);
                  returnVal.msg += std::to_string(info) + " .";
                }
              else
                returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                  ScalapackErrorCode::SUCCESS);
            }
          // if eigenvectors are queried copy eigenvectors to original matrix
          // as the temporary matrix eigenvectors has identical dimensions and
          // block-cyclic distribution we simply swap the local array
          if (compute_eigenvectors)
            this->values.swap(eigenvectors->values);

          // adapt the size of ev to fit m upon return
          while (ev.size() > static_cast<size_type>(m))
            ev.pop_back();
        }
      /*
       * send number of computed eigenvalues to inactive processes
       */
      grid->send_to_inactive(&m, 1);

      /*
       * inactive processes have to resize array of eigenvalues
       */
      if (!grid->is_process_active())
        ev.resize(m);
      /*
       * send the eigenvalues to processors not being part of the process grid
       */
      grid->send_to_inactive(ev.data(), ev.size());

      /*
       * if only eigenvalues are queried the content of the matrix will be
       * destroyed if the eigenpairs are queried matrix A on exit stores the
       * eigenvectors in the columns
       */
      if (compute_eigenvectors)
        {
          property = LAPACKSupport::Property::general;
          state    = LAPACKSupport::eigenvalues;
        }
      else
        state = LAPACKSupport::unusable;

      return ev;
    }


    template <typename NumberType>
    std::vector<double>
    ScaLAPACKMatrix<NumberType>::eigenpairs_hermitian_by_index_MRRR(
      const std::pair<unsigned int, unsigned int> &index_limits,
      const bool                                   compute_eigenvectors)
    {
      ScalapackError returnVal;
      // Check validity of index limits.
      DFTEFE_Assert(index_limits.first < static_cast<unsigned int>(n_rows));
      DFTEFE_Assert(index_limits.second < static_cast<unsigned int>(n_rows));

      const std::pair<unsigned int, unsigned int> idx =
        std::make_pair(std::min(index_limits.first, index_limits.second),
                       std::max(index_limits.first, index_limits.second));

      // Compute all eigenvalues/eigenvectors.
      if (idx.first == 0 && idx.second == static_cast<unsigned int>(n_rows - 1))
        return eigenpairs_hermitian_MRRR(compute_eigenvectors);
      else
        return eigenpairs_hermitian_MRRR(compute_eigenvectors, idx);
    }


    template <typename NumberType>
    std::vector<double>
    ScaLAPACKMatrix<NumberType>::eigenpairs_hermitian_MRRR(
      const bool                                   compute_eigenvectors,
      const std::pair<unsigned int, unsigned int> &eigenvalue_idx,
      const std::pair<double, double> &            eigenvalue_limits)
    {
      ScalapackError returnVal;
      DFTEFE_AssertWithMsg(
        state == LAPACKSupport::matrix,
        "Matrix has to be in Matrix state before calling this function.");
      DFTEFE_AssertWithMsg(property == LAPACKSupport::hermitian,
                           "Matrix has to be hermitian for this operation.");

      std::lock_guard<std::mutex> lock(std::mutex);

      const bool use_values = (std::isnan(eigenvalue_limits.first) ||
                               std::isnan(eigenvalue_limits.second)) ?
                                false :
                                true;
      const bool use_indices =
        ((eigenvalue_idx.first == std::numeric_limits<unsigned int>::max()) ||
         (eigenvalue_idx.second == std::numeric_limits<unsigned int>::max())) ?
          false :
          true;

      DFTEFE_AssertWithMsg(
        !(use_values && use_indices),
        "Prescribing both the index and value range for the eigenvalues is ambiguous");

      // If computation of eigenvectors is not required, use a sufficiently
      // small distributed matrix.
      std::unique_ptr<ScaLAPACKMatrix<NumberType>> eigenvectors =
        compute_eigenvectors ?
          std::make_unique<ScaLAPACKMatrix<NumberType>>(n_rows,
                                                        grid,
                                                        row_block_size) :
          std::make_unique<ScaLAPACKMatrix<NumberType>>(
            grid->get_process_grid_rows(),
            grid->get_process_grid_columns(),
            grid,
            1,
            1);

      eigenvectors->property = property;
      // Number of eigenvalues to be returned from psyevr; upon successful exit
      // ev contains the m seclected eigenvalues in ascending order.
      int                 m = n_rows;
      std::vector<double> ev(n_rows);

      // Number of eigenvectors to be returned;
      // Upon successful exit the first m=nz columns contain the selected
      // eigenvectors (only if jobz=='V').
      int nz = 0;

      if (grid->is_process_active())
        {
          int info = 0;
          /*
           * For jobz==N only eigenvalues are computed, for jobz='V' also the
           * eigenvectors of the matrix are computed.
           */
          char jobz = compute_eigenvectors ? 'V' : 'N';
          // Default value is to compute all eigenvalues and optionally
          // eigenvectors.
          char   range = 'A';
          double vl    = 0.0;
          double vu    = 0.0;
          int    il = 1, iu = 1;

          // Index range for eigenvalues is not specified.
          if (!use_indices)
            {
              // Interval for eigenvalues is not specified and consequently all
              // eigenvalues/eigenpairs will be computed.
              if (!use_values)
                {
                  range = 'A';
                }
              else
                {
                  range = 'V';
                  vl =
                    std::min(eigenvalue_limits.first, eigenvalue_limits.second);
                  vu =
                    std::max(eigenvalue_limits.first, eigenvalue_limits.second);
                }
            }
          else
            {
              range = 'I';
              // As Fortran starts counting/indexing from 1 unlike C/C++, where
              // it starts from 0.
              il = std::min(eigenvalue_idx.first, eigenvalue_idx.second) + 1;
              iu = std::max(eigenvalue_idx.first, eigenvalue_idx.second) + 1;
            }
          NumberType *A_loc = this->values.data();

          /*
           * By setting lwork to -1 a workspace query for optimal length of work
           * is performed.
           */
          int         lwork  = -1;
          int         liwork = -1;
          NumberType *eigenvectors_loc =
            (compute_eigenvectors ? eigenvectors->values.data() : nullptr);
          work.resize(1);
          iwork.resize(1);

          psyevr(&jobz,
                 &range,
                 &uplo,
                 &n_rows,
                 A_loc,
                 &submatrix_row,
                 &submatrix_column,
                 descriptor,
                 &vl,
                 &vu,
                 &il,
                 &iu,
                 &m,
                 &nz,
                 ev.data(),
                 eigenvectors_loc,
                 &eigenvectors->submatrix_row,
                 &eigenvectors->submatrix_column,
                 eigenvectors->descriptor,
                 work.data(),
                 &lwork,
                 iwork.data(),
                 &liwork,
                 &info);
          if (info != 0)
            {
              returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                ScalapackErrorCode::
                  FAILED_STANDARD_HERMITIAN_EIGENPROBLEM_MRRR);
              returnVal.msg += std::to_string(info) + " .";
            }
          else
            returnVal =
              ScalapackErrorMsg::isSuccessAndMsg(ScalapackErrorCode::SUCCESS);

          lwork = lworkFromWork(work);
          work.resize(lwork);
          liwork = iwork[0];
          iwork.resize(liwork);

          psyevr(&jobz,
                 &range,
                 &uplo,
                 &n_rows,
                 A_loc,
                 &submatrix_row,
                 &submatrix_column,
                 descriptor,
                 &vl,
                 &vu,
                 &il,
                 &iu,
                 &m,
                 &nz,
                 ev.data(),
                 eigenvectors_loc,
                 &eigenvectors->submatrix_row,
                 &eigenvectors->submatrix_column,
                 eigenvectors->descriptor,
                 work.data(),
                 &lwork,
                 iwork.data(),
                 &liwork,
                 &info);

          if (info != 0)
            {
              returnVal = ScalapackErrorMsg::isSuccessAndMsg(
                ScalapackErrorCode::
                  FAILED_STANDARD_HERMITIAN_EIGENPROBLEM_MRRR);
              returnVal.msg += std::to_string(info) + " .";
            }
          else
            returnVal =
              ScalapackErrorMsg::isSuccessAndMsg(ScalapackErrorCode::SUCCESS);

          if (compute_eigenvectors)
            DFTEFE_AssertWithMsg(
              m == nz,
              "psyevr failed to compute all eigenvectors for the selected eigenvalues");

          // If eigenvectors are queried, copy eigenvectors to original matrix.
          // As the temporary matrix eigenvectors has identical dimensions and
          // block-cyclic distribution we simply swap the local array.
          if (compute_eigenvectors)
            this->values.swap(eigenvectors->values);

          // Adapt the size of ev to fit m upon return.
          while (ev.size() > static_cast<size_type>(m))
            ev.pop_back();
        }
      /*
       * Send number of computed eigenvalues to inactive processes.
       */
      grid->send_to_inactive(&m, 1);

      /*
       * Inactive processes have to resize array of eigenvalues.
       */
      if (!grid->is_process_active())
        ev.resize(m);
      /*
       * Send the eigenvalues to processors not being part of the process grid.
       */
      grid->send_to_inactive(ev.data(), ev.size());

      /*
       * If only eigenvalues are queried, the content of the matrix will be
       * destroyed. If the eigenpairs are queried, matrix A on exit stores the
       * eigenvectors in the columns.
       */
      if (compute_eigenvectors)
        {
          property = LAPACKSupport::Property::general;
          state    = LAPACKSupport::eigenvalues;
        }
      else
        state = LAPACKSupport::unusable;

      return ev;
    }

    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::scale_columns(
      const std::vector<NumberType> &factors)
    {
      if (grid->is_process_active())
        {
          DFTEFE_AssertWithMsg(this->n() == factors.size(),
                               ("Dimension mismatch between " + this->n() +
                                " and " + factors.size()));

          for (unsigned int i = 0; i < this->local_n(); ++i)
            {
              const NumberType s = factors[this->global_column(i)];
              for (unsigned int j = 0; j < this->local_m(); ++j)
                this->local_el(j, i) *= s;
            }
        }
    }

    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::scale_rows(
      const std::vector<NumberType> &factors)
    {
      if (grid->is_process_active())
        {
          DFTEFE_AssertWithMsg(this->m() == factors.size(),
                               ("Dimension mismatch between " + this->m() +
                                " and " + factors.size()));
          for (unsigned int i = 0; i < this->local_m(); ++i)
            {
              const NumberType s = factors[this->global_row(i)];
              for (unsigned int j = 0; j < this->local_n(); ++j)
                this->local_el(i, j) *= s;
            }
        }
    }

    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::scale_columns_realfactors(
      const std::vector<double> &factors)
    {
      if (grid->is_process_active())
        {
          DFTEFE_AssertWithMsg(this->n() == factors.size(),
                               ("Dimension mismatch between " + this->n() +
                                " and " + factors.size()));

          for (unsigned int i = 0; i < this->local_n(); ++i)
            {
              const NumberType s = NumberType(factors[this->global_column(i)]);
              for (unsigned int j = 0; j < this->local_m(); ++j)
                this->local_el(j, i) *= s;
            }
        }
    }

    template <typename NumberType>
    void
    ScaLAPACKMatrix<NumberType>::scale_rows_realfactors(
      const std::vector<double> &factors)
    {
      if (grid->is_process_active())
        {
          DFTEFE_AssertWithMsg(this->m() == factors.size(),
                               ("Dimension mismatch between " + this->m() +
                                " and " + factors.size()));
          for (unsigned int i = 0; i < this->local_m(); ++i)
            {
              const NumberType s = NumberType(factors[this->global_row(i)]);
              for (unsigned int j = 0; j < this->local_n(); ++j)
                this->local_el(i, j) *= s;
            }
        }
    }

    template class ScaLAPACKMatrix<double>;
    template class ScaLAPACKMatrix<std::complex<double>>;
  } // namespace linearAlgebra
} // namespace dftefe
