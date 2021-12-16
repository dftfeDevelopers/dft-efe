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
 * @author Sambit Das.
 */

#ifdef DFTEFE_WITH_DEVICE
#  ifndef dftefeDeviceLAContextsSingleton_h
#    define dftefeDeviceLAContextsSingleton_h

#    ifdef DFTEFE_WITH_DEVICE_CUDA
#      include <linearAlgebra/DeviceLATypeConfig.cuh>
#    endif

namespace dftefe
{
  namespace linearAlgebra
  {
    class DeviceLAContextsSingleton
    {
    protected:
      static DeviceLAContextsSingleton *d_instPtr;

    public:
      static DeviceLAContextsSingleton *
      getInstance();

      void
      createDeviceBlasHandle();

      void
      destroyDeviceBlasHandle();

      deviceBlasHandleType &
      getDeviceBlasHandle();

    private:
      DeviceLAContextsSingleton()
      {}

      deviceBlasHandleType d_blasHandle;
    };
  } // namespace linearAlgebra

} // namespace dftefe

#  endif
#endif // DFTEFE_WITH_DEVICE
