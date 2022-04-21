//
// File:      EnrichedFunctionManager.cc
// Package:   fem
//
// Finite Element Method.
//
#if defined(HAVE_CONFIG_H)
#include "dft_config.h"
#endif // HAVE_CONFIG_H

#include "EnrichedFunctionManager.h"
#include <dft/NuclearPositionsReader.h>
#include <utils/point/Point.h>

#if defined(HAVE_ALGORITHM)
#include <algorithm>
#else
#error algorithm header file not available.
#endif // HAVE_ALGORITHM
//
//
//

namespace dft {

    
  //
  // Constructor.
  //
  EnrichedFunctionManager::EnrichedFunctionManager()
  {
    
    //
    // 
    //
    return; 

  }


  EnrichedFunctionManager::~EnrichedFunctionManager()
  {
    
     
    return;
  }  

}
