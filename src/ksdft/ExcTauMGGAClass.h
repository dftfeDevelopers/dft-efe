#ifndef DFTEFE_ExcTauMGGAClass_H
#define DFTEFE_ExcTauMGGAClass_H

#include <xc.h>
#include <ksdft/ExcSSDFunctionalBaseClass.h>

namespace dftefe
{
  namespace ksdft
  {
  template <dftefe::utils::MemorySpace memorySpace>
  class ExcTauMGGAClass : public ExcSSDFunctionalBaseClass<memorySpace>
  {
  public:
    ExcTauMGGAClass(std::shared_ptr<xc_func_type> &funcXPtr,
                    std::shared_ptr<xc_func_type> &funcCPtr,
                    std::string                    XCType);

    ~ExcTauMGGAClass();

    void
    checkInputOutputDataAttributesConsistency(
      const std::vector<xcRemainderOutputDataAttributes> &outputDataAttributes)
      const override;

  private:
    void
    computeRhoTauDependentXCData(
      const std::unordered_map<
        DensityDescrAttr,
        typename ExcSSDFunctionalBaseClass<memorySpace>::AttrStorage>
        &densityAttrVals,
      const std::unordered_map<
        WfcDescrAttr,
        typename ExcSSDFunctionalBaseClass<memorySpace>::AttrStorage>
        &wfcAttrVals,
      std::unordered_map<
        xcRemainderOutputDataAttributes,
        dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::HOST>>
        &xDataOut,
      std::unordered_map<
        xcRemainderOutputDataAttributes,
        dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::HOST>>
        &cDataout) const override;

    std::shared_ptr<xc_func_type> d_funcXPtr;
    std::shared_ptr<xc_func_type> d_funcCPtr;
    std::string                   d_XCType;
  };
  } // namespace ksdft
} // namespace dftefe

#endif
