#ifndef dftefeTriangulationDealii_h
#define dftefeTriangulationDealii_h

namespace dftefe {

namespace basis
{

<template unsigned int dim>
class TriangulationDealii : public  TriangulationBase
{
    public :
    TriangulationDealii();
    ~TriangulationDealii();
    
     TriangulationBase & getTriangulationObject() ;
     void     refine_global (const unsigned int times=1) ;
    
     void     coarsen_global (const unsigned int times=1) ;
    
     void     clear_user_flags () ;
    
     void     execute_coarsening_and_refinement () ;
     unsigned int     n_locally_owned_active_cells () const  ;
     dealii::types::global_cell_index     n_global_active_cells () const  ;
     dealii::types::subdomain_id     locally_owned_subdomain () const ;
     std::vector< types::boundary_id >     get_boundary_ids () const ;

      TriaCellBase  &   begin_active (const unsigned int level=0) const ;
    
     TriaCellBase &     end_active (const unsigned int level) const ;
    
    
    unsigned int getDim() const ;
    
    
    private :
    
    dealii::parallel::distributed::Triangulation<dim> triangDealii;
    
    std::vector<TriaCellDealii<dim> > triaVectorCell;
    
    
    
}; // end of class TriangulationDealii

}// end of namespace basis

} // end of namespace dftefe
#endif
