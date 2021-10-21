#ifndef dftefeTriangulationBase_h
#define dftefeTriangulationBase_h


namespace dftefe {

namespace basis
{
/**
 * @brief An abstract class for the triangulation class. The derived class specialises this class to dealii and otehr specialisations if required.
 **/
  class TriangulationBase
{
    public :
    TriangulationBase();
    ~TriangulationBase();
    
    
    virtual TriangulationBase & getTriangulationObject() = 0;
    virtual void     refine_global (const unsigned int times=1) = 0;
    
    virtual void     coarsen_global (const unsigned int times=1) = 0;
    
    virtual void     clear_user_flags () = 0;
    
    virtual void     execute_coarsening_and_refinement () = 0;
    virtual unsigned int     n_locally_owned_active_cells () const = 0 ;
    virtual dealii::types::global_cell_index     n_global_active_cells () const = 0 ;
    virtual dealii::types::subdomain_id     locally_owned_subdomain () const = 0;
    virtual std::vector< types::boundary_id >     get_boundary_ids () const = 0;

    virtual  TriaCellBase  &   begin_active () const = 0;
    
    virtual TriaCellBase &     end_active () const = 0;
    
    
    virtual  unsigned int getDim() const = 0 ;
    
    
      
}; // end of class TriangulationBase
} // end of namespace basis
}// end of namespace dftefe
#endif
