//// 3D simulations of HW
//// This version uses indexed operators
//// which reduce the number of loops over the domain
////  Profiling markers and ranges are set if USE_NVTX is defined
////  Baesed on Ben Duddson, Steven Glenn code, Yining Qin update 0521-2020
////  GPU processing is enabled if BOUT_ENABLE_CUDA is defined

#include <iostream>
#include <cstdlib>
#include <gpu_functions.hxx>
#include <bout/physicsmodel.hxx>
#include <smoothing.hxx>
#include <invert_laplace.hxx>
#include <derivs.hxx>
#include <bout/single_index_ops.hxx>
#include "RAJA/RAJA.hpp"         // using RAJA lib
#include <cuda_profiler_api.h>


class HW3D : public PhysicsModel {
public:
  Field3D n, vort;  // Evolving density and vorticity
  Field3D phi;      // Electrostatic potential

  // Model parameters
  BoutReal alpha;      // Adiabaticity (~conductivity)
  BoutReal kappa;      // Density gradient drive
  BoutReal Dvort, Dn;  // Diffusion
  std::unique_ptr<Laplacian> phiSolver; // Laplacian solver for vort -> phi


  int init(bool UNUSED(restart)) {

    auto& options = Options::root()["hw"];
    alpha = options["alpha"].withDefault(1.0);
    kappa = options["kappa"].withDefault(0.1);
    Dvort = options["Dvort"].doc("Vorticity diffusion (normalised)").withDefault(1e-2);
    Dn = options["Dn"].doc("Density diffusion (normalised)").withDefault(1e-2);

    SOLVE_FOR(n, vort);
    SAVE_REPEAT(phi);
    
    phiSolver = Laplacian::create();
    phi = 0.; // Starting phi

    return 0;
  }

  int rhs(BoutReal UNUSED(time)) {
    // Solve for potential
    phi = phiSolver->solve(vort, phi);    
    Field3D phi_minus_n = phi - n;
    
    // Communicate variables
    mesh->communicate(n, vort, phi, phi_minus_n);

    // Create accessors which enable fast access
    auto n_acc = FieldAccessor<>(n);
    auto vort_acc = FieldAccessor<>(vort);
    auto phi_acc = FieldAccessor<>(phi);
    auto phi_minus_n_acc = FieldAccessor<>(phi_minus_n);
    BoutReal* gpu_n_ddt= const_cast<BoutReal*>(n.timeDeriv()->operator()(0,0)); // copy ddt(n) to __device__
    BoutReal* gpu_vort_ddt = const_cast<BoutReal*>(vort.timeDeriv()->operator()(0,0)); // copy ddt(vort) to __device__                       


    auto region = n.getRegion("RGN_NOBNDRY"); // Region object
    auto indices = region.getIndices();   // A std::vector of Ind3D objects
    int indices_size = indices.size();

//  Copy data to __device__
    RAJA_data_copy(n,vort, phi,phi_minus_n,phi_acc,phi_minus_n_acc, indices_size);

//  GPU loop RAJA_DEVICE 
    RAJA::forall<EXEC_POL>(RAJA::RangeSegment(0, indices.size()), [=] RAJA_DEVICE (int i) {
	
	  	BoutReal div_current = alpha * gpu_Div_par_Grad_par(phi_minus_n_acc, i);
	
  		gpu_n_ddt[i]= - gpu_bracket_par(phi_acc, n_acc, i)
	            		          - div_current
         	                  - kappa * gpu_DZZ_par(phi_acc, i)
                            + Dn *gpu_Delp2_par(n_acc, i) ;	

      gpu_vort_ddt[i]= - gpu_bracket_par(phi_acc, vort_acc, i)
                                     - div_current
                                     + Dvort *gpu_Delp2_par(vort_acc, i) ;   
	
	});

     return 0;
  }
};

// Define a main() function
BOUTMAIN(HW3D);
