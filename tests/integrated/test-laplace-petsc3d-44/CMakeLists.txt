bout_add_integrated_test(test-laplace-petsc3d-44
  SOURCES test-laplace3d.cxx
  EXTRA_FILES
    data_circular_core/BOUT.inp
    data_circular_core-sol/BOUT.inp
    data_slab_core/BOUT.inp
    data_slab_sol/BOUT.inp
  USE_RUNTEST
  )
