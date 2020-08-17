#ifndef BOUT_HYPRE_INTERFACE_H
#define BOUT_HYPRE_INTERFACE_H

#ifdef BOUT_HAS_HYPRE

#include "boutcomm.hxx"
#include "field.hxx"
#include "utils.hxx"
#include "bout/globalindexer.hxx"

#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_utilities.h"
#include "_hypre_utilities.h"

#include <memory>

namespace bout {

#if 1   //HYPRE with Cuda enabled
#define HypreMalloc(P, SIZE)  cudaMallocManaged(P, SIZE)
#define HypreFree(P)  cudaFree(P)
#else
#define HypreMalloc(P, SIZE)  malloc(P, SIZE)
#define HypreFree(P)  free(P)
#endif



template <class T>
class HypreVector;
template <class T>
void swap(HypreVector<T>& lhs, HypreVector<T>& rhs);

// TODO: check correctness of initialise/assemble
// TODO: set sizes
// TODO: set contiguous blocks at once

template <class T>
class HypreVector {
  MPI_Comm comm;
  HYPRE_BigInt jlower, jupper, vsize;
  HYPRE_IJVector hypre_vector{nullptr};
  HYPRE_ParVector parallel_vector{nullptr};
  IndexerPtr indexConverter;
  CELL_LOC location;
  bool initialised{false};
  bool have_indices;
  HYPRE_BigInt *I{nullptr};
  HYPRE_Complex *V{nullptr};

public:
  static_assert(bout::utils::is_Field<T>::value, "HypreVector only works with Fields");
  using ind_type = typename T::ind_type;

  HypreVector() = default;
  ~HypreVector() {
    // Destroy may print an error if the vector is already null
    if (hypre_vector == nullptr) {
      return;
    }
    // Also destroys parallel_vector
    HYPRE_IJVectorDestroy(hypre_vector);
    HypreFree(I);  
    HypreFree(V);
  }

  // Disable copy, at least for now: not clear that HYPRE_IJVector is
  // copy-able
  HypreVector(const HypreVector<T>&) = delete;
  auto operator=(const HypreVector<T>&) = delete;

  HypreVector(HypreVector<T>&& other) {
    std::swap(hypre_vector, other.hypre_vector);
    std::swap(parallel_vector, other.parallel_vector);
    indexConverter = other.indexConverter;
    location = other.location;
    initialised = other.initialised;
    other.initialised = false;
  }

  HypreVector<T>& operator=(HypreVector<T>&& other) {
    std::swap(hypre_vector, other.hypre_vector);
    std::swap(parallel_vector, other.parallel_vector);
    indexConverter = other.indexConverter;
    location = other.location;
    initialised = other.initialised;
    other.initialised = false;
    return *this;
  }

  explicit HypreVector(const T& f) {
    comm = std::is_same<T, FieldPerp>::value ? f.getMesh()->getXcomm() : BoutComm::get();
    indexConverter = GlobalIndexer::getInstance(f.getMesh());

    const auto& region = f.getRegion("RGN_ALL_THIN");    
    jlower=static_cast<HYPRE_BigInt>(indexConverter->getGlobal(*std::begin(region)));
    jupper=static_cast<HYPRE_BigInt>(indexConverter->getGlobal(*--std::end(region)));
    vsize = jupper - jlower + 1;

    HYPRE_IJVectorCreate(comm, jlower, jupper, &hypre_vector);
    HYPRE_IJVectorSetObjectType(hypre_vector, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(hypre_vector);

    location = f.getLocation();
    initialised = true;

    HypreMalloc(&I, vsize*sizeof(HYPRE_BigInt)); 
    HypreMalloc(&V, vsize*sizeof(HYPRE_Complex));
    importValuesFromField(f);
  }

  // Mesh can't be const yet, need const-correctness on Mesh;
  // GlobalIndexer ctor also modifies mesh -- FIXME
  explicit HypreVector(Mesh& mesh) {
    comm = std::is_same<T, FieldPerp>::value ? mesh.getXcomm() : BoutComm::get();
    indexConverter = GlobalIndexer::getInstance(&mesh);

    const auto& region = mesh.getRegion<T>("RGN_ALL_THIN");
    jlower = static_cast<HYPRE_BigInt>(indexConverter->getGlobal(*std::begin(region)));
    jupper = static_cast<HYPRE_BigInt>(indexConverter->getGlobal(*--std::end(region)));
    vsize = jupper - jlower + 1;

    HYPRE_IJVectorCreate(comm, jlower, jupper, &hypre_vector);
    HYPRE_IJVectorSetObjectType(hypre_vector, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(hypre_vector);
    initialised = true;
    location = CELL_LOC::centre;

    HypreMalloc(&I, vsize*sizeof(HYPRE_BigInt)); 
    HypreMalloc(&V, vsize*sizeof(HYPRE_Complex));
    have_indices = false;
  }  

  void getIndices(T &f)
  {
    int vec_i = 0;
    BOUT_FOR_SERIAL(i, f.getRegion("RGN_ALL_THIN")) {
      const auto index = static_cast<HYPRE_BigInt>(indexConverter->getGlobal(i));
      if (index != -1) {
        I[vec_i] = index;
        vec_i ++;
      }
    }
    ASSERT2(vec_i == vsize);
    have_indices = true;
  }

  void importValuesFromField(T &f) {
    int vec_i = 0;
    BOUT_FOR_SERIAL(i, f.getRegion("RGN_ALL_THIN")) {
      const auto index = static_cast<HYPRE_BigInt>(indexConverter->getGlobal(i));
      if (index != -1) {
        I[vec_i] = index;        
        V[vec_i] = f[i];
        vec_i ++;
      }
    }
    ASSERT2(vec_i == vsize);
    HYPRE_IJVectorSetValues(hypre_vector, vsize, I, V);
    assemble();
    have_indices = true;
  }

  void exportValuesToField(T &f) {
    if (!have_indices) {
      getIndices(f);
    }
    HYPRE_IJVectorGetValues(hypre_vector, vsize, I, V);
    int count = 0;
    BOUT_FOR_SERIAL(i, f.getRegion("RGN_ALL_THIN")) {
      const auto index = static_cast<HYPRE_BigInt>(indexConverter->getGlobal(i));
      if (index != -1) {
        f[i] = static_cast<BoutReal>(V[count]);
        count ++;
      }
    }
  }

  HypreVector<T>& operator=(const T& f) {
    HypreVector<T> result(f);
    *this = std::move(result);
    return *this;
  }

  void assemble() {
    HYPRE_IJVectorAssemble(hypre_vector);
    HYPRE_IJVectorGetObject(hypre_vector, reinterpret_cast<void**>(&parallel_vector));
  }

  HYPRE_IJVector get() { return hypre_vector; }
  const HYPRE_IJVector& get() const { return hypre_vector; }

  HYPRE_ParVector getParallel() { return parallel_vector; }
  const HYPRE_ParVector& getParallel() const { return parallel_vector; }

  friend void swap(HypreVector<T>& lhs, HypreVector<T>& rhs) {
    using std::swap;
    swap(lhs.hypre_vector, rhs.hypre_vector);
    swap(lhs.parallel_vector, rhs.parallel_vector);
    swap(lhs.indexConverter, rhs.indexConverter);
    swap(lhs.location, rhs.location);
    swap(lhs.initialised, rhs.initialised);
  }
};

template <class T>
class HypreMatrix;
template <class T>
void swap(HypreMatrix<T>& lhs, HypreMatrix<T>& rhs);

template <class T>
class HypreMatrix {
  MPI_Comm comm;
  HYPRE_BigInt ilower, iupper;
  std::shared_ptr<HYPRE_IJMatrix> hypre_matrix{nullptr};
  HYPRE_ParCSRMatrix parallel_matrix{nullptr};
  IndexerPtr index_converter;
  CELL_LOC location;
  bool initialised{false};
  int yoffset{0};
  bool assembled{false};
  HYPRE_BigInt max_num_cols, num_rows;
  HYPRE_BigInt *num_cols; 
  HYPRE_BigInt *I;
  HYPRE_BigInt **J;
  HYPRE_Complex **V;

public:
  static_assert(bout::utils::is_Field<T>::value, "HypreMatrix only works with Fields");
  using ind_type = typename T::ind_type;

  HypreMatrix() = default;
  HypreMatrix(const HypreMatrix<T>&) = delete;
  HypreMatrix(HypreMatrix<T>&& other) {
    using std::swap;
    swap(hypre_matrix, other.hypre_matrix);
    swap(parallel_matrix, other.parallel_matrix);
    index_converter = other.index_converter;
    initialised = other.initialised;
    yoffset = other.yoffset;
    other.initialised = false;
  }

  ~HypreMatrix() {
    if (hypre_matrix != nullptr) {
      HYPRE_IJMatrixDestroy(*hypre_matrix);
    }

    for (int i = 0; i < num_rows; ++ i) {
      delete [] J[i];
      delete [] V[i];
    }
    delete [] J;
    delete [] V;
  }

  HypreMatrix<T>& operator=(const HypreMatrix<T>&) = delete;
  HypreMatrix<T>& operator=(HypreMatrix<T>&& other) {
    using std::swap;
    swap(hypre_matrix, other.hypre_matrix);
    swap(parallel_matrix, other.parallel_matrix);
    index_converter = other.index_converter;
    initialised = other.initialised;
    yoffset = other.yoffset;
    other.initialised = false;
    return *this;
  }

  explicit HypreMatrix(Mesh& mesh, HYPRE_BigInt max_num_cols_) {
    comm = std::is_same<T, FieldPerp>::value ? mesh.getXcomm() : BoutComm::get();
    index_converter = GlobalIndexer::getInstance(&mesh);
    const auto& region = mesh.getRegion<T>("RGN_ALL_THIN");

    ilower = static_cast<HYPRE_BigInt>(index_converter->getGlobal(*std::begin(region)));
    iupper = static_cast<HYPRE_BigInt>(index_converter->getGlobal(*--std::end(region)));
    num_rows = iupper - ilower + 1;

    max_num_cols = max_num_cols_;
    HypreMalloc(&I, num_rows*sizeof(HYPRE_BigInt)); 
    HypreMalloc(&num_cols, num_rows*sizeof(HYPRE_BigInt)); 
    J = new HYPRE_BigInt*[num_rows];
    V = new HYPRE_Complex*[num_rows];
    for (HYPRE_BigInt i = 0; i < num_rows; ++i) {
      num_cols[i] = 0;
      I[i] = ilower + i;
      J[i] = new HYPRE_BigInt[max_num_cols];
      V[i] = new HYPRE_Complex[max_num_cols];
    }

    HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &*hypre_matrix);
    HYPRE_IJMatrixSetObjectType(*hypre_matrix, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(*hypre_matrix);

    initialised = true;
  }


  BoutReal getVal(const ind_type& row, const ind_type& column) const {
    const HYPRE_BigInt global_row = index_converter->getGlobal(row);
    const HYPRE_BigInt global_column = index_converter->getGlobal(column);
#if CHECKLEVEL >= 1
    if (global_row < 0 or global_column < 0) {
      throw BoutException(
          "Invalid HypreMatrix element at row = ({}, {}, {}), column = ({}, {}, {})",
          row.x(), row.y(), row.z(), column.x(), column.y(), column.z());
    }
#endif
    return getVal(global_row, global_column);
  }

  BoutReal getVal(const HYPRE_BigInt row, const HYPRE_BigInt column) const {
    HYPRE_Complex value = 0.0;
    HYPRE_BigInt i = row - ilower;
    ASSERT2(i >= 0 && i < num_rows);
    for(HYPRE_BigInt col_ind = 0; col_ind < num_cols[i]; ++col_ind) {
      if (J[i][col_ind] == column) {
        value = V[i][col_ind];
        break;
      }
    }
    return static_cast<BoutReal>(value);
  }

  void setVal(const ind_type& row, const ind_type& column, BoutReal value) {
    const HYPRE_BigInt global_row = index_converter->getGlobal(row);
    const HYPRE_BigInt global_column = index_converter->getGlobal(column);
#if CHECKLEVEL >= 1
    if (global_row < 0 or global_column < 0) {
      throw BoutException(
          "Invalid HypreMatrix element at row = ({}, {}, {}), column = ({}, {}, {})",
          row.x(), row.y(), row.z(), column.x(), column.y(), column.z());
    }
#endif
    return setVal(global_row, global_column, value);
  }

  void setVal(const HYPRE_BigInt row, const HYPRE_BigInt column, BoutReal value) {
    HYPRE_BigInt i = row - ilower;
    ASSERT2(i >= 0 && i < num_rows);
    bool value_set = false;
    for(HYPRE_BigInt col_ind = 0; col_ind < num_cols[i]; ++col_ind) {
      if (J[i][col_ind] == column) {
        V[i][col_ind] = value;
        value_set  = true;
        break;
      }
    }

    if (!value_set)
    {
      ASSERT2(num_cols[i] != max_num_cols);
      V[i][num_cols[i]] = value;
      num_cols[i] ++;
    }
  }

  void assemble() {
    HYPRE_BigInt num_entries = 0;
    HYPRE_BigInt *cols;
    HYPRE_Complex *vals;
    for (HYPRE_BigInt i = 0; i < num_rows; ++i) {
      num_entries += num_cols[i];
    }
    HypreMalloc(&cols, num_entries*sizeof(HYPRE_BigInt)); 
    HypreMalloc(&vals, num_entries*sizeof(HYPRE_Complex));
    int entry = 0;
    for (HYPRE_BigInt i = 0; i < num_rows; ++i) {
      for(HYPRE_BigInt col_ind = 0; col_ind < num_cols[i]; ++col_ind) {
        cols[entry] = J[i][col_ind];
        vals[entry] = V[i][col_ind];
      }
    }
    HYPRE_IJMatrixSetValues(*hypre_matrix, num_rows, num_cols, I, cols, vals);

    HYPRE_IJMatrixAssemble(*hypre_matrix);
    HYPRE_IJMatrixGetObject(*hypre_matrix, reinterpret_cast<void**>(&parallel_matrix));
    assembled = true;

    HypreFree(cols);
    HypreFree(vals);
  }

  bool isAssembled() const { return assembled; }

  HypreMatrix<T> yup(int index = 0) { return ynext(index + 1); }
  HypreMatrix<T> ydown(int index = 0) { return ynext(-index - 1); }
  HypreMatrix<T> ynext(int dir) {
    if (std::is_same<T, FieldPerp>::value and ((yoffset + dir) != 0)) {
      throw BoutException("Can not get ynext for FieldPerp");
    }
    HypreMatrix<T> result;
    result.hypre_matrix = hypre_matrix;
    result.index_converter = index_converter;
    result.yoffset = std::is_same<T, Field2D>::value ? 0 : yoffset + dir;
    result.initialised = initialised;
    return result;
  }

  HYPRE_IJMatrix get() { return *hypre_matrix; }
  const HYPRE_IJMatrix& get() const { return *hypre_matrix; }

  HYPRE_ParCSRMatrix getParallel() { return parallel_matrix; }
  const HYPRE_ParCSRMatrix& getParallel() const { return parallel_matrix; }
};


// This is a specific Hypre matrix system.  here we will be setting is the system
// AMG(Px=b) A x = b
// and we will be solvingthe system using the preconditioned conjugate gradient 
// method utilizing the AMG preconditioner on the system Px = b.  In some cases
// it will be advantage to us a P that is an approximation to A instead of P = A
template <class T>
class HypreSystem
{
private:
  HypreMatrix<T> *P{nullptr};
  HypreMatrix<T> *A{nullptr};
  HypreVector<T> *x{nullptr};
  HypreVector<T> *b{nullptr};
  MPI_Comm comm;
  HYPRE_Solver solver;
  HYPRE_Solver precon;
public:
  HypreSystem(Mesh& mesh)
  {
    HYPRE_Init();
# if 1
    hypre_HandleDefaultExecPolicy(hypre_handle()) = HYPRE_EXEC_DEVICE;
#endif

    comm = std::is_same<T, FieldPerp>::value ? mesh.getXcomm() : BoutComm::get();
    HYPRE_ParCSRPCGCreate(comm, &solver);
    HYPRE_PCGSetTol(solver, 1e-7);
    HYPRE_PCGSetMaxIter(solver, 200);

    HYPRE_BoomerAMGCreate(&precon);
    HYPRE_BoomerAMGSetOldDefault(precon);
    HYPRE_BoomerAMGSetRelaxType(precon, 18);  // 18 or 7 for GPU implementation
    HYPRE_BoomerAMGSetRelaxOrder(precon, false); // must be false for GPU
    HYPRE_BoomerAMGSetCoarsenType(precon, 8); // must be PMIS (8) for GPU 
    HYPRE_BoomerAMGSetInterpType(precon, 3); // must be 3 or 15 for GPU 
    HYPRE_BoomerAMGSetNumSweeps(precon, 1);
    HYPRE_BoomerAMGSetMaxLevels(precon, 20);
    HYPRE_BoomerAMGSetKeepTranspose(precon, 1);
    //TODO:  add methods to change some of these defaults
  }

  ~HypreSystem() {
    HYPRE_Finalize();
  }

  void setMatrix(HypreMatrix<T>* A_) {
    A=A_;
  }

  void setupAMG(HypreMatrix<T>* P_) {
    P = P_;
    HYPRE_ParCSRPCGSetPrecond(solver, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, precon);
    HYPRE_BoomerAMGSetup(precon, P->getParallel(), nullptr, nullptr);    
  }

  void setSolutionVector(HypreVector<T>* x_) {
    x = x_;
  }

  void setRHSVector(HypreVector<T>* b_) {
    b = b_;
  }

  HypreMatrix<T>* getMatrix() {
    return A;
  }

  HypreMatrix<T>* getAMGMatrix() {
    return P;
  }  

  HypreVector<T>* getRHS() {
    return b;
  }

  HypreVector<T>* getSolution() {
    return x;
  }

  HYPRE_PtrToParSolverFcn SetupFcn() const { 
    return (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSetup; 
  }
  
  HYPRE_PtrToParSolverFcn SolveFcn() const { 
    return (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSolve; 
  }

  void solve()
  {
    ASSERT2(A != nullptr);
    ASSERT2(x != nullptr);
    ASSERT2(b != nullptr);

    HYPRE_ParCSRPCGSolve(solver, A->getParallel(), b->getParallel(), x->getParallel());   
  }


};


} // namespace bout

#endif // BOUT_HAS_HYPRE

#endif // BOUT_HYPRE_INTERFACE_H
