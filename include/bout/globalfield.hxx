/*
 * Provides global gather/scatter operations for fields
 *
 */

class GlobalField;
class GlobalField2D;
//class GlobalField3D;

#ifndef __GLOBALFIELD_H__
#define __GLOBALFIELD_H__

#include "mesh.hxx"

class GlobalField {
public:
  virtual bool valid() const = 0;  ///< Is the data valid on any processor?
  bool dataIsLocal() const {return valid() && (data_on_proc == mype);} ///< Data is on this processor

  // Data access by index
  BoutReal& operator()(int jx, int jy, int jz) {return data[jz + nz*jy + nz*ny*jx];}
  const BoutReal& operator()(int jx, int jy, int jz) const {return data[jz + nz*jy + nz*ny*jx];}

  // Direct data access
  BoutReal* getData() {return data;}
protected:
  GlobalField(Mesh *m, int xsize, int ysize, int zsize);
  
  Mesh *mesh;
  
  int nx, ny, nz;
  BoutReal *data;
  int data_on_proc; // Which processor is this data on?

  MPI_Comm comm;
  int npes, mype;
  
  void proc_local_origin(int proc, int *x, int *y, int *z = NULL);
  void proc_origin(int proc, int *x, int *y, int *z = NULL);  ///< Return the global origin of processor proc
  void proc_size(int proc, int *lx, int *ly, int *lz = NULL); ///< Return the array size of processor proc
private:
  GlobalField();
  
};

class GlobalField2D : public GlobalField {
public:
  GlobalField2D(Mesh *m, int proc = 0);
  virtual ~GlobalField2D();
  
  bool valid() const {return false;}
  
  void gather(const Field2D &f); ///< Gather all data onto one processor
  const Field2D scatter() const; ///< Scatter data back from one to many processors
  
  /// Assignment from a 2D field. Shorthand for a gather, and must be called on all processors
  /// The scatter assignment operator needs to be a member of Field2D.
  GlobalField& operator=(const Field2D &rhs) {
    gather(rhs);
    return *this;
  }
  
  // Data access by index
  BoutReal& operator()(int jx, int jy) {return GlobalField::operator()(jx, jy, 0);}
  const BoutReal& operator()(int jx, int jy) const {return GlobalField::operator()(jx, jy, 0);}
  
protected:
  

private:
  GlobalField2D();
  
  BoutReal** buffer;
  
  int msg_len(int proc) const;
  
};


/*
class GlobalField3D : public GlobalField {
public:
  
};
*/

#endif // __GLOBALFIELD_H__
