/*!************************************************************************
 * \file deriv_methods.hxx
 *
 * Definitions of available derivative methods
 *
 **************************************************************************
 * Copyright 2018
 *    D.Dickinson, P.Hill
 *
 * Contact: Ben Dudson, bd512@york.ac.uk
 *
 * This file is part of BOUT++.
 *
 * BOUT++ is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * BOUT++ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with BOUT++.  If not, see <http://www.gnu.org/licenses/>.
 *
 **************************************************************************/

#include <bout_types.hxx>

#ifndef __DERIV_METHODS_H__
#define __DERIV_METHODS_H__

#include <functional>
#include <map>
#include <string>

////////////////////////////////////////////////////////////////////////////////
/// Non-staggered methods
////////////////////////////////////////////////////////////////////////////////

template <DIRECTION direction, STAGGER stagger, int nGuard, typename FieldType>
using Derivative = std::function<void(const FieldType &var, FieldType &result, const REGION region)>;

template <DIRECTION direction, STAGGER stagger, int nGuard, typename FieldType>
class DerivativeFactory {
public:
  using Derivative = ::Derivative<direction, stagger, nGuard, FieldType>;

  bool add(const std::string &name, Derivative derivative) {
    return derivative_map.emplace(name, derivative).second;
  }
  bool remove(const std::string &name) { return derivative_map.erase(name) == 1; }
  Derivative create(const std::string &name) {
    auto index = derivative_map.find(name);
    if (index != std::end(derivative_map)) {
      return index->second;
    }
    throw BoutException("Could not find %s", name.c_str());
  }

  std::vector<std::string> listAvailable() {
    std::vector<std::string> available;
    for (const auto &name : derivative_map) {
      available.push_back(name.first);
    }
    return available;
  }

  static DerivativeFactory &getInstance() {
    static DerivativeFactory instance;
    return instance;
  }

protected:
  std::map<std::string, Derivative> derivative_map;
  DerivativeFactory() {}
};

template <DIRECTION direction, STAGGER stagger, int nGuard, typename FieldType,
          template <DIRECTION, class> class T>
class RegisterDerivative {
public:
  using Derivative = ::Derivative<direction, stagger, nGuard, FieldType>;
  RegisterDerivative(const std::string &name) {
    Derivative derivative = T<direction, FieldType>();
    DerivativeFactory<direction, stagger, nGuard, FieldType>::getInstance().add(
        name, derivative);
  }
};

#define REGISTER_BOUT_DERIV_NAME(name) register_bout_deriv_##name

#define BOUT_DERIV(name, key, nGuards, ...)				\
  template<DIRECTION direction, typename T>					\
  class name {                                        \
  public:								\
  const int nGuardsRequired = nGuards;\
  const std::string shortName = key;\
  BoutReal apply(const stencil &f) const;				\
  void operator()(const T &var, T &result, const REGION region) const {	\
    BOUT_FOR(i, var.getRegion(region)) {				\
      result[i] = apply(populateStencil<direction, STAGGER::None, nGuards>(var, i));	\
    }									\
    return;								\
  }									\
  };\
  namespace { \
    RegisterDerivative<DIRECTION::X, STAGGER::None, 1, Field3D, DDX_C2> REGISTER_BOUT_DERIV_NAME(name)(key); \
  } \
  template<DIRECTION direction, typename T>					\
  BoutReal name<direction, T>::apply(const stencil &f) const

#define BOUT_VDERIV(name, key, nGuards, ...)					\
  class name {								\
  public:								\
  const BoutReal apply(const BoutReal vc, const stencil &f) const;	\
  template<DIRECTION direction, typename T>					\
    void operator()(const T& vel, const T &var, T &result, const REGION region) const { \
      BOUT_FOR(i, var.getRegion(region)) {				\
      result[i] = apply(vel[i], populateStencil<direction, STAGGER::None, 1>(var, i)); \
    }									\
    return;								\
  }									\
  };\
  const BoutReal name::apply(const BoutReal vc, const stencil &f) const 

#define BOUT_FDERIV(name, key, nGuards, ...)					\
  class name {								\
  public:								\
  const BoutReal apply(const stencil &v, const stencil &f) const;	\
  template<DIRECTION direction, typename T>					\
  void operator()(const T& vel, const T &var, T &result, const REGION region) const { \
    BOUT_FOR(i, var.getRegion(region)) {				\
      result[i] = apply(\
			populateStencil<direction, STAGGER::None, 1>(vel, i),\
			populateStencil<direction, STAGGER::None, 1>(var, i)); \
    }									\
    return;								\
  }									\
  };\
  const BoutReal name::apply(const stencil &v, const stencil &f) const 

////////////////////////////////////////////////////////////////////////////////
/// Simple non-staggered methods
///
/// Basic derivative methods.
/// All expect to have an input grid cell at the same location as the output
/// Hence convert cell centred values -> centred values, or left -> left
///
////////////////////////////////////////////////////////////////////////////////

//////////////////////////////
//--- First order derivatives
//////////////////////////////
/// central, 2nd order
BOUT_DERIV(DDX_C2, "C2", 1) {return (f.p-f.m)*0.5;}

/// central, 4th order
BOUT_DERIV(DDX_C4, "C4", 2) {return (8. * f.p - 8. * f.m + f.mm - f.pp) / 12.; }

/// Central WENO method, 2nd order (reverts to 1st order near shocks)
BOUT_DERIV(DDX_CWENO2, "W2", 1) {
  BoutReal isl, isr, isc;  // Smoothness indicators
  BoutReal al, ar, ac, sa; // Un-normalised weights
  BoutReal dl, dr, dc;     // Derivatives using different stencils

  dc = 0.5 * (f.p - f.m);
  dl = f.c - f.m;
  dr = f.p - f.c;

  isl = SQ(dl);
  isr = SQ(dr);
  isc = (13. / 3.) * SQ(f.p - 2. * f.c + f.m) + 0.25 * SQ(f.p - f.m);

  al = 0.25 / SQ(WENO_SMALL + isl);
  ar = 0.25 / SQ(WENO_SMALL + isr);
  ac = 0.5 / SQ(WENO_SMALL + isc);
  sa = al + ar + ac;

  return (al * dl + ar * dr + ac * dc) / sa;
}

// Smoothing 2nd order derivative
BOUT_DERIV(DDX_S2, "S2", 2) {
  // 4th-order differencing
  BoutReal result = (8. * f.p - 8. * f.m + f.mm - f.pp) / 12.;

  result += SIGN(f.c) * (f.pp - 4. * f.p + 6. * f.c - 4. * f.m + f.mm) / 12.;

  return result;
}

//////////////////////////////
//--- Second order derivatives
//////////////////////////////

/// Second derivative: Central, 2nd order
BOUT_DERIV(D2DX2_C2, "C2_2", 1) { return f.p + f.m - 2. * f.c; }

/// Second derivative: Central, 4th order
BOUT_DERIV(D2DX2_C4, "C4_2", 2) {return (-f.pp + 16. * f.p - 30. * f.c + 16. * f.m - f.mm) / 12.;}

//////////////////////////////
//--- Fourth order derivatives
//////////////////////////////
BOUT_DERIV(D4DX4_C2, "C2_4", 2) { return (f.pp - 4. * f.p + 6. * f.c - 4. * f.m + f.mm); }

////////////////////////////////////////////////////////////////////////////////
/// Upwind non-staggered methods
///
/// Basic derivative methods.
/// All expect to have an input grid cell at the same location as the output
/// Hence convert cell centred values -> centred values, or left -> left
///
////////////////////////////////////////////////////////////////////////////////
std::tuple<BoutReal, BoutReal> vUpDown(const BoutReal v){
  return std::tuple<BoutReal, BoutReal>{ 0.5*(v + fabs(v)), 0.5*(v - fabs(v))};
}
/// Upwinding: Central, 2nd order
BOUT_VDERIV(VDDX_C2, "C2", 1) { return vc * 0.5 * (f.p - f.m);}

/// Upwinding: Central, 4th order
BOUT_VDERIV(VDDX_C4, "C4", 2) { return vc * (8. * f.p - 8. * f.m + f.mm - f.pp) / 12.;}

/// upwind, 1st order
BOUT_VDERIV(VDDX_U1, "U1", 1) { //No vec
  // Existing form doesn't vectorise due to branching
  return vc >= 0.0 ? vc * (f.c - f.m) : vc * (f.p - f.c);
  // Alternative form would but may involve more operations
  const auto vSplit = vUpDown(vc); 
  return (std::get<0>(vSplit)*(f.p-f.c)
	  + std::get<1>(vSplit) * (f.c-f.m));
}

/// upwind, 2nd order
BOUT_VDERIV(VDDX_U2, "U2", 2) { //No vec
  // Existing form doesn't vectorise due to branching  
  return vc >= 0.0 ? vc * (1.5 * f.c - 2.0 * f.m + 0.5 * f.mm)
    : vc * (-0.5 * f.pp + 2.0 * f.p - 1.5 * f.c);
  // Alternative form would but may involve more operations
  const auto vSplit = vUpDown(vc); 
  return (std::get<0>(vSplit) * (1.5 * f.c - 2.0 * f.m + 0.5 * f.mm)
	  + std::get<1>(vSplit) * (-0.5 * f.pp + 2.0 * f.p - 1.5 * f.c));

}

/// upwind, 3rd order
BOUT_VDERIV(VDDX_U3, "U3", 2) { //No vec
  // Existing form doesn't vectorise due to branching
  return vc >= 0.0 ? vc*(4.*f.p - 12.*f.m + 2.*f.mm + 6.*f.c)/12.
    : vc*(-4.*f.m + 12.*f.p - 2.*f.pp - 6.*f.c)/12.;
  // Alternative form would but may involve more operations
  const auto vSplit = vUpDown(vc);
  return (std::get<0>(vSplit) * (4.*f.p - 12.*f.m + 2.*f.mm + 6.*f.c)
	  + std::get<1>(vSplit)*(-4.*f.m + 12.*f.p - 2.*f.pp - 6.*f.c))/12.;
  
}

/// 3rd-order WENO scheme
BOUT_VDERIV(VDDX_WENO3, "W3", 2) { //No vec
  BoutReal deriv, w, r;
  // Existing form doesn't vectorise due to branching
  
  if (vc > 0.0) {
    // Left-biased stencil

    r = (WENO_SMALL + SQ(f.c - 2.0 * f.m + f.mm)) /
        (WENO_SMALL + SQ(f.p - 2.0 * f.c + f.m));

    deriv = (-f.mm + 3. * f.m - 3. * f.c + f.p);

  } else {
    // Right-biased

    r = (WENO_SMALL + SQ(f.pp - 2.0 * f.p + f.c)) /
        (WENO_SMALL + SQ(f.p  - 2.0 * f.c + f.m));

    deriv = (-f.m + 3. * f.c - 3. * f.p + f.pp);
  }
  
  w = 1.0 / (1.0 + 2.0 * r * r);
  deriv = 0.5 * ((f.p - f.m) - w * deriv);
  
  return vc * deriv;
}

///-----------------------------------------------------------------
/// 3rd-order CWENO. Uses the upwinding code and split flux
BOUT_DERIV(DDX_CWENO3, "W3", 2) {
  BoutReal a, ma = fabs(f.c);
  // Split flux
  a = fabs(f.m);
  if (a > ma)
    ma = a;
  a = fabs(f.p);
  if (a > ma)
    ma = a;
  a = fabs(f.mm);
  if (a > ma)
    ma = a;
  a = fabs(f.pp);
  if (a > ma)
    ma = a;

  stencil sp, sm;

  sp.mm = f.mm + ma;
  sp.m = f.m + ma;
  sp.c = f.c + ma;
  sp.p = f.p + ma;
  sp.pp = f.pp + ma;

  sm.mm = ma - f.mm;
  sm.m = ma - f.m;
  sm.c = ma - f.c;
  sm.p = ma - f.p;
  sm.pp = ma - f.pp;

  VDDX_WENO3 upwindOp;
  return upwindOp.apply(0.5, sp) + upwindOp.apply(-0.5, sm);
}
///-----------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////
/// Flux non-staggered methods
///
/// Basic derivative methods.
/// All expect to have an input grid cell at the same location as the output
/// Hence convert cell centred values -> centred values, or left -> left
///
////////////////////////////////////////////////////////////////////////////////

BOUT_FDERIV(FDDX_U1, "U1", 1) { //No vec

  // Velocity at lower end
  BoutReal vs = 0.5 * (v.m + v.c);
  BoutReal result = (vs >= 0.0) ? vs * f.m : vs * f.c;
  // and at upper
  vs = 0.5 * (v.c + v.p);
  // Existing form doesn't vectorise due to branching
  result -= (vs >= 0.0) ? vs * f.c : vs * f.p;
  return - result;

  // Alternative form would but may involve more operations
  const auto vSplit = vUpDown(vs);
  return result - std::get<0>(vSplit) * f.c + std::get<1>(vSplit) * f.p;
}

BOUT_FDERIV(FDDX_C2, "C2", 2) { return 0.5 * (v.p * f.p - v.m * f.m);}

BOUT_FDERIV(FDDX_C4, "C4", 2) { return (8. * v.p * f.p - 8. * v.m * f.m + v.mm * f.mm - v.pp * f.pp) / 12.;}


////////////////////////////////////////////////////////////////////////////////
/// Staggered methods
///
/// Map Centre -> Low or Low -> Centre
///
/// These expect the output grid cell to be at a different location to the input
/// 
/// The stencil no longer has a value in 'C' (centre)
/// instead, points are shifted as follows:
///
///  mm  -> -3/2 h
///  m   -> -1/2 h
///  p   -> +1/2 h
///  pp  -? +3/2 h
///
/// NOTE: Cell widths (dx, dy, dz) are currently defined as centre->centre
/// for the methods above. This is currently not taken account of, so large
/// variations in cell size will cause issues.
////////////////////////////////////////////////////////////////////////////////

#define BOUT_STAGGERED_DERIV(name, key, nGuards, ...)				\
  class name {								\
  public:								\
  const int nGuardsRequired = nGuards;					\
  const std::string shortName = key;					\
  const BoutReal apply(const stencil &f) const;				\
  template<DIRECTION direction, STAGGER stagger, typename T>		\
  void operator()(const T &var, T &result, const REGION region) const {	\
    BOUT_FOR(i, var.getRegion(region)) {				\
      result[i] = apply(populateStencil<direction, stagger, nGuards>(var, i)); \
    }									\
    return;								\
  }									\
  };									\
  const BoutReal name::apply(const stencil &f) const 
  
#define BOUT_STAGGERED_VDERIV(name, key, nGuards, ...)			\
  class name {								\
  public:								\
  const int nGuardsRequired = nGuards;					\
  const std::string shortName = key;					\
  const BoutReal apply(const stencil &v, const stencil &f) const;	\
  template<DIRECTION direction, STAGGER stagger, typename T>		\
  void operator()(const T& vel, const T &var, T &result, const REGION region) const { \
    BOUT_FOR(i, var.getRegion(region)) {				\
      result[i] = apply(						\
			populateStencil<direction, stagger, nGuards>(vel, i), \
			populateStencil<direction, STAGGER::None, nGuards>(var, i)); \
    }									\
    return;								\
  }									\
  };									\
  const BoutReal name::apply(const stencil &v, const stencil &f) const 
  
#define BOUT_STAGGERED_FDERIV(name, key, nGuards, ...)			\
  class name {								\
  public:								\
  const int nGuardsRequired = nGuards;					\
  const std::string shortName = key;					\
  const BoutReal apply(const stencil &v, const stencil &f) const;	\
  template<DIRECTION direction, STAGGER stagger, typename T>		\
  void operator()(const T& vel, const T &var, T &result, const REGION region) const { \
    BOUT_FOR(i, var.getRegion(region)) {				\
      result[i] = apply(						\
			populateStencil<direction, stagger, nGuards>(vel, i), \
			populateStencil<direction, STAGGER::None, nGuards>(var, i)); \
    }									\
    return;								\
  }									\
  };									\
  const BoutReal name::apply(const stencil &v, const stencil &f) const 

////////////////////////////////////////////////////////////////////////////////
/// Standard methods
////////////////////////////////////////////////////////////////////////////////  
BOUT_STAGGERED_DERIV(DDX_C2_stag, "C2", 1) { return f.p - f.m; }

BOUT_STAGGERED_DERIV(DDX_C4_stag, "C4", 2) { return (27. * (f.p - f.m) - (f.pp - f.mm)) / 24.; }

BOUT_STAGGERED_DERIV(D2DX2_C2_stag, "C2_2", 2) { return (f.pp + f.mm - f.p - f.m) / 2.; }

////////////////////////////////////////////////////////////////////////////////
/// Upwind methods
////////////////////////////////////////////////////////////////////////////////
BOUT_STAGGERED_VDERIV(VDDX_U1_stag, "U1", 1) {
  // Lower cell boundary
  BoutReal result = (v.m >= 0) ? v.m * f.m : v.m * f.c;

  // Upper cell boundary
  result -= (v.p >= 0) ? v.p * f.c : v.p * f.p;
  result *= -1;

  // result is now d/dx(v*f), but want v*d/dx(f) so subtract f*d/dx(v)
  result -= f.c * (v.p - v.m);
  return result;
}

BOUT_STAGGERED_VDERIV(VDDX_U2_stag, "U2", 2) {
  // Calculate d(v*f)/dx = (v*f)[i+1/2] - (v*f)[i-1/2]

  // Upper cell boundary
  BoutReal result = (v.p >= 0.) ? v.p * (1.5*f.c - 0.5*f.m) : v.p * (1.5*f.p - 0.5*f.pp);

  // Lower cell boundary
  result -= (v.m >= 0.) ? v.m * (1.5*f.m - 0.5*f.mm) : v.m * (1.5*f.c - 0.5*f.p);

  // result is now d/dx(v*f), but want v*d/dx(f) so subtract f*d/dx(v)
  result -= f.c * (v.p - v.m);

  return result;
}

BOUT_STAGGERED_VDERIV(VDDX_C2_stag, "C2", 1) {
  // Result is needed at location of f: interpolate v to f's location and take an
  // unstaggered derivative of f
  return 0.5 * (v.p + v.m) * 0.5 * (f.p - f.m);
}

BOUT_STAGGERED_VDERIV(VDDX_C4_stag, "C4", 2) {
  // Result is needed at location of f: interpolate v to f's location and take an
  // unstaggered derivative of f
  return (9. * (v.m + v.p) - v.mm - v.pp) / 16. * (8. * f.p - 8. * f.m + f.mm - f.pp) /
         12.;
}

////////////////////////////////////////////////////////////////////////////////
/// Flux methods
////////////////////////////////////////////////////////////////////////////////
BOUT_STAGGERED_FDERIV(FDDX_U1_stag, "U1", 1) {
  // Lower cell boundary
  BoutReal result = (v.m >= 0) ? v.m * f.m : v.m * f.c;

  // Upper cell boundary
  result -= (v.p >= 0) ? v.p * f.c : v.p * f.p;

  return - result;
}
#endif