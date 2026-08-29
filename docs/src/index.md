# [FiniteDiffWENO5.jl](@id home)

FiniteDiffWENO5.jl is a Julia package that implements fifth-order finite-difference weighted essentially non-oscillatory (WENO) schemes for solving hyperbolic partial differential equations (PDEs) in 1D, 2D and 3D on regular grids.

Scalar transport explicitly selects `form = :nonconservative` for
$\mathbf{v} \cdot \nabla u$ or `form = :conservative` for
$\nabla \cdot (\mathbf{v} u)`. Either form can use collocated or staggered
velocity data; staggered normal velocities are interpolated to scalar cell centres
before the spatial operator is applied. Partitioned multiphase transport uses the
non-conservative material form with shared nonlinear weights and simplex limiting.

The core of the package is written in pure Julia, focusing on performance using CPUs, but GPU support is available using KernelAbstractions.jl and Chmy.jl via an extension.

## Installation

FiniteDiffWENO5.jl is a registered package and may be installed directly with the following command in the Julia REPL

```julia-repl
julia>]
  pkg> add FiniteDiffWENO5
  pkg> test FiniteDiffWENO5
```
