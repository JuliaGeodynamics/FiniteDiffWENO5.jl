# FiniteDiffWENO5

[![Build Status](https://github.com/JuliaGeodynamics/FiniteDiffWENO5.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGeodynamics/FiniteDiffWENO5.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![][docs-dev-img]][docs-dev-url]
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://juliageodynamics.github.io/FiniteDiffWENO5.jl/dev/

FiniteDiffWENO5.jl is a Julia package that implements a finite difference fifth order Weighted Essentially Non-Oscillatory (WENO) method on regular grids for advection terms in partial differential equations for 1D, 2D, and 3D problems. The current implementation is based on the WENO-Z scheme from [Borges et al. (2008)](10.1016/j.jcp.2007.11.038).

The package solves both the non-conservative form of the advection term ($\mathbf{v} \cdot \nabla u$) and the conservative form ($`\nabla \cdot (\mathbf{v} u)`$), independently of whether the advection velocity is collocated with $u$ or staggered on the sides of the cells (all four combinations are fifth-order accurate). The time integration is performed using a third-order Strong Stability Preserving Runge-Kutta (SSP-RK3) method. Periodic, extrapolated, and prescribed-inflow boundaries are supported.

The core of the package is written in pure Julia, focusing on performance using CPUs but GPU support is available using KernelAbstractions.jl and Chmy.jl via 2 separate extensions.

## Installation

FiniteDiffWENO5.jl is a registered Julia package and can be installed directly using the package manager:

```julia-repl
julia>]
  pkg> add FiniteDiffWENO5
```

And you can test the package with:

```julia-repl
julia>]
  pkg> test FiniteDiffWENO5
```

## Features

The main API consists of `WENOScheme`, `MultiphaseWENOScheme`, and `WENO_step!`.
`WENOScheme` transports one scalar or several unrelated fields. `MultiphaseWENOScheme`
simultaneously transports material fractions constrained by `0 ≤ ϕₖ ≤ 1` and
`Σₖϕₖ = 1`. The grid and initial conditions are defined by the user.

## Example

To see more examples, refer to the folder examples or the test folder.
Here is a simple example of using the package to solve the 1D linear advection equation with periodic boundary conditions and classical initial conditions:

```julia
using FiniteDiffWENO5
using GLMakie

# Number of grid points
nx = 200

# domain size
x_min = -1.0
x_max = 1.0
Lx = x_max - x_min

x = range(x_min, stop = x_max, length = nx)

# Courant number
CFL = 0.4
period = 4

# Parameters for Shu test
z = -0.7
δ = 0.005
β = log(2) / (36 * δ^2)
v = 0.5
α = 10

# Functions
G(x, β, z) = exp.(-β .* (x .- z) .^ 2)
F(x, α, a) = sqrt.(max.(1 .- α^2 .* (x .- a) .^ 2, 0.0))

# Grid x assumed defined
c0_vec = zeros(length(x))

# Gaussian-like smooth bump at x in [-0.8, -0.6]
idx = (x .>= -0.8) .& (x .<= -0.6)
c0_vec[idx] .= (1 / 6) .* (G(x[idx], β, z - δ) .+ 4 .* G(x[idx], β, z) .+ G(x[idx], β, z + δ))

# Heaviside step at x in [-0.4, -0.2]
idx = (x .>= -0.4) .& (x .<= -0.2)
c0_vec[idx] .= 1.0

# Piecewise linear ramp at x in [0, 0.2]
# Triangular spike at x=0.1, base width 0.2
idx = abs.(x .- 0.1) .<= 0.1
c0_vec[idx] .= 1 .- 10 .* abs.(x[idx] .- 0.1)

# Elliptic/smooth bell at x in [0.4, 0.6]
idx = (x .>= 0.4) .& (x .<= 0.6)
c0_vec[idx] .= (1 / 6) .* (F(x[idx], α, v - δ) .+ 4 .* F(x[idx], α, v) .+ F(x[idx], α, v + δ))

c = copy(c0_vec)
# Here we create a WENO scheme for a staggered grid with periodic boundaries.
# stag = true means that the advection velocity is defined on the sides
# of the cells and should be of size nx+1 compared to the scalar field u.
weno = WENOScheme(c; form = :conservative, boundary = (PeriodicBC(), PeriodicBC()), stag = true)

# advection velocity, here we use a constant velocity of 1.0.
# It should be provided as a NamedTuple
v = (; x = ones(nx + 1))

# grid size
Δx = x[2] - x[1]
Δt = CFL * Δx^(5 / 3)

tmax = period * (Lx + Δx) / maximum(abs.(v.x))

t = 0

# timeloop
while t < tmax
    # here, u is updated in place and contains the solution
    # at the next time step after the call to WENO_step!
    WENO_step!(c, v, weno, Δt, Δx)

    t += Δt

    if t + Δt > tmax
        Δt = tmax - t
    end
end
```

Which produces the following result:

![](/docs/src/assets/1D_linear_advection.png)

## Boundary conditions

Boundary tuples are ordered as lower/upper faces in every dimension. For 2D
this is `(west, east, bottom, top)`. Three typed conditions are available:

```julia
PeriodicBC()                 # wrap the field across the domain
ExtrapolateBC()              # constant continuation from the interior
PrescribedInflowBC(value)    # prescribed exterior state at inflow only
```

`PrescribedInflowBC` is sign-aware through the upwind numerical flux. Its value
is selected when the normal velocity enters the domain and ignored when the
same face is outflow. The value may be a scalar or an array over the tangential
face dimensions:

```julia
west_profile = range(300.0, 500.0, length = ny)
bc = AdvectionBC(
    west = PrescribedInflowBC(west_profile),
    east = ExtrapolateBC(),
    bot = ExtrapolateBC(),
    top = ExtrapolateBC(),
)
weno = WENOScheme(c; form = :conservative, boundary = bc, stag = true)
```

Legacy integer tuples remain accepted. Codes `0` and `1` normalize to
`ExtrapolateBC()`, matching their historical numerical behavior, while code `2`
normalizes to `PeriodicBC()`.

## Multi-field advection

If you have multiple scalar fields (e.g. different chemical components) that share the same velocity, you can advect them all in a single call by passing a tuple of arrays. The same `WENOScheme` buffers are reused for each field, so there is no extra memory overhead. Each field gets its own `u_min` / `u_max` bounds for the Zhang-Shu limiter.

```julia
using FiniteDiffWENO5

nx, ny = 200, 200
Lx = 1.0
Δx, Δy = Lx / nx, Lx / ny

# Three chemical components with different initial distributions
c1 = rand(nx, ny)
c2 = zeros(nx, ny)
c3 = ones(nx, ny)

# Shared velocity field
v = (; x = ones(nx, ny), y = 0.5 .* ones(nx, ny))

# Create the WENO scheme from any one of the fields (they must all have the same size and type)
weno = WENOScheme(c1; form = :nonconservative, boundary = (2, 2, 2, 2), stag = false)

Δt = 0.7 * min(Δx, Δy)^(5 / 3)

# Advect all three fields in one call with per-field limiter bounds
WENO_step!((c1, c2, c3), v, weno, Δt, Δx, Δy;
    u_min = (0.0, 0.0, 0.0),
    u_max = (1.0, 1.0, 1.0))
```

This also works with the KernelAbstractions and Chmy backends:

```julia
# KernelAbstractions
WENO_step!((c1, c2, c3), v, weno, Δt, Δx, Δy, backend;
    u_min = (0.0, 0.0, 0.0), u_max = (1.0, 1.0, 1.0))

# Chmy
WENO_step!((c1, c2, c3), v, weno, Δt, Δx, Δy, grid, arch;
    u_min = (0.0, 0.0, 0.0), u_max = (1.0, 1.0, 1.0))
```

The tuple interface above treats the fields independently and does not preserve their
sum. For material fractions, use the simultaneous simplex operator:

```julia
ϕ1 = fill(0.2, nx, ny)
ϕ2 = fill(0.3, nx, ny)
ϕ3 = 1 .- ϕ1 .- ϕ2
phases = (ϕ1, ϕ2, ϕ3)

bc = ntuple(_ -> PeriodicBC(), 4)
scheme = MultiphaseWENOScheme(phases; boundary = bc, stag = true)
WENO_step!(phases, v, scheme, Δt, Δx, Δy)
```

The input must already be a valid composition. Under a stable explicit CFL, shared
WENO-Z weights and one common convex limiter preserve the bounds and pointwise sum
without post-step clipping or renormalisation. See
`examples/2D/2D_multiphase_rotation.jl` for a complete three-phase example.

## Funding & author

The development of this package was supported by the TRIGGER project funded by the German Federal Ministry for Economic Affairs and Energy (BMWK).

Author: Hugo Dominguez (hdomingu@uni-mainz.de).
