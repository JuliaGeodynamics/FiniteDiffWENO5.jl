# FiniteDiffWENO5

[![Build Status](https://github.com/Iddingsite/FiniteDiffWENO5.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Iddingsite/FiniteDiffWENO5.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![][docs-dev-img]][docs-dev-url]
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://juliageodynamics.github.io/FiniteDiffWENO5.jl/dev/

FiniteDiffWENO5.jl is a Julia package that implements a finite difference fifth order Weighted Essentially Non-Oscillatory (WENO) method on regular grids for advection terms in partial differential equations for 1D, 2D, and 3D problems. The current implementation is based on the WENO-Z scheme from [Borges et al. (2008)](10.1016/j.jcp.2007.11.038).

Currently, the package focuses on non-conservative form of the advection terms ($\mathbf{v} \cdot \nabla u$) on collocated grid, and conservative form ($\nabla \cdot$ ($\mathbf{v} u$)) on staggered grid with the advection velocity located on the sides of the cells. The time integration is performed using a third-order Strong Stability Preserving Runge-Kutta (SSP-RK3) method. Periodic and homogeneous Neumann and Dirichlet boundaries are currently supported.

The core of the package is written in pure Julia, focusing on performance using CPUs but GPU support is available using KernelAbstractions.jl and Chmy.jl via 2 separate extensions.

## Features

The package currently exports only two main functions: `WENOScheme()`, that is used to create a WENO scheme struct containing all the necessary information for the WENO method, and `WENO_step!()`, that performs one step of the time integration using the WENO-Z method and a 3rd-order Runge-Kutta method. The grid and the initial condition must be defined by the user.

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
# here we create a WENO scheme for staggered grid, boundary (2,2) means periodic BCs on both sides.
# 0 means homogeneous Neumann and 1 means homogeneous Dirichlet BCs.
# stag = true means that the advection velocity is defined on the sides
# of the cells and should be of size nx+1 compared to the scalar field u.
weno = WENOScheme(c; boundary = (2, 2), stag = true)

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


## Funding & author

The development of this package was supported by the TRIGGER project funded by the German Federal Ministry for Economic Affairs and Energy (BMWK).

Author: Hugo Dominguez (hdomingu@univ-mainz.de).
