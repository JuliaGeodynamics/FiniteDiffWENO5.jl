abstract type AbstractWENO end

@kwdef struct WENOScheme{T, TArray, TFlux, N_boundary} <: AbstractWENO
    # upwind and downwind constants
    γ::NTuple{3, T} = T.((0.1, 0.6, 0.3))
    # betas' constants
    χ::NTuple{2, T} = T.((13 / 12, 1 / 4))
    # stencil weights
    ζ::NTuple{5, T} = T.((1 / 3, 7 / 6, 11 / 6, 1 / 6, 5 / 6))
    # tolerance to machine precision of the type T
    ϵ::T = eps(T)
    # staggered grid or not (velocities on cell faces or cell centers)
    stag::Bool
    # use Zhang-Shu limiter
    lim_ZS::Bool
    # boundary conditions
    boundary::NTuple{N_boundary, Int}
    # multithreading
    multithreading::Bool
    # simple upwind for debugging
    upwind_mode::Bool = false
    # fluxes as NamedTuples
    fl::TFlux
    fr::TFlux
    # semi-discretisation of the advection term
    du::TArray
    # temporary array for the time stepping
    ut::TArray
end

"""
    WENOScheme(c0::Array{T, N}; boundary::NTuple=ntuple(i -> 0, N*2), stag::Bool=false,  multithreading::Bool=false) where {T, N}

Structure containing the Weighted Essentially Non-Oscillatory (WENO) scheme of order 5 constants and arrays for N-dimensional data of type T. The formulation is from Borges et al. 2008.

# Arguments
- `c0::Array{T, N}`: The input field for which the WENO scheme is to be created. Only used to get the type and size.
- `boundary::NTuple{2N, Int}`: A tuple specifying the boundary conditions for each dimension (0: homogeneous Neumann, 1: homogeneous Dirichlet, 2: periodic). Default to homogeneous Neumann (0).
- `stag::Bool`: Whether the grid is staggered (velocities on cell faces) or not (velocities on cell centers). Default to false.
- `lim_ZS::Bool`: Whether to use the Zhang-Shu (2010) limiter. Default to false.
- `multithreading::Bool`: Whether to use multithreading (only for 2D and 3D). Default to true.
- `upwind_mode::Bool`: Whether to use a simple upwind scheme for debugging purposes. Default to false.

# Fields
- `γ::NTuple{3, T}`: Upwind and downwind constants.
- `χ::NTuple{2, T}`: Betas' constants.
- `ζ::NTuple{5, T}`: Stencil weights.
- `ϵ::T`: Tolerance, fixed to machine precision.
- `stag::Bool`: Whether the grid is staggered (velocities on cell faces) or not (velocities on cell centers).
- `boundary::NTuple{N_boundary, Int}`: Boundary conditions for each dimension (0: homogeneous Neumann, 1: homogeneous Dirichlet, 2: periodic). Default to homogeneous Neumann.
- `lim_ZS::Bool`: Whether to use the Zhang-Shu limiter.
- `multithreading::Bool`: Whether to use multithreading (only for 2D and 3D).
- `fl::NamedTuple`: Fluxes in the left direction for each dimension.
- `fr::NamedTuple`: Fluxes in the right direction for each dimension.
- `du::Array{T, N}`: Semi-discretisation of the advection term.
- `ut::Array{T, N}`: Temporary array for intermediate calculations using Runge-Kutta.
"""
function WENOScheme(c0::Array{T, N}; boundary::NTuple = ntuple(i -> 0, N * 2), stag::Bool = false, lim_ZS::Bool = false, multithreading::Bool = true, upwind_mode::Bool = false) where {T, N}

    # check that boundary conditions are correctly defined
    @assert length(boundary) == 2N "Boundary conditions must be a tuple of length $(2N) for $(N)D data."
    # check that boundary conditions are either 0 (homogeneous Neumann) or 1 (homogeneous Dirichlet) or 2 (periodic)
    @assert all(b in (0, 1, 2) for b in boundary) "Boundary conditions must be either 0 (homogeneous Neumann), 1 (homogeneous Dirichlet) or 2 (periodic)."

    # dimension labels
    labels = (:x, :y, :z)[1:min(N, 3)]
    sizes = size(c0)

    # helper to expand size in a given dimension
    function flux_size(d)
        return ntuple(i -> sizes[i] + (i == d ? 1 : 0), min(N, 3))
    end

    # construct NamedTuples for left and right fluxes
    fl = NamedTuple{labels}(ntuple(d -> zeros(T, flux_size(d)), min(N, 3)))
    fr = NamedTuple{labels}(ntuple(d -> zeros(T, flux_size(d)), min(N, 3)))

    # semi-discretisation array
    du = zeros(T, size(c0))

    # temporary array for Runge-Kutta
    ut = zeros(T, size(c0))

    # boundary conditions tuple length
    N_boundary = 2 * N

    TFlux = typeof(fl)
    TArray = typeof(du)

    return WENOScheme{T, TArray, TFlux, N_boundary}(stag = stag, boundary = boundary, lim_ZS = lim_ZS, multithreading = multithreading, upwind_mode = upwind_mode, fl = fl, fr = fr, du = du, ut = ut)
end
