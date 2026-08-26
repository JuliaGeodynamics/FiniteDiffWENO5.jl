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
    WENOScheme(c0::AbstractArray{T, N}; boundary::NTuple=ntuple(i -> 0, N*2), stag::Bool=false,  multithreading::Bool=false) where {T, N}

Structure containing the Weighted Essentially Non-Oscillatory (WENO) scheme of order 5 constants and arrays for N-dimensional data of type T. The formulation is from Borges et al. 2008.

# Arguments
- `c0::AbstractArray{T, N}`: The input field for which the WENO scheme is to be created. Only used to get the type and size.
- `boundary::NTuple{2N, Int}`: A tuple specifying the boundary conditions for each dimension (0: homogeneous Dirichlet, 1: homogeneous Neumann, 2: periodic). Default to homogeneous Dirichlet (0).
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
- `boundary::NTuple{N_boundary, Int}`: Boundary conditions for each dimension (0: homogeneous Dirichlet, 1: homogeneous Neumann, 2: periodic). Default to homogeneous Dirichlet.
- `lim_ZS::Bool`: Whether to use the Zhang-Shu limiter.
- `multithreading::Bool`: Whether to use multithreading (only for 2D and 3D).
- `fl::NamedTuple`: Fluxes in the left direction for each dimension.
- `fr::NamedTuple`: Fluxes in the right direction for each dimension.
- `du::AbstractArray{T, N}`: Semi-discretisation of the advection term.
- `ut::AbstractArray{T, N}`: Temporary array for intermediate calculations using Runge-Kutta.
"""
function WENOScheme(c0::AbstractArray{T, N}; boundary::NTuple = ntuple(i -> 0, N * 2), stag::Bool = false, lim_ZS::Bool = false, multithreading::Bool = true, upwind_mode::Bool = false) where {T, N}

    validate_boundary(boundary, N)

    # dimension labels
    labels = (:x, :y, :z)[1:min(N, 3)]
    sizes = size(c0)

    # allocate a zeroed buffer of the same array type as `c0`
    zeros_like(dims) = fill!(similar(c0, T, dims), zero(T))

    # construct NamedTuples for left and right fluxes
    fl = NamedTuple{labels}(ntuple(d -> zeros_like(flux_size(sizes, d, N)), min(N, 3)))
    fr = NamedTuple{labels}(ntuple(d -> zeros_like(flux_size(sizes, d, N)), min(N, 3)))

    # semi-discretisation array
    du = zeros_like(sizes)

    # temporary array for Runge-Kutta
    ut = zeros_like(sizes)

    # boundary conditions tuple length
    N_boundary = length(boundary)

    TFlux = typeof(fl)
    TArray = typeof(du)

    return WENOScheme{T, TArray, TFlux, N_boundary}(stag = stag, boundary = boundary, lim_ZS = lim_ZS, multithreading = multithreading, upwind_mode = upwind_mode, fl = fl, fr = fr, du = du, ut = ut)
end
