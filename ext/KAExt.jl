module KAExt
using FiniteDiffWENO5
using FiniteDiffWENO5: zhang_shu_limit, weno5_reconstruction_upwind, weno5_reconstruction_downwind
using MuladdMacro
using UnPack
using KernelAbstractions

import FiniteDiffWENO5: WENOScheme, WENO_step!

# stolen from Chmy.jl to reproduce the behaviour with KA.jl
struct Offset{O} end

Offset(o::Vararg{Integer}) = Offset{o}()
Offset(o::Tuple{Vararg{Integer}}) = Offset{o}()
Offset(o::CartesianIndex) = Offset{o.I}()
Offset() = Offset{0}()

Base.:+(::Offset{O1}, ::Offset{O2}) where {O1, O2} = Offset((O1 .+ O2)...)
Base.:+(::Offset{O}, tp::Tuple{Vararg{Integer}}) where {O} = O .+ tp
Base.:+(::Offset{O}, tp::CartesianIndex) where {O} = CartesianIndex(O .+ Tuple(tp))

Base.:+(tp, off::Offset) = off + tp

const Offset0 = Offset{(0,)}()

"""
WENOScheme(c0::AbstractArray{T, N}, backend::Backend; boundary=(2, 2), stag=true) where {T, N}

Create a WENO scheme structure for the given field `c` using the specified `backend` from KernelAbstractions.jl.

# Arguments
- `c0::AbstractArray{T, N}`: The input field for which the WENO scheme is to be created. Only used to get the type and size.
- `backend::Backend`: The KernelAbstractions backend to be used (e.g., CPU(), CUDA(), etc.).
- `boundary::NTuple{2N, Int}`: A tuple specifying the boundary conditions for each dimension (0: homogeneous Neumann, 1: homogeneous Dirichlet, 2: periodic). Default is periodic (2).
- `stag::Bool`: Whether the grid is staggered (velocities on cell faces) or not (velocities on cell centers).
"""
function WENOScheme(c0::AbstractArray{T, N}, backend::Backend; boundary::NTuple = (2, 2), stag::Bool = true, lim_ZS::Bool = false, upwind_mode::Bool = false) where {T, N}

    @assert get_backend(c0) == backend "The type of the input field must match the specified backend."

    # check that boundary conditions are correctly defined
    @assert length(boundary) == 2N "Boundary conditions must be a tuple of length $(2N) for $(N)D data."
    # check that boundary conditions are either 0 (homogeneous Neumann) or 1 (homogeneous Dirichlet) or 2 (periodic)
    @assert all(b in (0, 1, 2) for b in boundary) "Boundary conditions must be either 0 (homogeneous Neumann), 1 (homogeneous Dirichlet) or 2 (periodic)."

    # multithreading is always on in this case
    multithreading = true

    backend = get_backend(c0)

    N_boundary = 2 * N

    # helper to expand size in a given dimension
    @inline function flux_size(d)
        return ntuple(i -> size(c0, i) + (i == d ? 1 : 0), min(N, 3))
    end

    # construct NamedTuples for left and right fluxes
    labels = (:x, :y, :z)[1:min(N, 3)]
    fl = NamedTuple{labels}(ntuple(d -> KernelAbstractions.zeros(backend, T, flux_size(d)), min(N, 3)))
    fr = NamedTuple{labels}(ntuple(d -> KernelAbstractions.zeros(backend, T, flux_size(d)), min(N, 3)))

    du = KernelAbstractions.zeros(backend, T, size(c0))
    ut = KernelAbstractions.zeros(backend, T, size(c0))

    TFlux = typeof(fl)
    TArray = typeof(du)

    return WENOScheme{T, TArray, TFlux, N_boundary}(stag = stag, boundary = boundary, multithreading = multithreading, lim_ZS = lim_ZS, fl = fl, fr = fr, du = du, ut = ut, upwind_mode = upwind_mode)
end

include("KAExt1D.jl")
include("KAExt2D.jl")
include("KAExt3D.jl")

"""
    WENO_step!(u::T_KA,
               v::NamedTuple{names, <:Tuple{<:T_KA}},
               weno::FiniteDiffWENO5.WENOScheme, Δt, Δx,
               backend::Backend;
               u_min = 0.0, u_max = 0.0) where T_KA <: AbstractField{<:Real} where names

Advance the solution `u` by one time step using the 3rd-order Runge-Kutta method with WENO5 spatial discretization using Chmy.jl fields in 1D.

# Arguments
- `u::T_field`: The current solution field to be updated in place.
- `v::NamedTuple{names, <:Tuple{<:T_field}}`: The velocity field (can be staggered or not based on `weno.stag`). Needs to be a NamedTuple with field `:x`.
- `weno::WENOScheme`: The WENO scheme structure containing necessary parameters and fields.
- `Δt`: The time step size.
- `Δx`: The spatial grid size.
- `backend::Backend`: The KernelAbstractions backend in use (e.g., CPU(), CUDABackend(), etc.).
- `u_min`: Minimum value of `u` for the Zhang-Shu positivity limiter.
- `u_max`: Maximum value of `u` for the Zhang-Shu positivity limiter.
"""
function WENO_step!(u::T_KA, v::NamedTuple{(:x,), <:Tuple{<:AbstractArray{<:Real}}}, weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, backend::Backend; u_min = 0.0, u_max = 0.0) where {T_KA <: AbstractVector{<:Real}}

    @assert get_backend(u) == backend
    @assert get_backend(v.x) == backend

    #! do things here for halos and such for clusters for boundaries probably

    @unpack ut, du, fl, fr, stag, lim_ZS, boundary, χ, γ, ζ, ϵ, upwind_mode = weno

    nx = size(u, 1)
    Δx_ = inv(Δx)

    if !upwind_mode

        kernel_flux_1D = WENO_flux_KA_1D(backend)
        kernel_semi_discretisation_1D = WENO_semi_discretisation_weno5_KA_1D!(backend)

        kernel_flux_1D(fl.x, fr.x, u, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = length(fl.x))
        kernel_semi_discretisation_1D(du, fl, fr, v, stag, Δx_, nothing, Offset0, ndrange = length(du))

        ut .= @muladd u .- Δt .* du

        kernel_flux_1D(fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = length(fl.x))
        kernel_semi_discretisation_1D(du, fl, fr, v, stag, Δx_, nothing, Offset0, ndrange = length(du))

        ut .= @muladd 0.75 .* u .+ 0.25 .* ut .- 0.25 .* Δt .* du

        kernel_flux_1D(fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = length(fl.x))
        kernel_semi_discretisation_1D(du, fl, fr, v, stag, Δx_, nothing, Offset0, ndrange = length(du))

        u .= @muladd inv(3.0) .* u .+ 2.0 / 3.0 .* ut .- 2.0 / 3.0 .* Δt .* du
    else
        kernel_upwind = upwind_update_KA_1D!(backend)

        kernel_upwind(u, v, nx, Δx_, Δt, stag, boundary, nothing, Offset0, ndrange = length(u))

    end

    return nothing
end


"""
    WENO_step!(u::T_field,
               v::NamedTuple{(:x, :y), <:Tuple{Vararg{AbstractArray{<:Real}, 2}}},
               weno::FiniteDiffWENO5.WENOScheme,
               Δt, Δx, Δy,
               backend::Backend;
               u_min = 0.0, u_max = 0.0) where T_field <: AbstractField{<:Real} where names

Advance the solution `u` by one time step using the 3rd-order Runge-Kutta method with WENO5 spatial discretization using Chmy.jl fields in 2D.

# Arguments
- `u::T_KA`: Current solution field to be updated in place.
- `v::NamedTuple{(:x, :y), <:Tuple{Vararg{AbstractArray{<:Real}, 2}}}`: Velocity field (can be staggered or not based on `weno.stag`).
- `weno::WENOScheme`: WENO scheme structure containing necessary parameters and fields.
- `Δt`: Time step size.
- `Δx`: Spatial grid size.
- `Δy`: Spatial grid size.
- `backend::Backend`: KernelAbstractions backend in use (e.g., CPU(), CUDABackend(), etc.).
- `u_min`: Minimum value of `u` for the Zhang-Shu positivity limiter.
- `u_max`: Maximum value of `u` for the Zhang-Shu positivity limiter.
"""
function WENO_step!(u::T_KA, v::NamedTuple{(:x, :y), <:Tuple{Vararg{AbstractArray{<:Real}, 2}}}, weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, Δy, backend::Backend; u_min = 0.0, u_max = 0.0) where {T_KA <: AbstractArray{<:Real, 2}}

    @assert get_backend(u) == backend
    @assert get_backend(v.x) == backend
    @assert get_backend(v.y) == backend

    #! do things here for halos and such for clusters for boundaries probably

    @unpack ut, du, fl, fr, stag, lim_ZS, boundary, χ, γ, ζ, ϵ, upwind_mode = weno

    nx = size(u, 1)
    ny = size(u, 2)
    Δx_ = inv(Δx)
    Δy_ = inv(Δy)

    if !upwind_mode
        flx_l = size(fl.x)
        fly_l = size(fl.y)
        du_l = size(du)

        kernel_flux_2D_x = WENO_flux_KA_2D_x(backend)
        kernel_flux_2D_y = WENO_flux_KA_2D_y(backend)
        kernel_semi_discretisation_2D = WENO_semi_discretisation_weno5_KA_2D!(backend)

        kernel_flux_2D_x(fl.x, fr.x, u, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = flx_l)
        kernel_flux_2D_y(fl.y, fr.y, u, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = fly_l)
        kernel_semi_discretisation_2D(du, fl, fr, v, stag, Δx_, Δy_, nothing, Offset0, ndrange = du_l)

        ut .= @muladd u .- Δt .* du


        kernel_flux_2D_x(fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = flx_l)
        kernel_flux_2D_y(fl.y, fr.y, ut, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = fly_l)
        kernel_semi_discretisation_2D(du, fl, fr, v, stag, Δx_, Δy_, nothing, Offset0, ndrange = du_l)

        ut .= @muladd 0.75 .* u .+ 0.25 .* ut .- 0.25 .* Δt .* du

        kernel_flux_2D_x(fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = flx_l)
        kernel_flux_2D_y(fl.y, fr.y, ut, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = fly_l)
        kernel_semi_discretisation_2D(du, fl, fr, v, stag, Δx_, Δy_, nothing, Offset0, ndrange = du_l)

        u .= @muladd inv(3.0) .* u .+ 2.0 / 3.0 .* ut .- 2.0 / 3.0 .* Δt .* du
    else
        u_l = size(u)

        kernel_upwind = upwind_update_KA_2D!(backend)

        kernel_upwind(u, v, nx, ny, Δx_, Δy_, Δt, stag, boundary, nothing, Offset0, ndrange = u_l)
    end

    return nothing
end


"""
    WENO_step!(u::T_KA,
               v::NamedTuple{(:x, :y, :z), <:Tuple{Vararg{AbstractArray{<:Real}, 3}}},
               weno::FiniteDiffWENO5.WENOScheme,
               Δt, Δx, Δy, Δz,
               backend::Backend;
               u_min = 0.0, u_max = 0.0) where T_KA <: AbstractArray{<:Real, 3}

Advance the solution `u` by one time step using the 3rd-order Runge-Kutta method with WENO5 spatial discretization using Chmy.jl fields in 3D.

# Arguments
- `u::T_KA`: Current solution field to be updated in place.
- `v::NamedTuple{names, <:Tuple{<:T_KA}}`: Velocity field (can be staggered or not based on `weno.stag`). Needs to be a NamedTuple with fields `:x`, `:y` and `:z`.
- `weno::WENOScheme`: WENO scheme structure containing necessary parameters and fields.
- `Δt`: Time step size.
- `Δx`: Spatial grid size.
- `Δy`: Spatial grid size.
- `Δz`: Spatial grid size.
- `backend::Backend`: Computational backend to use (e.g., CPU, GPU).
"""
function WENO_step!(u::T_KA, v::NamedTuple{(:x, :y, :z), <:Tuple{Vararg{AbstractArray{<:Real}, 3}}}, weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, Δy, Δz, backend::Backend; u_min = 0.0, u_max = 0.0) where {T_KA <: AbstractArray{<:Real, 3}}

    @assert get_backend(u) == backend
    @assert get_backend(v.x) == backend
    @assert get_backend(v.y) == backend
    @assert get_backend(v.z) == backend

    #! do things here for halos and such for clusters for boundaries probably

    nx = size(u, 1)
    ny = size(u, 2)
    nz = size(u, 3)
    Δx_ = inv(Δx)
    Δy_ = inv(Δy)
    Δz_ = inv(Δz)

    @unpack ut, du, fl, fr, stag, lim_ZS, boundary, χ, γ, ζ, ϵ, upwind_mode = weno

    if !upwind_mode

        flx_l = size(fl.x)
        fly_l = size(fl.y)
        flz_l = size(fl.z)
        du_l = size(du)

        kernel_flux_3D_x = WENO_flux_KA_3D_x(backend)
        kernel_flux_3D_y = WENO_flux_KA_3D_y(backend)
        kernel_flux_3D_z = WENO_flux_KA_3D_z(backend)
        kernel_semi_discretisation_3D = WENO_semi_discretisation_weno5_KA_3D!(backend)


        kernel_flux_3D_x(fl.x, fr.x, u, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = flx_l)
        kernel_flux_3D_y(fl.y, fr.y, u, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = fly_l)
        kernel_flux_3D_z(fl.z, fr.z, u, boundary, nz, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = flz_l)
        kernel_semi_discretisation_3D(du, fl, fr, v, stag, Δx_, Δy_, Δz_, nothing, Offset0, ndrange = du_l)

        ut .= @muladd u .- Δt .* du

        kernel_flux_3D_x(fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = flx_l)
        kernel_flux_3D_y(fl.y, fr.y, ut, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = fly_l)
        kernel_flux_3D_z(fl.z, fr.z, ut, boundary, nz, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = flz_l)
        kernel_semi_discretisation_3D(du, fl, fr, v, stag, Δx_, Δy_, Δz_, nothing, Offset0, ndrange = du_l)

        ut .= @muladd 0.75 .* u .+ 0.25 .* ut .- 0.25 .* Δt .* du

        kernel_flux_3D_x(fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = flx_l)
        kernel_flux_3D_y(fl.y, fr.y, ut, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = fly_l)
        kernel_flux_3D_z(fl.z, fr.z, ut, boundary, nz, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = flz_l)
        kernel_semi_discretisation_3D(du, fl, fr, v, stag, Δx_, Δy_, Δz_, nothing, Offset0, ndrange = du_l)

        u .= @muladd inv(3.0) .* u .+ 2.0 / 3.0 .* ut .- 2.0 / 3.0 .* Δt .* du
    else
        u_l = size(u)

        kernel_upwind = upwind_update_KA_3D!(backend)

        kernel_upwind(u, v, nx, ny, nz, Δx_, Δy_, Δz_, Δt, stag, boundary, nothing, Offset0, ndrange = u_l)

    end

    return nothing
end


end
