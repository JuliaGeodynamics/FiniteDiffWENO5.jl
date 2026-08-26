module ChmyExt
using FiniteDiffWENO5
using FiniteDiffWENO5: zhang_shu_limit, weno5_reconstruction_upwind, weno5_reconstruction_downwind, validate_boundary, left_index, right_index
using MuladdMacro
using Chmy
using KernelAbstractions

import FiniteDiffWENO5: WENOScheme, WENO_step!

# Velocity can be passed either as a plain NamedTuple of fields or as a Chmy `VectorField`
# (which behaves like a NamedTuple via `getproperty` but isn't one).
const Velocity1D = Union{NamedTuple{(:x,), <:Tuple{<:AbstractField{<:Real, 1}}}, VectorField{1, <:Tuple{<:AbstractField{<:Real, 1}}}}
const Velocity2D = Union{NamedTuple{(:x, :y), <:Tuple{Vararg{AbstractField{<:Real}, 2}}}, VectorField{2, <:Tuple{Vararg{AbstractField{<:Real}, 2}}}}
const Velocity3D = Union{NamedTuple{(:x, :y, :z), <:Tuple{Vararg{AbstractField{<:Real}, 3}}}, VectorField{3, <:Tuple{Vararg{AbstractField{<:Real}, 3}}}}


"""
WENOScheme(u::AbstractField{T, N},
           grid::StructuredGrid;
           boundary=(2, 2), stag=true) where {T, N}

Create a WENO scheme structure for the given field `u` on the specified `grid` using Chmy.jl.

# Arguments
- `c0::AbstractField{T, N}`: Input field for which the WENO scheme is to be created. Only used to get the type and size.
- `grid::StructuredGrid`: Computational grid.
- `boundary::NTuple{2N, Int}`: Tuple specifying the boundary conditions for each dimension (0: homogeneous Dirichlet, 1: homogeneous Neumann, 2: periodic). Default is periodic (2).
- `stag::Bool`: Whether the grid is staggered (velocities on cell faces) or not (velocities on cell centers).
"""
function WENOScheme(c0::AbstractField{T, N}, grid::StructuredGrid; boundary::NTuple = (2, 2), stag::Bool = true, lim_ZS::Bool = false, upwind_mode = false) where {T, N}

    validate_boundary(boundary, N)

    # multithreading is always on in this case with chmy.jl
    multithreading = true

    backend = get_backend(c0)

    N_boundary = 2 * N

    fl = VectorField(backend, grid)
    fr = VectorField(backend, grid)
    du = Field(backend, grid, Center())
    ut = Field(backend, grid, Center())

    TFlux = typeof(fl)
    TArray = typeof(du)

    return WENOScheme{T, TArray, TFlux, N_boundary}(stag = stag, boundary = boundary, multithreading = multithreading, lim_ZS = lim_ZS, fl = fl, fr = fr, du = du, ut = ut, upwind_mode = upwind_mode)
end

function WENOScheme(c0::AbstractField; kwargs...)
    error(
        """
        You called `WENOScheme(c0)` with a `$(typeof(c0))`, which is a subtype of `AbstractField`.

        To construct a WENO scheme for Chmy.jl fields, you must also provide the computational grid:
            WENOScheme(c0::AbstractField, grid::StructuredGrid; kwargs...)

        Example:
            grid = UniformGrid(arch; origin=(0.0, 0.0), extent=(Lx, Lx), dims=(nx, ny))
            weno = WENOScheme(c0, grid; boundary=(2,2,2,2), stag=false)
        """
    )
end

# I am reimporting the files from the discretisation of KA.jl here.
# I didn't find a better way because @kernel functions are not real functions that we can extend from the base package
# and we can't access an extension from another extension
# but at least we don't have to duplicate physically the code
include("KAExt1D.jl")
include("KAExt2D.jl")
include("KAExt3D.jl")


"""
    WENO_step!(u::T_field,
               v::Velocity1D,
               weno::FiniteDiffWENO5.WENOScheme,
               Δt, Δx,
               grid::StructuredGrid, arch;
               u_min = 0.0, u_max = 0.0) where {T_field <: AbstractField{<:Real, 1}}

Advance the solution `u` by one time step using the 3rd-order Runge-Kutta method with WENO5 spatial discretization using Chmy.jl fields in 1D.

# Arguments
- `u::T_field`: Current solution field to be updated in place.
- `v::Velocity1D`: Velocity field (can be staggered or not based on `weno.stag`).
- `weno::WENOScheme`: WENO scheme structure containing necessary parameters and fields.
- `Δt`: Time step size.
- `Δx`: Spatial grid size.
- `grid::StructuredGrid`: Computational grid from Chmy.
- `arch::Backend`: The KernelAbstractions backend in use (e.g., CPU(), CUDABackend(), etc.).
- `u_min`: Minimum value of `u` for the Zhang-Shu positivity limiter.
- `u_max`: Maximum value of `u` for the Zhang-Shu positivity limiter.
"""
function WENO_step!(u::T_field, v::Velocity1D, weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, grid::StructuredGrid, arch; u_min = 0.0, u_max = 0.0) where {T_field <: AbstractField{<:Real, 1}}

    @assert get_backend(u) == get_backend(v.x)

    launch = Launcher(arch, grid)

    #! do things here for halos and such for clusters for boundaries probably

    nx = grid.axes[1].length
    Δx_ = inv(Δx)

    (; ut, du, fl, fr, stag, lim_ZS, boundary, χ, γ, ζ, ϵ, upwind_mode) = weno

    if !upwind_mode

        launch(arch, grid, WENO_flux_KA_1D => (fl.x, fr.x, u, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
        launch(arch, grid, WENO_semi_discretisation_weno5_KA_1D! => (du, fl, fr, v, stag, Δx_, grid))

        interior(ut) .= @muladd interior(u) .- Δt .* interior(du)

        launch(arch, grid, WENO_flux_KA_1D => (fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
        launch(arch, grid, WENO_semi_discretisation_weno5_KA_1D! => (du, fl, fr, v, stag, Δx_, grid))

        interior(ut) .= @muladd 0.75 .* interior(u) .+ 0.25 .* interior(ut) .- 0.25 .* Δt .* interior(du)

        launch(arch, grid, WENO_flux_KA_1D => (fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
        launch(arch, grid, WENO_semi_discretisation_weno5_KA_1D! => (du, fl, fr, v, stag, Δx_, grid))

        interior(u) .= @muladd inv(3.0) .* interior(u) .+ 2.0 / 3.0 .* interior(ut) .- 2.0 / 3.0 .* Δt .* interior(du)
    else
        launch(arch, grid, upwind_update_KA_1D! => (u, v, nx, Δx_, Δt, stag, boundary, grid))
    end

    return nothing
end


"""
    WENO_step!(u::T_field,
               v::Velocity2D,
               weno::FiniteDiffWENO5.WENOScheme,
               Δt, Δx,
               grid::StructuredGrid, arch;
               u_min = 0.0, u_max = 0.0) where {T_field <: AbstractField{<:Real, 2}}

Advance the solution `u` by one time step using the 3rd-order Runge-Kutta method with WENO5 spatial discretization using Chmy.jl fields in 2D.

# Arguments
- `u::T_field`: Current solution field to be updated in place.
- `v::Velocity2D`: The velocity field (can be staggered or not based on `weno.stag`).
- `weno::WENOScheme`: WENO scheme structure containing necessary parameters and fields.
- `Δt`: Time step size.
- `Δx`: Spatial grid size.
- `grid::StructuredGrid`: Computational grid from Chmy.
- `arch::Backend`: The KernelAbstractions backend in use (e.g., CPU(), CUDABackend(), etc.).
- `u_min`: Minimum value of `u` for the Zhang-Shu positivity limiter.
- `u_max`: Maximum value of `u` for the Zhang-Shu positivity limiter.
"""
function WENO_step!(u::T_field, v::Velocity2D, weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, Δy, grid::StructuredGrid, arch; u_min = 0.0, u_max = 0.0) where {T_field <: AbstractField{<:Real, 2}}

    @assert get_backend(u) == get_backend(v.x)
    @assert get_backend(u) == get_backend(v.y)

    launch = Launcher(arch, grid)

    #! do things here for halos and such for clusters for boundaries probably

    nx = grid.axes[1].length
    ny = grid.axes[2].length
    Δx_ = inv(Δx)
    Δy_ = inv(Δy)

    (; ut, du, fl, fr, stag, lim_ZS, boundary, χ, γ, ζ, ϵ, upwind_mode) = weno

    if !upwind_mode
        launch(arch, grid, WENO_flux_KA_2D_x => (fl.x, fr.x, u, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
        launch(arch, grid, WENO_flux_KA_2D_y => (fl.y, fr.y, u, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
        launch(arch, grid, WENO_semi_discretisation_weno5_KA_2D! => (du, fl, fr, v, stag, Δx_, Δy_, grid))

        interior(ut) .= @muladd interior(u) .- Δt .* interior(du)

        launch(arch, grid, WENO_flux_KA_2D_x => (fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
        launch(arch, grid, WENO_flux_KA_2D_y => (fl.y, fr.y, ut, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
        launch(arch, grid, WENO_semi_discretisation_weno5_KA_2D! => (du, fl, fr, v, stag, Δx_, Δy_, grid))

        interior(ut) .= @muladd 0.75 .* interior(u) .+ 0.25 .* interior(ut) .- 0.25 .* Δt .* interior(du)

        launch(arch, grid, WENO_flux_KA_2D_x => (fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
        launch(arch, grid, WENO_flux_KA_2D_y => (fl.y, fr.y, ut, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
        launch(arch, grid, WENO_semi_discretisation_weno5_KA_2D! => (du, fl, fr, v, stag, Δx_, Δy_, grid))

        interior(u) .= @muladd inv(3.0) .* interior(u) .+ 2.0 / 3.0 .* interior(ut) .- 2.0 / 3.0 .* Δt .* interior(du)
    else
        launch(arch, grid, upwind_update_KA_2D! => (u, v, nx, ny, Δx_, Δy_, Δt, stag, boundary, grid))
    end

    return nothing
end


"""
    WENO_step!(u::T_field,
               v::Velocity3D,
               weno::FiniteDiffWENO5.WENOScheme,
               Δt, Δx, Δy, Δz,
               grid::StructuredGrid, arch;
               u_min = 0.0, u_max = 0.0) where T_field <: AbstractArray{<:Real, 3}

Advance the solution `u` by one time step using the 3rd-order Runge-Kutta method with WENO5 spatial discretization using Chmy.jl fields in 3D.

# Arguments
- `u::T_field`: Current solution field to be updated in place.
- `v::Velocity3D`: Velocity field (can be staggered or not based on `weno.stag`).
- `weno::WENOScheme`: WENO scheme structure containing necessary parameters and fields.
- `Δt`: Time step size.
- `Δx`: Spatial grid size.
- `Δy`: Spatial grid size.
- `Δz`: Spatial grid size.
- `grid::StructuredGrid`: Computational grid from Chmy.
- `arch::Backend`: The KernelAbstractions backend in use (e.g., CPU(), CUDABackend(), etc.).
- `u_min`: Minimum value of `u` for the Zhang-Shu positivity limiter.
- `u_max`: Maximum value of `u` for the Zhang-Shu positivity limiter.
"""
function WENO_step!(u::T_field, v::Velocity3D, weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, Δy, Δz, grid::StructuredGrid, arch; u_min = 0.0, u_max = 0.0) where {T_field <: AbstractArray{<:Real, 3}}

    @assert get_backend(u) == get_backend(v.x)
    @assert get_backend(u) == get_backend(v.y)
    @assert get_backend(u) == get_backend(v.z)

    launch = Launcher(arch, grid)

    #! do things here for halos and such for clusters for boundaries probably

    nx = grid.axes[1].length
    ny = grid.axes[2].length
    nz = grid.axes[3].length
    Δx_ = inv(Δx)
    Δy_ = inv(Δy)
    Δz_ = inv(Δz)

    (; ut, du, fl, fr, stag, lim_ZS, boundary, χ, γ, ζ, ϵ, upwind_mode) = weno

    if !upwind_mode
        launch(arch, grid, WENO_flux_KA_3D_x => (fl.x, fr.x, u, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
        launch(arch, grid, WENO_flux_KA_3D_y => (fl.y, fr.y, u, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
        launch(arch, grid, WENO_flux_KA_3D_z => (fl.z, fr.z, u, boundary, nz, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
        launch(arch, grid, WENO_semi_discretisation_weno5_KA_3D! => (du, fl, fr, v, stag, Δx_, Δy_, Δz_, grid))

        interior(ut) .= @muladd interior(u) .- Δt .* interior(du)

        launch(arch, grid, WENO_flux_KA_3D_x => (fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
        launch(arch, grid, WENO_flux_KA_3D_y => (fl.y, fr.y, ut, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
        launch(arch, grid, WENO_flux_KA_3D_z => (fl.z, fr.z, ut, boundary, nz, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
        launch(arch, grid, WENO_semi_discretisation_weno5_KA_3D! => (du, fl, fr, v, stag, Δx_, Δy_, Δz_, grid))

        interior(ut) .= @muladd 0.75 .* interior(u) .+ 0.25 .* interior(ut) .- 0.25 .* Δt .* interior(du)

        launch(arch, grid, WENO_flux_KA_3D_x => (fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
        launch(arch, grid, WENO_flux_KA_3D_y => (fl.y, fr.y, ut, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
        launch(arch, grid, WENO_flux_KA_3D_z => (fl.z, fr.z, ut, boundary, nz, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
        launch(arch, grid, WENO_semi_discretisation_weno5_KA_3D! => (du, fl, fr, v, stag, Δx_, Δy_, Δz_, grid))

        interior(u) .= @muladd inv(3.0) .* interior(u) .+ 2.0 / 3.0 .* interior(ut) .- 2.0 / 3.0 .* Δt .* interior(du)
    else
        launch(arch, grid, upwind_update_KA_3D! => (u, v, nx, ny, nz, Δx_, Δy_, Δz_, Δt, stag, boundary, grid))
    end

    return nothing
end


# Multi-field advection (u = (c1, c2, ...) sharing v and WENOScheme buffers) is
# handled generically for every dimension and backend by the `WENO_step!(u::Tuple, ...)`
# method in FiniteDiffWENO5's src/utils.jl.

end
