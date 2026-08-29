module ChmyExt
using FiniteDiffWENO5
using FiniteDiffWENO5: zhang_shu_limit, limit_simplex, weno5_reconstruction_upwind, weno5_reconstruction_downwind, multiphase_reconstruction_upwind, multiphase_reconstruction_downwind, validate_boundary, validate_multiphase_boundary, normalize_boundary_faces, left_index, right_index, inflow_value, multiphase_inflow_value
using MuladdMacro
using Chmy
using KernelAbstractions

import FiniteDiffWENO5: WENOScheme, MultiphaseWENOScheme, WENO_step!

# Velocity can be passed either as a plain NamedTuple of fields or as a Chmy `VectorField`
# (which behaves like a NamedTuple via `getproperty` but isn't one).
const Velocity1D = Union{NamedTuple{(:x,), <:Tuple{<:AbstractField{<:Real, 1}}}, VectorField{1, <:Tuple{<:AbstractField{<:Real, 1}}}}
const Velocity2D = Union{NamedTuple{(:x, :y), <:Tuple{Vararg{AbstractField{<:Real}, 2}}}, VectorField{2, <:Tuple{Vararg{AbstractField{<:Real}, 2}}}}
const Velocity3D = Union{NamedTuple{(:x, :y, :z), <:Tuple{Vararg{AbstractField{<:Real}, 3}}}, VectorField{3, <:Tuple{Vararg{AbstractField{<:Real}, 3}}}}

function adapt_multiphase_profile(component::AbstractArray, backend)
    get_backend(component) == backend && return component
    adapted = KernelAbstractions.zeros(backend, eltype(component), size(component)...)
    copyto!(adapted, component)
    synchronize(backend)
    return adapted
end

adapt_multiphase_profile(component, backend) = component

function validate_multiphase_chmy_boundary(boundary, backend, N, sizes, NP, ::Type{T}) where {T}
    faces = normalize_boundary_faces(boundary, N)
    backend_faces = map(faces) do bc
        if bc isa PrescribedInflowBC && bc.value isa Tuple
            PrescribedInflowBC(map(component -> adapt_multiphase_profile(component, backend), bc.value))
        else
            bc
        end
    end
    host_faces = map(backend_faces) do bc
        if bc isa PrescribedInflowBC && bc.value isa Tuple
            PrescribedInflowBC(
                map(
                    component -> component isa AbstractArray ? Array(component) : component,
                    bc.value
                )
            )
        else
            bc
        end
    end
    validate_multiphase_boundary(host_faces, N, sizes, NP, T)
    return backend_faces
end


"""
WENOScheme(u::AbstractField{T, N},
           grid::StructuredGrid;
           form::Symbol, boundary=nothing, stag=true) where {T, N}

Create a WENO scheme structure for the given field `u` on the specified `grid` using Chmy.jl.

# Arguments
- `c0::AbstractField{T, N}`: Input field for which the WENO scheme is to be created. Only used to get the type and size.
- `grid::StructuredGrid`: Computational grid.
- `form::Symbol`: Required scalar PDE form, either `:conservative` or
  `:nonconservative`.
- `boundary`: Ordered tuple of typed advection boundaries or an
  `AdvectionBC`. The default is `ExtrapolateBC()` on every face.
- `stag::Bool`: Whether the grid is staggered (velocities on cell faces) or not (velocities on cell centers).
"""
function WENOScheme(c0::AbstractField{T, N}, grid::StructuredGrid; boundary = nothing,
                    form::Symbol, stag::Bool = true,
                    lim_ZS::Bool = false, upwind_mode = false) where {T, N}

    boundary === nothing && (boundary = ntuple(i -> ExtrapolateBC(), N * 2))
    sizes = ntuple(i -> grid.axes[i].length, N)
    boundary = validate_boundary(boundary, N, sizes)
    upwind_mode && any(b -> b isa PrescribedInflowBC, boundary) && throw(
        ArgumentError(
            "PrescribedInflowBC is supported by WENO5 reconstruction, " *
                "but not by upwind_mode"
        )
    )

    # multithreading is always on in this case with chmy.jl
    multithreading = true

    backend = get_backend(c0)
    form_tag = FiniteDiffWENO5.advection_form(form)
    FiniteDiffWENO5.validate_scalar_options(form_tag, stag, lim_ZS, upwind_mode)

    fl = VectorField(backend, grid)
    fr = VectorField(backend, grid)
    du = Field(backend, grid, Center())
    ut = Field(backend, grid, Center())
    labels = (:x, :y, :z)[1:N]
    vcenter = stag ?
        NamedTuple{labels}(ntuple(_ -> Field(backend, grid, Center()), N)) : nothing
    vperiodic = stag ? FiniteDiffWENO5.velocity_periodicity(boundary, labels) : nothing

    TFlux = typeof(fl)
    TArray = typeof(du)

    return WENOScheme{T, TArray, TFlux, typeof(vcenter), typeof(vperiodic), typeof(form_tag), typeof(boundary)}(
        stag = stag, form = form_tag, boundary = boundary, multithreading = multithreading,
        lim_ZS = lim_ZS, fl = fl, fr = fr, du = du, ut = ut, vcenter = vcenter,
        vperiodic = vperiodic,
        upwind_mode = upwind_mode,
    )
end

function MultiphaseWENOScheme(
        phases::Tuple{A, Vararg{A, M}}, grid::StructuredGrid{N};
        boundary = nothing, stag::Bool = false, multithreading::Bool = true,
    ) where {T, N, A <: AbstractField{T, N}, M}
    NP = M + 1
    NP >= 2 || throw(
        ArgumentError(
            "MultiphaseWENOScheme requires at least two phases, got $NP. " *
                "Use WENOScheme for a single field."
        )
    )
    1 <= N <= 3 || throw(
        ArgumentError(
            "MultiphaseWENOScheme supports 1D, 2D, and 3D fields, got $(N)D"
        )
    )
    sizes = ntuple(d -> grid.axes[d].length, Val(N))
    backend = get_backend(first(phases))
    for q in 1:NP
        size(phases[q]) == sizes || throw(
            DimensionMismatch(
                "phase $q has size $(size(phases[q])) but the grid center has size $sizes"
            )
        )
        @assert get_backend(phases[q]) == backend
        all(loc -> loc isa Center, location(phases[q])) || throw(
            ArgumentError(
                "multiphase fields must be cell-centred, phase $q is at $(location(phases[q]))"
            )
        )
    end

    boundary === nothing && (boundary = ntuple(_ -> ExtrapolateBC(), 2N))
    boundary = validate_multiphase_chmy_boundary(boundary, backend, N, sizes, NP, T)
    labels = (:x, :y, :z)[1:N]
    phase_count = Val(NP)
    direction_count = Val(N)
    direction_location(d) = ntuple(i -> i == d ? Vertex() : Center(), direction_count)
    fl = NamedTuple{labels}(
        ntuple(direction_count) do d
            ntuple(_ -> Field(backend, grid, direction_location(d), T), phase_count)
        end
    )
    fr = NamedTuple{labels}(
        ntuple(direction_count) do d
            ntuple(_ -> Field(backend, grid, direction_location(d), T), phase_count)
        end
    )
    du = ntuple(_ -> Field(backend, grid, Center(), T), phase_count)
    ut = ntuple(_ -> Field(backend, grid, Center(), T), phase_count)
    vcenter = stag ?
        NamedTuple{labels}(ntuple(_ -> Field(backend, grid, Center(), T), direction_count)) :
        nothing
    vperiodic = stag ? FiniteDiffWENO5.velocity_periodicity(boundary, labels) : nothing

    return MultiphaseWENOScheme{
        T, NP, typeof(du), typeof(fl), typeof(vcenter), typeof(vperiodic), typeof(boundary),
    }(
        stag = stag, boundary = boundary, multithreading = multithreading,
        fl = fl, fr = fr, du = du, ut = ut, vcenter = vcenter, vperiodic = vperiodic,
    )
end

function WENOScheme(c0::AbstractField; kwargs...)
    error(
        """
        You called `WENOScheme(c0)` with a `$(typeof(c0))`, which is a subtype of `AbstractField`.

        To construct a WENO scheme for Chmy.jl fields, you must also provide the computational grid:
            WENOScheme(c0::AbstractField, grid::StructuredGrid; kwargs...)

        Example:
            grid = UniformGrid(arch; origin=(0.0, 0.0), extent=(Lx, Lx), dims=(nx, ny))
            weno = WENOScheme(
                c0, grid;
                form=:nonconservative, boundary=(2,2,2,2), stag=false,
            )
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
include("KAMultiphase1D.jl")
include("KAMultiphase2D.jl")
include("KAMultiphase3D.jl")

"""Stage update with the shared simplex limiter, matching the CPU/KA multiphase stepper."""
function launch_multiphase_limited_update_chmy!(
        dest, initial, stage, du, a, b, Δt, phase_count, grid, backend,
    )
    kernel = multiphase_RK_limited_update_KA!(backend)
    offset = Offset(ntuple(_ -> 0, Val(ndims(grid))))
    dest_interior = map(interior, dest)
    initial_interior = map(interior, initial)
    stage_interior = map(interior, stage)
    du_interior = map(interior, du)
    kernel(
        dest_interior, initial_interior, stage_interior, du_interior,
        a, b, Δt, phase_count, grid, offset; ndrange = size(first(dest_interior))
    )
    synchronize(backend)
    return nothing
end

"""Interpolate a staggered scalar velocity to cell centres once per step (Chmy, 1D)."""
function prepare_velocity_chmy_1D!(weno, v, nx, grid, arch, launch)
    weno.stag || return v
    weno.vcenter === nothing && return v
    v === weno.vcenter && return v
    periodic = weno.vperiodic.x
    kernel = nx >= FiniteDiffWENO5.eno5_minimum_cells(periodic) ?
        eno5_face_to_center_KA_1D! : linear_face_to_center_KA_1D!
    launch(arch, grid, kernel => (weno.vcenter.x, v.x, nx, periodic, grid))
    return weno.vcenter
end

"""Chmy counterpart in 2D."""
function prepare_velocity_chmy_2D!(weno, v, nx, ny, grid, arch, launch)
    weno.stag || return v
    weno.vcenter === nothing && return v
    v === weno.vcenter && return v
    px, py = weno.vperiodic.x, weno.vperiodic.y
    kx = nx >= FiniteDiffWENO5.eno5_minimum_cells(px) ?
        eno5_face_to_center_KA_2D_x! : linear_face_to_center_KA_2D_x!
    ky = ny >= FiniteDiffWENO5.eno5_minimum_cells(py) ?
        eno5_face_to_center_KA_2D_y! : linear_face_to_center_KA_2D_y!
    launch(arch, grid, kx => (weno.vcenter.x, v.x, nx, px, grid))
    launch(arch, grid, ky => (weno.vcenter.y, v.y, ny, py, grid))
    return weno.vcenter
end

"""Chmy counterpart in 3D."""
function prepare_velocity_chmy_3D!(weno, v, nx, ny, nz, grid, arch, launch)
    weno.stag || return v
    weno.vcenter === nothing && return v
    v === weno.vcenter && return v
    px, py, pz = weno.vperiodic.x, weno.vperiodic.y, weno.vperiodic.z
    kx = nx >= FiniteDiffWENO5.eno5_minimum_cells(px) ?
        eno5_face_to_center_KA_3D_x! : linear_face_to_center_KA_3D_x!
    ky = ny >= FiniteDiffWENO5.eno5_minimum_cells(py) ?
        eno5_face_to_center_KA_3D_y! : linear_face_to_center_KA_3D_y!
    kz = nz >= FiniteDiffWENO5.eno5_minimum_cells(pz) ?
        eno5_face_to_center_KA_3D_z! : linear_face_to_center_KA_3D_z!
    launch(arch, grid, kx => (weno.vcenter.x, v.x, nx, px, grid))
    launch(arch, grid, ky => (weno.vcenter.y, v.y, ny, py, grid))
    launch(arch, grid, kz => (weno.vcenter.z, v.z, nz, pz, grid))
    return weno.vcenter
end

"""Interpolate a staggered multiphase velocity to cell centres once per step (Chmy, 1D)."""
function prepare_multiphase_velocity_chmy_1D!(scheme, v, nx, grid, arch, launch)
    scheme.stag || return v
    scheme.vcenter === nothing && return v
    periodic = scheme.vperiodic.x
    kernel = nx >= FiniteDiffWENO5.eno5_minimum_cells(periodic) ?
        eno5_face_to_center_KA_1D! : linear_face_to_center_KA_1D!
    launch(arch, grid, kernel => (scheme.vcenter.x, v.x, nx, periodic, grid))
    return scheme.vcenter
end

"""Chmy multiphase counterpart in 2D."""
function prepare_multiphase_velocity_chmy_2D!(scheme, v, nx, ny, grid, arch, launch)
    scheme.stag || return v
    scheme.vcenter === nothing && return v
    px, py = scheme.vperiodic.x, scheme.vperiodic.y
    kx = nx >= FiniteDiffWENO5.eno5_minimum_cells(px) ?
        eno5_face_to_center_KA_2D_x! : linear_face_to_center_KA_2D_x!
    ky = ny >= FiniteDiffWENO5.eno5_minimum_cells(py) ?
        eno5_face_to_center_KA_2D_y! : linear_face_to_center_KA_2D_y!
    launch(arch, grid, kx => (scheme.vcenter.x, v.x, nx, px, grid))
    launch(arch, grid, ky => (scheme.vcenter.y, v.y, ny, py, grid))
    return scheme.vcenter
end

"""Chmy multiphase counterpart in 3D."""
function prepare_multiphase_velocity_chmy_3D!(scheme, v, nx, ny, nz, grid, arch, launch)
    scheme.stag || return v
    scheme.vcenter === nothing && return v
    px, py, pz = scheme.vperiodic.x, scheme.vperiodic.y, scheme.vperiodic.z
    kx = nx >= FiniteDiffWENO5.eno5_minimum_cells(px) ?
        eno5_face_to_center_KA_3D_x! : linear_face_to_center_KA_3D_x!
    ky = ny >= FiniteDiffWENO5.eno5_minimum_cells(py) ?
        eno5_face_to_center_KA_3D_y! : linear_face_to_center_KA_3D_y!
    kz = nz >= FiniteDiffWENO5.eno5_minimum_cells(pz) ?
        eno5_face_to_center_KA_3D_z! : linear_face_to_center_KA_3D_z!
    launch(arch, grid, kx => (scheme.vcenter.x, v.x, nx, px, grid))
    launch(arch, grid, ky => (scheme.vcenter.y, v.y, ny, py, grid))
    launch(arch, grid, kz => (scheme.vcenter.z, v.z, nz, pz, grid))
    return scheme.vcenter
end

function WENO_step!(
        phases::Tuple{A, Vararg{A, M}}, v::Velocity1D,
        scheme::MultiphaseWENOScheme{T, NP}, Δt, Δx,
        grid::StructuredGrid{1}, arch,
    ) where {M, A <: AbstractField{<:Real, 1}, T, NP}
    M + 1 == NP || throw(
        DimensionMismatch(
            "scheme was built for $NP phases but $(M + 1) were given"
        )
    )
    backend = get_backend(first(phases))
    for q in 1:NP
        @assert get_backend(phases[q]) == backend
    end
    @assert get_backend(v.x) == backend

    launch = Launcher(arch, grid)
    (; fl, fr, ut, du, boundary, χ, γ, ζ, ϵ) = scheme
    nx = grid.axes[1].length
    Δx_ = inv(Δx)
    phase_count = Val(NP)
    vcell = prepare_multiphase_velocity_chmy_1D!(scheme, v, nx, grid, arch, launch)

    function launch_stage!(dest, stage, a, b)
        launch(
            arch, grid, multiphase_WENO_flux_KA_1D! =>
                (fl.x, fr.x, stage, boundary, nx, χ, γ, ζ, ϵ, phase_count, grid)
        )
        launch(
            arch, grid, multiphase_semi_collocated_KA_1D! =>
                (du, fl, fr, vcell, Δx_, phase_count, grid)
        )
        launch_multiphase_limited_update_chmy!(
            dest, phases, stage, du, a, b, Δt, phase_count, grid, backend
        )
        return nothing
    end

    launch_stage!(ut, phases, 0.0, 1.0)
    launch_stage!(ut, ut, 0.75, 0.25)
    launch_stage!(phases, ut, 1.0 / 3.0, 2.0 / 3.0)
    return nothing
end

function WENO_step!(
        phases::Tuple{A, Vararg{A, M}}, v::Velocity2D,
        scheme::MultiphaseWENOScheme{T, NP}, Δt, Δx, Δy,
        grid::StructuredGrid{2}, arch,
    ) where {M, A <: AbstractField{<:Real, 2}, T, NP}
    M + 1 == NP || throw(
        DimensionMismatch(
            "scheme was built for $NP phases but $(M + 1) were given"
        )
    )
    backend = get_backend(first(phases))
    for q in 1:NP
        @assert get_backend(phases[q]) == backend
    end
    @assert get_backend(v.x) == backend
    @assert get_backend(v.y) == backend

    launch = Launcher(arch, grid)
    (; fl, fr, ut, du, boundary, χ, γ, ζ, ϵ) = scheme
    nx, ny = map(axis -> axis.length, grid.axes)
    Δx_, Δy_ = inv(Δx), inv(Δy)
    phase_count = Val(NP)
    vcell = prepare_multiphase_velocity_chmy_2D!(scheme, v, nx, ny, grid, arch, launch)

    function launch_stage!(dest, stage, a, b)
        launch(
            arch, grid, multiphase_WENO_flux_KA_2D_x! =>
                (fl.x, fr.x, stage, boundary, nx, χ, γ, ζ, ϵ, phase_count, grid)
        )
        launch(
            arch, grid, multiphase_WENO_flux_KA_2D_y! =>
                (fl.y, fr.y, stage, boundary, ny, χ, γ, ζ, ϵ, phase_count, grid)
        )
        launch(
            arch, grid, multiphase_semi_collocated_KA_2D! =>
                (du, fl, fr, vcell, Δx_, Δy_, phase_count, grid)
        )
        launch_multiphase_limited_update_chmy!(
            dest, phases, stage, du, a, b, Δt, phase_count, grid, backend
        )
        return nothing
    end

    launch_stage!(ut, phases, 0.0, 1.0)
    launch_stage!(ut, ut, 0.75, 0.25)
    launch_stage!(phases, ut, 1.0 / 3.0, 2.0 / 3.0)
    return nothing
end

function WENO_step!(
        phases::Tuple{A, Vararg{A, M}}, v::Velocity3D,
        scheme::MultiphaseWENOScheme{T, NP}, Δt, Δx, Δy, Δz,
        grid::StructuredGrid{3}, arch,
    ) where {M, A <: AbstractField{<:Real, 3}, T, NP}
    M + 1 == NP || throw(
        DimensionMismatch(
            "scheme was built for $NP phases but $(M + 1) were given"
        )
    )
    backend = get_backend(first(phases))
    for q in 1:NP
        @assert get_backend(phases[q]) == backend
    end
    @assert get_backend(v.x) == backend
    @assert get_backend(v.y) == backend
    @assert get_backend(v.z) == backend

    launch = Launcher(arch, grid)
    (; fl, fr, ut, du, boundary, χ, γ, ζ, ϵ) = scheme
    nx, ny, nz = map(axis -> axis.length, grid.axes)
    Δx_, Δy_, Δz_ = inv(Δx), inv(Δy), inv(Δz)
    phase_count = Val(NP)
    vcell = prepare_multiphase_velocity_chmy_3D!(scheme, v, nx, ny, nz, grid, arch, launch)

    function launch_stage!(dest, stage, a, b)
        launch(
            arch, grid, multiphase_WENO_flux_KA_3D_x! =>
                (fl.x, fr.x, stage, boundary, nx, χ, γ, ζ, ϵ, phase_count, grid)
        )
        launch(
            arch, grid, multiphase_WENO_flux_KA_3D_y! =>
                (fl.y, fr.y, stage, boundary, ny, χ, γ, ζ, ϵ, phase_count, grid)
        )
        launch(
            arch, grid, multiphase_WENO_flux_KA_3D_z! =>
                (fl.z, fr.z, stage, boundary, nz, χ, γ, ζ, ϵ, phase_count, grid)
        )
        launch(
            arch, grid, multiphase_semi_collocated_KA_3D! =>
                (du, fl, fr, vcell, Δx_, Δy_, Δz_, phase_count, grid)
        )
        launch_multiphase_limited_update_chmy!(
            dest, phases, stage, du, a, b, Δt, phase_count, grid, backend
        )
        return nothing
    end

    launch_stage!(ut, phases, 0.0, 1.0)
    launch_stage!(ut, ut, 0.75, 0.25)
    launch_stage!(phases, ut, 1.0 / 3.0, 2.0 / 3.0)
    return nothing
end


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

    (; ut, du, fl, fr, stag, lim_ZS, boundary, χ, γ, ζ, ϵ, upwind_mode, form) = weno

    if !upwind_mode
        vcell = prepare_velocity_chmy_1D!(weno, v, nx, grid, arch, launch)
        conservative = FiniteDiffWENO5.is_conservative(form)
        α = conservative ? FiniteDiffWENO5.lf_speed(interior(vcell.x), maximum) : zero(eltype(u))

        function stage!(state)
            if conservative
                launch(arch, grid, WENO_conservative_flux_KA_1D! => (fl.x, fr.x, state, vcell.x, α, boundary, nx, χ, γ, ζ, ϵ, grid))
                launch(arch, grid, WENO_conservative_divergence_KA_1D! => (du, fl.x, fr.x, Δx_, grid))
            else
                launch(arch, grid, WENO_flux_KA_1D => (fl.x, fr.x, state, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
                launch(arch, grid, WENO_semi_discretisation_weno5_KA_1D! => (du, fl, fr, vcell, Δx_, grid))
            end
            return nothing
        end

        stage!(u)
        interior(ut) .= @muladd interior(u) .- Δt .* interior(du)

        stage!(ut)
        interior(ut) .= @muladd 0.75 .* interior(u) .+ 0.25 .* interior(ut) .- 0.25 .* Δt .* interior(du)

        stage!(ut)
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

    (; ut, du, fl, fr, stag, lim_ZS, boundary, χ, γ, ζ, ϵ, upwind_mode, form) = weno

    if !upwind_mode
        vcell = prepare_velocity_chmy_2D!(weno, v, nx, ny, grid, arch, launch)
        conservative = FiniteDiffWENO5.is_conservative(form)
        αx = conservative ? FiniteDiffWENO5.lf_speed(interior(vcell.x), maximum) : zero(eltype(u))
        αy = conservative ? FiniteDiffWENO5.lf_speed(interior(vcell.y), maximum) : zero(eltype(u))

        function stage!(state)
            if conservative
                launch(arch, grid, WENO_conservative_flux_KA_2D_x! => (fl.x, fr.x, state, vcell.x, αx, boundary, nx, χ, γ, ζ, ϵ, grid))
                launch(arch, grid, WENO_conservative_flux_KA_2D_y! => (fl.y, fr.y, state, vcell.y, αy, boundary, ny, χ, γ, ζ, ϵ, grid))
                launch(arch, grid, WENO_conservative_divergence_KA_2D! => (du, fl, fr, Δx_, Δy_, grid))
            else
                launch(arch, grid, WENO_flux_KA_2D_x => (fl.x, fr.x, state, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
                launch(arch, grid, WENO_flux_KA_2D_y => (fl.y, fr.y, state, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
                launch(arch, grid, WENO_semi_discretisation_weno5_KA_2D! => (du, fl, fr, vcell, Δx_, Δy_, grid))
            end
            return nothing
        end

        stage!(u)
        interior(ut) .= @muladd interior(u) .- Δt .* interior(du)

        stage!(ut)
        interior(ut) .= @muladd 0.75 .* interior(u) .+ 0.25 .* interior(ut) .- 0.25 .* Δt .* interior(du)

        stage!(ut)
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

    (; ut, du, fl, fr, stag, lim_ZS, boundary, χ, γ, ζ, ϵ, upwind_mode, form) = weno

    if !upwind_mode
        vcell = prepare_velocity_chmy_3D!(weno, v, nx, ny, nz, grid, arch, launch)
        conservative = FiniteDiffWENO5.is_conservative(form)
        αx = conservative ? FiniteDiffWENO5.lf_speed(interior(vcell.x), maximum) : zero(eltype(u))
        αy = conservative ? FiniteDiffWENO5.lf_speed(interior(vcell.y), maximum) : zero(eltype(u))
        αz = conservative ? FiniteDiffWENO5.lf_speed(interior(vcell.z), maximum) : zero(eltype(u))

        function stage!(state)
            if conservative
                launch(arch, grid, WENO_conservative_flux_KA_3D_x! => (fl.x, fr.x, state, vcell.x, αx, boundary, nx, χ, γ, ζ, ϵ, grid))
                launch(arch, grid, WENO_conservative_flux_KA_3D_y! => (fl.y, fr.y, state, vcell.y, αy, boundary, ny, χ, γ, ζ, ϵ, grid))
                launch(arch, grid, WENO_conservative_flux_KA_3D_z! => (fl.z, fr.z, state, vcell.z, αz, boundary, nz, χ, γ, ζ, ϵ, grid))
                launch(arch, grid, WENO_conservative_divergence_KA_3D! => (du, fl, fr, Δx_, Δy_, Δz_, grid))
            else
                launch(arch, grid, WENO_flux_KA_3D_x => (fl.x, fr.x, state, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
                launch(arch, grid, WENO_flux_KA_3D_y => (fl.y, fr.y, state, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
                launch(arch, grid, WENO_flux_KA_3D_z => (fl.z, fr.z, state, boundary, nz, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, grid))
                launch(arch, grid, WENO_semi_discretisation_weno5_KA_3D! => (du, fl, fr, vcell, Δx_, Δy_, Δz_, grid))
            end
            return nothing
        end

        stage!(u)
        interior(ut) .= @muladd interior(u) .- Δt .* interior(du)

        stage!(ut)
        interior(ut) .= @muladd 0.75 .* interior(u) .+ 0.25 .* interior(ut) .- 0.25 .* Δt .* interior(du)

        stage!(ut)
        interior(u) .= @muladd inv(3.0) .* interior(u) .+ 2.0 / 3.0 .* interior(ut) .- 2.0 / 3.0 .* Δt .* interior(du)
    else
        launch(arch, grid, upwind_update_KA_3D! => (u, v, nx, ny, nz, Δx_, Δy_, Δz_, Δt, stag, boundary, grid))
    end

    return nothing
end


# Chmy's `VectorField` is not a `NamedTuple`, so the generic tuple forwarder cannot
# prepare its staggered velocity.  These specializations prepare once and then pass
# the cell-centred velocity through the scalar Chmy steppers.
function WENO_step!(
        u::Tuple{A, Vararg{A, M}}, v::Velocity1D,
        weno::FiniteDiffWENO5.WENOScheme, Δt, Δx,
        grid::StructuredGrid{1}, arch;
        u_min::Tuple{Vararg{Real}}, u_max::Tuple{Vararg{Real}},
    ) where {A <: AbstractField{<:Real, 1}, M}
    length(u) == length(u_min) == length(u_max) || throw(DimensionMismatch(
        "one lower and upper bound is required for each field",
    ))
    nx = grid.axes[1].length
    vstep = prepare_velocity_chmy_1D!(weno, v, nx, grid, arch, Launcher(arch, grid))
    for q in eachindex(u)
        WENO_step!(u[q], vstep, weno, Δt, Δx, grid, arch; u_min = u_min[q], u_max = u_max[q])
    end
    return nothing
end

function WENO_step!(
        u::Tuple{A, Vararg{A, M}}, v::Velocity2D,
        weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, Δy,
        grid::StructuredGrid{2}, arch;
        u_min::Tuple{Vararg{Real}}, u_max::Tuple{Vararg{Real}},
    ) where {A <: AbstractField{<:Real, 2}, M}
    length(u) == length(u_min) == length(u_max) || throw(DimensionMismatch(
        "one lower and upper bound is required for each field",
    ))
    nx, ny = grid.axes[1].length, grid.axes[2].length
    vstep = prepare_velocity_chmy_2D!(weno, v, nx, ny, grid, arch, Launcher(arch, grid))
    for q in eachindex(u)
        WENO_step!(u[q], vstep, weno, Δt, Δx, Δy, grid, arch; u_min = u_min[q], u_max = u_max[q])
    end
    return nothing
end

function WENO_step!(
        u::Tuple{A, Vararg{A, M}}, v::Velocity3D,
        weno::FiniteDiffWENO5.WENOScheme, Δt, Δx, Δy, Δz,
        grid::StructuredGrid{3}, arch;
        u_min::Tuple{Vararg{Real}}, u_max::Tuple{Vararg{Real}},
    ) where {A <: AbstractField{<:Real, 3}, M}
    length(u) == length(u_min) == length(u_max) || throw(DimensionMismatch(
        "one lower and upper bound is required for each field",
    ))
    nx, ny, nz = grid.axes[1].length, grid.axes[2].length, grid.axes[3].length
    vstep = prepare_velocity_chmy_3D!(weno, v, nx, ny, nz, grid, arch, Launcher(arch, grid))
    for q in eachindex(u)
        WENO_step!(u[q], vstep, weno, Δt, Δx, Δy, Δz, grid, arch; u_min = u_min[q], u_max = u_max[q])
    end
    return nothing
end

end
