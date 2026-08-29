module KAExt
using FiniteDiffWENO5
using FiniteDiffWENO5: zhang_shu_limit, limit_simplex, weno5_reconstruction_upwind, weno5_reconstruction_downwind, multiphase_reconstruction_upwind, multiphase_reconstruction_downwind, validate_boundary, validate_multiphase_boundary, normalize_boundary_faces, flux_size, left_index, right_index, inflow_value, multiphase_inflow_value
using MuladdMacro
using KernelAbstractions

import FiniteDiffWENO5: WENOScheme, MultiphaseWENOScheme, WENO_step!

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
WENOScheme(c0::AbstractArray{T, N}, backend::Backend;
           form::Symbol, boundary=nothing, stag=true) where {T, N}

Create a WENO scheme structure for the given field `c` using the specified `backend` from KernelAbstractions.jl.

# Arguments
- `c0::AbstractArray{T, N}`: The input field for which the WENO scheme is to be created. Only used to get the type and size.
- `backend::Backend`: The KernelAbstractions backend to be used (e.g., CPU(), CUDA(), etc.).
- `form::Symbol`: Required scalar PDE form, either `:conservative` or
  `:nonconservative`.
- `boundary`: Ordered tuple of typed advection boundaries or an
  `AdvectionBC`. The default is `ExtrapolateBC()` on every face.
- `stag::Bool`: Whether the grid is staggered (velocities on cell faces) or not (velocities on cell centers).
"""
function WENOScheme(c0::AbstractArray{T, N}, backend::Backend; boundary = nothing,
                    form::Symbol, stag::Bool = true,
                    lim_ZS::Bool = false, upwind_mode::Bool = false) where {T, N}

    @assert get_backend(c0) == backend "The type of the input field must match the specified backend."

    boundary === nothing && (boundary = ntuple(i -> ExtrapolateBC(), N * 2))
    boundary = validate_boundary(boundary, N, size(c0))
    upwind_mode && any(b -> b isa PrescribedInflowBC, boundary) && throw(
        ArgumentError(
            "PrescribedInflowBC is supported by WENO5 reconstruction, " *
                "but not by upwind_mode"
        )
    )

    # multithreading is always on in this case
    multithreading = true

    backend = get_backend(c0)

    # construct NamedTuples for left and right fluxes
    labels = (:x, :y, :z)[1:min(N, 3)]
    sizes = size(c0)
    form_tag = FiniteDiffWENO5.advection_form(form)
    FiniteDiffWENO5.validate_scalar_options(form_tag, stag, lim_ZS, upwind_mode)
    fl = NamedTuple{labels}(ntuple(d -> KernelAbstractions.zeros(backend, T, flux_size(sizes, d, N)), min(N, 3)))
    fr = NamedTuple{labels}(ntuple(d -> KernelAbstractions.zeros(backend, T, flux_size(sizes, d, N)), min(N, 3)))

    du = KernelAbstractions.zeros(backend, T, size(c0))
    ut = KernelAbstractions.zeros(backend, T, size(c0))
    # Cell-centred velocity buffers, needed whenever a staggered velocity must be
    # interpolated before the operator sees it. Allocated on the same backend.
    vcenter = stag ?
        NamedTuple{labels}(ntuple(_ -> KernelAbstractions.zeros(backend, T, sizes), min(N, 3))) :
        nothing
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

function validate_multiphase_backend_boundary(boundary, backend, N, sizes, NP, ::Type{T}) where {T}
    faces = normalize_boundary_faces(boundary, N)
    host_faces = map(faces) do bc
        if bc isa PrescribedInflowBC && bc.value isa Tuple
            host_value = map(bc.value) do component
                if component isa AbstractArray
                    @assert get_backend(component) == backend "inflow profiles must use the specified backend"
                    Array(component)
                else
                    component
                end
            end
            PrescribedInflowBC(host_value)
        else
            bc
        end
    end
    validate_multiphase_boundary(host_faces, N, sizes, NP, T)
    return faces
end

function MultiphaseWENOScheme(
        phases::Tuple{A, Vararg{A, M}}, backend::Backend;
        boundary = nothing, stag::Bool = false, multithreading::Bool = true,
    ) where {T, N, A <: AbstractArray{T, N}, M}
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

    reference_axes = axes(first(phases))
    for k in 1:NP
        axes(phases[k]) == reference_axes || throw(
            DimensionMismatch(
                "all phases must share axes, phase 1 has $reference_axes but phase $k has " *
                    "$(axes(phases[k]))"
            )
        )
        @assert get_backend(phases[k]) == backend "phase $k must use the specified backend"
    end

    boundary === nothing && (boundary = ntuple(_ -> ExtrapolateBC(), 2N))
    boundary = validate_multiphase_backend_boundary(
        boundary, backend, N, size(first(phases)), NP, T
    )

    labels = (:x, :y, :z)[1:min(N, 3)]
    sizes = size(first(phases))
    valNP = Val(NP)
    backend_zeros(dims) = KernelAbstractions.zeros(backend, T, dims...)
    fl = NamedTuple{labels}(
        ntuple(min(N, 3)) do d
            ntuple(_ -> backend_zeros(flux_size(sizes, d, N)), valNP)
        end
    )
    fr = NamedTuple{labels}(
        ntuple(min(N, 3)) do d
            ntuple(_ -> backend_zeros(flux_size(sizes, d, N)), valNP)
        end
    )
    du = ntuple(_ -> backend_zeros(sizes), valNP)
    ut = ntuple(_ -> backend_zeros(sizes), valNP)
    vcenter = stag ? NamedTuple{labels}(ntuple(_ -> backend_zeros(sizes), min(N, 3))) : nothing
    vperiodic = stag ? FiniteDiffWENO5.velocity_periodicity(boundary, labels) : nothing

    return MultiphaseWENOScheme{
        T, NP, typeof(du), typeof(fl), typeof(vcenter), typeof(vperiodic), typeof(boundary),
    }(
        stag = stag, boundary = boundary, multithreading = multithreading,
        fl = fl, fr = fr, du = du, ut = ut, vcenter = vcenter, vperiodic = vperiodic,
    )
end

include("KAExt1D.jl")
include("KAExt2D.jl")
include("KAExt3D.jl")
include("KAMultiphase1D.jl")
include("KAMultiphase2D.jl")
include("KAMultiphase3D.jl")

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

    (; ut, du, fl, fr, stag, lim_ZS, boundary, χ, γ, ζ, ϵ, upwind_mode, form, vcenter) = weno

    nx = size(u, 1)
    Δx_ = inv(Δx)

    if !upwind_mode
        # Prepare the staggered velocity once for all three RK stages, exactly as
        # the CPU path does, so both backends evaluate the same operator.
        vcell = prepare_velocity_KA_1D!(weno, v, nx, backend)
        conservative = FiniteDiffWENO5.is_conservative(form)

        kernel_flux_1D = WENO_flux_KA_1D(backend)
        kernel_semi_discretisation_1D = WENO_semi_discretisation_weno5_KA_1D!(backend)
        kernel_cons_flux = WENO_conservative_flux_KA_1D!(backend)
        kernel_cons_div = WENO_conservative_divergence_KA_1D!(backend)
        α = conservative ? FiniteDiffWENO5.lf_speed(vcell.x, maximum) : zero(eltype(u))

        function stage!(state)
            if conservative
                kernel_cons_flux(
                    fl.x, fr.x, state, vcell.x, α, boundary, nx, χ, γ, ζ, ϵ,
                    nothing, Offset0, ndrange = length(fl.x),
                )
                synchronize(backend)
                kernel_cons_div(du, fl.x, fr.x, Δx_, nothing, Offset0, ndrange = length(du))
            else
                kernel_flux_1D(fl.x, fr.x, state, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = length(fl.x))
                synchronize(backend)
                kernel_semi_discretisation_1D(du, fl, fr, vcell, Δx_, nothing, Offset0, ndrange = length(du))
            end
            synchronize(backend)
            return nothing
        end

        stage!(u)
        ut .= @muladd u .- Δt .* du

        stage!(ut)
        ut .= @muladd 0.75 .* u .+ 0.25 .* ut .- 0.25 .* Δt .* du

        stage!(ut)
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

    (; ut, du, fl, fr, stag, lim_ZS, boundary, χ, γ, ζ, ϵ, upwind_mode, form) = weno

    nx = size(u, 1)
    ny = size(u, 2)
    Δx_ = inv(Δx)
    Δy_ = inv(Δy)

    if !upwind_mode
        flx_l = size(fl.x)
        fly_l = size(fl.y)
        du_l = size(du)

        vcell = prepare_velocity_KA_2D!(weno, v, nx, ny, backend)
        conservative = FiniteDiffWENO5.is_conservative(form)

        kernel_flux_2D_x = WENO_flux_KA_2D_x(backend)
        kernel_flux_2D_y = WENO_flux_KA_2D_y(backend)
        kernel_semi_discretisation_2D = WENO_semi_discretisation_weno5_KA_2D!(backend)
        kernel_cons_x = WENO_conservative_flux_KA_2D_x!(backend)
        kernel_cons_y = WENO_conservative_flux_KA_2D_y!(backend)
        kernel_cons_div = WENO_conservative_divergence_KA_2D!(backend)
        αx = conservative ? FiniteDiffWENO5.lf_speed(vcell.x, maximum) : zero(eltype(u))
        αy = conservative ? FiniteDiffWENO5.lf_speed(vcell.y, maximum) : zero(eltype(u))

        function stage2!(state)
            if conservative
                kernel_cons_x(fl.x, fr.x, state, vcell.x, αx, boundary, nx, χ, γ, ζ, ϵ, nothing, Offset0, ndrange = flx_l)
                kernel_cons_y(fl.y, fr.y, state, vcell.y, αy, boundary, ny, χ, γ, ζ, ϵ, nothing, Offset0, ndrange = fly_l)
                synchronize(backend)
                kernel_cons_div(du, fl, fr, Δx_, Δy_, nothing, Offset0, ndrange = du_l)
            else
                kernel_flux_2D_x(fl.x, fr.x, state, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = flx_l)
                kernel_flux_2D_y(fl.y, fr.y, state, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = fly_l)
                synchronize(backend)
                kernel_semi_discretisation_2D(du, fl, fr, vcell, Δx_, Δy_, nothing, Offset0, ndrange = du_l)
            end
            synchronize(backend)
            return nothing
        end

        stage2!(u)
        ut .= @muladd u .- Δt .* du

        stage2!(ut)
        ut .= @muladd 0.75 .* u .+ 0.25 .* ut .- 0.25 .* Δt .* du

        stage2!(ut)
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

    (; ut, du, fl, fr, stag, lim_ZS, boundary, χ, γ, ζ, ϵ, upwind_mode, form) = weno

    if !upwind_mode

        flx_l = size(fl.x)
        fly_l = size(fl.y)
        flz_l = size(fl.z)
        du_l = size(du)

        vcell = prepare_velocity_KA_3D!(weno, v, nx, ny, nz, backend)
        conservative = FiniteDiffWENO5.is_conservative(form)

        kernel_flux_3D_x = WENO_flux_KA_3D_x(backend)
        kernel_flux_3D_y = WENO_flux_KA_3D_y(backend)
        kernel_flux_3D_z = WENO_flux_KA_3D_z(backend)
        kernel_semi_discretisation_3D = WENO_semi_discretisation_weno5_KA_3D!(backend)
        kernel_cons_x = WENO_conservative_flux_KA_3D_x!(backend)
        kernel_cons_y = WENO_conservative_flux_KA_3D_y!(backend)
        kernel_cons_z = WENO_conservative_flux_KA_3D_z!(backend)
        kernel_cons_div = WENO_conservative_divergence_KA_3D!(backend)
        αx = conservative ? FiniteDiffWENO5.lf_speed(vcell.x, maximum) : zero(eltype(u))
        αy = conservative ? FiniteDiffWENO5.lf_speed(vcell.y, maximum) : zero(eltype(u))
        αz = conservative ? FiniteDiffWENO5.lf_speed(vcell.z, maximum) : zero(eltype(u))

        function stage3!(state)
            if conservative
                kernel_cons_x(fl.x, fr.x, state, vcell.x, αx, boundary, nx, χ, γ, ζ, ϵ, nothing, Offset0, ndrange = flx_l)
                kernel_cons_y(fl.y, fr.y, state, vcell.y, αy, boundary, ny, χ, γ, ζ, ϵ, nothing, Offset0, ndrange = fly_l)
                kernel_cons_z(fl.z, fr.z, state, vcell.z, αz, boundary, nz, χ, γ, ζ, ϵ, nothing, Offset0, ndrange = flz_l)
                synchronize(backend)
                kernel_cons_div(du, fl, fr, Δx_, Δy_, Δz_, nothing, Offset0, ndrange = du_l)
            else
                kernel_flux_3D_x(fl.x, fr.x, state, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = flx_l)
                kernel_flux_3D_y(fl.y, fr.y, state, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = fly_l)
                kernel_flux_3D_z(fl.z, fr.z, state, boundary, nz, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, nothing, Offset0, ndrange = flz_l)
                synchronize(backend)
                kernel_semi_discretisation_3D(du, fl, fr, vcell, Δx_, Δy_, Δz_, nothing, Offset0, ndrange = du_l)
            end
            synchronize(backend)
            return nothing
        end

        stage3!(u)
        ut .= @muladd u .- Δt .* du

        stage3!(ut)
        ut .= @muladd 0.75 .* u .+ 0.25 .* ut .- 0.25 .* Δt .* du

        stage3!(ut)
        u .= @muladd inv(3.0) .* u .+ 2.0 / 3.0 .* ut .- 2.0 / 3.0 .* Δt .* du
    else
        u_l = size(u)

        kernel_upwind = upwind_update_KA_3D!(backend)

        kernel_upwind(u, v, nx, ny, nz, Δx_, Δy_, Δz_, Δt, stag, boundary, nothing, Offset0, ndrange = u_l)

    end

    return nothing
end


# Multi-field advection (u = (c1, c2, ...) sharing v and WENOScheme buffers) is
# handled generically for every dimension and backend by the `WENO_step!(u::Tuple, ...)`
# method in FiniteDiffWENO5's src/utils.jl.

end
