abstract type AbstractWENO end

@kwdef struct WENOScheme{T, TArray, TFlux, TVelocity, TPeriodicity, TForm, TBoundary, TExtent, TTopology, THaloBuffers} <: AbstractWENO
    γ::NTuple{3, T} = T.((0.1, 0.6, 0.3))
    χ::NTuple{2, T} = T.((13 / 12, 1 / 4))
    ζ::NTuple{5, T} = T.((1 / 3, 7 / 6, 11 / 6, 1 / 6, 5 / 6))
    ϵ::T = eps(T)
    stag::Bool
    form::TForm
    lim_ZS::Bool
    boundary::TBoundary
    multithreading::Bool
    upwind_mode::Bool = false
    fl::TFlux
    fr::TFlux
    du::TArray
    ut::TArray
    vcenter::TVelocity
    vperiodic::TPeriodicity
    extent::TExtent
    topology::TTopology = NoTopology()
    halo_buffers::THaloBuffers = EmptyHaloBuffers()
end

"""
    WENOScheme(c0::AbstractArray{T, N}; form::Symbol, boundary=nothing, stag=false,
               multithreading=true, lim_ZS=false, upwind_mode=false) where {T, N}

WENO5-Z scheme and work buffers for an `N`-dimensional field.

# Arguments
- `c0`: Supplies the array type and size; its values are not read.
- `boundary`: Face conditions (`ExtrapolateBC`, `PeriodicBC`, or
  `PrescribedInflowBC`) as a tuple or `AdvectionBC`; defaults to extrapolation.
  Legacy codes `0`/`1` mean extrapolation and `2` means periodic.
- `form`: `:conservative` (`∂u/∂t + ∇·(v u) = 0`) or
  `:nonconservative` (`∂u/∂t + v·∇u = 0`).
- `stag`: Use face-centered velocities when `true`.
- `lim_ZS`: Enable the Zhang-Shu limiter.
- `multithreading`: Enable threading in 2D or 3D.
- `upwind_mode`: Use the simple debugging upwind scheme.
"""
function WENOScheme(
        c0::AbstractArray{T, N}; boundary = nothing, form::Symbol,
        stag::Bool = false, lim_ZS::Bool = false, multithreading::Bool = true,
        upwind_mode::Bool = false
    ) where {T, N}

    boundary === nothing && (boundary = ntuple(i -> ExtrapolateBC(), N * 2))
    faces = validate_boundary(boundary, N, size(c0))
    extent = default_extent(size(c0), default_global_periodic(faces, N))
    return _build_weno_scheme(c0, extent, faces; form, stag, lim_ZS, multithreading, upwind_mode)
end

"""Periodicity inferred from low faces; pair validation happens later."""
default_global_periodic(faces, N) = ntuple(d -> faces[2d - 1] isa PeriodicBC, N)

"""Use local ghost filling, not periodic ENO5 indexing, on padded axes."""
function _resolved_vperiodic(faces, labels, extent::PaddedExtent{N}) where {N}
    base = velocity_periodicity(faces, labels)
    return NamedTuple{labels}(ntuple(d -> extent.pad[d] > 0 ? false : base[d], N))
end

"""Allocate a scheme from already validated boundary faces."""
function _build_weno_scheme(
        c0::AbstractArray{T, N}, extent::PaddedExtent{N}, faces;
        form::Symbol, stag::Bool, lim_ZS::Bool, multithreading::Bool, upwind_mode::Bool,
        topology = NoTopology(),
    ) where {T, N}
    upwind_mode && any(b -> b isa PrescribedInflowBC, faces) && throw(
        ArgumentError(
            "PrescribedInflowBC is supported by WENO5 reconstruction, " *
                "but not by upwind_mode"
        )
    )
    # The upwind kernel has no halo exchange schedule.
    upwind_mode && !(topology isa NoTopology) && throw(
        ArgumentError(
            "upwind_mode=true is not supported on a distributed topology " *
                "(it reads a neighbour index directly and has no halo exchange " *
                "of its own)"
        )
    )

    labels = (:x, :y, :z)[1:min(N, 3)]
    sizes = size(c0)
    form_tag = advection_form(form)
    validate_scalar_options(form_tag, stag, lim_ZS, upwind_mode)

    zeros_like(dims) = fill!(similar(c0, T, dims), zero(T))

    fl = NamedTuple{labels}(ntuple(d -> zeros_like(flux_size(sizes, d, N)), min(N, 3)))
    fr = NamedTuple{labels}(ntuple(d -> zeros_like(flux_size(sizes, d, N)), min(N, 3)))

    du = zeros_like(sizes)

    ut = zeros_like(sizes)

    # Staggered transport prepares cell-centered velocity once per step.
    vcenter = stag ? NamedTuple{labels}(ntuple(_ -> zeros_like(sizes), min(N, 3))) : nothing
    vperiodic = stag ? _resolved_vperiodic(faces, labels, extent) : nothing
    halo_buffers = halo_buffers_for(topology, extent, stag, T)

    TFlux = typeof(fl)
    TArray = typeof(du)

    return WENOScheme{
        T, TArray, TFlux, typeof(vcenter), typeof(vperiodic), typeof(form_tag),
        typeof(faces), typeof(extent), typeof(topology), typeof(halo_buffers),
    }(
        stag = stag, form = form_tag, boundary = faces, lim_ZS = lim_ZS,
        multithreading = multithreading, upwind_mode = upwind_mode, fl = fl, fr = fr,
        du = du, ut = ut, vcenter = vcenter, vperiodic = vperiodic, extent = extent,
        topology = topology, halo_buffers = halo_buffers,
    )
end

"""
    padded_weno_scheme(c0, halo; boundary, form, stag=false, lim_ZS=false,
                        multithreading=true, upwind_mode=false,
                        global_size=nothing, global_periodic=nothing,
                        geometry=:cell)

Build a padded scheme from a fully sized `c0` and per-axis `halo`.
`boundary` must already be resolved and may contain `ProcessBC`. Topology
constructors supply the global extent and periodicity before calling this
allocation helper.
"""
function padded_weno_scheme(
        c0::AbstractArray{T, N}, halo::NTuple{N, Int}; boundary, form::Symbol,
        stag::Bool = false, lim_ZS::Bool = false, multithreading::Bool = true,
        upwind_mode::Bool = false,
        global_size::NTuple{N, Int} = size(c0) .- 2 .* halo,
        global_periodic::Union{NTuple{N, Bool}, Nothing} = nothing,
        geometry::Symbol = :cell,
        topology = NoTopology(),
    ) where {T, N}

    faces, owned = _resolve_padded_extent(size(c0), halo, boundary, :padded_weno_scheme)

    gperiodic = global_periodic === nothing ? default_global_periodic(faces, N) : global_periodic
    extent = PaddedExtent{N}(owned, halo, global_size, gperiodic, geometry)
    return _build_weno_scheme(c0, extent, faces; form, stag, lim_ZS, multithreading, upwind_mode, topology)
end
