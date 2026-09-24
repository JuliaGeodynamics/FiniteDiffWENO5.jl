abstract type AbstractWENO end

@kwdef struct WENOScheme{T, TArray, TFlux, TVelocity, TPeriodicity, TForm, TBoundary, TExtent, TTopology} <: AbstractWENO
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
    # selected PDE form
    form::TForm
    # use Zhang-Shu limiter
    lim_ZS::Bool
    # boundary conditions
    boundary::TBoundary
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
    # prepared cell-centred velocity, used by staggered material transport
    vcenter::TVelocity
    # paired-periodic flag per velocity direction, cached with the scheme
    vperiodic::TPeriodicity
    # allocated and owned extents
    extent::TExtent
    # distributed topology, or `NoTopology()` for an unpadded scheme
    topology::TTopology = NoTopology()
end

"""
    WENOScheme(c0::AbstractArray{T, N}; form::Symbol, boundary=nothing, stag=false,
               multithreading=true, lim_ZS=false, upwind_mode=false) where {T, N}

Structure containing the Weighted Essentially Non-Oscillatory (WENO) scheme of order 5 constants and arrays for N-dimensional data of type T. The formulation is from Borges et al. 2008.

# Arguments
- `c0::AbstractArray{T, N}`: The input field for which the WENO scheme is to be created. Only used to get the type and size.
- `boundary`: Ordered tuple of `ExtrapolateBC()`, `PeriodicBC()`, or
  `PrescribedInflowBC(value)` conditions, or an `AdvectionBC`. The default is
  `ExtrapolateBC()` on every face. Legacy integer tuples remain accepted: `0`
  and `1` map to `ExtrapolateBC()`, and `2` maps to `PeriodicBC()`.
- `stag::Bool`: Whether the grid is staggered (velocities on cell faces) or not (velocities on cell centers). Default to false.
- `form::Symbol`: Required scalar PDE form: `:conservative` for
  `∂u/∂t + ∇·(v u) = 0`, or `:nonconservative` for `∂u/∂t + v·∇u = 0`.
- `lim_ZS::Bool`: Whether to use the Zhang-Shu (2010) limiter. Default to false.
- `multithreading::Bool`: Whether to use multithreading (only for 2D and 3D). Default to true.
- `upwind_mode::Bool`: Whether to use a simple upwind scheme for debugging purposes. Default to false.

# Fields
- `γ::NTuple{3, T}`: Upwind and downwind constants.
- `χ::NTuple{2, T}`: Betas' constants.
- `ζ::NTuple{5, T}`: Stencil weights.
- `ϵ::T`: Tolerance, fixed to machine precision.
- `stag::Bool`: Whether the grid is staggered (velocities on cell faces) or not (velocities on cell centers).
- `boundary`: Normalized tuple of typed advection boundary conditions.
- `lim_ZS::Bool`: Whether to use the Zhang-Shu limiter.
- `multithreading::Bool`: Whether to use multithreading (only for 2D and 3D).
- `fl::NamedTuple`: Fluxes in the left direction for each dimension.
- `fr::NamedTuple`: Fluxes in the right direction for each dimension.
- `du::AbstractArray{T, N}`: Semi-discretisation of the advection term.
- `ut::AbstractArray{T, N}`: Temporary array for intermediate calculations using Runge-Kutta.
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

"""Per-axis serial periodicity implied by a face tuple: both faces of a pair
`PeriodicBC` (mismatched pairs are caught later by `velocity_periodicity`, so
this may over-report `true` for an inconsistent tuple; it exists only to seed
`PaddedExtent.global_periodic`)."""
default_global_periodic(faces, N) = ntuple(d -> faces[2d - 1] isa PeriodicBC, N)

"""Resolve `vperiodic`: paired-periodicity validated as before, then forced
`false` on every axis with nonzero pad — a padded axis always drives the
ENO5 interpolation with `periodic = false`, even a physically periodic one on
a single rank; the wrap is then supplied by `fill_physical_ghosts!`."""
function _resolved_vperiodic(faces, labels, extent::PaddedExtent{N}) where {N}
    base = velocity_periodicity(faces, labels)
    return NamedTuple{labels}(ntuple(d -> extent.pad[d] > 0 ? false : base[d], N))
end

"""Shared allocation logic for both the public unpadded constructor above and
`padded_weno_scheme` below. `faces` must already be a validated (or, for the
padded path, resolved) tuple of `AbstractAdvectionBoundary`; this function
does not call `validate_boundary` itself."""
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
    # `upwind_mode` is a separate single-stage debug kernel that reads a
    # neighbour index directly and has no exchange schedule of its own — it
    # is rejected on any real topology, not silently left wrong.
    upwind_mode && !(topology isa NoTopology) && throw(
        ArgumentError(
            "upwind_mode=true is not supported on a distributed topology " *
                "(it reads a neighbour index directly and has no halo exchange " *
                "of its own)"
        )
    )

    # dimension labels
    labels = (:x, :y, :z)[1:min(N, 3)]
    sizes = size(c0)
    form_tag = advection_form(form)
    validate_scalar_options(form_tag, stag, lim_ZS, upwind_mode)

    # allocate a zeroed buffer of the same array type as `c0`
    zeros_like(dims) = fill!(similar(c0, T, dims), zero(T))

    # construct NamedTuples for left and right fluxes
    fl = NamedTuple{labels}(ntuple(d -> zeros_like(flux_size(sizes, d, N)), min(N, 3)))
    fr = NamedTuple{labels}(ntuple(d -> zeros_like(flux_size(sizes, d, N)), min(N, 3)))

    # semi-discretisation array
    du = zeros_like(sizes)

    # temporary array for Runge-Kutta
    ut = zeros_like(sizes)

    # Both staggered forms (conservative split-flux and non-conservative material
    # transport) need velocity prepared to cell centres once per step; the collocated
    # path passes its supplied velocity straight through and needs no buffer.
    vcenter = stag ? NamedTuple{labels}(ntuple(_ -> zeros_like(sizes), min(N, 3))) : nothing
    vperiodic = stag ? _resolved_vperiodic(faces, labels, extent) : nothing

    TFlux = typeof(fl)
    TArray = typeof(du)

    return WENOScheme{
        T, TArray, TFlux, typeof(vcenter), typeof(vperiodic), typeof(form_tag),
        typeof(faces), typeof(extent), typeof(topology),
    }(
        stag = stag, form = form_tag, boundary = faces, lim_ZS = lim_ZS,
        multithreading = multithreading, upwind_mode = upwind_mode, fl = fl, fr = fr,
        du = du, ut = ut, vcenter = vcenter, vperiodic = vperiodic, extent = extent,
        topology = topology,
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

    faces = boundary_faces(boundary)
    length(faces) == 2N || throw(
        ArgumentError(
            "boundary must contain $(2N) face conditions for $(N)D data, got $(length(faces))"
        )
    )
    all(b -> b isa AbstractAdvectionBoundary, faces) || throw(
        ArgumentError(
            "padded_weno_scheme expects an already-resolved boundary tuple of " *
                "AbstractAdvectionBoundary instances, got $(typeof(faces))"
        )
    )

    owned = size(c0) .- 2 .* halo
    all(>=(0), owned) || throw(
        ArgumentError(
            "halo $halo exceeds the allocated size $(size(c0)) — owned extent " *
                "would be $owned"
        )
    )

    gperiodic = global_periodic === nothing ? default_global_periodic(faces, N) : global_periodic
    extent = PaddedExtent{N}(owned, halo, global_size, gperiodic, geometry)
    return _build_weno_scheme(c0, extent, faces; form, stag, lim_ZS, multithreading, upwind_mode, topology)
end
