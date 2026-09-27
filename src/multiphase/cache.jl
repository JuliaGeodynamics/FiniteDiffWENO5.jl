@kwdef struct MultiphaseWENOScheme{T, NP, TArray, TFlux, TVelocity, TPeriodicity, TBoundary, TExtent, TTopology, THaloBuffers} <: AbstractWENO
    γ::NTuple{3, T} = T.((0.1, 0.6, 0.3))
    χ::NTuple{2, T} = T.((13 / 12, 1 / 4))
    ζ::NTuple{5, T} = T.((1 / 3, 7 / 6, 11 / 6, 1 / 6, 5 / 6))
    ϵ::T = eps(T)
    stag::Bool
    boundary::TBoundary
    multithreading::Bool
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
    MultiphaseWENOScheme(phases::Tuple; boundary=nothing, stag=false, multithreading=true)

WENO5-Z scheme for fractions satisfying `0 ≤ ϕₖ ≤ 1` and `Σₖϕₖ = 1`.
Phases share reconstruction weights and a Zhang-Shu limiter coefficient so
the sum is preserved. Use `WENOScheme` for unrelated fields.

# Arguments
- `phases`: At least two arrays with identical axes, element type, and concrete
  array type. Values are not read during construction.
- `boundary`: Face conditions shared by all phases; defaults to extrapolation.
- `stag`: Use face-centered velocities when `true`.
- `multithreading`: Enable threading in 2D or 3D.

The simplex limiter is always enabled; upwind mode and custom bounds are not
supported.
"""
function _validate_phases(phases::Tuple{Vararg{Any, NP}}) where {NP}
    NP >= 2 || throw(
        ArgumentError(
            "MultiphaseWENOScheme requires at least two phases, got $NP. " *
                "Use WENOScheme for a single field."
        )
    )

    all(p -> p isa AbstractArray, phases) || throw(
        ArgumentError(
            "MultiphaseWENOScheme requires a tuple of arrays, got " *
                "$(map(typeof, phases))"
        )
    )

    c0 = first(phases)
    T = eltype(c0)
    N = ndims(c0)
    1 <= N <= 3 || throw(
        ArgumentError(
            "MultiphaseWENOScheme supports 1D, 2D, and 3D fields, got $(N)D"
        )
    )

    for k in 2:NP
        p = phases[k]
        eltype(p) === T || throw(
            ArgumentError(
                "all phases must share an element type, phase 1 is $(T) but phase $k is " *
                    "$(eltype(p))"
            )
        )
        ndims(p) == N || throw(
            DimensionMismatch(
                "all phases must share a dimensionality, phase 1 is $(N)D but phase $k is " *
                    "$(ndims(p))D"
            )
        )
        axes(p) == axes(c0) || throw(
            DimensionMismatch(
                "all phases must share axes, phase 1 has $(axes(c0)) but phase $k has " *
                    "$(axes(p))"
            )
        )
        typeof(p) === typeof(c0) || throw(
            ArgumentError(
                "all phases must share a concrete array type, phase 1 is $(typeof(c0)) " *
                    "but phase $k is $(typeof(p))"
            )
        )
    end
    return c0, T, N
end

"""Allocate multiphase buffers from validated or resolved boundary faces."""
function _build_multiphase_scheme(
        phases::Tuple{Vararg{Any, NP}}, extent::PaddedExtent{N}, faces;
        stag::Bool, multithreading::Bool, topology = NoTopology(),
    ) where {NP, N}
    c0 = first(phases)
    T = eltype(c0)

    labels = (:x, :y, :z)[1:min(N, 3)]
    sizes = size(c0)
    zeros_like(dims) = fill!(similar(c0, T, dims), zero(T))
    valNP = Val(NP)

    fl = NamedTuple{labels}(
        ntuple(min(N, 3)) do d
            ntuple(_ -> zeros_like(flux_size(sizes, d, N)), valNP)
        end
    )
    fr = NamedTuple{labels}(
        ntuple(min(N, 3)) do d
            ntuple(_ -> zeros_like(flux_size(sizes, d, N)), valNP)
        end
    )

    du = ntuple(_ -> zeros_like(sizes), valNP)
    ut = ntuple(_ -> zeros_like(sizes), valNP)

    vcenter = stag ? NamedTuple{labels}(ntuple(_ -> zeros_like(sizes), Val(N))) : nothing
    vperiodic = stag ? _resolved_vperiodic(faces, labels, extent) : nothing
    halo_buffers = halo_buffers_for_multiphase(topology, extent, stag, T, Val(NP))

    return MultiphaseWENOScheme{
        T, NP, typeof(du), typeof(fl), typeof(vcenter), typeof(vperiodic),
        typeof(faces), typeof(extent), typeof(topology), typeof(halo_buffers),
    }(
        stag = stag, boundary = faces, multithreading = multithreading,
        fl = fl, fr = fr, du = du, ut = ut, vcenter = vcenter, vperiodic = vperiodic,
        extent = extent, topology = topology, halo_buffers = halo_buffers,
    )
end

function MultiphaseWENOScheme(
        phases::Tuple{Vararg{Any, NP}};
        boundary = nothing, stag::Bool = false, multithreading::Bool = true,
    ) where {NP}
    c0, T, N = _validate_phases(phases)

    boundary === nothing && (boundary = ntuple(i -> ExtrapolateBC(), N * 2))
    faces = validate_multiphase_boundary(boundary, N, size(c0), NP, T)
    extent = default_extent(size(c0), default_global_periodic(faces, N))
    return _build_multiphase_scheme(phases, extent, faces; stag, multithreading)
end

"""
    padded_multiphase_scheme(phases, halo; boundary, stag=false,
                              multithreading=true, global_size=nothing,
                              global_periodic=nothing, geometry=:cell,
                              topology=NoTopology())

Build a padded multiphase scheme from resolved boundaries, including
`ProcessBC`. The topology constructor supplies the halo and extents.
"""
function padded_multiphase_scheme(
        phases::Tuple{Vararg{Any, NP}}, halo::NTuple{N, Int}; boundary,
        stag::Bool = false, multithreading::Bool = true,
        global_size::Union{NTuple{N, Int}, Nothing} = nothing,
        global_periodic::Union{NTuple{N, Bool}, Nothing} = nothing,
        geometry::Symbol = :cell,
        topology = NoTopology(),
    ) where {NP, N}
    c0, T, N2 = _validate_phases(phases)
    N2 == N || throw(ArgumentError("halo has $N entries but phases are $(N2)D"))

    faces, owned = _resolve_padded_extent(size(c0), halo, boundary, :padded_multiphase_scheme)

    gsize = global_size === nothing ? owned : global_size
    gperiodic = global_periodic === nothing ? default_global_periodic(faces, N) : global_periodic
    extent = PaddedExtent{N}(owned, halo, gsize, gperiodic, geometry)
    return _build_multiphase_scheme(phases, extent, faces; stag, multithreading, topology)
end

"""
    nphases(scheme::MultiphaseWENOScheme)

Number of phases carried by `scheme`, available as a compile-time constant.
"""
@inline nphases(::MultiphaseWENOScheme{T, NP}) where {T, NP} = NP
