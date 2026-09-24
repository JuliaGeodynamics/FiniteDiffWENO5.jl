# Topology accessors keep scheme construction and stage hooks independent of MPI.

"""
    build_topology_weno_scheme(c0, topo; geometry = :cell, boundary, form,
                                stag = false, lim_ZS = false,
                                multithreading = true, upwind_mode = false)

Build a padded scheme from topology accessors. Validate boundaries against the
owned extent, replace nonphysical faces with `ProcessBC`, and check the array
size before allocation.
"""
function build_topology_weno_scheme(
        c0::AbstractArray{T, N}, topo; geometry::Symbol = :cell, boundary,
        form::Symbol, stag::Bool = false, lim_ZS::Bool = false,
        multithreading::Bool = true, upwind_mode::Bool = false,
    ) where {T, N}
    weno_ndims(topo) == N || throw(
        ArgumentError("topology is $(weno_ndims(topo))D but c0 is $(N)D")
    )

    owned = weno_owned_size(topo; geometry)
    user_faces = validate_boundary(boundary, N, owned)
    resolved = resolve_boundary(user_faces, topo)

    halo = weno_halo(topo)
    all(>=(3), halo) || throw(
        ArgumentError(
            "halo $halo from topology $(typeof(topo)) is thinner than the WENO5 reconstruction " *
                "stencil (3) on at least one axis — building a scheme on a topology padded for a " *
                "different stencil (e.g. a Stokes halo) would silently produce wrong ghost values " *
                "instead of an error"
        )
    )
    # A staggered low-side send needs h+1 owned entries to avoid forwarding
    # an unreceived seam ghost. Check every topology provider here.
    min_owned = stag ? halo .+ 1 : halo
    all(owned .>= min_owned) || throw(
        ArgumentError(
            "owned cells per rank $owned from topology $(typeof(topo)) must be at least " *
                "$min_owned on every axis (halo $halo" * (stag ? ", +1 for stag=true" : "") *
                ") — a thinner subdomain makes an exchange forward a not-yet-received ghost as " *
                "if it were owned data"
        )
    )
    expected_size = owned .+ 2 .* halo
    size(c0) == expected_size || throw(
        DimensionMismatch(
            "c0 has size $(size(c0)), expected $expected_size " *
                "(owned $owned + 2×halo $halo on topology $(typeof(topo)))"
        )
    )

    global_size = weno_global_size(topo; geometry)
    global_periodic = weno_periodic(topo)

    return padded_weno_scheme(
        c0, halo; boundary = resolved, form, stag, lim_ZS, multithreading, upwind_mode,
        global_size, global_periodic, geometry, topology = topo,
    )
end

"""
    WENOScheme(c0::AbstractArray{T,N}, topo::SerialTopology; geometry = :cell,
               boundary, form, stag = false, lim_ZS = false,
               multithreading = true, upwind_mode = false)

Build a padded single-rank scheme that uses the topology stage hooks.
"""
function WENOScheme(
        c0::AbstractArray{T, N}, topo::SerialTopology{N}; geometry::Symbol = :cell,
        boundary, form::Symbol, stag::Bool = false, lim_ZS::Bool = false,
        multithreading::Bool = true, upwind_mode::Bool = false,
    ) where {T, N}
    return build_topology_weno_scheme(
        c0, topo; geometry, boundary, form, stag, lim_ZS, multithreading, upwind_mode,
    )
end

# Exchange and refill the RK-stage array before each scalar operator call.

"""
    sync_stage!(weno, a)

Exchange halo values and refill physical ghosts before the operator reads `a`.
RK combinations write across the padded array, including its ghosts.
"""
sync_stage!(weno::WENOScheme, a) = _sync_stage!(weno.topology, weno, a)

_sync_stage!(::NoTopology, weno::WENOScheme, a) = a

function _sync_stage!(topo, weno::WENOScheme, a)
    weno_exchange_halo!(a, topo; geometry = weno.extent.geometry)
    fill_physical_ghosts!(a, weno.extent, weno.boundary)
    return a
end

# Reduce conservative-form Lax-Friedrichs speeds across the topology.

"""
    scheme_lf_speeds(weno, v)

Return the prepared velocity's Lax-Friedrichs speeds. Topology-backed
conservative schemes reduce `lf_speed` over owned cells, with all components
in one collective. Nonconservative schemes need no speed reduction.
"""
scheme_lf_speeds(weno::WENOScheme, v) = _scheme_lf_speeds(weno.topology, weno, v)

_scheme_lf_speeds(::NoTopology, weno::WENOScheme, v) = lf_speeds(weno.form, v)

function _scheme_lf_speeds(topo, weno::WENOScheme, v)
    is_conservative(weno.form) || return nothing
    names = keys(v)
    N = length(names)
    geometry = weno.extent.geometry
    local_speeds = ntuple(N) do d
        component = getproperty(v, names[d])
        lf_speed(owned_window(component, topo; geometry, stagger = nothing))
    end
    reduced = weno_allreduce_max(local_speeds, topo)
    return NamedTuple{names}(reduced)
end

# Multiphase transport uses the same stage hooks without LF speed reduction.

"""
    build_topology_multiphase_scheme(phases, topo; geometry = :cell, boundary,
                                      stag = false, multithreading = true)

Build a padded multiphase scheme from topology accessors.
"""
function build_topology_multiphase_scheme(
        phases::Tuple{Vararg{Any, NP}}, topo; geometry::Symbol = :cell, boundary,
        stag::Bool = false, multithreading::Bool = true,
    ) where {NP}
    c0 = first(phases)
    N = ndims(c0)
    weno_ndims(topo) == N || throw(
        ArgumentError("topology is $(weno_ndims(topo))D but the phases are $(N)D")
    )

    owned = weno_owned_size(topo; geometry)
    user_faces = validate_multiphase_boundary(boundary, N, owned, NP, eltype(c0))
    resolved = resolve_boundary(user_faces, topo)

    halo = weno_halo(topo)
    all(>=(3), halo) || throw(
        ArgumentError(
            "halo $halo from topology $(typeof(topo)) is thinner than the WENO5 reconstruction " *
                "stencil (3) on at least one axis — building a scheme on a topology padded for a " *
                "different stencil (e.g. a Stokes halo) would silently produce wrong ghost values " *
                "instead of an error"
        )
    )
    min_owned = stag ? halo .+ 1 : halo
    all(owned .>= min_owned) || throw(
        ArgumentError(
            "owned cells per rank $owned from topology $(typeof(topo)) must be at least " *
                "$min_owned on every axis (halo $halo" * (stag ? ", +1 for stag=true" : "") *
                ") — a thinner subdomain makes an exchange forward a not-yet-received ghost as " *
                "if it were owned data"
        )
    )
    expected_size = owned .+ 2 .* halo
    size(c0) == expected_size || throw(
        DimensionMismatch(
            "phases have size $(size(c0)), expected $expected_size " *
                "(owned $owned + 2×halo $halo on topology $(typeof(topo)))"
        )
    )

    global_size = weno_global_size(topo; geometry)
    global_periodic = weno_periodic(topo)

    return padded_multiphase_scheme(
        phases, halo; boundary = resolved, stag, multithreading,
        global_size, global_periodic, geometry, topology = topo,
    )
end

"""
    MultiphaseWENOScheme(phases::Tuple, topo::SerialTopology; geometry = :cell,
                          boundary, stag = false, multithreading = true)

Build a padded single-rank multiphase scheme.
"""
function MultiphaseWENOScheme(
        phases::Tuple{Vararg{Any, NP}}, topo::SerialTopology{N}; geometry::Symbol = :cell,
        boundary, stag::Bool = false, multithreading::Bool = true,
    ) where {NP, N}
    return build_topology_multiphase_scheme(phases, topo; geometry, boundary, stag, multithreading)
end

"""
    sync_stage!(scheme::MultiphaseWENOScheme, state)

Exchange and refill every phase before `multiphase_WENO_flux!` reads `state`.
Topology providers may exchange the tuple as one operation.
"""
sync_stage!(scheme::MultiphaseWENOScheme, state) = _sync_stage!(scheme.topology, scheme, state)

_sync_stage!(::NoTopology, scheme::MultiphaseWENOScheme, state) = state

function _sync_stage!(topo, scheme::MultiphaseWENOScheme, state)
    weno_exchange_halo!(state, topo; geometry = scheme.extent.geometry)
    for phase in state
        fill_physical_ghosts!(phase, scheme.extent, scheme.boundary)
    end
    return state
end
