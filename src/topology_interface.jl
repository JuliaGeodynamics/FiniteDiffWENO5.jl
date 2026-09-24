# Topology providers implement these accessors without sharing a supertype.
# SerialTopology exercises padding and halo hooks without MPI.

"""
    AbstractWENOTopology

Optional supertype for a domain decomposition driving a padded WENO scheme.
Subtyping it is not required: the scheme builders only call the accessor
functions (`weno_ndims`, `weno_halo`, `weno_owned_size`, `weno_global_size`,
`weno_global_offset`, `weno_periodic`, `weno_physical_low`,
`weno_physical_high`, `weno_exchange_halo!`, `weno_allreduce_max`,
`weno_allreduce_min`). Extend them under their qualified `FiniteDiffWENO5.`
names.
"""
abstract type AbstractWENOTopology end

"""Report a missing topology accessor."""
_no_topology_method(name, topo) = throw(
    ArgumentError(
        "$name has no method for $(typeof(topo)). Load the extension that " *
            "implements the WENO topology interface for this type " *
            "(e.g. FiniteDiffWENO5's MPI extension for `WENOCartesianTopology`), " *
            "or add the missing accessor " *
            "method for your own topology type."
    )
)

"""Number of spatial dimensions `N` this topology partitions."""
weno_ndims(topo) = _no_topology_method(:weno_ndims, topo)

"""Ghost width per axis, `NTuple{N,Int}`."""
weno_halo(topo) = _no_topology_method(:weno_halo, topo)

"""Owned entry count per axis on the selected cell or vertex lattice."""
weno_owned_size(topo; geometry::Symbol = :cell) = _no_topology_method(:weno_owned_size, topo)

"""Global entry count per axis; vertex counts include one extra per axis."""
weno_global_size(topo; geometry::Symbol = :cell) = _no_topology_method(:weno_global_size, topo)

"""Zero-based global offset of this rank's first owned entry."""
weno_global_offset(topo; geometry::Symbol = :cell) = _no_topology_method(:weno_global_offset, topo)

"""Topology periodicity per axis; boundary periodicity must match."""
weno_periodic(topo) = _no_topology_method(:weno_periodic, topo)

"""Whether this rank owns each low physical face. Single-rank axes count as physical."""
weno_physical_low(topo) = _no_topology_method(:weno_physical_low, topo)

"""Whether this rank owns each high physical face."""
weno_physical_high(topo) = _no_topology_method(:weno_physical_high, topo)

"""
    weno_exchange_halo!(field, topo; geometry = :cell, stagger = nothing)

Fill `field`'s ghost cells from neighbouring ranks on `topo`. `stagger = d`
means `field` is face-staggered along axis `d` on the cell lattice (one extra
entry on the high side of its own normal axis); `stagger = nothing` means
cell-centred on the selected `geometry` lattice. `geometry = :vertex`
requires `stagger = nothing`.
"""
weno_exchange_halo!(field::AbstractArray, topo; geometry::Symbol = :cell, stagger::Union{Nothing, Int} = nothing) =
    _no_topology_method(:weno_exchange_halo!, topo)

"""
    weno_exchange_halo!(fields::Tuple, topo; geometry = :cell, stagger = nothing)

Exchange each array using the single-array method. A topology may override
this method to fuse messages."""
function weno_exchange_halo!(fields::Tuple, topo; geometry::Symbol = :cell, stagger::Union{Nothing, Int} = nothing)
    for field in fields
        weno_exchange_halo!(field, topo; geometry, stagger)
    end
    return fields
end

"""Collective elementwise maximum over a number or tuple. Tuple components
must be reduced separately; Julia compares whole tuples lexicographically."""
weno_allreduce_max(value, topo) = _no_topology_method(:weno_allreduce_max, topo)

"""Collective elementwise minimum — see [`weno_allreduce_max`](@ref)."""
weno_allreduce_min(value, topo) = _no_topology_method(:weno_allreduce_min, topo)

# Default marker for schemes without padding or communication.

"""
    NoTopology()

Default marker for unpadded schemes. Topology accessors throw on this type.
"""
struct NoTopology end

# Single-rank topology with explicit padding.

"""
    SerialTopology(global_dims::NTuple{N,Int}; halo = 3, periodic = false)

Single-rank topology: owned size equals global size, zero offset, both faces
of every axis physical, `weno_exchange_halo!` a no-op, and the reductions the
identity. `halo`/`periodic` may each be given as a scalar (applied to every
axis) or an `NTuple{N}`.
"""
struct SerialTopology{N} <: AbstractWENOTopology
    global_dims::NTuple{N, Int}
    halo::NTuple{N, Int}
    periodic::NTuple{N, Bool}
end

function SerialTopology(
        global_dims::NTuple{N, Int};
        halo::Union{Int, NTuple{N, Int}} = 3,
        periodic::Union{Bool, NTuple{N, Bool}} = false,
    ) where {N}
    halo_t = halo isa Int ? ntuple(_ -> halo, N) : halo
    periodic_t = periodic isa Bool ? ntuple(_ -> periodic, N) : periodic
    return SerialTopology{N}(global_dims, halo_t, periodic_t)
end

weno_ndims(::SerialTopology{N}) where {N} = N
weno_halo(topo::SerialTopology) = topo.halo
weno_owned_size(topo::SerialTopology; geometry::Symbol = :cell) =
    geometry === :vertex ? topo.global_dims .+ 1 : topo.global_dims
weno_global_size(topo::SerialTopology; geometry::Symbol = :cell) = weno_owned_size(topo; geometry)
weno_global_offset(topo::SerialTopology{N}; geometry::Symbol = :cell) where {N} = ntuple(_ -> 0, N)
weno_periodic(topo::SerialTopology) = topo.periodic
weno_physical_low(topo::SerialTopology{N}) where {N} = ntuple(_ -> true, N)
weno_physical_high(topo::SerialTopology{N}) where {N} = ntuple(_ -> true, N)

function weno_exchange_halo!(field::AbstractArray, topo::SerialTopology; geometry::Symbol = :cell, stagger::Union{Nothing, Int} = nothing)
    geometry === :vertex && stagger !== nothing && throw(
        ArgumentError("geometry = :vertex requires stagger = nothing")
    )
    return field # single rank: no neighbour to exchange with
end

weno_allreduce_max(value, ::SerialTopology) = value
weno_allreduce_min(value, ::SerialTopology) = value

# Resolve user boundaries for each rank.

"""
    resolve_boundary(boundary, topo)

Replace nonphysical faces with `ProcessBC()` and keep physical faces unchanged.
Any topology implementing the accessors can use this method. Reject periodicity
that disagrees with the topology to avoid wrapping rank-local values.
"""
function resolve_boundary(boundary, topo)
    N = weno_ndims(topo)
    faces = boundary_faces(boundary)
    length(faces) == 2N || throw(
        ArgumentError(
            "boundary must contain $(2N) face conditions for a $(N)D topology, got $(length(faces))"
        )
    )

    topo_periodic = weno_periodic(topo)
    for d in 1:N
        bc_lo, bc_hi = faces[2d - 1], faces[2d]
        (bc_lo isa PeriodicBC) == (bc_hi isa PeriodicBC) || throw(
            ArgumentError("boundary axis $d pairs a periodic face with a non-periodic one")
        )
        (bc_lo isa PeriodicBC) == topo_periodic[d] || throw(
            ArgumentError(
                "boundary axis $d is " * ((bc_lo isa PeriodicBC) ? "periodic" : "non-periodic") *
                    ", but the topology's axis $d is " * (topo_periodic[d] ? "periodic" : "non-periodic")
            )
        )
    end

    phys_lo = weno_physical_low(topo)
    phys_hi = weno_physical_high(topo)
    return ntuple(2N) do face
        d = (face + 1) ÷ 2
        physical = isodd(face) ? phys_lo[d] : phys_hi[d]
        physical ? faces[face] : ProcessBC(topo_periodic[d] ? PeriodicBC() : ExtrapolateBC())
    end
end

# Helpers shared by topology providers.

"""
    allocate_weno_field(topo; T = Float64, geometry = :cell, stagger = nothing)

A zeroed, padded `Array{T,N}` sized exactly as a `WENOScheme`/
`MultiphaseWENOScheme` field buffer on `topo` would be:
`weno_owned_size(topo; geometry) .+ 2 .* weno_halo(topo)`, plus one entry
along `stagger` (cell geometry only). `geometry = :vertex` requires
`stagger = nothing`.
"""
function allocate_weno_field(topo; T::Type = Float64, geometry::Symbol = :cell, stagger::Union{Nothing, Int} = nothing)
    geometry === :vertex && stagger !== nothing && throw(
        ArgumentError("geometry = :vertex requires stagger = nothing")
    )
    N = weno_ndims(topo)
    owned = weno_owned_size(topo; geometry)
    halo = weno_halo(topo)
    sizes = ntuple(d -> owned[d] + 2halo[d] + (d == stagger ? 1 : 0), N)
    return zeros(T, sizes)
end

"""
    owned_window(a, topo; geometry = :cell, stagger = nothing)

A view of owned entries. A face-staggered array includes its extra high face
only on a physical, nonperiodic high boundary.
"""
function owned_window(a::AbstractArray, topo; geometry::Symbol = :cell, stagger::Union{Nothing, Int} = nothing)
    geometry === :vertex && stagger !== nothing && throw(
        ArgumentError("geometry = :vertex requires stagger = nothing")
    )
    N = weno_ndims(topo)
    owned = weno_owned_size(topo; geometry)
    halo = weno_halo(topo)
    phys_hi = weno_physical_high(topo)
    ranges = ntuple(N) do d
        lo = halo[d] + 1
        extra = (d == stagger && phys_hi[d] && !weno_periodic(topo)[d]) ? 1 : 0
        lo:(halo[d] + owned[d] + extra)
    end
    return view(a, ranges...)
end

"""
    weno_global_ranges(topo; geometry = :cell)

This rank's owned global indices per axis, `NTuple{N,UnitRange{Int}}`, from
`weno_global_offset` and `weno_owned_size`."""
function weno_global_ranges(topo; geometry::Symbol = :cell)
    N = weno_ndims(topo)
    offset = weno_global_offset(topo; geometry)
    owned = weno_owned_size(topo; geometry)
    return ntuple(d -> (offset[d] + 1):(offset[d] + owned[d]), N)
end

# CFL inputs must agree across ranks so each rank takes the same number of steps.

"""
    weno_cfl_dt(topo, velocity::Union{Tuple, NamedTuple}, spacing, cfl; geometry = :cell, staggered = true)

Compute a global CFL timestep from component maxima over owned entries.
Reduce all maxima together, then sum `vmax[d] / spacing[d]` in axis order.
Set `staggered = false` for collocated components, including vertex fields.
"""
function weno_cfl_dt(topo, velocity::Union{Tuple, NamedTuple}, spacing, cfl; geometry::Symbol = :cell, staggered::Bool = true)
    N = length(velocity)
    vmax_local = ntuple(N) do d
        component = velocity[d]
        ow = owned_window(component, topo; geometry, stagger = staggered ? d : nothing)
        maximum(abs, ow)
    end
    vmax = weno_allreduce_max(vmax_local, topo)
    speed = zero(eltype(spacing))
    for d in 1:N
        speed += vmax[d] / spacing[d]
    end
    return iszero(speed) ? typeof(speed)(Inf) : cfl / speed
end

"""Compute the same CFL formula over whole arrays for an unpadded scheme."""
function weno_cfl_dt(::NoTopology, velocity::Union{Tuple, NamedTuple}, spacing, cfl; geometry::Symbol = :cell, staggered::Bool = true)
    N = length(velocity)
    speed = zero(eltype(spacing))
    for d in 1:N
        speed += maximum(abs, velocity[d]) / spacing[d]
    end
    return iszero(speed) ? typeof(speed)(Inf) : cfl / speed
end

"""
    weno_substeps(topo, duration, dt_cfl; debug = false)

Compute the substep count from a shared duration and collective CFL timestep.
With `debug = true`, check agreement across ranks before advancing.
"""
function weno_substeps(topo, duration, dt_cfl; debug::Bool = false)
    # No early return: under `debug`, every rank must reach the reductions below.
    n = (iszero(duration) || isinf(dt_cfl)) ? 0 : max(1, ceil(Int, duration / dt_cfl))
    if debug && !(topo isa NoTopology)
        n_min = weno_allreduce_min(n, topo)
        n_max = weno_allreduce_max(n, topo)
        n_min == n_max || throw(
            ArgumentError(
                "weno_substeps: ranks disagree on the substep count ($n_min vs $n_max) " *
                    "despite collective inputs — this should be impossible; check that " *
                    "`duration` and `dt_cfl` are truly identical on every rank"
            )
        )
    end
    return n
end
