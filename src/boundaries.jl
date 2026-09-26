"""Supertype for boundary treatments used by the advection operator."""
abstract type AbstractAdvectionBoundary end

"""Periodic continuation of the advected field."""
struct PeriodicBC <: AbstractAdvectionBoundary end

"""Constant extrapolation from the nearest interior cell."""
struct ExtrapolateBC <: AbstractAdvectionBoundary end

"""
    PrescribedInflowBC(value)

Prescribe the exterior upwind state at an inflow boundary. At outflow the
interior WENO reconstruction is used, so `value` is not imposed. `value` may
be a scalar or an array over the tangential boundary dimensions.
"""
struct PrescribedInflowBC{T} <: AbstractAdvectionBoundary
    value::T
end

"""
An interior process-seam face under a padded/distributed decomposition.
Retain the global axis's indexing policy so reconstruction compiles with the
same arithmetic as serial (periodic indexing can inhibit loop vectorization).
With a sufficient halo, neither wrapping nor clamping is reached by an owned
stencil; both keep scratch ghost computations in bounds. The tag still skips
physical ghost filling and inflow installation. User constructors reject it.
"""
struct ProcessBC{B <: Union{PeriodicBC, ExtrapolateBC}} <: AbstractAdvectionBoundary
    indexing::B
end
ProcessBC() = ProcessBC(ExtrapolateBC())

"""A dimension-independent wrapper around an ordered tuple of face conditions."""
struct AdvectionBC{B <: Tuple}
    faces::B
end

AdvectionBC(faces::Vararg{AbstractAdvectionBoundary}) = AdvectionBC(faces)

"""Convenience constructor for two-dimensional west/east/bottom/top faces."""
function AdvectionBC(;
        west = ExtrapolateBC(),
        east = ExtrapolateBC(),
        bot = ExtrapolateBC(),
        top = ExtrapolateBC(),
    )
    return AdvectionBC((west, east, bot, top))
end

boundary_faces(boundary::AdvectionBC) = boundary.faces
boundary_faces(boundary::Tuple) = boundary

valid_boundary(boundary::AbstractAdvectionBoundary) = true
valid_boundary(boundary::ProcessBC) = false
valid_boundary(boundary::Integer) = boundary in (0, 1, 2)
valid_boundary(boundary) = false

normalize_boundary(boundary::AbstractAdvectionBoundary) = boundary
normalize_boundary(boundary::Integer) = boundary == 2 ? PeriodicBC() : ExtrapolateBC()

function validate_inflow_value(boundary::PrescribedInflowBC, expected_size, face)
    value = boundary.value
    if value isa Real
        isfinite(value) || throw(
            ArgumentError(
                "PrescribedInflowBC on face $face must be finite, got $value"
            )
        )
    elseif value isa AbstractArray{<:Real}
        size(value) == expected_size || throw(
            DimensionMismatch(
                "PrescribedInflowBC on face $face requires a value array of size " *
                    "$expected_size, got $(size(value))"
            )
        )
        all(isfinite, value) || throw(
            ArgumentError(
                "PrescribedInflowBC on face $face contains a nonfinite value"
            )
        )
    else
        throw(
            ArgumentError(
                "PrescribedInflowBC on face $face requires a real scalar or array, " *
                    "got $(typeof(value))"
            )
        )
    end
    return nothing
end

validate_inflow_value(::Any, expected_size, face) = nothing

function tangential_size(sizes::NTuple{N, Int}, dimension) where {N}
    return ntuple(i -> sizes[i < dimension ? i : i + 1], N - 1)
end

"""
    normalize_boundary_faces(boundary, N)

Check the face count and entry kinds, then map legacy integer codes onto typed boundary
conditions. This is the half of `validate_boundary` that does not inspect the *value*
carried by a `PrescribedInflowBC`, so it can be shared by the scalar route and by the
multiphase route, whose inflow values are tuples that the scalar validator rejects.
"""
function normalize_boundary_faces(boundary, N)
    faces = boundary_faces(boundary)
    length(faces) == 2N || throw(
        ArgumentError(
            "boundary must contain $(2N) face conditions for $(N)D data, got " *
                "$(length(faces))"
        )
    )
    all(valid_boundary, faces) || throw(
        ArgumentError(
            "boundary entries must be PeriodicBC(), ExtrapolateBC(), " *
                "PrescribedInflowBC(value), or a legacy integer code 0, 1, or 2"
        )
    )
    return map(normalize_boundary, faces)
end

function validate_boundary(boundary, N, sizes = nothing)
    faces = normalize_boundary_faces(boundary, N)

    if sizes !== nothing
        for face in eachindex(faces)
            dimension = (face + 1) ÷ 2
            validate_inflow_value(
                faces[face], tangential_size(sizes, dimension), face
            )
        end
    end
    return faces
end

inflow_value(boundary::PrescribedInflowBC{<:Real}, indices...) = boundary.value
inflow_value(boundary::PrescribedInflowBC{<:AbstractArray}, indices...) =
    boundary.value[indices...]

# Typed boundaries use either periodic indexing or the existing constant
# extrapolation. Prescribed inflow values are installed directly into the
# exterior upwind state at the physical face after reconstruction.
left_index(i, d, nx, ::PeriodicBC) = mod1(i - d, nx)
right_index(i, d, nx, ::PeriodicBC) = mod1(i + d, nx)
left_index(i, d, nx, ::ExtrapolateBC) = max(i - d, 1)
right_index(i, d, nx, ::ExtrapolateBC) = min(i + d, nx)
left_index(i, d, nx, ::PrescribedInflowBC) = max(i - d, 1)
right_index(i, d, nx, ::PrescribedInflowBC) = min(i + d, nx)
# `ProcessBC` delegates to the global indexing policy — see the type's
# docstring. With a halo at least as wide as the reconstruction stencil the
# wrap/clamp never fires on an owned face; it keeps ghost-face reads
# memory-safe under `@inbounds`.
left_index(i, d, nx, b::ProcessBC) = left_index(i, d, nx, b.indexing)
right_index(i, d, nx, b::ProcessBC) = right_index(i, d, nx, b.indexing)

# Install inflow flux at physical faces and across owned tangential entries.
# The inflow value array is indexed relative to the owned extent.

"""
    _inflow_ranges(extent::PaddedExtent{N}, d, upper)

Index ranges for axis `d`'s (`upper` ? high : low) face: `d` fixed to one
index, every other axis spanning its owned entries.
"""
function _inflow_ranges(extent::PaddedExtent{N}, d, upper) where {N}
    face = upper ? extent.pad[d] + extent.owned[d] + 1 : extent.pad[d] + 1
    return ntuple(N) do ax
        ax == d ? (face:face) : (extent.pad[ax] + 1):(extent.pad[ax] + extent.owned[ax])
    end
end

"""
    _tangential_offset(I::CartesianIndex, extent::PaddedExtent{N}, d)

`I`'s coordinates on every axis but `d`, each shifted to a 1-based owned-window
offset — the argument order `inflow_value`/`multiphase_inflow_value` expect.
"""
function _tangential_offset(I::CartesianIndex, extent::PaddedExtent{N}, d) where {N}
    return ntuple(N - 1) do k
        axis = k < d ? k : k + 1
        I[axis] - extent.pad[axis]
    end
end

apply_axis_inflow!(flux, ::Any, extent, d, upper) = nothing

function apply_axis_inflow!(
        flux::AbstractArray{T, N}, boundary::PrescribedInflowBC, extent::PaddedExtent{N}, d, upper,
    ) where {T, N}
    @inbounds for I in CartesianIndices(_inflow_ranges(extent, d, upper))
        flux[I] = inflow_value(boundary, _tangential_offset(I, extent, d)...)
    end
    return nothing
end

# Three per-arity methods, not one loop over `enumerate(keys(fl))`: `boundary`
# is a heterogeneous tuple, so a runtime-computed index into it (`boundary[2d-1]`)
# is not type-stable and allocates (measured). Every index/field access below
# is a compile-time literal instead.
function apply_inflow_boundaries!(fl::NamedTuple{(:x,)}, fr, boundary, extent)
    apply_axis_inflow!(fl.x, boundary[1], extent, 1, false)
    apply_axis_inflow!(fr.x, boundary[2], extent, 1, true)
    return nothing
end

function apply_inflow_boundaries!(fl::NamedTuple{(:x, :y)}, fr, boundary, extent)
    apply_axis_inflow!(fl.x, boundary[1], extent, 1, false)
    apply_axis_inflow!(fr.x, boundary[2], extent, 1, true)
    apply_axis_inflow!(fl.y, boundary[3], extent, 2, false)
    apply_axis_inflow!(fr.y, boundary[4], extent, 2, true)
    return nothing
end

function apply_inflow_boundaries!(fl::NamedTuple{(:x, :y, :z)}, fr, boundary, extent)
    apply_axis_inflow!(fl.x, boundary[1], extent, 1, false)
    apply_axis_inflow!(fr.x, boundary[2], extent, 1, true)
    apply_axis_inflow!(fl.y, boundary[3], extent, 2, false)
    apply_axis_inflow!(fr.y, boundary[4], extent, 2, true)
    apply_axis_inflow!(fl.z, boundary[5], extent, 3, false)
    apply_axis_inflow!(fr.z, boundary[6], extent, 3, true)
    return nothing
end

# Fill physical ghosts with serial clamp or wrap values. Prescribed inflow
# changes boundary fluxes, so ghost filling does not write its prescribed state.

@inline _ghost_kind(::PeriodicBC) = :periodic
@inline _ghost_kind(::ExtrapolateBC) = :extrapolate
@inline _ghost_kind(::PrescribedInflowBC) = :extrapolate
@inline _ghost_kind(::ProcessBC) = :none

"""
    _fill_ghost_axis!(a, d, pad, n_phys, owned_lo, owned_hi, lo_bc, hi_bc)

Fill every entry of `a` outside `[owned_lo, owned_hi]` along axis `d`. Low
positions (`< owned_lo`) use `lo_bc`, high positions (`> owned_hi`) use
`hi_bc`. `ExtrapolateBC`/`PrescribedInflowBC` clamp to the nearest owned edge;
`PeriodicBC` wraps modulo `n_phys` cells starting at padded index `pad + 1`;
`ProcessBC` is left untouched (its ghosts come from a halo exchange).
"""
function _fill_ghost_axis!(a::AbstractArray{T, N}, d, pad, n_phys, owned_lo, owned_hi, lo_bc, hi_bc) where {T, N}
    lo_kind = _ghost_kind(lo_bc)
    hi_kind = _ghost_kind(hi_bc)
    (lo_kind === :none && hi_kind === :none) && return a
    @inbounds for I in CartesianIndices(a)
        g = I[d]
        owned_lo <= g <= owned_hi && continue
        kind = g < owned_lo ? lo_kind : hi_kind
        kind === :none && continue
        src = kind === :periodic ? mod1(g - pad, n_phys) + pad : (g < owned_lo ? owned_lo : owned_hi)
        Isrc = CartesianIndex(ntuple(k -> k == d ? src : I[k], N))
        a[I] = a[Isrc]
    end
    return a
end

"""
    fill_physical_ghosts!(a::AbstractArray, extent::PaddedExtent, boundary)

Cell-centred ghost fill: for each axis with nonzero pad, write the pad
entries of `a` that the resolved `boundary` implies on that axis' two faces.
`a` may be the advected field, `ut`, or a cell-centred (`stag = false`)
velocity component — the fill never depends on what the array represents,
only on the boundary kind, which is what makes it safe to reuse across all
of them.
"""
function fill_physical_ghosts!(a::AbstractArray{T, N}, extent::PaddedExtent{N}, boundary) where {T, N}
    faces = boundary_faces(boundary)
    for d in 1:N
        pad = extent.pad[d]
        pad == 0 && continue
        n_phys = extent.owned[d]
        owned_lo = pad + 1
        owned_hi = pad + n_phys
        _fill_ghost_axis!(a, d, pad, n_phys, owned_lo, owned_hi, faces[2d - 1], faces[2d])
    end
    return a
end

"""
    fill_physical_ghosts!(v::NamedTuple, extent::PaddedExtent, boundary)

Face-staggered ghost fill for a velocity `NamedTuple` whose component at
position `direction` (`:x`, `:y`, `:z`, ...) carries one extra entry on the
high side of its own normal axis (`direction`). Every axis other than a
component's own normal axis is filled exactly like the cell-centred case.
Along its own normal axis, a periodic face treats the physical `n_own + 1`-th
face as the duplicate of physical face 1 — written by the wrap, never read as
independent data — matching how the serial ENO5 interpolation ignores
`face[n+1]` under periodicity.
"""
function fill_physical_ghosts!(v::NamedTuple, extent::PaddedExtent{N}, boundary) where {N}
    faces = boundary_faces(boundary)
    names = keys(v)
    for direction in eachindex(names)
        component = getproperty(v, names[direction])
        for d in 1:N
            pad = extent.pad[d]
            pad == 0 && continue
            n_phys = extent.owned[d]
            lo_bc, hi_bc = faces[2d - 1], faces[2d]
            owned_lo = pad + 1
            # A component is face-staggered along its OWN axis only if it
            # structurally carries the extra entry there (`size == n_phys +
            # 2·pad + 1`) — e.g. the raw face velocity passed into
            # `prepare_velocity!`. A NamedTuple of otherwise cell-centred
            # arrays (e.g. `weno.vcenter`, which shares `du`'s shape on every
            # axis) must NOT take the extra-face branch merely because it is
            # a NamedTuple; dispatch on this structural check, not on type.
            own_axis_staggered = d == direction && size(component, d) == n_phys + 2pad + 1
            extra_face = own_axis_staggered && !(hi_bc isa PeriodicBC)
            owned_hi = pad + n_phys + (extra_face ? 1 : 0)
            _fill_ghost_axis!(component, d, pad, n_phys, owned_lo, owned_hi, lo_bc, hi_bc)
        end
    end
    return v
end
