# Fifth-order ENO point interpolation from normal staggered faces to scalar
# cell centres. This is deliberately separate from finite-difference WENO flux
# reconstruction: the latter reconstructs sliding-average fluxes, not point values.

const ENO5_MIDPOINT_NUMERATORS = (
    (-5, 28, -70, 140, 35),
    (3, -20, 90, 60, -5),
    (-5, 60, 90, -20, 3),
    (35, 140, -70, 28, -5),
)

@inline eno5_row(s, i) = ENO5_MIDPOINT_NUMERATORS[s - i + 4]

@inline function eno5_difference_valid(s, p, n, periodic)
    return periodic || (1 <= s && s + p <= n + 1)
end

"""
    ENO5PhysicalRestriction

Physical cell and face bounds for ENO5 interpolation near boundaries.
Restrict stencils at extrapolation and inflow faces only for owned cells;
ghost cells use the allocated-extent bound.
"""
struct ENO5PhysicalRestriction
    owned_lo::Int
    owned_hi::Int
    face_lo::Int
    face_hi::Int
    restrict_lo::Bool
    restrict_hi::Bool
end

@inline function eno5_difference_valid(s, p, n, periodic, i::Int, r::ENO5PhysicalRestriction)
    eno5_difference_valid(s, p, n, periodic) || return false
    (r.owned_lo <= i <= r.owned_hi) || return true
    r.restrict_lo && s < r.face_lo && return false
    r.restrict_hi && s + p > r.face_hi && return false
    return true
end

"""
    eno5_face_to_center!(center, face; periodic)

Interpolate a normal velocity stored at cell faces to the corresponding scalar
cell centres. In the periodic case, `face[n+1]`, when present, is a duplicate
and is ignored; logical face `n+1` aliases face `1`.
"""
function eno5_face_to_center!(center::AbstractVector, face::AbstractVector; periodic::Bool)
    n = length(center)
    if periodic
        n >= 5 || throw(ArgumentError("periodic ENO5 interpolation requires at least 5 cells, got $n"))
        length(face) in (n, n + 1) || throw(
            DimensionMismatch(
                "periodic face velocity must have $n or $(n + 1) samples, got $(length(face))",
            )
        )
    else
        n >= 4 || throw(ArgumentError("nonperiodic ENO5 interpolation requires at least 4 cells, got $n"))
        length(face) == n + 1 || throw(
            DimensionMismatch(
                "nonperiodic face velocity must have $(n + 1) samples, got $(length(face))",
            )
        )
    end

    denominator = oftype(first(face), 128)
    @inbounds for i in eachindex(center)
        I = CartesianIndex(i)
        s = eno5_stencil_start(face, I, 1, i, n, periodic)
        row = eno5_row(s, i)
        value = zero(eltype(face))
        for r in 0:4
            value += row[r + 1] * eno5_face_sample(face, I, 1, s + r, n, periodic)
        end
        center[i] = value / denominator
    end
    return center
end

@inline function eno5_face_sample(face, I::CartesianIndex{N}, direction, q, n, periodic) where {N}
    index = ntuple(d -> d == direction ? (periodic ? mod1(q, n) : q) : I[d], N)
    return face[index...]
end

function eno5_undivided_difference(face, I::CartesianIndex, direction, s, p, n, periodic)
    value = zero(eltype(face))
    for r in 0:p
        coefficient = isodd(p - r) ? -binomial(p, r) : binomial(p, r)
        value += coefficient * eno5_face_sample(face, I, direction, s + r, n, periodic)
    end
    return value
end

function eno5_stencil_start(face, I::CartesianIndex, direction, i, n, periodic)
    s = i
    for p in 2:4
        left_ok = eno5_difference_valid(s - 1, p, n, periodic)
        right_ok = eno5_difference_valid(s, p, n, periodic)
        (!left_ok && !right_ok) && throw(
            ArgumentError(
                "ENO5 has no valid stencil at cell $i with $n cells",
            )
        )
        left = left_ok ? abs(eno5_undivided_difference(face, I, direction, s - 1, p, n, periodic)) : Inf
        right = right_ok ? abs(eno5_undivided_difference(face, I, direction, s, p, n, periodic)) : Inf
        left_ok && (!right_ok || left <= right) && (s -= 1)
    end
    return s
end

"""
    eno5_stencil_start(face, I, direction, i, n, periodic, r::ENO5PhysicalRestriction)

Resolved-BC-aware variant of [`eno5_stencil_start`](@ref): identical
recurrence, restricted per [`ENO5PhysicalRestriction`](@ref). A new method,
not a replacement — the unrestricted 4-/6-argument methods above are used
unchanged by every existing caller (including the KernelAbstractions
kernels), which never carry padding.
"""
function eno5_stencil_start(face, I::CartesianIndex, direction, i, n, periodic, r::ENO5PhysicalRestriction)
    s = i
    for p in 2:4
        left_ok = eno5_difference_valid(s - 1, p, n, periodic, i, r)
        right_ok = eno5_difference_valid(s, p, n, periodic, i, r)
        (!left_ok && !right_ok) && throw(
            ArgumentError(
                "ENO5 has no valid stencil at cell $i with $n cells",
            )
        )
        left = left_ok ? abs(eno5_undivided_difference(face, I, direction, s - 1, p, n, periodic)) : Inf
        right = right_ok ? abs(eno5_undivided_difference(face, I, direction, s, p, n, periodic)) : Inf
        left_ok && (!right_ok || left <= right) && (s -= 1)
    end
    return s
end

# The 1D CPU and KA kernels use the same direction-generic recurrence. These
# narrow forwarding methods keep their compact call sites without maintaining a
# second stencil-selection implementation.
@inline eno5_face_sample(face::AbstractVector, q, n, periodic) =
    eno5_face_sample(face, CartesianIndex(1), 1, q, n, periodic)
@inline eno5_undivided_difference(face::AbstractVector, s, p, n, periodic) =
    eno5_undivided_difference(face, CartesianIndex(1), 1, s, p, n, periodic)
@inline eno5_stencil_start(face::AbstractVector, i, n, periodic) =
    eno5_stencil_start(face, CartesianIndex(i), 1, i, n, periodic)
@inline eno5_stencil_start(face::AbstractVector, i, n, periodic, r::ENO5PhysicalRestriction) =
    eno5_stencil_start(face, CartesianIndex(i), 1, i, n, periodic, r)

function validate_face_to_center_direction(
        center::AbstractArray, face::AbstractArray,
        direction::Int; periodic::Bool
    )
    N = ndims(center)
    N == ndims(face) || throw(
        DimensionMismatch(
            "center velocity is $(N)D but face velocity is $(ndims(face))D",
        )
    )
    1 <= direction <= N || throw(ArgumentError("invalid velocity direction $direction for $(N)D field"))
    n = size(center, direction)
    for d in 1:N
        if d != direction && size(face, d) != size(center, d)
            throw(DimensionMismatch("tangential face-velocity axis $d must match the center field"))
        end
    end
    if periodic
        size(face, direction) in (n, n + 1) || throw(
            DimensionMismatch(
                "periodic normal face-velocity axis must have $n or $(n + 1) samples",
            )
        )
    else
        size(face, direction) == n + 1 || throw(
            DimensionMismatch(
                "nonperiodic normal face-velocity axis must have $(n + 1) samples",
            )
        )
    end

    return nothing
end

function validate_staggered_velocity!(
        center::NamedTuple, face::NamedTuple;
        periodic::NamedTuple
    )
    keys(center) == keys(face) == keys(periodic) || throw(
        ArgumentError(
            "center velocity, face velocity, and periodicity must use the same direction labels",
        )
    )
    for direction in eachindex(keys(center))
        name = keys(center)[direction]
        validate_face_to_center_direction(
            getproperty(center, name), getproperty(face, name), direction;
            periodic = getproperty(periodic, name),
        )
    end
    return nothing
end

function eno5_face_to_center_direction!(
        center::AbstractArray, face::AbstractArray,
        direction::Int; periodic::Bool, restriction::Union{Nothing, ENO5PhysicalRestriction} = nothing
    )
    validate_face_to_center_direction(center, face, direction; periodic)
    n = size(center, direction)
    n >= eno5_minimum_cells(periodic) || throw(
        ArgumentError(
            "$(periodic ? "periodic" : "nonperiodic") ENO5 interpolation requires at least " *
                "$(eno5_minimum_cells(periodic)) cells, got $n",
        )
    )
    denominator = oftype(first(face), 128)
    @inbounds for I in CartesianIndices(center)
        i = I[direction]
        s = restriction === nothing ?
            eno5_stencil_start(face, I, direction, i, n, periodic) :
            eno5_stencil_start(face, I, direction, i, n, periodic, restriction)
        row = eno5_row(s, i)
        value = zero(eltype(face))
        for r in 0:4
            value += row[r + 1] * eno5_face_sample(face, I, direction, s + r, n, periodic)
        end
        center[I] = value / denominator
    end
    return center
end

"""Smallest cell count in one direction that admits a five-face ENO5 stencil."""
@inline eno5_minimum_cells(periodic::Bool) = periodic ? 5 : 4

"""
Second-order fallback for directions too small to carry the ENO5 stencil.

A grid with fewer cells than the stencil cannot support fifth-order interpolation
at all, so the alternative to this fallback is refusing to run. The two bracketing
faces average to the cell centre exactly for affine data, which is the best a
three- or four-cell direction admits; order is limited by the grid, not by choice.
"""
function linear_face_to_center_direction!(center, face, direction::Int; periodic::Bool)
    n = size(center, direction)
    @inbounds for I in CartesianIndices(center)
        i = I[direction]
        lo = eno5_face_sample(face, I, direction, i, n, periodic)
        hi = eno5_face_sample(face, I, direction, i + 1, n, periodic)
        center[I] = 0.5 * (lo + hi)
    end
    return center
end

"""
Interpolate one velocity component, choosing ENO5 when the direction is large
enough for its stencil and the second-order average otherwise. The choice
keys off `global_cells`/`global_periodic` (defaulting to `size(center,
direction)`/`periodic` — today's behaviour), not the padded allocated extent:
under padding `size(center, direction)` is the allocated size, so a thin
*global* axis that serial interpolates linearly must not switch to ENO5
merely because padding made the allocated extent look big enough.
"""
function face_to_center_direction!(
        center, face, direction::Int; periodic::Bool,
        global_cells::Int = size(center, direction), global_periodic::Bool = periodic,
        restriction::Union{Nothing, ENO5PhysicalRestriction} = nothing,
    )
    return if global_cells >= eno5_minimum_cells(global_periodic)
        eno5_face_to_center_direction!(center, face, direction; periodic, restriction)
    else
        linear_face_to_center_direction!(center, face, direction; periodic)
    end
end

# Preparation entry point: applies the per-direction policy above. The strict
# `eno5_*` routines keep their minimum-size preconditions; the choice of when to
# use them belongs here, not inside the stencil code.
function eno5_face_to_center!(
        center::NamedTuple, face::NamedTuple; periodic::NamedTuple,
        global_sizes::Union{NamedTuple, Nothing} = nothing,
        global_periodic::Union{NamedTuple, Nothing} = nothing,
        restrictions::Union{NamedTuple, Nothing} = nothing,
    )
    validate_staggered_velocity!(center, face; periodic)
    names = keys(center)
    ntuple(Val(length(names))) do direction
        name = names[direction]
        gcells = global_sizes === nothing ? size(getproperty(center, name), direction) : getproperty(global_sizes, name)
        gperiodic = global_periodic === nothing ? getproperty(periodic, name) : getproperty(global_periodic, name)
        restriction = restrictions === nothing ? nothing : getproperty(restrictions, name)
        face_to_center_direction!(
            getproperty(center, name), getproperty(face, name), direction;
            periodic = getproperty(periodic, name), global_cells = gcells,
            global_periodic = gperiodic, restriction,
        )
    end
    return center
end

"""
Build the resolved-BC-aware restriction for one axis, or `nothing` when the
axis is unpadded (every existing serial/KA/Chmy scheme) or when neither face
is `ExtrapolateBC`/`PrescribedInflowBC` (a fully periodic or process-seam
axis gets the full stencil, so no restriction object is needed at all).
"""
function _eno5_restriction(extent::PaddedExtent{N}, boundary, d) where {N}
    pad = extent.pad[d]
    pad == 0 && return nothing
    faces = boundary_faces(boundary)
    lo_bc, hi_bc = faces[2d - 1], faces[2d]
    restrict_lo = lo_bc isa ExtrapolateBC || lo_bc isa PrescribedInflowBC
    restrict_hi = hi_bc isa ExtrapolateBC || hi_bc isa PrescribedInflowBC
    (restrict_lo || restrict_hi) || return nothing
    owned_lo = pad + 1
    owned_hi = pad + extent.owned[d]
    return ENO5PhysicalRestriction(owned_lo, owned_hi, owned_lo, owned_hi + 1, restrict_lo, restrict_hi)
end

"""Interpolate face velocity into `weno.vcenter` using global and boundary
stencil bounds. Callers supply distinct source and destination arrays."""
function _interpolate_velocity!(weno::WENOScheme, velocity)
    extent = weno.extent
    labels = keys(weno.vcenter)
    N = length(labels)
    if all(==(0), extent.pad)
        eno5_face_to_center!(weno.vcenter, velocity; periodic = weno.vperiodic)
    else
        global_sizes = NamedTuple{labels}(ntuple(d -> extent.global_size[d], N))
        global_periodic = NamedTuple{labels}(ntuple(d -> extent.global_periodic[d], N))
        restrictions = NamedTuple{labels}(ntuple(d -> _eno5_restriction(extent, weno.boundary, d), N))
        eno5_face_to_center!(
            weno.vcenter, velocity; periodic = weno.vperiodic,
            global_sizes, global_periodic, restrictions,
        )
    end
    return weno.vcenter
end

"""
Prepare velocity once for all Runge–Kutta stages. Topology-backed schemes
exchange face velocity before interpolation and conservative centred velocity
afterward.
"""
function prepare_velocity!(weno::WENOScheme, velocity)
    return _prepare_velocity!(weno.topology, weno, velocity)
end

"""Interpolate staggered velocity without halo exchange."""
function _prepare_velocity!(::NoTopology, weno::WENOScheme, velocity)
    weno.stag || return velocity
    weno.vcenter === nothing && return velocity
    velocity === weno.vcenter && return velocity
    _interpolate_velocity!(weno, velocity)
    return weno.vcenter
end

"""
Exchange face velocity before interpolation when `stag = true`. Conservative
schemes then exchange centred velocity because reconstruction reads its ghosts;
with `stag = false`, these are the caller's arrays. Check the `vcenter` identity
first so repeated tuple-field calls do not repeat exchanges.
"""
function _prepare_velocity!(topo, weno::WENOScheme, velocity)
    weno.vcenter !== nothing && velocity === weno.vcenter && return velocity

    geometry = weno.extent.geometry

    if weno.stag
        for (d, name) in enumerate(keys(velocity))
            weno_exchange_halo!(getproperty(velocity, name), topo; geometry, stagger = d)
        end
        fill_physical_ghosts!(velocity, weno.extent, weno.boundary)
        _interpolate_velocity!(weno, velocity)
        voperator = weno.vcenter
    else
        voperator = velocity
    end

    if is_conservative(weno.form)
        for name in keys(voperator)
            weno_exchange_halo!(getproperty(voperator, name), topo; geometry, stagger = nothing)
        end
        fill_physical_ghosts!(voperator, weno.extent, weno.boundary)
    end

    return voperator
end

"""Interpolate `velocity` into `scheme.vcenter`, mirroring
`_interpolate_velocity!` exactly for `MultiphaseWENOScheme`."""
function _interpolate_velocity_multiphase!(scheme::MultiphaseWENOScheme, velocity)
    extent = scheme.extent
    labels = keys(scheme.vcenter)
    N = length(labels)
    if all(==(0), extent.pad)
        eno5_face_to_center!(scheme.vcenter, velocity; periodic = scheme.vperiodic)
    else
        global_sizes = NamedTuple{labels}(ntuple(d -> extent.global_size[d], N))
        global_periodic = NamedTuple{labels}(ntuple(d -> extent.global_periodic[d], N))
        restrictions = NamedTuple{labels}(ntuple(d -> _eno5_restriction(extent, scheme.boundary, d), N))
        eno5_face_to_center!(
            scheme.vcenter, velocity; periodic = scheme.vperiodic,
            global_sizes, global_periodic, restrictions,
        )
    end
    return scheme.vcenter
end

"""
Prepare face-staggered CPU velocity once for all multiphase RK stages.
Dispatches on `scheme.topology`, mirroring the scalar `prepare_velocity!`.
`MultiphaseWENOScheme` has no `form` field (material transport only), so
there is no row 2 here — only row 1 (`stag = true`, exchange + refill the
FACE velocity, then interpolate). With `stag = false` the guard order does
not matter (there is nothing conditional on the form to protect), so this
checks `scheme.stag` first, unlike the scalar topology method.
"""
function prepare_velocity!(scheme::MultiphaseWENOScheme, velocity)
    return _prepare_velocity_multiphase!(scheme.topology, scheme, velocity)
end

function _prepare_velocity_multiphase!(::NoTopology, scheme::MultiphaseWENOScheme, velocity)
    scheme.stag || return velocity
    scheme.vcenter === nothing && return velocity
    velocity === scheme.vcenter && return velocity
    _interpolate_velocity_multiphase!(scheme, velocity)
    return scheme.vcenter
end

function _prepare_velocity_multiphase!(topo, scheme::MultiphaseWENOScheme, velocity)
    scheme.stag || return velocity
    scheme.vcenter === nothing && return velocity
    velocity === scheme.vcenter && return velocity

    geometry = scheme.extent.geometry
    for (d, name) in enumerate(keys(velocity))
        weno_exchange_halo!(getproperty(velocity, name), topo; geometry, stagger = d)
    end
    fill_physical_ghosts!(velocity, scheme.extent, scheme.boundary)
    _interpolate_velocity_multiphase!(scheme, velocity)
    return scheme.vcenter
end
