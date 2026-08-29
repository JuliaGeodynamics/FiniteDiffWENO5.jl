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

# The 1D CPU and KA kernels use the same direction-generic recurrence. These
# narrow forwarding methods keep their compact call sites without maintaining a
# second stencil-selection implementation.
@inline eno5_face_sample(face::AbstractVector, q, n, periodic) =
    eno5_face_sample(face, CartesianIndex(1), 1, q, n, periodic)
@inline eno5_undivided_difference(face::AbstractVector, s, p, n, periodic) =
    eno5_undivided_difference(face, CartesianIndex(1), 1, s, p, n, periodic)
@inline eno5_stencil_start(face::AbstractVector, i, n, periodic) =
    eno5_stencil_start(face, CartesianIndex(i), 1, i, n, periodic)

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
        direction::Int; periodic::Bool
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
        s = eno5_stencil_start(face, I, direction, i, n, periodic)
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
enough for its stencil and the second-order average otherwise.
"""
function face_to_center_direction!(center, face, direction::Int; periodic::Bool)
    return if size(center, direction) >= eno5_minimum_cells(periodic)
        eno5_face_to_center_direction!(center, face, direction; periodic)
    else
        linear_face_to_center_direction!(center, face, direction; periodic)
    end
end

# Preparation entry point: applies the per-direction policy above. The strict
# `eno5_*` routines keep their minimum-size preconditions; the choice of when to
# use them belongs here, not inside the stencil code.
function eno5_face_to_center!(center::NamedTuple, face::NamedTuple; periodic::NamedTuple)
    validate_staggered_velocity!(center, face; periodic)
    names = keys(center)
    ntuple(Val(length(names))) do direction
        name = names[direction]
        face_to_center_direction!(
            getproperty(center, name), getproperty(face, name), direction;
            periodic = getproperty(periodic, name)
        )
    end
    return center
end

"""Prepare face-staggered CPU velocity once for all Runge--Kutta stages."""
function prepare_velocity!(weno::WENOScheme, velocity)
    weno.stag || return velocity
    weno.vcenter === nothing && return velocity
    velocity === weno.vcenter && return velocity
    eno5_face_to_center!(weno.vcenter, velocity; periodic = weno.vperiodic)
    return weno.vcenter
end

"""Prepare face-staggered CPU velocity once for all multiphase RK stages."""
function prepare_velocity!(scheme::MultiphaseWENOScheme, velocity)
    scheme.stag || return velocity
    scheme.vcenter === nothing && return velocity
    velocity === scheme.vcenter && return velocity
    eno5_face_to_center!(scheme.vcenter, velocity; periodic = scheme.vperiodic)
    return scheme.vcenter
end
