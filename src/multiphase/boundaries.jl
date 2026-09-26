# Multiphase boundary compositions.
#
# A prescribed inflow for a phase vector is a tuple of components, one per phase, each a
# scalar or a tangential array. The scalar `validate_inflow_value` deliberately rejects
# tuples, which is what stops a `WENOScheme` from silently accepting a phase vector; the
# validation here is a separate route rather than a widening of that one.

"""
    validate_multiphase_inflow(bc, expected_size, face, NP, T)

Validate one prescribed inflow composition. Every component must be finite and inside
`[0,1]`, and the composition must sum to one at every tangential point.
"""
function validate_multiphase_inflow(bc::PrescribedInflowBC, expected_size, face, NP, ::Type{T}) where {T}
    value = bc.value
    value isa Tuple || throw(
        ArgumentError(
            "PrescribedInflowBC on face $face of a multiphase scheme requires a tuple of " *
                "$NP components, got $(typeof(value))"
        )
    )
    length(value) == NP || throw(
        ArgumentError(
            "PrescribedInflowBC on face $face requires one component per phase " *
                "($NP), got $(length(value))"
        )
    )

    for k in eachindex(value)
        c = value[k]
        if c isa Real
            isfinite(c) || throw(
                ArgumentError(
                    "PrescribedInflowBC on face $face, phase $k must be finite, got $c"
                )
            )
            zero(T) <= c <= one(T) || throw(
                ArgumentError(
                    "PrescribedInflowBC on face $face, phase $k must lie in [0,1], got $c"
                )
            )
        elseif c isa AbstractArray{<:Real}
            size(c) == expected_size || throw(
                DimensionMismatch(
                    "PrescribedInflowBC on face $face, phase $k requires a value array of " *
                        "size $expected_size, got $(size(c))"
                )
            )
            all(isfinite, c) || throw(
                ArgumentError(
                    "PrescribedInflowBC on face $face, phase $k contains a nonfinite value"
                )
            )
            all(x -> zero(T) <= x <= one(T), c) || throw(
                ArgumentError(
                    "PrescribedInflowBC on face $face, phase $k contains a value outside [0,1]"
                )
            )
        else
            throw(
                ArgumentError(
                    "PrescribedInflowBC on face $face, phase $k requires a real scalar or " *
                        "array, got $(typeof(c))"
                )
            )
        end
    end

    tol = 64 * eps(T)
    if all(c -> c isa Real, value)
        abs(sum(value) - one(T)) <= tol || throw(
            ArgumentError(
                "PrescribedInflowBC on face $face must sum to one across phases within " *
                    "$tol, got $(sum(value))"
            )
        )
    else
        total = zeros(T, expected_size)
        for c in value
            total .+= c
        end
        err = isempty(total) ? zero(T) : maximum(abs, total .- one(T))
        err <= tol || throw(
            ArgumentError(
                "PrescribedInflowBC on face $face must sum to one at every tangential " *
                    "point within $tol, largest deviation is $err"
            )
        )
    end
    return nothing
end

validate_multiphase_inflow(::Any, expected_size, face, NP, ::Type{T}) where {T} = nothing

"""
    validate_multiphase_boundary(boundary, N, sizes, NP, T)

Normalize face conditions and validate any prescribed inflow compositions.

Uses [`normalize_boundary_faces`](@ref) rather than `validate_boundary`, because the
latter also runs the scalar inflow validator, which rejects the tuple values that a
multiphase inflow is made of.
"""
function validate_multiphase_boundary(boundary, N, sizes, NP, ::Type{T}) where {T}
    faces = normalize_boundary_faces(boundary, N)
    for face in eachindex(faces)
        dimension = (face + 1) ÷ 2
        validate_multiphase_inflow(
            faces[face], tangential_size(sizes, dimension), face, NP, T
        )
    end
    return faces
end

@inline inflow_component(value::Real, indices...) = value
@inline inflow_component(value::AbstractArray, indices...) = @inbounds value[indices...]

"""
    multiphase_inflow_value(bc, k, indices...)

Component `k` of a prescribed inflow composition at the given tangential indices. Scalar
components ignore the indices; array components are indexed by them. Construction-time
validation guarantees no other component kind reaches this function.
"""
@inline multiphase_inflow_value(bc::PrescribedInflowBC, k, indices...) =
    inflow_component(bc.value[k], indices...)

# --- installation into the face buffers -------------------------------------------
#
# This deliberately does NOT reuse `apply_inflow_boundaries!`/`apply_axis_inflow!`.
# Those fall back to `apply_axis_inflow!(flux, ::Any, extent, d, upper) = nothing` for
# any unhandled `flux` type, so an `NTuple` of phase arrays would match only the
# fallback and the prescribed composition would be discarded with no error.
#
# Here the no-op method dispatches on the *boundary* being a non-inflow condition, so a
# `PrescribedInflowBC` paired with a wrong-shaped buffer raises a `MethodError` instead of
# silently doing nothing.

const _NoInflowBC = Union{PeriodicBC, ExtrapolateBC, ProcessBC}

apply_multiphase_axis_inflow!(flux, ::_NoInflowBC, extent, d, upper) = nothing

function apply_multiphase_axis_inflow!(
        flux::Tuple{A, Vararg{A, M}}, bc::PrescribedInflowBC, extent::PaddedExtent{N}, d, upper,
    ) where {M, N, A <: AbstractArray{<:Any, N}}
    ranges = _inflow_ranges(extent, d, upper)
    for k in 1:(M + 1)
        f = @inbounds flux[k]
        @inbounds for I in CartesianIndices(ranges)
            f[I] = multiphase_inflow_value(bc, k, _tangential_offset(I, extent, d)...)
        end
    end
    return nothing
end

# Three per-arity methods, not one loop — same reasoning as
# `apply_inflow_boundaries!` in `boundaries.jl` (a runtime index into the
# heterogeneous `boundary` tuple allocates).
function apply_multiphase_inflow_boundaries!(fl::NamedTuple{(:x,)}, fr, boundary, extent)
    apply_multiphase_axis_inflow!(fl.x, boundary[1], extent, 1, false)
    apply_multiphase_axis_inflow!(fr.x, boundary[2], extent, 1, true)
    return nothing
end

function apply_multiphase_inflow_boundaries!(fl::NamedTuple{(:x, :y)}, fr, boundary, extent)
    apply_multiphase_axis_inflow!(fl.x, boundary[1], extent, 1, false)
    apply_multiphase_axis_inflow!(fr.x, boundary[2], extent, 1, true)
    apply_multiphase_axis_inflow!(fl.y, boundary[3], extent, 2, false)
    apply_multiphase_axis_inflow!(fr.y, boundary[4], extent, 2, true)
    return nothing
end

function apply_multiphase_inflow_boundaries!(fl::NamedTuple{(:x, :y, :z)}, fr, boundary, extent)
    apply_multiphase_axis_inflow!(fl.x, boundary[1], extent, 1, false)
    apply_multiphase_axis_inflow!(fr.x, boundary[2], extent, 1, true)
    apply_multiphase_axis_inflow!(fl.y, boundary[3], extent, 2, false)
    apply_multiphase_axis_inflow!(fr.y, boundary[4], extent, 2, true)
    apply_multiphase_axis_inflow!(fl.z, boundary[5], extent, 3, false)
    apply_multiphase_axis_inflow!(fr.z, boundary[6], extent, 3, true)
    return nothing
end
