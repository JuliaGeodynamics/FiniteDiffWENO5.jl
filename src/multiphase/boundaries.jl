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
# These deliberately do NOT reuse `apply_inflow_boundaries!` and its `apply_*_inflow!`
# family. Those dispatch on `flux::AbstractVector`/`AbstractMatrix` and fall back to
# `apply_lower_inflow!(flux, ::Any) = nothing`, so an `NTuple` of phase arrays would match
# only the fallback and the prescribed composition would be discarded with no error.
#
# Here the no-op methods dispatch on the *boundary* being a non-inflow condition, so a
# `PrescribedInflowBC` paired with a wrong-shaped buffer raises a `MethodError` instead of
# silently doing nothing.

const _NoInflowBC = Union{PeriodicBC, ExtrapolateBC}

apply_multiphase_lower_inflow!(flux, ::_NoInflowBC) = nothing
apply_multiphase_upper_inflow!(flux, ::_NoInflowBC) = nothing
apply_multiphase_x_lower_inflow!(flux, ::_NoInflowBC) = nothing
apply_multiphase_x_upper_inflow!(flux, ::_NoInflowBC) = nothing
apply_multiphase_y_lower_inflow!(flux, ::_NoInflowBC) = nothing
apply_multiphase_y_upper_inflow!(flux, ::_NoInflowBC) = nothing
apply_multiphase_z_lower_inflow!(flux, ::_NoInflowBC) = nothing
apply_multiphase_z_upper_inflow!(flux, ::_NoInflowBC) = nothing

function apply_multiphase_lower_inflow!(
        flux::Tuple{A, Vararg{A, M}}, bc::PrescribedInflowBC,
    ) where {M, A <: AbstractVector}
    for k in 1:(M + 1)
        @inbounds flux[k][begin] = multiphase_inflow_value(bc, k)
    end
    return nothing
end

function apply_multiphase_upper_inflow!(
        flux::Tuple{A, Vararg{A, M}}, bc::PrescribedInflowBC,
    ) where {M, A <: AbstractVector}
    for k in 1:(M + 1)
        @inbounds flux[k][end] = multiphase_inflow_value(bc, k)
    end
    return nothing
end

function apply_multiphase_x_lower_inflow!(
        flux::Tuple{A, Vararg{A, M}}, bc::PrescribedInflowBC,
    ) where {M, A <: AbstractMatrix}
    for k in 1:(M + 1)
        f = @inbounds flux[k]
        @inbounds for j in axes(f, 2)
            f[begin, j] = multiphase_inflow_value(bc, k, j)
        end
    end
    return nothing
end

function apply_multiphase_x_upper_inflow!(
        flux::Tuple{A, Vararg{A, M}}, bc::PrescribedInflowBC,
    ) where {M, A <: AbstractMatrix}
    for k in 1:(M + 1)
        f = @inbounds flux[k]
        @inbounds for j in axes(f, 2)
            f[end, j] = multiphase_inflow_value(bc, k, j)
        end
    end
    return nothing
end

function apply_multiphase_y_lower_inflow!(
        flux::Tuple{A, Vararg{A, M}}, bc::PrescribedInflowBC,
    ) where {M, A <: AbstractMatrix}
    for k in 1:(M + 1)
        f = @inbounds flux[k]
        @inbounds for i in axes(f, 1)
            f[i, begin] = multiphase_inflow_value(bc, k, i)
        end
    end
    return nothing
end

function apply_multiphase_y_upper_inflow!(
        flux::Tuple{A, Vararg{A, M}}, bc::PrescribedInflowBC,
    ) where {M, A <: AbstractMatrix}
    for k in 1:(M + 1)
        f = @inbounds flux[k]
        @inbounds for i in axes(f, 1)
            f[i, end] = multiphase_inflow_value(bc, k, i)
        end
    end
    return nothing
end

function apply_multiphase_x_lower_inflow!(
        flux::Tuple{A, Vararg{A, M}}, bc::PrescribedInflowBC,
    ) where {M, A <: AbstractArray{<:Any, 3}}
    for k in 1:(M + 1)
        f = @inbounds flux[k]
        @inbounds for m in axes(f, 3), j in axes(f, 2)
            f[begin, j, m] = multiphase_inflow_value(bc, k, j, m)
        end
    end
    return nothing
end

function apply_multiphase_x_upper_inflow!(
        flux::Tuple{A, Vararg{A, M}}, bc::PrescribedInflowBC,
    ) where {M, A <: AbstractArray{<:Any, 3}}
    for k in 1:(M + 1)
        f = @inbounds flux[k]
        @inbounds for m in axes(f, 3), j in axes(f, 2)
            f[end, j, m] = multiphase_inflow_value(bc, k, j, m)
        end
    end
    return nothing
end

function apply_multiphase_y_lower_inflow!(
        flux::Tuple{A, Vararg{A, M}}, bc::PrescribedInflowBC,
    ) where {M, A <: AbstractArray{<:Any, 3}}
    for k in 1:(M + 1)
        f = @inbounds flux[k]
        @inbounds for m in axes(f, 3), i in axes(f, 1)
            f[i, begin, m] = multiphase_inflow_value(bc, k, i, m)
        end
    end
    return nothing
end

function apply_multiphase_y_upper_inflow!(
        flux::Tuple{A, Vararg{A, M}}, bc::PrescribedInflowBC,
    ) where {M, A <: AbstractArray{<:Any, 3}}
    for k in 1:(M + 1)
        f = @inbounds flux[k]
        @inbounds for m in axes(f, 3), i in axes(f, 1)
            f[i, end, m] = multiphase_inflow_value(bc, k, i, m)
        end
    end
    return nothing
end

function apply_multiphase_z_lower_inflow!(
        flux::Tuple{A, Vararg{A, M}}, bc::PrescribedInflowBC,
    ) where {M, A <: AbstractArray{<:Any, 3}}
    for k in 1:(M + 1)
        f = @inbounds flux[k]
        @inbounds for j in axes(f, 2), i in axes(f, 1)
            f[i, j, begin] = multiphase_inflow_value(bc, k, i, j)
        end
    end
    return nothing
end

function apply_multiphase_z_upper_inflow!(
        flux::Tuple{A, Vararg{A, M}}, bc::PrescribedInflowBC,
    ) where {M, A <: AbstractArray{<:Any, 3}}
    for k in 1:(M + 1)
        f = @inbounds flux[k]
        @inbounds for j in axes(f, 2), i in axes(f, 1)
            f[i, j, end] = multiphase_inflow_value(bc, k, i, j)
        end
    end
    return nothing
end

function apply_multiphase_inflow_boundaries!(fl::NamedTuple{(:x,)}, fr, boundary)
    apply_multiphase_lower_inflow!(fl.x, boundary[1])
    apply_multiphase_upper_inflow!(fr.x, boundary[2])
    return nothing
end

function apply_multiphase_inflow_boundaries!(fl::NamedTuple{(:x, :y)}, fr, boundary)
    apply_multiphase_x_lower_inflow!(fl.x, boundary[1])
    apply_multiphase_x_upper_inflow!(fr.x, boundary[2])
    apply_multiphase_y_lower_inflow!(fl.y, boundary[3])
    apply_multiphase_y_upper_inflow!(fr.y, boundary[4])
    return nothing
end

function apply_multiphase_inflow_boundaries!(fl::NamedTuple{(:x, :y, :z)}, fr, boundary)
    apply_multiphase_x_lower_inflow!(fl.x, boundary[1])
    apply_multiphase_x_upper_inflow!(fr.x, boundary[2])
    apply_multiphase_y_lower_inflow!(fl.y, boundary[3])
    apply_multiphase_y_upper_inflow!(fr.y, boundary[4])
    apply_multiphase_z_lower_inflow!(fl.z, boundary[5])
    apply_multiphase_z_upper_inflow!(fr.z, boundary[6])
    return nothing
end
