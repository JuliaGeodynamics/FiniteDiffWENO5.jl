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
valid_boundary(boundary::Integer) = boundary in (0, 1, 2)
valid_boundary(boundary) = false

normalize_boundary(boundary::AbstractAdvectionBoundary) = boundary
normalize_boundary(boundary::Integer) = boundary == 2 ? PeriodicBC() : ExtrapolateBC()

function validate_inflow_value(boundary::PrescribedInflowBC, expected_size, face)
    value = boundary.value
    if value isa Real
        isfinite(value) || throw(ArgumentError(
            "PrescribedInflowBC on face $face must be finite, got $value"))
    elseif value isa AbstractArray{<:Real}
        size(value) == expected_size || throw(DimensionMismatch(
            "PrescribedInflowBC on face $face requires a value array of size " *
            "$expected_size, got $(size(value))"))
        all(isfinite, value) || throw(ArgumentError(
            "PrescribedInflowBC on face $face contains a nonfinite value"))
    else
        throw(ArgumentError(
            "PrescribedInflowBC on face $face requires a real scalar or array, " *
            "got $(typeof(value))"))
    end
    return nothing
end

validate_inflow_value(::Any, expected_size, face) = nothing

function tangential_size(sizes::NTuple{N,Int}, dimension) where {N}
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
    length(faces) == 2N || throw(ArgumentError(
        "boundary must contain $(2N) face conditions for $(N)D data, got " *
        "$(length(faces))"))
    all(valid_boundary, faces) || throw(ArgumentError(
        "boundary entries must be PeriodicBC(), ExtrapolateBC(), " *
        "PrescribedInflowBC(value), or a legacy integer code 0, 1, or 2"))
    return map(normalize_boundary, faces)
end

function validate_boundary(boundary, N, sizes = nothing)
    faces = normalize_boundary_faces(boundary, N)

    if sizes !== nothing
        for face in eachindex(faces)
            dimension = (face + 1) ÷ 2
            validate_inflow_value(
                faces[face], tangential_size(sizes, dimension), face)
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

apply_lower_inflow!(flux, ::Any) = nothing
apply_upper_inflow!(flux, ::Any) = nothing

function apply_lower_inflow!(flux::AbstractVector, boundary::PrescribedInflowBC)
    flux[begin] = inflow_value(boundary)
    return nothing
end

function apply_upper_inflow!(flux::AbstractVector, boundary::PrescribedInflowBC)
    flux[end] = inflow_value(boundary)
    return nothing
end

function apply_x_lower_inflow!(flux, boundary::PrescribedInflowBC)
    @inbounds for k in axes(flux, 3), j in axes(flux, 2)
        flux[begin, j, k] = inflow_value(boundary, j, k)
    end
    return nothing
end
apply_x_lower_inflow!(flux, ::Any) = nothing

function apply_x_lower_inflow!(flux::AbstractMatrix, boundary::PrescribedInflowBC)
    @inbounds for j in axes(flux, 2)
        flux[begin, j] = inflow_value(boundary, j)
    end
    return nothing
end

function apply_x_upper_inflow!(flux, boundary::PrescribedInflowBC)
    @inbounds for k in axes(flux, 3), j in axes(flux, 2)
        flux[end, j, k] = inflow_value(boundary, j, k)
    end
    return nothing
end
apply_x_upper_inflow!(flux, ::Any) = nothing

function apply_x_upper_inflow!(flux::AbstractMatrix, boundary::PrescribedInflowBC)
    @inbounds for j in axes(flux, 2)
        flux[end, j] = inflow_value(boundary, j)
    end
    return nothing
end

function apply_y_lower_inflow!(flux, boundary::PrescribedInflowBC)
    @inbounds for k in axes(flux, 3), i in axes(flux, 1)
        flux[i, begin, k] = inflow_value(boundary, i, k)
    end
    return nothing
end
apply_y_lower_inflow!(flux, ::Any) = nothing

function apply_y_lower_inflow!(flux::AbstractMatrix, boundary::PrescribedInflowBC)
    @inbounds for i in axes(flux, 1)
        flux[i, begin] = inflow_value(boundary, i)
    end
    return nothing
end

function apply_y_upper_inflow!(flux, boundary::PrescribedInflowBC)
    @inbounds for k in axes(flux, 3), i in axes(flux, 1)
        flux[i, end, k] = inflow_value(boundary, i, k)
    end
    return nothing
end
apply_y_upper_inflow!(flux, ::Any) = nothing

function apply_y_upper_inflow!(flux::AbstractMatrix, boundary::PrescribedInflowBC)
    @inbounds for i in axes(flux, 1)
        flux[i, end] = inflow_value(boundary, i)
    end
    return nothing
end

function apply_z_lower_inflow!(flux, boundary::PrescribedInflowBC)
    @inbounds for j in axes(flux, 2), i in axes(flux, 1)
        flux[i, j, begin] = inflow_value(boundary, i, j)
    end
    return nothing
end
apply_z_lower_inflow!(flux, ::Any) = nothing

function apply_z_upper_inflow!(flux, boundary::PrescribedInflowBC)
    @inbounds for j in axes(flux, 2), i in axes(flux, 1)
        flux[i, j, end] = inflow_value(boundary, i, j)
    end
    return nothing
end
apply_z_upper_inflow!(flux, ::Any) = nothing

function apply_inflow_boundaries!(fl::NamedTuple{(:x,)}, fr, boundary)
    apply_lower_inflow!(fl.x, boundary[1])
    apply_upper_inflow!(fr.x, boundary[2])
    return nothing
end

function apply_inflow_boundaries!(fl::NamedTuple{(:x, :y)}, fr, boundary)
    apply_x_lower_inflow!(fl.x, boundary[1])
    apply_x_upper_inflow!(fr.x, boundary[2])
    apply_y_lower_inflow!(fl.y, boundary[3])
    apply_y_upper_inflow!(fr.y, boundary[4])
    return nothing
end

function apply_inflow_boundaries!(fl::NamedTuple{(:x, :y, :z)}, fr, boundary)
    apply_x_lower_inflow!(fl.x, boundary[1])
    apply_x_upper_inflow!(fr.x, boundary[2])
    apply_y_lower_inflow!(fl.y, boundary[3])
    apply_y_upper_inflow!(fr.y, boundary[4])
    apply_z_lower_inflow!(fl.z, boundary[5])
    apply_z_upper_inflow!(fr.z, boundary[6])
    return nothing
end
