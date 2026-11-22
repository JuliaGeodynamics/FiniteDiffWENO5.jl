# Conservative transport by local Lax-Friedrichs splitting (Mishra, Parés-Pulido &
# Pressel 2020): `f± = ½(v u ± α u)` with, per face, `α = max |v|` over the six
# cells its WENO stencil reads. Monotone, fifth order, and α = 0 where v = 0.

@inline lf_split_plus(u, v, α) = 0.5 * (v * u + α * u)
@inline lf_split_minus(u, v, α) = 0.5 * (v * u - α * u)

"""Max |v| over the six stencil cells of one face."""
@inline stencil_lf_speed(v1, v2, v3, v4, v5, v6) =
    max(abs(v1), abs(v2), abs(v3), abs(v4), abs(v5), abs(v6))

"""
    boundary_face_speed(velocity, face, first_cell, last_cell)

`stencil_lf_speed` of a boundary face. `velocity` is one line along the face's
axis; the stencil cells `face - 3 … face + 2` are clamped to the domain
`first_cell:last_cell`, so ghost cells are never read.
"""
@inline function boundary_face_speed(velocity, face, first_cell, last_cell)
    cell(offset) = clamp(face + offset, first_cell, last_cell)
    return stencil_lf_speed(
        velocity[cell(-3)], velocity[cell(-2)], velocity[cell(-1)],
        velocity[cell(0)], velocity[cell(1)], velocity[cell(2)],
    )
end

# `PrescribedInflowBC` faces get the first-order Lax-Friedrichs flux
# `f⁺(u_ext) + f⁻(u_int)` from the prescribed exterior state. Faces and edge cells
# sit at `extent.pad`/`extent.owned`, as in `boundaries.jl`.
function apply_conservative_inflow_1d!(fl, fr, u, v, boundary, extent::PaddedExtent{1})
    bL, bR = boundary[1], boundary[2]
    lo_face = extent.pad[1] + 1
    hi_face = extent.pad[1] + extent.owned[1] + 1
    lo_cell = extent.pad[1] + 1
    hi_cell = extent.pad[1] + extent.owned[1]
    if bL isa PrescribedInflowBC
        α = boundary_face_speed(v, lo_face, lo_cell, hi_cell)
        @inbounds begin
            fl.x[lo_face] = lf_split_plus(inflow_value(bL), v[lo_cell], α)
            fr.x[lo_face] = lf_split_minus(u[lo_cell], v[lo_cell], α)
        end
    end
    if bR isa PrescribedInflowBC
        α = boundary_face_speed(v, hi_face, lo_cell, hi_cell)
        @inbounds begin
            fl.x[hi_face] = lf_split_plus(u[hi_cell], v[hi_cell], α)
            fr.x[hi_face] = lf_split_minus(inflow_value(bR), v[hi_cell], α)
        end
    end
    return nothing
end

# Higher-dimensional counterparts. Tangential indices follow the same convention
# as `apply_axis_inflow!` in `boundaries.jl`: the remaining axes in their natural
# order, looped over the owned window and offset into the (owned-sized) inflow
# value array by the tangential pad.
function apply_conservative_inflow_2d!(fl, fr, u, vx, vy, boundary, extent::PaddedExtent{2})
    bLx, bRx, bLy, bRy = boundary
    px, py = extent.pad[1], extent.pad[2]
    ox, oy = extent.owned[1], extent.owned[2]
    if bLx isa PrescribedInflowBC
        face, cell = px + 1, px + 1
        @inbounds for j in (py + 1):(py + oy)
            α = boundary_face_speed(view(vx, :, j), face, px + 1, px + ox)
            fl.x[face, j] = lf_split_plus(inflow_value(bLx, j - py), vx[cell, j], α)
            fr.x[face, j] = lf_split_minus(u[cell, j], vx[cell, j], α)
        end
    end
    if bRx isa PrescribedInflowBC
        face, cell = px + ox + 1, px + ox
        @inbounds for j in (py + 1):(py + oy)
            α = boundary_face_speed(view(vx, :, j), face, px + 1, px + ox)
            fl.x[face, j] = lf_split_plus(u[cell, j], vx[cell, j], α)
            fr.x[face, j] = lf_split_minus(inflow_value(bRx, j - py), vx[cell, j], α)
        end
    end
    if bLy isa PrescribedInflowBC
        face, cell = py + 1, py + 1
        @inbounds for i in (px + 1):(px + ox)
            α = boundary_face_speed(view(vy, i, :), face, py + 1, py + oy)
            fl.y[i, face] = lf_split_plus(inflow_value(bLy, i - px), vy[i, cell], α)
            fr.y[i, face] = lf_split_minus(u[i, cell], vy[i, cell], α)
        end
    end
    if bRy isa PrescribedInflowBC
        face, cell = py + oy + 1, py + oy
        @inbounds for i in (px + 1):(px + ox)
            α = boundary_face_speed(view(vy, i, :), face, py + 1, py + oy)
            fl.y[i, face] = lf_split_plus(u[i, cell], vy[i, cell], α)
            fr.y[i, face] = lf_split_minus(inflow_value(bRy, i - px), vy[i, cell], α)
        end
    end
    return nothing
end

function apply_conservative_inflow_3d!(fl, fr, u, vx, vy, vz, boundary, extent::PaddedExtent{3})
    bLx, bRx, bLy, bRy, bLz, bRz = boundary
    px, py, pz = extent.pad[1], extent.pad[2], extent.pad[3]
    ox, oy, oz = extent.owned[1], extent.owned[2], extent.owned[3]
    if bLx isa PrescribedInflowBC
        face, cell = px + 1, px + 1
        @inbounds for k in (pz + 1):(pz + oz), j in (py + 1):(py + oy)
            α = boundary_face_speed(view(vx, :, j, k), face, px + 1, px + ox)
            fl.x[face, j, k] = lf_split_plus(inflow_value(bLx, j - py, k - pz), vx[cell, j, k], α)
            fr.x[face, j, k] = lf_split_minus(u[cell, j, k], vx[cell, j, k], α)
        end
    end
    if bRx isa PrescribedInflowBC
        face, cell = px + ox + 1, px + ox
        @inbounds for k in (pz + 1):(pz + oz), j in (py + 1):(py + oy)
            α = boundary_face_speed(view(vx, :, j, k), face, px + 1, px + ox)
            fl.x[face, j, k] = lf_split_plus(u[cell, j, k], vx[cell, j, k], α)
            fr.x[face, j, k] = lf_split_minus(inflow_value(bRx, j - py, k - pz), vx[cell, j, k], α)
        end
    end
    if bLy isa PrescribedInflowBC
        face, cell = py + 1, py + 1
        @inbounds for k in (pz + 1):(pz + oz), i in (px + 1):(px + ox)
            α = boundary_face_speed(view(vy, i, :, k), face, py + 1, py + oy)
            fl.y[i, face, k] = lf_split_plus(inflow_value(bLy, i - px, k - pz), vy[i, cell, k], α)
            fr.y[i, face, k] = lf_split_minus(u[i, cell, k], vy[i, cell, k], α)
        end
    end
    if bRy isa PrescribedInflowBC
        face, cell = py + oy + 1, py + oy
        @inbounds for k in (pz + 1):(pz + oz), i in (px + 1):(px + ox)
            α = boundary_face_speed(view(vy, i, :, k), face, py + 1, py + oy)
            fl.y[i, face, k] = lf_split_plus(u[i, cell, k], vy[i, cell, k], α)
            fr.y[i, face, k] = lf_split_minus(inflow_value(bRy, i - px, k - pz), vy[i, cell, k], α)
        end
    end
    if bLz isa PrescribedInflowBC
        face, cell = pz + 1, pz + 1
        @inbounds for j in (py + 1):(py + oy), i in (px + 1):(px + ox)
            α = boundary_face_speed(view(vz, i, j, :), face, pz + 1, pz + oz)
            fl.z[i, j, face] = lf_split_plus(inflow_value(bLz, i - px, j - py), vz[i, j, cell], α)
            fr.z[i, j, face] = lf_split_minus(u[i, j, cell], vz[i, j, cell], α)
        end
    end
    if bRz isa PrescribedInflowBC
        face, cell = pz + oz + 1, pz + oz
        @inbounds for j in (py + 1):(py + oy), i in (px + 1):(px + ox)
            α = boundary_face_speed(view(vz, i, j, :), face, pz + 1, pz + oz)
            fl.z[i, j, face] = lf_split_plus(u[i, j, cell], vz[i, j, cell], α)
            fr.z[i, j, face] = lf_split_minus(inflow_value(bRz, i - px, j - py), vz[i, j, cell], α)
        end
    end
    return nothing
end

"""
    conservative_face_velocity(weno, v, given)

Face velocity for the wall flux (`given`, else `v`), or `nothing` if there is none.
"""
function conservative_face_velocity(weno, v, given)
    (is_conservative(weno.form) && weno.stag) || return nothing
    f = given === nothing ? v : given
    return f === weno.vcenter ? nothing : f
end

"""Component `name` of `conservative_face_velocity`, or `nothing`."""
function wall_face_velocity(weno, v, given, name::Symbol)
    f = conservative_face_velocity(weno, v, given)
    return f === nothing ? nothing : getproperty(f, name)
end

# `ExtrapolateBC` wall: `v_face · u_boundary_cell`, zero when `v_face = 0`.
apply_axis_wall!(fl, fr, u, vface, ::Any, extent, d, upper) = nothing

function apply_axis_wall!(
        fl::AbstractArray{T, N}, fr, u, vface, ::ExtrapolateBC, extent::PaddedExtent{N}, d, upper,
    ) where {T, N}
    shift = upper ? CartesianIndex(ntuple(k -> k == d ? 1 : 0, N)) : zero(CartesianIndex{N})
    @inbounds for I in CartesianIndices(_inflow_ranges(extent, d, upper))
        fl[I] = vface[I] * u[I - shift]
        fr[I] = zero(T)
    end
    return nothing
end

apply_conservative_walls!(fl, fr, u, ::Nothing, boundary, extent) = nothing

function apply_conservative_walls!(fl::NamedTuple{(:x,)}, fr, u, vface::NamedTuple, boundary, extent)
    apply_axis_wall!(fl.x, fr.x, u, vface.x, boundary[1], extent, 1, false)
    apply_axis_wall!(fl.x, fr.x, u, vface.x, boundary[2], extent, 1, true)
    return nothing
end

function apply_conservative_walls!(fl::NamedTuple{(:x, :y)}, fr, u, vface::NamedTuple, boundary, extent)
    apply_axis_wall!(fl.x, fr.x, u, vface.x, boundary[1], extent, 1, false)
    apply_axis_wall!(fl.x, fr.x, u, vface.x, boundary[2], extent, 1, true)
    apply_axis_wall!(fl.y, fr.y, u, vface.y, boundary[3], extent, 2, false)
    apply_axis_wall!(fl.y, fr.y, u, vface.y, boundary[4], extent, 2, true)
    return nothing
end

function apply_conservative_walls!(fl::NamedTuple{(:x, :y, :z)}, fr, u, vface::NamedTuple, boundary, extent)
    apply_axis_wall!(fl.x, fr.x, u, vface.x, boundary[1], extent, 1, false)
    apply_axis_wall!(fl.x, fr.x, u, vface.x, boundary[2], extent, 1, true)
    apply_axis_wall!(fl.y, fr.y, u, vface.y, boundary[3], extent, 2, false)
    apply_axis_wall!(fl.y, fr.y, u, vface.y, boundary[4], extent, 2, true)
    apply_axis_wall!(fl.z, fr.z, u, vface.z, boundary[5], extent, 3, false)
    apply_axis_wall!(fl.z, fr.z, u, vface.z, boundary[6], extent, 3, true)
    return nothing
end

"""
    conservative_semi_discretisation_weno5!(du, u, vcell, weno, nx, Δx_; vface = nothing)

Evaluate `∂ₓ(v u)` in 1D. `vcell` is collocated with `u`; `vface` is the raw
face velocity for the wall flux, or `nothing`.
"""
function conservative_semi_discretisation_weno5!(
        du::AbstractVector, u::AbstractVector, vcell, weno::WENOScheme, nx, Δx_;
        vface = nothing,
    )
    (; fl, fr, boundary, χ, γ, ζ, ϵ, multithreading, extent) = weno
    v = vcell.x
    size(v) == size(u) || throw(
        DimensionMismatch(
            "conservative transport needs a cell-centred velocity of size $(size(u)), got $(size(v))",
        )
    )

    bL = boundary[1]
    bR = boundary[2]

    @inbounds @maybe_threads multithreading for i in axes(fl.x, 1)
        iwww = left_index(i, 3, nx, bL)
        iww = left_index(i, 2, nx, bL)
        iw = left_index(i, 1, nx, bL)
        ie = right_index(i, 0, nx, bR)
        iee = right_index(i, 1, nx, bR)
        ieee = right_index(i, 2, nx, bR)
        α = stencil_lf_speed(v[iwww], v[iww], v[iw], v[ie], v[iee], v[ieee])

        fl.x[i] = weno5_reconstruction_upwind(
            lf_split_plus(u[iwww], v[iwww], α), lf_split_plus(u[iww], v[iww], α),
            lf_split_plus(u[iw], v[iw], α), lf_split_plus(u[ie], v[ie], α),
            lf_split_plus(u[iee], v[iee], α), χ, γ, ζ, ϵ,
        )
        fr.x[i] = weno5_reconstruction_downwind(
            lf_split_minus(u[iww], v[iww], α), lf_split_minus(u[iw], v[iw], α),
            lf_split_minus(u[ie], v[ie], α), lf_split_minus(u[iee], v[iee], α),
            lf_split_minus(u[ieee], v[ieee], α), χ, γ, ζ, ϵ,
        )
    end

    apply_conservative_inflow_1d!(fl, fr, u, v, boundary, extent)
    apply_conservative_walls!(fl, fr, u, vface, boundary, extent)

    @inbounds @maybe_threads multithreading for i in eachindex(du)
        du[i] = ((fl.x[i + 1] + fr.x[i + 1]) - (fl.x[i] + fr.x[i])) * Δx_
    end
    return nothing
end

"""
    conservative_semi_discretisation_weno5!(du, u, vcell, weno, nx, ny, Δx_, Δy_)

Two-dimensional counterpart; each direction is split with its own velocity.
"""
function conservative_semi_discretisation_weno5!(
        du::AbstractMatrix, u::AbstractMatrix, vcell, weno::WENOScheme, nx, ny, Δx_, Δy_; vface = nothing,
    )
    (; fl, fr, boundary, χ, γ, ζ, ϵ, multithreading, extent) = weno
    vx, vy = vcell.x, vcell.y
    (size(vx) == size(u) && size(vy) == size(u)) || throw(
        DimensionMismatch(
            "conservative transport needs cell-centred velocities of size $(size(u))",
        )
    )

    bLx, bRx, bLy, bRy = boundary

    @inbounds @maybe_threads multithreading for I in CartesianIndices(fl.x)
        i, j = Tuple(I)
        iwww = left_index(i, 3, nx, bLx)
        iww = left_index(i, 2, nx, bLx)
        iw = left_index(i, 1, nx, bLx)
        ie = right_index(i, 0, nx, bRx)
        iee = right_index(i, 1, nx, bRx)
        ieee = right_index(i, 2, nx, bRx)
        α = stencil_lf_speed(vx[iwww, j], vx[iww, j], vx[iw, j], vx[ie, j], vx[iee, j], vx[ieee, j])

        fl.x[I] = weno5_reconstruction_upwind(
            lf_split_plus(u[iwww, j], vx[iwww, j], α), lf_split_plus(u[iww, j], vx[iww, j], α),
            lf_split_plus(u[iw, j], vx[iw, j], α), lf_split_plus(u[ie, j], vx[ie, j], α),
            lf_split_plus(u[iee, j], vx[iee, j], α), χ, γ, ζ, ϵ,
        )
        fr.x[I] = weno5_reconstruction_downwind(
            lf_split_minus(u[iww, j], vx[iww, j], α), lf_split_minus(u[iw, j], vx[iw, j], α),
            lf_split_minus(u[ie, j], vx[ie, j], α), lf_split_minus(u[iee, j], vx[iee, j], α),
            lf_split_minus(u[ieee, j], vx[ieee, j], α), χ, γ, ζ, ϵ,
        )
    end

    @inbounds @maybe_threads multithreading for I in CartesianIndices(fl.y)
        i, j = Tuple(I)
        jwww = left_index(j, 3, ny, bLy)
        jww = left_index(j, 2, ny, bLy)
        jw = left_index(j, 1, ny, bLy)
        je = right_index(j, 0, ny, bRy)
        jee = right_index(j, 1, ny, bRy)
        jeee = right_index(j, 2, ny, bRy)
        α = stencil_lf_speed(vy[i, jwww], vy[i, jww], vy[i, jw], vy[i, je], vy[i, jee], vy[i, jeee])

        fl.y[I] = weno5_reconstruction_upwind(
            lf_split_plus(u[i, jwww], vy[i, jwww], α), lf_split_plus(u[i, jww], vy[i, jww], α),
            lf_split_plus(u[i, jw], vy[i, jw], α), lf_split_plus(u[i, je], vy[i, je], α),
            lf_split_plus(u[i, jee], vy[i, jee], α), χ, γ, ζ, ϵ,
        )
        fr.y[I] = weno5_reconstruction_downwind(
            lf_split_minus(u[i, jww], vy[i, jww], α), lf_split_minus(u[i, jw], vy[i, jw], α),
            lf_split_minus(u[i, je], vy[i, je], α), lf_split_minus(u[i, jee], vy[i, jee], α),
            lf_split_minus(u[i, jeee], vy[i, jeee], α), χ, γ, ζ, ϵ,
        )
    end

    apply_conservative_inflow_2d!(fl, fr, u, vx, vy, boundary, extent)
    apply_conservative_walls!(fl, fr, u, vface, boundary, extent)

    @inbounds @maybe_threads multithreading for I in CartesianIndices(du)
        i, j = Tuple(I)
        du[I] = @muladd ((fl.x[i + 1, j] + fr.x[i + 1, j]) - (fl.x[I] + fr.x[I])) * Δx_ +
            ((fl.y[i, j + 1] + fr.y[i, j + 1]) - (fl.y[I] + fr.y[I])) * Δy_
    end
    return nothing
end

"""
    conservative_semi_discretisation_weno5!(du, u, vcell, weno, nx, ny, nz, Δx_, Δy_, Δz_)

Three-dimensional counterpart of the split-flux operator.
"""
function conservative_semi_discretisation_weno5!(
        du::AbstractArray{<:Real, 3}, u::AbstractArray{<:Real, 3}, vcell, weno::WENOScheme,
        nx, ny, nz, Δx_, Δy_, Δz_; vface = nothing,
    )
    (; fl, fr, boundary, χ, γ, ζ, ϵ, multithreading, extent) = weno
    vx, vy, vz = vcell.x, vcell.y, vcell.z
    all(w -> size(w) == size(u), (vx, vy, vz)) || throw(
        DimensionMismatch(
            "conservative transport needs cell-centred velocities of size $(size(u))",
        )
    )

    bLx, bRx, bLy, bRy, bLz, bRz = boundary

    @inbounds @maybe_threads multithreading for I in CartesianIndices(fl.x)
        i, j, k = Tuple(I)
        iwww = left_index(i, 3, nx, bLx)
        iww = left_index(i, 2, nx, bLx)
        iw = left_index(i, 1, nx, bLx)
        ie = right_index(i, 0, nx, bRx)
        iee = right_index(i, 1, nx, bRx)
        ieee = right_index(i, 2, nx, bRx)
        α = stencil_lf_speed(vx[iwww, j, k], vx[iww, j, k], vx[iw, j, k], vx[ie, j, k], vx[iee, j, k], vx[ieee, j, k])

        fl.x[I] = weno5_reconstruction_upwind(
            lf_split_plus(u[iwww, j, k], vx[iwww, j, k], α),
            lf_split_plus(u[iww, j, k], vx[iww, j, k], α),
            lf_split_plus(u[iw, j, k], vx[iw, j, k], α),
            lf_split_plus(u[ie, j, k], vx[ie, j, k], α),
            lf_split_plus(u[iee, j, k], vx[iee, j, k], α), χ, γ, ζ, ϵ,
        )
        fr.x[I] = weno5_reconstruction_downwind(
            lf_split_minus(u[iww, j, k], vx[iww, j, k], α),
            lf_split_minus(u[iw, j, k], vx[iw, j, k], α),
            lf_split_minus(u[ie, j, k], vx[ie, j, k], α),
            lf_split_minus(u[iee, j, k], vx[iee, j, k], α),
            lf_split_minus(u[ieee, j, k], vx[ieee, j, k], α), χ, γ, ζ, ϵ,
        )
    end

    @inbounds @maybe_threads multithreading for I in CartesianIndices(fl.y)
        i, j, k = Tuple(I)
        jwww = left_index(j, 3, ny, bLy)
        jww = left_index(j, 2, ny, bLy)
        jw = left_index(j, 1, ny, bLy)
        je = right_index(j, 0, ny, bRy)
        jee = right_index(j, 1, ny, bRy)
        jeee = right_index(j, 2, ny, bRy)
        α = stencil_lf_speed(vy[i, jwww, k], vy[i, jww, k], vy[i, jw, k], vy[i, je, k], vy[i, jee, k], vy[i, jeee, k])

        fl.y[I] = weno5_reconstruction_upwind(
            lf_split_plus(u[i, jwww, k], vy[i, jwww, k], α),
            lf_split_plus(u[i, jww, k], vy[i, jww, k], α),
            lf_split_plus(u[i, jw, k], vy[i, jw, k], α),
            lf_split_plus(u[i, je, k], vy[i, je, k], α),
            lf_split_plus(u[i, jee, k], vy[i, jee, k], α), χ, γ, ζ, ϵ,
        )
        fr.y[I] = weno5_reconstruction_downwind(
            lf_split_minus(u[i, jww, k], vy[i, jww, k], α),
            lf_split_minus(u[i, jw, k], vy[i, jw, k], α),
            lf_split_minus(u[i, je, k], vy[i, je, k], α),
            lf_split_minus(u[i, jee, k], vy[i, jee, k], α),
            lf_split_minus(u[i, jeee, k], vy[i, jeee, k], α), χ, γ, ζ, ϵ,
        )
    end

    @inbounds @maybe_threads multithreading for I in CartesianIndices(fl.z)
        i, j, k = Tuple(I)
        kwww = left_index(k, 3, nz, bLz)
        kww = left_index(k, 2, nz, bLz)
        kw = left_index(k, 1, nz, bLz)
        ke = right_index(k, 0, nz, bRz)
        kee = right_index(k, 1, nz, bRz)
        keee = right_index(k, 2, nz, bRz)
        α = stencil_lf_speed(vz[i, j, kwww], vz[i, j, kww], vz[i, j, kw], vz[i, j, ke], vz[i, j, kee], vz[i, j, keee])

        fl.z[I] = weno5_reconstruction_upwind(
            lf_split_plus(u[i, j, kwww], vz[i, j, kwww], α),
            lf_split_plus(u[i, j, kww], vz[i, j, kww], α),
            lf_split_plus(u[i, j, kw], vz[i, j, kw], α),
            lf_split_plus(u[i, j, ke], vz[i, j, ke], α),
            lf_split_plus(u[i, j, kee], vz[i, j, kee], α), χ, γ, ζ, ϵ,
        )
        fr.z[I] = weno5_reconstruction_downwind(
            lf_split_minus(u[i, j, kww], vz[i, j, kww], α),
            lf_split_minus(u[i, j, kw], vz[i, j, kw], α),
            lf_split_minus(u[i, j, ke], vz[i, j, ke], α),
            lf_split_minus(u[i, j, kee], vz[i, j, kee], α),
            lf_split_minus(u[i, j, keee], vz[i, j, keee], α), χ, γ, ζ, ϵ,
        )
    end

    apply_conservative_inflow_3d!(fl, fr, u, vx, vy, vz, boundary, extent)
    apply_conservative_walls!(fl, fr, u, vface, boundary, extent)

    @inbounds @maybe_threads multithreading for I in CartesianIndices(du)
        i, j, k = Tuple(I)
        du[I] = @muladd ((fl.x[i + 1, j, k] + fr.x[i + 1, j, k]) - (fl.x[I] + fr.x[I])) * Δx_ +
            ((fl.y[i, j + 1, k] + fr.y[i, j + 1, k]) - (fl.y[I] + fr.y[I])) * Δy_ +
            ((fl.z[i, j, k + 1] + fr.z[i, j, k + 1]) - (fl.z[I] + fr.z[I])) * Δz_
    end
    return nothing
end
