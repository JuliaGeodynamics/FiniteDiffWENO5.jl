# Conservative transport by global Lax-Friedrichs flux splitting.
#
# The previous conservative operator differenced `v_face * û_face`, reconstructing
# the transported state and multiplying it by a face velocity. Mishra, Parés-Pulido
# & Pressel (2020), Theorem 1 shows that construction is at most second order for a
# variable velocity, regardless of the reconstruction order: finite-difference WENO
# reconstructs the sliding-average of the *flux function*, so multiplying a
# reconstructed `u` by a velocity does not produce the flux that must be
# differenced.
#
# The correct construction forms the point flux `fᵢ = vᵢuᵢ` at cell centres, splits
# it globally as
#
#     f⁺ = ½(f + αu),   f⁻ = ½(f − αu),   α = max|v|
#
# so that `∂f⁺/∂u ≥ 0` and `∂f⁻/∂u ≤ 0`, reconstructs `f⁺` with the left-biased and
# `f⁻` with the right-biased WENO operator, and differences their sum. The split
# stencil values are formed locally from `(u, v)` inside the reconstruction loop, so
# no full-domain `f⁺`/`f⁻` scratch arrays are allocated.

@inline lf_split_plus(u, v, α) = 0.5 * (v * u + α * u)
@inline lf_split_minus(u, v, α) = 0.5 * (v * u - α * u)

"""Largest wave speed used for the global Lax-Friedrichs splitting constant."""
@inline function lf_speed(v)
    α = zero(eltype(v))
    @inbounds for i in eachindex(v)
        a = abs(v[i])
        a > α && (α = a)
    end
    return α
end

"""Backend-reduction variant of [`lf_speed`](@ref) for device-resident arrays."""
@inline lf_speed(v, reduce_abs) = reduce_abs(abs, v)

"""Directional global Lax--Friedrichs speeds, computed once per velocity field."""
@inline lf_speeds(form::AbstractAdvectionForm, velocity::NamedTuple) =
    is_conservative(form) ? map(lf_speed, velocity) : nothing

# `PrescribedInflowBC` prescribes an exterior transported *state*. The
# non-conservative path applies that state directly to a reconstructed `u`, but a
# conservative face needs a numerical *flux*. Writing the state into a flux array
# would be a units error, so the boundary face instead gets the first-order
# monotone Lax-Friedrichs flux built from the prescribed exterior state and the
# adjacent interior state,
#
#     F = ½(f(u_L) + f(u_R)) − ½α(u_R − u_L),
#
# which is exactly `f⁺(u_L) + f⁻(u_R)` — the same split the interior uses, at first
# order. Accuracy at the boundary is therefore first order by construction; the
# plan documents that no high-order inflow closure is claimed here.
function apply_conservative_inflow_1d!(fl, fr, u, v, α, boundary)
    bL, bR = boundary[1], boundary[2]
    if bL isa PrescribedInflowBC
        @inbounds begin
            fl.x[begin] = lf_split_plus(inflow_value(bL), v[begin], α)
            fr.x[begin] = lf_split_minus(u[begin], v[begin], α)
        end
    end
    if bR isa PrescribedInflowBC
        @inbounds begin
            fl.x[end] = lf_split_plus(u[end], v[end], α)
            fr.x[end] = lf_split_minus(inflow_value(bR), v[end], α)
        end
    end
    return nothing
end

# Higher-dimensional counterparts. Tangential indices follow the same convention
# as `apply_x_lower_inflow!` and friends in `boundaries.jl`: the remaining axes in
# their natural order.
function apply_conservative_inflow_2d!(fl, fr, u, vx, vy, αx, αy, boundary)
    bLx, bRx, bLy, bRy = boundary
    if bLx isa PrescribedInflowBC
        @inbounds for j in axes(fl.x, 2)
            fl.x[begin, j] = lf_split_plus(inflow_value(bLx, j), vx[begin, j], αx)
            fr.x[begin, j] = lf_split_minus(u[begin, j], vx[begin, j], αx)
        end
    end
    if bRx isa PrescribedInflowBC
        @inbounds for j in axes(fr.x, 2)
            fl.x[end, j] = lf_split_plus(u[end, j], vx[end, j], αx)
            fr.x[end, j] = lf_split_minus(inflow_value(bRx, j), vx[end, j], αx)
        end
    end
    if bLy isa PrescribedInflowBC
        @inbounds for i in axes(fl.y, 1)
            fl.y[i, begin] = lf_split_plus(inflow_value(bLy, i), vy[i, begin], αy)
            fr.y[i, begin] = lf_split_minus(u[i, begin], vy[i, begin], αy)
        end
    end
    if bRy isa PrescribedInflowBC
        @inbounds for i in axes(fr.y, 1)
            fl.y[i, end] = lf_split_plus(u[i, end], vy[i, end], αy)
            fr.y[i, end] = lf_split_minus(inflow_value(bRy, i), vy[i, end], αy)
        end
    end
    return nothing
end

function apply_conservative_inflow_3d!(fl, fr, u, vx, vy, vz, αx, αy, αz, boundary)
    bLx, bRx, bLy, bRy, bLz, bRz = boundary
    if bLx isa PrescribedInflowBC
        @inbounds for k in axes(fl.x, 3), j in axes(fl.x, 2)
            fl.x[begin, j, k] = lf_split_plus(inflow_value(bLx, j, k), vx[begin, j, k], αx)
            fr.x[begin, j, k] = lf_split_minus(u[begin, j, k], vx[begin, j, k], αx)
        end
    end
    if bRx isa PrescribedInflowBC
        @inbounds for k in axes(fr.x, 3), j in axes(fr.x, 2)
            fl.x[end, j, k] = lf_split_plus(u[end, j, k], vx[end, j, k], αx)
            fr.x[end, j, k] = lf_split_minus(inflow_value(bRx, j, k), vx[end, j, k], αx)
        end
    end
    if bLy isa PrescribedInflowBC
        @inbounds for k in axes(fl.y, 3), i in axes(fl.y, 1)
            fl.y[i, begin, k] = lf_split_plus(inflow_value(bLy, i, k), vy[i, begin, k], αy)
            fr.y[i, begin, k] = lf_split_minus(u[i, begin, k], vy[i, begin, k], αy)
        end
    end
    if bRy isa PrescribedInflowBC
        @inbounds for k in axes(fr.y, 3), i in axes(fr.y, 1)
            fl.y[i, end, k] = lf_split_plus(u[i, end, k], vy[i, end, k], αy)
            fr.y[i, end, k] = lf_split_minus(inflow_value(bRy, i, k), vy[i, end, k], αy)
        end
    end
    if bLz isa PrescribedInflowBC
        @inbounds for j in axes(fl.z, 2), i in axes(fl.z, 1)
            fl.z[i, j, begin] = lf_split_plus(inflow_value(bLz, i, j), vz[i, j, begin], αz)
            fr.z[i, j, begin] = lf_split_minus(u[i, j, begin], vz[i, j, begin], αz)
        end
    end
    if bRz isa PrescribedInflowBC
        @inbounds for j in axes(fr.z, 2), i in axes(fr.z, 1)
            fl.z[i, j, end] = lf_split_plus(u[i, j, end], vz[i, j, end], αz)
            fr.z[i, j, end] = lf_split_minus(inflow_value(bRz, i, j), vz[i, j, end], αz)
        end
    end
    return nothing
end

"""
    conservative_semi_discretisation_weno5!(du, u, vcell, weno, nx, Δx_)

Evaluate `∂ₓ(v u)` in 1D from a globally split point flux. `vcell` must be
collocated with `u`; a face-staggered velocity is prepared by `prepare_velocity!`
before this call.
"""
function conservative_semi_discretisation_weno5!(
        du::AbstractVector, u::AbstractVector, vcell, weno::WENOScheme, nx, Δx_, α,
    )
    (; fl, fr, boundary, χ, γ, ζ, ϵ, multithreading) = weno
    v = vcell.x
    size(v) == size(u) || throw(DimensionMismatch(
        "conservative transport needs a cell-centred velocity of size $(size(u)), got $(size(v))",
    ))

    bL = boundary[1]
    bR = boundary[2]

    @inbounds @maybe_threads multithreading for i in axes(fl.x, 1)
        iwww = left_index(i, 3, nx, bL)
        iww = left_index(i, 2, nx, bL)
        iw = left_index(i, 1, nx, bL)
        ie = right_index(i, 0, nx, bR)
        iee = right_index(i, 1, nx, bR)
        ieee = right_index(i, 2, nx, bR)

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

    apply_conservative_inflow_1d!(fl, fr, u, v, α, boundary)

    @inbounds @maybe_threads multithreading for i in eachindex(du)
        du[i] = ((fl.x[i + 1] + fr.x[i + 1]) - (fl.x[i] + fr.x[i])) * Δx_
    end
    return nothing
end

"""
    conservative_semi_discretisation_weno5!(du, u, vcell, weno, nx, ny, Δx_, Δy_)

Two-dimensional counterpart. Each direction is split independently with its own
Lax-Friedrichs constant, as the directional fluxes are differenced independently.
"""
function conservative_semi_discretisation_weno5!(
        du::AbstractMatrix, u::AbstractMatrix, vcell, weno::WENOScheme, nx, ny, Δx_, Δy_, αx, αy,
    )
    (; fl, fr, boundary, χ, γ, ζ, ϵ, multithreading) = weno
    vx, vy = vcell.x, vcell.y
    (size(vx) == size(u) && size(vy) == size(u)) || throw(DimensionMismatch(
        "conservative transport needs cell-centred velocities of size $(size(u))",
    ))

    bLx, bRx, bLy, bRy = boundary

    @inbounds @maybe_threads multithreading for I in CartesianIndices(fl.x)
        i, j = Tuple(I)
        iwww = left_index(i, 3, nx, bLx)
        iww = left_index(i, 2, nx, bLx)
        iw = left_index(i, 1, nx, bLx)
        ie = right_index(i, 0, nx, bRx)
        iee = right_index(i, 1, nx, bRx)
        ieee = right_index(i, 2, nx, bRx)

        fl.x[I] = weno5_reconstruction_upwind(
            lf_split_plus(u[iwww, j], vx[iwww, j], αx), lf_split_plus(u[iww, j], vx[iww, j], αx),
            lf_split_plus(u[iw, j], vx[iw, j], αx), lf_split_plus(u[ie, j], vx[ie, j], αx),
            lf_split_plus(u[iee, j], vx[iee, j], αx), χ, γ, ζ, ϵ,
        )
        fr.x[I] = weno5_reconstruction_downwind(
            lf_split_minus(u[iww, j], vx[iww, j], αx), lf_split_minus(u[iw, j], vx[iw, j], αx),
            lf_split_minus(u[ie, j], vx[ie, j], αx), lf_split_minus(u[iee, j], vx[iee, j], αx),
            lf_split_minus(u[ieee, j], vx[ieee, j], αx), χ, γ, ζ, ϵ,
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

        fl.y[I] = weno5_reconstruction_upwind(
            lf_split_plus(u[i, jwww], vy[i, jwww], αy), lf_split_plus(u[i, jww], vy[i, jww], αy),
            lf_split_plus(u[i, jw], vy[i, jw], αy), lf_split_plus(u[i, je], vy[i, je], αy),
            lf_split_plus(u[i, jee], vy[i, jee], αy), χ, γ, ζ, ϵ,
        )
        fr.y[I] = weno5_reconstruction_downwind(
            lf_split_minus(u[i, jww], vy[i, jww], αy), lf_split_minus(u[i, jw], vy[i, jw], αy),
            lf_split_minus(u[i, je], vy[i, je], αy), lf_split_minus(u[i, jee], vy[i, jee], αy),
            lf_split_minus(u[i, jeee], vy[i, jeee], αy), χ, γ, ζ, ϵ,
        )
    end

    apply_conservative_inflow_2d!(fl, fr, u, vx, vy, αx, αy, boundary)

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
        nx, ny, nz, Δx_, Δy_, Δz_, αx, αy, αz,
    )
    (; fl, fr, boundary, χ, γ, ζ, ϵ, multithreading) = weno
    vx, vy, vz = vcell.x, vcell.y, vcell.z
    all(w -> size(w) == size(u), (vx, vy, vz)) || throw(DimensionMismatch(
        "conservative transport needs cell-centred velocities of size $(size(u))",
    ))

    bLx, bRx, bLy, bRy, bLz, bRz = boundary

    @inbounds @maybe_threads multithreading for I in CartesianIndices(fl.x)
        i, j, k = Tuple(I)
        iwww = left_index(i, 3, nx, bLx)
        iww = left_index(i, 2, nx, bLx)
        iw = left_index(i, 1, nx, bLx)
        ie = right_index(i, 0, nx, bRx)
        iee = right_index(i, 1, nx, bRx)
        ieee = right_index(i, 2, nx, bRx)

        fl.x[I] = weno5_reconstruction_upwind(
            lf_split_plus(u[iwww, j, k], vx[iwww, j, k], αx),
            lf_split_plus(u[iww, j, k], vx[iww, j, k], αx),
            lf_split_plus(u[iw, j, k], vx[iw, j, k], αx),
            lf_split_plus(u[ie, j, k], vx[ie, j, k], αx),
            lf_split_plus(u[iee, j, k], vx[iee, j, k], αx), χ, γ, ζ, ϵ,
        )
        fr.x[I] = weno5_reconstruction_downwind(
            lf_split_minus(u[iww, j, k], vx[iww, j, k], αx),
            lf_split_minus(u[iw, j, k], vx[iw, j, k], αx),
            lf_split_minus(u[ie, j, k], vx[ie, j, k], αx),
            lf_split_minus(u[iee, j, k], vx[iee, j, k], αx),
            lf_split_minus(u[ieee, j, k], vx[ieee, j, k], αx), χ, γ, ζ, ϵ,
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

        fl.y[I] = weno5_reconstruction_upwind(
            lf_split_plus(u[i, jwww, k], vy[i, jwww, k], αy),
            lf_split_plus(u[i, jww, k], vy[i, jww, k], αy),
            lf_split_plus(u[i, jw, k], vy[i, jw, k], αy),
            lf_split_plus(u[i, je, k], vy[i, je, k], αy),
            lf_split_plus(u[i, jee, k], vy[i, jee, k], αy), χ, γ, ζ, ϵ,
        )
        fr.y[I] = weno5_reconstruction_downwind(
            lf_split_minus(u[i, jww, k], vy[i, jww, k], αy),
            lf_split_minus(u[i, jw, k], vy[i, jw, k], αy),
            lf_split_minus(u[i, je, k], vy[i, je, k], αy),
            lf_split_minus(u[i, jee, k], vy[i, jee, k], αy),
            lf_split_minus(u[i, jeee, k], vy[i, jeee, k], αy), χ, γ, ζ, ϵ,
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

        fl.z[I] = weno5_reconstruction_upwind(
            lf_split_plus(u[i, j, kwww], vz[i, j, kwww], αz),
            lf_split_plus(u[i, j, kww], vz[i, j, kww], αz),
            lf_split_plus(u[i, j, kw], vz[i, j, kw], αz),
            lf_split_plus(u[i, j, ke], vz[i, j, ke], αz),
            lf_split_plus(u[i, j, kee], vz[i, j, kee], αz), χ, γ, ζ, ϵ,
        )
        fr.z[I] = weno5_reconstruction_downwind(
            lf_split_minus(u[i, j, kww], vz[i, j, kww], αz),
            lf_split_minus(u[i, j, kw], vz[i, j, kw], αz),
            lf_split_minus(u[i, j, ke], vz[i, j, ke], αz),
            lf_split_minus(u[i, j, kee], vz[i, j, kee], αz),
            lf_split_minus(u[i, j, keee], vz[i, j, keee], αz), χ, γ, ζ, ϵ,
        )
    end

    apply_conservative_inflow_3d!(fl, fr, u, vx, vy, vz, αx, αy, αz, boundary)

    @inbounds @maybe_threads multithreading for I in CartesianIndices(du)
        i, j, k = Tuple(I)
        du[I] = @muladd ((fl.x[i + 1, j, k] + fr.x[i + 1, j, k]) - (fl.x[I] + fr.x[I])) * Δx_ +
            ((fl.y[i, j + 1, k] + fr.y[i, j + 1, k]) - (fl.y[I] + fr.y[I])) * Δy_ +
            ((fl.z[i, j, k + 1] + fr.z[i, j, k + 1]) - (fl.z[I] + fr.z[I])) * Δz_
    end
    return nothing
end
