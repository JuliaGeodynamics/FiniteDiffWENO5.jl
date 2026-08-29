@kernel inbounds = true function WENO_flux_KA_1D(fl, fr, u, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, g, O)

    I = @index(Global, NTuple)
    I = I + O
    i = I[1]

    iwww = left_index(i, 3, nx, boundary[1])
    iww = left_index(i, 2, nx, boundary[1])
    iw = left_index(i, 1, nx, boundary[1])
    ie = right_index(i, 0, nx, boundary[2])
    iee = right_index(i, 1, nx, boundary[2])
    ieee = right_index(i, 2, nx, boundary[2])

    u1 = u[iwww]
    u2 = u[iww]
    u3 = u[iw]
    u4 = u[ie]
    u5 = u[iee]
    u6 = u[ieee]

    fl[i] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
    fr[i] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

    if lim_ZS
        # --- Zhang-Shu positivity limiter ---
        ϵθ = 1.0e-16 # small number to avoid division by zero

        fl[i] = zhang_shu_limit(fl[i], u3, u_min, u_max, ϵθ)
        fr[i] = zhang_shu_limit(fr[i], u4, u_min, u_max, ϵθ)
    end

    if i == 1 && boundary[1] isa PrescribedInflowBC
        fl[i] = inflow_value(boundary[1])
    end
    if i == nx + 1 && boundary[2] isa PrescribedInflowBC
        fr[i] = inflow_value(boundary[2])
    end
end

@kernel inbounds = true function WENO_semi_discretisation_weno5_KA_1D!(du, fl, fr, v, Δx_, g, O)

    I = @index(Global, NTuple)
    I = I + O
    i = I[1]

    du[i] = @muladd max(v.x[i], 0) * (fl.x[i + 1] - fl.x[i]) * Δx_ +
        min(v.x[i], 0) * (fr.x[i + 1] - fr.x[i]) * Δx_
end

@kernel inbounds = true function upwind_update_KA_1D!(
        u, v, nx, Δx_, Δt, stag, boundary, g, O
    )
    I = @index(Global, NTuple)
    I = I + O
    i = I[1]

    iL = left_index(i, 1, nx, boundary[1])
    iR = right_index(i, 1, nx, boundary[2])

    if stag
        # velocity defined at faces
        u[i] -= @muladd Δt * (
            max(v.x[i], 0) * (u[i] - u[iL]) +
                min(v.x[iR], 0) * (u[iR] - u[i])
        ) * Δx_
    else
        # velocity defined at centers
        u[i] -= @muladd Δt * (
            max(v.x[i], 0) * (u[i] - u[iL]) +
                min(v.x[i], 0) * (u[iR] - u[i])
        ) * Δx_
    end
end

# Interpolate a staggered normal velocity to cell centres on the device, reusing
# the same per-cell stencil selection and coefficient row as the CPU path so the
# two backends produce identical values rather than merely similar ones.
@kernel inbounds = true function eno5_face_to_center_KA_1D!(center, face, n, periodic, g, O)
    I = @index(Global, NTuple)
    I = I + O
    i = I[1]
    if 1 <= i <= n
        s = FiniteDiffWENO5.eno5_stencil_start(face, i, n, periodic)
        row = FiniteDiffWENO5.eno5_row(s, i)
        value = zero(eltype(face))
        for r in 0:4
            value += row[r + 1] * FiniteDiffWENO5.eno5_face_sample(face, s + r, n, periodic)
        end
        center[i] = value / oftype(value, 128)
    end
end

# Second-order fallback for directions smaller than the ENO5 stencil, matching
# `linear_face_to_center_direction!` on the CPU.
@kernel inbounds = true function linear_face_to_center_KA_1D!(center, face, n, periodic, g, O)
    I = @index(Global, NTuple)
    I = I + O
    i = I[1]
    if 1 <= i <= n
        lo = FiniteDiffWENO5.eno5_face_sample(face, i, n, periodic)
        hi = FiniteDiffWENO5.eno5_face_sample(face, i + 1, n, periodic)
        center[i] = 0.5 * (lo + hi)
    end
end

# Conservative transport by global Lax-Friedrichs splitting; the device counterpart
# of `conservative_semi_discretisation_weno5!`. Split stencil values are formed
# locally from (u, v), so no full-domain f± scratch arrays are needed.
@kernel inbounds = true function WENO_conservative_flux_KA_1D!(
        fl, fr, u, v, α, boundary, nx, χ, γ, ζ, ϵ, g, O,
    )
    I = @index(Global, NTuple)
    I = I + O
    i = I[1]
    bL = boundary[1]
    bR = boundary[2]

    iwww = FiniteDiffWENO5.left_index(i, 3, nx, bL)
    iww = FiniteDiffWENO5.left_index(i, 2, nx, bL)
    iw = FiniteDiffWENO5.left_index(i, 1, nx, bL)
    ie = FiniteDiffWENO5.right_index(i, 0, nx, bR)
    iee = FiniteDiffWENO5.right_index(i, 1, nx, bR)
    ieee = FiniteDiffWENO5.right_index(i, 2, nx, bR)

    fl[i] = FiniteDiffWENO5.weno5_reconstruction_upwind(
        FiniteDiffWENO5.lf_split_plus(u[iwww], v[iwww], α),
        FiniteDiffWENO5.lf_split_plus(u[iww], v[iww], α),
        FiniteDiffWENO5.lf_split_plus(u[iw], v[iw], α),
        FiniteDiffWENO5.lf_split_plus(u[ie], v[ie], α),
        FiniteDiffWENO5.lf_split_plus(u[iee], v[iee], α), χ, γ, ζ, ϵ,
    )
    fr[i] = FiniteDiffWENO5.weno5_reconstruction_downwind(
        FiniteDiffWENO5.lf_split_minus(u[iww], v[iww], α),
        FiniteDiffWENO5.lf_split_minus(u[iw], v[iw], α),
        FiniteDiffWENO5.lf_split_minus(u[ie], v[ie], α),
        FiniteDiffWENO5.lf_split_minus(u[iee], v[iee], α),
        FiniteDiffWENO5.lf_split_minus(u[ieee], v[ieee], α), χ, γ, ζ, ϵ,
    )

    # First-order monotone conservative inflow closure, folded into the same
    # full-domain kernel (rather than a separate small-ndrange launch) because
    # Chmy's `Launcher` always covers the whole grid and cannot be given a
    # custom, smaller launch domain.
    if i == 1 && bL isa FiniteDiffWENO5.PrescribedInflowBC
        fl[i] = FiniteDiffWENO5.lf_split_plus(FiniteDiffWENO5.inflow_value(bL), v[begin], α)
        fr[i] = FiniteDiffWENO5.lf_split_minus(u[begin], v[begin], α)
    end
    if i == nx + 1 && bR isa FiniteDiffWENO5.PrescribedInflowBC
        fl[i] = FiniteDiffWENO5.lf_split_plus(u[end], v[end], α)
        fr[i] = FiniteDiffWENO5.lf_split_minus(FiniteDiffWENO5.inflow_value(bR), v[end], α)
    end
end

@kernel inbounds = true function WENO_conservative_divergence_KA_1D!(du, fl, fr, Δx_, g, O)
    I = @index(Global, NTuple)
    I = I + O
    i = I[1]
    du[i] = ((fl[i + 1] + fr[i + 1]) - (fl[i] + fr[i])) * Δx_
end

"""
    prepare_velocity_KA_1D!(weno, v, nx, backend)

Device counterpart of `prepare_velocity!`: map a staggered normal velocity onto
cell centres once per step. Collocated schemes pass their velocity straight
through, matching the CPU behaviour of returning the input untouched.
"""
function prepare_velocity_KA_1D!(weno, v, nx, backend)
    weno.stag || return v
    weno.vcenter === nothing && return v

    periodic = weno.vperiodic.x
    FiniteDiffWENO5.validate_staggered_velocity!(weno.vcenter, v; periodic = weno.vperiodic)

    center = weno.vcenter.x
    kernel = nx >= FiniteDiffWENO5.eno5_minimum_cells(periodic) ?
        eno5_face_to_center_KA_1D!(backend) : linear_face_to_center_KA_1D!(backend)
    kernel(center, v.x, nx, periodic, nothing, Offset0, ndrange = length(center))
    synchronize(backend)
    return (; x = center)
end
