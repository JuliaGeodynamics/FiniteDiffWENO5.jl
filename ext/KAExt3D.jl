@kernel function WENO_flux_KA_3D_x(fl, fr, u, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, g, O)

    I = @index(Global, NTuple)
    I = I + O
    i, j, k = I[1], I[2], I[3]
    n, m, p = size(fl)

    if 1 <= i <= n && 1 <= j <= m && 1 <= k <= p

        iwww = left_index(i, 3, nx, boundary[1])
        iww = left_index(i, 2, nx, boundary[1])
        iw = left_index(i, 1, nx, boundary[1])
        ie = right_index(i, 0, nx, boundary[2])
        iee = right_index(i, 1, nx, boundary[2])
        ieee = right_index(i, 2, nx, boundary[2])

        u1 = u[iwww, j, k]
        u2 = u[iww, j, k]
        u3 = u[iw, j, k]
        u4 = u[ie, j, k]
        u5 = u[iee, j, k]
        u6 = u[ieee, j, k]

        fl[i, j, k] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr[i, j, k] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

        if lim_ZS
            ϵθ = 1.0e-16 # small number to avoid division by zero

            fl[i, j, k] = zhang_shu_limit(fl[i, j, k], u3, u_min, u_max, ϵθ)
            fr[i, j, k] = zhang_shu_limit(fr[i, j, k], u4, u_min, u_max, ϵθ)
        end

        if i == 1 && boundary[1] isa PrescribedInflowBC
            fl[i, j, k] = inflow_value(boundary[1], j, k)
        end
        if i == nx + 1 && boundary[2] isa PrescribedInflowBC
            fr[i, j, k] = inflow_value(boundary[2], j, k)
        end
    end
end

@kernel function WENO_flux_KA_3D_y(fl, fr, u, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, g, O)

    I = @index(Global, NTuple)
    I = I + O
    i, j, k = I[1], I[2], I[3]
    n, m, p = size(fl)

    if 1 <= i <= n && 1 <= j <= m && 1 <= k <= p

        jwww = left_index(j, 3, ny, boundary[3])
        jww = left_index(j, 2, ny, boundary[3])
        jw = left_index(j, 1, ny, boundary[3])
        je = right_index(j, 0, ny, boundary[4])
        jee = right_index(j, 1, ny, boundary[4])
        jeee = right_index(j, 2, ny, boundary[4])

        u1 = u[i, jwww, k]
        u2 = u[i, jww, k]
        u3 = u[i, jw, k]
        u4 = u[i, je, k]
        u5 = u[i, jee, k]
        u6 = u[i, jeee, k]

        fl[i, j, k] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr[i, j, k] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

        if lim_ZS
            ϵθ = 1.0e-16 # small number to avoid division by zero

            fl[i, j, k] = zhang_shu_limit(fl[i, j, k], u3, u_min, u_max, ϵθ)
            fr[i, j, k] = zhang_shu_limit(fr[i, j, k], u4, u_min, u_max, ϵθ)
        end

        if j == 1 && boundary[3] isa PrescribedInflowBC
            fl[i, j, k] = inflow_value(boundary[3], i, k)
        end
        if j == ny + 1 && boundary[4] isa PrescribedInflowBC
            fr[i, j, k] = inflow_value(boundary[4], i, k)
        end
    end
end

@kernel function WENO_flux_KA_3D_z(fl, fr, u, boundary, nz, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, g, O)

    I = @index(Global, NTuple)
    I = I + O
    i, j, k = I[1], I[2], I[3]
    n, m, p = size(fl)

    if 1 <= i <= n && 1 <= j <= m && 1 <= k <= p

        kwww = left_index(k, 3, nz, boundary[5])
        kww = left_index(k, 2, nz, boundary[5])
        kw = left_index(k, 1, nz, boundary[5])
        ke = right_index(k, 0, nz, boundary[6])
        kee = right_index(k, 1, nz, boundary[6])
        keee = right_index(k, 2, nz, boundary[6])

        u1 = u[i, j, kwww]
        u2 = u[i, j, kww]
        u3 = u[i, j, kw]
        u4 = u[i, j, ke]
        u5 = u[i, j, kee]
        u6 = u[i, j, keee]

        fl[i, j, k] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr[i, j, k] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

        if lim_ZS
            ϵθ = 1.0e-16 # small number to avoid division by zero

            fl[i, j, k] = zhang_shu_limit(fl[i, j, k], u3, u_min, u_max, ϵθ)
            fr[i, j, k] = zhang_shu_limit(fr[i, j, k], u4, u_min, u_max, ϵθ)
        end

        if k == 1 && boundary[5] isa PrescribedInflowBC
            fl[i, j, k] = inflow_value(boundary[5], i, j)
        end
        if k == nz + 1 && boundary[6] isa PrescribedInflowBC
            fr[i, j, k] = inflow_value(boundary[6], i, j)
        end
    end
end

@kernel function WENO_semi_discretisation_weno5_KA_3D!(du, fl, fr, v, Δx_, Δy_, Δz_, g, O)

    I = @index(Global, Cartesian)
    I = I + O

    i, j, k = I[1], I[2], I[3]

    m, n, p = size(du)

    if 1 <= i <= m && 1 <= j <= n && 1 <= k <= p
        du[I] = @muladd max(v.x[I], 0) * (fl.x[i + 1, j, k] - fl.x[I]) * Δx_ +
            min(v.x[I], 0) * (fr.x[i + 1, j, k] - fr.x[I]) * Δx_ +
            max(v.y[I], 0) * (fl.y[i, j + 1, k] - fl.y[I]) * Δy_ +
            min(v.y[I], 0) * (fr.y[i, j + 1, k] - fr.y[I]) * Δy_ +
            max(v.z[I], 0) * (fl.z[i, j, k + 1] - fl.z[I]) * Δz_ +
            min(v.z[I], 0) * (fr.z[i, j, k + 1] - fr.z[I]) * Δz_
    end
end

@kernel function upwind_update_KA_3D!(
        u, v, nx, ny, nz, Δx_, Δy_, Δz_, Δt, stag, boundary, g, O
    )
    I = @index(Global, NTuple)
    I = I + O

    i, j, k = I[1], I[2], I[3]

    iLx = left_index(i, 1, nx, boundary[1])
    iRx = right_index(i, 1, nx, boundary[2])
    jLy = left_index(j, 1, ny, boundary[3])
    jRy = right_index(j, 1, ny, boundary[4])
    kLz = left_index(k, 1, nz, boundary[5])
    kRz = right_index(k, 1, nz, boundary[6])

    # ---- Upwind update ----
    if stag
        # Velocities defined at faces
        u[i, j, k] -= @muladd Δt * (
            (
                max(v.x[i, j, k], 0) * (u[i, j, k] - u[iLx, j, k]) +
                    min(v.x[iRx, j, k], 0) * (u[iRx, j, k] - u[i, j, k])
            ) * Δx_ +
                (
                max(v.y[i, j, k], 0) * (u[i, j, k] - u[i, jLy, k]) +
                    min(v.y[i, jRy, k], 0) * (u[i, jRy, k] - u[i, j, k])
            ) * Δy_ +
                (
                max(v.z[i, j, k], 0) * (u[i, j, k] - u[i, j, kLz]) +
                    min(v.z[i, j, kRz], 0) * (u[i, j, kRz] - u[i, j, k])
            ) * Δz_
        )
    else
        # Velocities defined at cell centers
        u[i, j, k] -= @muladd Δt * (
            (
                max(v.x[i, j, k], 0) * (u[i, j, k] - u[iLx, j, k]) +
                    min(v.x[i, j, k], 0) * (u[iRx, j, k] - u[i, j, k])
            ) * Δx_ +
                (
                max(v.y[i, j, k], 0) * (u[i, j, k] - u[i, jLy, k]) +
                    min(v.y[i, j, k], 0) * (u[i, jRy, k] - u[i, j, k])
            ) * Δy_ +
                (
                max(v.z[i, j, k], 0) * (u[i, j, k] - u[i, j, kLz]) +
                    min(v.z[i, j, k], 0) * (u[i, j, kRz] - u[i, j, k])
            ) * Δz_
        )
    end
end

# ENO5 face-to-centre interpolation for one velocity component along its own
# staggered direction, and the second-order small-grid fallback, matching the
# 1D/2D counterparts.
# Conservative split-flux reconstruction along x, y, z independently, matching
# `conservative_semi_discretisation_weno5!` in 3D.
@kernel inbounds = true function WENO_conservative_flux_KA_3D_x!(fl, fr, u, v, α, boundary, nx, χ, γ, ζ, ϵ, g, O)
    I = @index(Global, NTuple)
    I = I + O
    i, j, k = I[1], I[2], I[3]
    bL, bR = boundary[1], boundary[2]
    iwww = left_index(i, 3, nx, bL); iww = left_index(i, 2, nx, bL); iw = left_index(i, 1, nx, bL)
    ie = right_index(i, 0, nx, bR); iee = right_index(i, 1, nx, bR); ieee = right_index(i, 2, nx, bR)
    fl[i, j, k] = weno5_reconstruction_upwind(
        FiniteDiffWENO5.lf_split_plus(u[iwww, j, k], v[iwww, j, k], α),
        FiniteDiffWENO5.lf_split_plus(u[iww, j, k], v[iww, j, k], α),
        FiniteDiffWENO5.lf_split_plus(u[iw, j, k], v[iw, j, k], α),
        FiniteDiffWENO5.lf_split_plus(u[ie, j, k], v[ie, j, k], α),
        FiniteDiffWENO5.lf_split_plus(u[iee, j, k], v[iee, j, k], α), χ, γ, ζ, ϵ,
    )
    fr[i, j, k] = weno5_reconstruction_downwind(
        FiniteDiffWENO5.lf_split_minus(u[iww, j, k], v[iww, j, k], α),
        FiniteDiffWENO5.lf_split_minus(u[iw, j, k], v[iw, j, k], α),
        FiniteDiffWENO5.lf_split_minus(u[ie, j, k], v[ie, j, k], α),
        FiniteDiffWENO5.lf_split_minus(u[iee, j, k], v[iee, j, k], α),
        FiniteDiffWENO5.lf_split_minus(u[ieee, j, k], v[ieee, j, k], α), χ, γ, ζ, ϵ,
    )

    if i == 1 && bL isa FiniteDiffWENO5.PrescribedInflowBC
        fl[i, j, k] = FiniteDiffWENO5.lf_split_plus(FiniteDiffWENO5.inflow_value(bL, j, k), v[begin, j, k], α)
        fr[i, j, k] = FiniteDiffWENO5.lf_split_minus(u[begin, j, k], v[begin, j, k], α)
    end
    if i == nx + 1 && bR isa FiniteDiffWENO5.PrescribedInflowBC
        fl[i, j, k] = FiniteDiffWENO5.lf_split_plus(u[end, j, k], v[end, j, k], α)
        fr[i, j, k] = FiniteDiffWENO5.lf_split_minus(FiniteDiffWENO5.inflow_value(bR, j, k), v[end, j, k], α)
    end
end

@kernel inbounds = true function WENO_conservative_flux_KA_3D_y!(fl, fr, u, v, α, boundary, ny, χ, γ, ζ, ϵ, g, O)
    I = @index(Global, NTuple)
    I = I + O
    i, j, k = I[1], I[2], I[3]
    bL, bR = boundary[3], boundary[4]
    jwww = left_index(j, 3, ny, bL); jww = left_index(j, 2, ny, bL); jw = left_index(j, 1, ny, bL)
    je = right_index(j, 0, ny, bR); jee = right_index(j, 1, ny, bR); jeee = right_index(j, 2, ny, bR)
    fl[i, j, k] = weno5_reconstruction_upwind(
        FiniteDiffWENO5.lf_split_plus(u[i, jwww, k], v[i, jwww, k], α),
        FiniteDiffWENO5.lf_split_plus(u[i, jww, k], v[i, jww, k], α),
        FiniteDiffWENO5.lf_split_plus(u[i, jw, k], v[i, jw, k], α),
        FiniteDiffWENO5.lf_split_plus(u[i, je, k], v[i, je, k], α),
        FiniteDiffWENO5.lf_split_plus(u[i, jee, k], v[i, jee, k], α), χ, γ, ζ, ϵ,
    )
    fr[i, j, k] = weno5_reconstruction_downwind(
        FiniteDiffWENO5.lf_split_minus(u[i, jww, k], v[i, jww, k], α),
        FiniteDiffWENO5.lf_split_minus(u[i, jw, k], v[i, jw, k], α),
        FiniteDiffWENO5.lf_split_minus(u[i, je, k], v[i, je, k], α),
        FiniteDiffWENO5.lf_split_minus(u[i, jee, k], v[i, jee, k], α),
        FiniteDiffWENO5.lf_split_minus(u[i, jeee, k], v[i, jeee, k], α), χ, γ, ζ, ϵ,
    )

    if j == 1 && bL isa FiniteDiffWENO5.PrescribedInflowBC
        fl[i, j, k] = FiniteDiffWENO5.lf_split_plus(FiniteDiffWENO5.inflow_value(bL, i, k), v[i, begin, k], α)
        fr[i, j, k] = FiniteDiffWENO5.lf_split_minus(u[i, begin, k], v[i, begin, k], α)
    end
    if j == ny + 1 && bR isa FiniteDiffWENO5.PrescribedInflowBC
        fl[i, j, k] = FiniteDiffWENO5.lf_split_plus(u[i, end, k], v[i, end, k], α)
        fr[i, j, k] = FiniteDiffWENO5.lf_split_minus(FiniteDiffWENO5.inflow_value(bR, i, k), v[i, end, k], α)
    end
end

@kernel inbounds = true function WENO_conservative_flux_KA_3D_z!(fl, fr, u, v, α, boundary, nz, χ, γ, ζ, ϵ, g, O)
    I = @index(Global, NTuple)
    I = I + O
    i, j, k = I[1], I[2], I[3]
    bL, bR = boundary[5], boundary[6]
    kwww = left_index(k, 3, nz, bL); kww = left_index(k, 2, nz, bL); kw = left_index(k, 1, nz, bL)
    ke = right_index(k, 0, nz, bR); kee = right_index(k, 1, nz, bR); keee = right_index(k, 2, nz, bR)
    fl[i, j, k] = weno5_reconstruction_upwind(
        FiniteDiffWENO5.lf_split_plus(u[i, j, kwww], v[i, j, kwww], α),
        FiniteDiffWENO5.lf_split_plus(u[i, j, kww], v[i, j, kww], α),
        FiniteDiffWENO5.lf_split_plus(u[i, j, kw], v[i, j, kw], α),
        FiniteDiffWENO5.lf_split_plus(u[i, j, ke], v[i, j, ke], α),
        FiniteDiffWENO5.lf_split_plus(u[i, j, kee], v[i, j, kee], α), χ, γ, ζ, ϵ,
    )
    fr[i, j, k] = weno5_reconstruction_downwind(
        FiniteDiffWENO5.lf_split_minus(u[i, j, kww], v[i, j, kww], α),
        FiniteDiffWENO5.lf_split_minus(u[i, j, kw], v[i, j, kw], α),
        FiniteDiffWENO5.lf_split_minus(u[i, j, ke], v[i, j, ke], α),
        FiniteDiffWENO5.lf_split_minus(u[i, j, kee], v[i, j, kee], α),
        FiniteDiffWENO5.lf_split_minus(u[i, j, keee], v[i, j, keee], α), χ, γ, ζ, ϵ,
    )

    if k == 1 && bL isa FiniteDiffWENO5.PrescribedInflowBC
        fl[i, j, k] = FiniteDiffWENO5.lf_split_plus(FiniteDiffWENO5.inflow_value(bL, i, j), v[i, j, begin], α)
        fr[i, j, k] = FiniteDiffWENO5.lf_split_minus(u[i, j, begin], v[i, j, begin], α)
    end
    if k == nz + 1 && bR isa FiniteDiffWENO5.PrescribedInflowBC
        fl[i, j, k] = FiniteDiffWENO5.lf_split_plus(u[i, j, end], v[i, j, end], α)
        fr[i, j, k] = FiniteDiffWENO5.lf_split_minus(FiniteDiffWENO5.inflow_value(bR, i, j), v[i, j, end], α)
    end
end

@kernel inbounds = true function WENO_conservative_divergence_KA_3D!(du, fl, fr, Δx_, Δy_, Δz_, g, O)
    I = @index(Global, Cartesian)
    I = I + O
    i, j, k = I[1], I[2], I[3]
    du[I] = @muladd ((fl.x[i + 1, j, k] + fr.x[i + 1, j, k]) - (fl.x[I] + fr.x[I])) * Δx_ +
        ((fl.y[i, j + 1, k] + fr.y[i, j + 1, k]) - (fl.y[I] + fr.y[I])) * Δy_ +
        ((fl.z[i, j, k + 1] + fr.z[i, j, k + 1]) - (fl.z[I] + fr.z[I])) * Δz_
end

@kernel inbounds = true function eno5_face_to_center_KA_3D_x!(center, face, nx, periodic, g, O)
    I = @index(Global, NTuple)
    I = I + O
    i, j, k = I[1], I[2], I[3]
    n1, n2, n3 = size(center)
    if 1 <= i <= n1 && 1 <= j <= n2 && 1 <= k <= n3
        C = CartesianIndex(i, j, k)
        s = FiniteDiffWENO5.eno5_stencil_start(face, C, 1, i, nx, periodic)
        row = FiniteDiffWENO5.eno5_row(s, i)
        value = zero(eltype(face))
        for r in 0:4
            value += row[r + 1] * FiniteDiffWENO5.eno5_face_sample(face, C, 1, s + r, nx, periodic)
        end
        center[i, j, k] = value / oftype(value, 128)
    end
end

@kernel inbounds = true function eno5_face_to_center_KA_3D_y!(center, face, ny, periodic, g, O)
    I = @index(Global, NTuple)
    I = I + O
    i, j, k = I[1], I[2], I[3]
    n1, n2, n3 = size(center)
    if 1 <= i <= n1 && 1 <= j <= n2 && 1 <= k <= n3
        C = CartesianIndex(i, j, k)
        s = FiniteDiffWENO5.eno5_stencil_start(face, C, 2, j, ny, periodic)
        row = FiniteDiffWENO5.eno5_row(s, j)
        value = zero(eltype(face))
        for r in 0:4
            value += row[r + 1] * FiniteDiffWENO5.eno5_face_sample(face, C, 2, s + r, ny, periodic)
        end
        center[i, j, k] = value / oftype(value, 128)
    end
end

@kernel inbounds = true function eno5_face_to_center_KA_3D_z!(center, face, nz, periodic, g, O)
    I = @index(Global, NTuple)
    I = I + O
    i, j, k = I[1], I[2], I[3]
    n1, n2, n3 = size(center)
    if 1 <= i <= n1 && 1 <= j <= n2 && 1 <= k <= n3
        C = CartesianIndex(i, j, k)
        s = FiniteDiffWENO5.eno5_stencil_start(face, C, 3, k, nz, periodic)
        row = FiniteDiffWENO5.eno5_row(s, k)
        value = zero(eltype(face))
        for r in 0:4
            value += row[r + 1] * FiniteDiffWENO5.eno5_face_sample(face, C, 3, s + r, nz, periodic)
        end
        center[i, j, k] = value / oftype(value, 128)
    end
end

@kernel inbounds = true function linear_face_to_center_KA_3D_x!(center, face, nx, periodic, g, O)
    I = @index(Global, NTuple)
    I = I + O
    i, j, k = I[1], I[2], I[3]
    n1, n2, n3 = size(center)
    if 1 <= i <= n1 && 1 <= j <= n2 && 1 <= k <= n3
        C = CartesianIndex(i, j, k)
        lo = FiniteDiffWENO5.eno5_face_sample(face, C, 1, i, nx, periodic)
        hi = FiniteDiffWENO5.eno5_face_sample(face, C, 1, i + 1, nx, periodic)
        center[i, j, k] = 0.5 * (lo + hi)
    end
end

@kernel inbounds = true function linear_face_to_center_KA_3D_y!(center, face, ny, periodic, g, O)
    I = @index(Global, NTuple)
    I = I + O
    i, j, k = I[1], I[2], I[3]
    n1, n2, n3 = size(center)
    if 1 <= i <= n1 && 1 <= j <= n2 && 1 <= k <= n3
        C = CartesianIndex(i, j, k)
        lo = FiniteDiffWENO5.eno5_face_sample(face, C, 2, j, ny, periodic)
        hi = FiniteDiffWENO5.eno5_face_sample(face, C, 2, j + 1, ny, periodic)
        center[i, j, k] = 0.5 * (lo + hi)
    end
end

@kernel inbounds = true function linear_face_to_center_KA_3D_z!(center, face, nz, periodic, g, O)
    I = @index(Global, NTuple)
    I = I + O
    i, j, k = I[1], I[2], I[3]
    n1, n2, n3 = size(center)
    if 1 <= i <= n1 && 1 <= j <= n2 && 1 <= k <= n3
        C = CartesianIndex(i, j, k)
        lo = FiniteDiffWENO5.eno5_face_sample(face, C, 3, k, nz, periodic)
        hi = FiniteDiffWENO5.eno5_face_sample(face, C, 3, k + 1, nz, periodic)
        center[i, j, k] = 0.5 * (lo + hi)
    end
end

"""Device counterpart of `prepare_velocity!` in 3D."""
function prepare_velocity_KA_3D!(weno, v, nx, ny, nz, backend)
    weno.stag || return v
    weno.vcenter === nothing && return v

    px, py, pz = weno.vperiodic.x, weno.vperiodic.y, weno.vperiodic.z
    FiniteDiffWENO5.validate_staggered_velocity!(weno.vcenter, v; periodic = weno.vperiodic)

    cx, cy, cz = weno.vcenter.x, weno.vcenter.y, weno.vcenter.z
    kx = nx >= FiniteDiffWENO5.eno5_minimum_cells(px) ?
        eno5_face_to_center_KA_3D_x!(backend) : linear_face_to_center_KA_3D_x!(backend)
    ky = ny >= FiniteDiffWENO5.eno5_minimum_cells(py) ?
        eno5_face_to_center_KA_3D_y!(backend) : linear_face_to_center_KA_3D_y!(backend)
    kz = nz >= FiniteDiffWENO5.eno5_minimum_cells(pz) ?
        eno5_face_to_center_KA_3D_z!(backend) : linear_face_to_center_KA_3D_z!(backend)
    kx(cx, v.x, nx, px, nothing, Offset0, ndrange = size(cx))
    ky(cy, v.y, ny, py, nothing, Offset0, ndrange = size(cy))
    kz(cz, v.z, nz, pz, nothing, Offset0, ndrange = size(cz))
    synchronize(backend)
    return (; x = cx, y = cy, z = cz)
end
