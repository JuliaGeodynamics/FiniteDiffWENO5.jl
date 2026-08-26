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

@kernel function WENO_semi_discretisation_weno5_KA_3D!(du, fl, fr, v, stag, Δx_, Δy_, Δz_, g, O)

    I = @index(Global, Cartesian)
    I = I + O

    i, j, k = I[1], I[2], I[3]

    m, n, p = size(du)

    if 1 <= i <= m && 1 <= j <= n && 1 <= k <= p
        if stag
            du[I] = @muladd (
                max(v.x[i + 1, j, k], 0) * fl.x[i + 1, j, k] +
                    min(v.x[i + 1, j, k], 0) * fr.x[i + 1, j, k] -
                    max(v.x[I], 0) * fl.x[I] -
                    min(v.x[I], 0) * fr.x[I]
            ) * Δx_ +
                (
                max(v.y[i, j + 1, k], 0) * fl.y[i, j + 1, k] +
                    min(v.y[i, j + 1, k], 0) * fr.y[i, j + 1, k] -
                    max(v.y[I], 0) * fl.y[I] -
                    min(v.y[I], 0) * fr.y[I]
            ) * Δy_ +
                (max(v.z[i, j, k + 1], 0) * fl.z[i, j, k + 1] + min(v.z[i, j, k + 1], 0) * fr.z[i, j, k + 1] - max(v.z[I], 0) * fl.z[I] - min(v.z[I], 0) * fr.z[I]) * Δz_
        else
            du[I] = @muladd max(v.x[I], 0) * (fl.x[i + 1, j, k] - fl.x[I]) * Δx_ + min(v.x[I], 0) * (fr.x[i + 1, j, k] - fr.x[I]) * Δx_ +
                max(v.y[I], 0) * (fl.y[i, j + 1, k] - fl.y[I]) * Δy_ + min(v.y[I], 0) * (fr.y[i, j + 1, k] - fr.y[I]) * Δy_ +
                max(v.z[I], 0) * (fl.z[i, j, k + 1] - fl.z[I]) * Δz_ + min(v.z[I], 0) * (fr.z[i, j, k + 1] - fr.z[I]) * Δz_
        end
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
