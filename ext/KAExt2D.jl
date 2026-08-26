@kernel function WENO_flux_KA_2D_x(fl, fr, u, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, g, O)

    I = @index(Global, NTuple)
    I = I + O
    i, j = I[1], I[2]
    n, m = size(fl)

    if 1 <= i <= n && 1 <= j <= m

        iwww = left_index(i, 3, nx, boundary[1])
        iww = left_index(i, 2, nx, boundary[1])
        iw = left_index(i, 1, nx, boundary[1])
        ie = right_index(i, 0, nx, boundary[2])
        iee = right_index(i, 1, nx, boundary[2])
        ieee = right_index(i, 2, nx, boundary[2])

        u1 = u[iwww, j]
        u2 = u[iww, j]
        u3 = u[iw, j]
        u4 = u[ie, j]
        u5 = u[iee, j]
        u6 = u[ieee, j]

        fl[i, j] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr[i, j] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

        if lim_ZS
            ϵθ = 1.0e-16 # small number to avoid division by zero

            fl[i, j] = zhang_shu_limit(fl[i, j], u3, u_min, u_max, ϵθ)
            fr[i, j] = zhang_shu_limit(fr[i, j], u4, u_min, u_max, ϵθ)
        end

        if i == 1 && boundary[1] isa PrescribedInflowBC
            fl[i, j] = inflow_value(boundary[1], j)
        end
        if i == nx + 1 && boundary[2] isa PrescribedInflowBC
            fr[i, j] = inflow_value(boundary[2], j)
        end
    end
end


@kernel function WENO_flux_KA_2D_y(fl, fr, u, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, g, O)

    I = @index(Global, NTuple)
    I = I + O
    i, j = I[1], I[2]
    n, m = size(fl)

    if 1 <= i <= n && 1 <= j <= m

        jwww = left_index(j, 3, ny, boundary[3])
        jww = left_index(j, 2, ny, boundary[3])
        jw = left_index(j, 1, ny, boundary[3])
        je = right_index(j, 0, ny, boundary[4])
        jee = right_index(j, 1, ny, boundary[4])
        jeee = right_index(j, 2, ny, boundary[4])

        u1 = u[i, jwww]
        u2 = u[i, jww]
        u3 = u[i, jw]
        u4 = u[i, je]
        u5 = u[i, jee]
        u6 = u[i, jeee]

        fl[i, j] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr[i, j] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

        if lim_ZS
            ϵθ = 1.0e-16 # small number to avoid division by zero

            fl[i, j] = zhang_shu_limit(fl[i, j], u3, u_min, u_max, ϵθ)
            fr[i, j] = zhang_shu_limit(fr[i, j], u4, u_min, u_max, ϵθ)
        end


        if j == 1 && boundary[3] isa PrescribedInflowBC
            fl[i, j] = inflow_value(boundary[3], i)
        end
        if j == ny + 1 && boundary[4] isa PrescribedInflowBC
            fr[i, j] = inflow_value(boundary[4], i)
        end
    end
end

@kernel function WENO_semi_discretisation_weno5_KA_2D!(du, fl, fr, v, stag, Δx_, Δy_, g, O)

    I = @index(Global, Cartesian)
    I = I + O
    i, j = I[1], I[2]

    m, n = size(du)

    if 1 <= i <= m && 1 <= j <= n
        if stag
            du[I] = @muladd (
                max(v.x[i + 1, j], 0) * fl.x[i + 1, j] +
                    min(v.x[i + 1, j], 0) * fr.x[i + 1, j] -
                    max(v.x[I], 0) * fl.x[I] -
                    min(v.x[I], 0) * fr.x[I]
            ) * Δx_ +
                (
                max(v.y[i, j + 1], 0) * fl.y[i, j + 1] +
                    min(v.y[i, j + 1], 0) * fr.y[i, j + 1] -
                    max(v.y[I], 0) * fl.y[I] -
                    min(v.y[I], 0) * fr.y[I]
            ) * Δy_
        else
            du[I] = @muladd max(v.x[I], 0) * (fl.x[i + 1, j] - fl.x[I]) * Δx_ + min(v.x[I], 0) * (fr.x[i + 1, j] - fr.x[I]) * Δx_ +
                max(v.y[I], 0) * (fl.y[i, j + 1] - fl.y[I]) * Δy_ + min(v.y[I], 0) * (fr.y[i, j + 1] - fr.y[I]) * Δy_
        end
    end
end

@kernel function upwind_update_KA_2D!(
        u, v, nx, ny, Δx_, Δy_, Δt, stag, boundary, g, O
    )
    I = @index(Global, NTuple)
    I = I + O

    i, j = I[1], I[2]

    iLx = left_index(i, 1, nx, boundary[1])
    iRx = right_index(i, 1, nx, boundary[2])
    jLy = left_index(j, 1, ny, boundary[3])
    jRy = right_index(j, 1, ny, boundary[4])

    # ---- Upwind update ----
    if stag
        # Velocities defined at faces
        u[i, j] -= @muladd Δt * (
            (
                max(v.x[i, j], 0) * (u[i, j] - u[iLx, j]) +
                    min(v.x[iRx, j], 0) * (u[iRx, j] - u[i, j])
            ) * Δx_ +
                (
                max(v.y[i, j], 0) * (u[i, j] - u[i, jLy]) +
                    min(v.y[i, jRy], 0) * (u[i, jRy] - u[i, j])
            ) * Δy_
        )
    else
        # Velocities defined at centers
        u[i, j] -= @muladd Δt * (
            (
                max(v.x[i, j], 0) * (u[i, j] - u[iLx, j]) +
                    min(v.x[i, j], 0) * (u[iRx, j] - u[i, j])
            ) * Δx_ +
                (
                max(v.y[i, j], 0) * (u[i, j] - u[i, jLy]) +
                    min(v.y[i, j], 0) * (u[i, jRy] - u[i, j])
            ) * Δy_
        )
    end
end
