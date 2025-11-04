@kernel function WENO_flux_KA_3D_x(fl, fr, u, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, g, O)

    I = @index(Global, NTuple)
    I = I + O
    i, j, k = I[1], I[2], I[3]
    n, m, p = size(fl)

    if 1 <= i <= n && 1 <= j <= m && 1 <= k <= p

        # Left boundary condition
        if boundary[1] == 0       # homogeneous Dirichlet
            iwww = clamp(i - 3, 1, nx)
            iww = clamp(i - 2, 1, nx)
            iw = clamp(i - 1, 1, nx)
        elseif boundary[1] == 1   # homogeneous Neumann
            iwww = max(i - 3, 1)
            iww = max(i - 2, 1)
            iw = max(i - 1, 1)
        elseif boundary[1] == 2   # Periodic
            iwww = mod1(i - 3, nx)
            iww = mod1(i - 2, nx)
            iw = mod1(i - 1, nx)
        end

        # Right boundary condition
        if boundary[2] == 0
            ie = clamp(i, 1, nx)
            iee = clamp(i + 1, 1, nx)
            ieee = clamp(i + 2, 1, nx)
        elseif boundary[2] == 1
            ie = min(i, nx)
            iee = min(i + 1, nx)
            ieee = min(i + 2, nx)
        elseif boundary[2] == 2
            ie = mod1(i, nx)
            iee = mod1(i + 1, nx)
            ieee = mod1(i + 2, nx)
        end

        u1 = u[iwww, j, k]
        u2 = u[iww, j, k]
        u3 = u[iw, j, k]
        u4 = u[ie, j, k]
        u5 = u[iee, j, k]
        u6 = u[ieee, j, k]

        fl[i, j, k] = FiniteDiffWENO5.weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr[i, j, k] = FiniteDiffWENO5.weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

        if lim_ZS
            # left interface (from left stencil)
            u_avg = u3
            θ_fl = min(
                1.0,
                abs((u_max - u_avg) / (fl[I...] - u_avg + ϵθ)),
                abs((u_avg - u_min) / (u_avg - fl[I...] + ϵθ))
            )
            fl[I...] = θ_fl * (fl[I...] - u_avg) + u_avg

            # right interface (from right stencil)
            u_avg = u4
            θ_fr = min(
                1.0,
                abs((u_max - u_avg) / (fr[I...] - u_avg + ϵθ)),
                abs((u_avg - u_min) / (u_avg - fr[I...] + ϵθ))
            )
            fr[I...] = θ_fr * (fr[I...] - u_avg) + u_avg
        end
    end
end

@kernel function WENO_flux_KA_3D_y(fl, fr, u, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, g, O)

    I = @index(Global, NTuple)
    I = I + O
    i, j, k = I[1], I[2], I[3]
    n, m, p = size(fl)

    if 1 <= i <= n && 1 <= j <= m && 1 <= k <= p

        # Left boundary condition
        if boundary[3] == 0       # homogeneous Dirichlet
            jwww = clamp(j - 3, 1, ny)
            jww = clamp(j - 2, 1, ny)
            jw = clamp(j - 1, 1, ny)
        elseif boundary[3] == 1   # homogeneous Neumann
            jwww = max(j - 3, 1)
            jww = max(j - 2, 1)
            jw = max(j - 1, 1)
        elseif boundary[3] == 2   # Periodic
            jwww = mod1(j - 3, ny)
            jww = mod1(j - 2, ny)
            jw = mod1(j - 1, ny)
        end

        # Right boundary condition
        if boundary[4] == 0
            je = clamp(j, 1, ny)
            jee = clamp(j + 1, 1, ny)
            jeee = clamp(j + 2, 1, ny)
        elseif boundary[4] == 1
            je = min(j, ny)
            jee = min(j + 1, ny)
            jeee = min(j + 2, ny)
        elseif boundary[4] == 2
            je = mod1(j, ny)
            jee = mod1(j + 1, ny)
            jeee = mod1(j + 2, ny)
        end

        u1 = u[i, jwww, k]
        u2 = u[i, jww, k]
        u3 = u[i, jw, k]
        u4 = u[i, je, k]
        u5 = u[i, jee, k]
        u6 = u[i, jeee, k]

        fl[i, j, k] = FiniteDiffWENO5.weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr[i, j, k] = FiniteDiffWENO5.weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

        if lim_ZS
            # left interface (from left stencil)
            u_avg = u3
            θ_fl = min(
                1.0,
                abs((u_max - u_avg) / (fl[I] - u_avg + ϵθ)),
                abs((u_avg - u_min) / (u_avg - fl[I] + ϵθ))
            )
            fl[I] = θ_fl * (fl[I] - u_avg) + u_avg

            # right interface (from right stencil)
            u_avg = u4
            θ_fr = min(
                1.0,
                abs((u_max - u_avg) / (fr[I] - u_avg + ϵθ)),
                abs((u_avg - u_min) / (u_avg - fr[I] + ϵθ))
            )
            fr[I] = θ_fr * (fr[I] - u_avg) + u_avg
        end
    end
end

@kernel function WENO_flux_KA_3D_z(fl, fr, u, boundary, nz, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, g, O)

    I = @index(Global, NTuple)
    I = I + O
    i, j, k = I[1], I[2], I[3]
    n, m, p = size(fl)

    if 1 <= i <= n && 1 <= j <= m && 1 <= k <= p

        # Left boundary condition
        if boundary[5] == 0       # homogeneous Dirichlet
            kwww = clamp(k - 3, 1, nz)
            kww = clamp(k - 2, 1, nz)
            kw = clamp(k - 1, 1, nz)
        elseif boundary[5] == 1   # homogeneous Neumann
            kwww = max(k - 3, 1)
            kww = max(k - 2, 1)
            kw = max(k - 1, 1)
        elseif boundary[5] == 2   # Periodic
            kwww = mod1(k - 3, nz)
            kww = mod1(k - 2, nz)
            kw = mod1(k - 1, nz)
        end

        # Right boundary condition
        if boundary[6] == 0
            ke = clamp(k, 1, nz)
            kee = clamp(k + 1, 1, nz)
            keee = clamp(k + 2, 1, nz)
        elseif boundary[6] == 1
            ke = min(k, nz)
            kee = min(k + 1, nz)
            keee = min(k + 2, nz)
        elseif boundary[6] == 2
            ke = mod1(k, nz)
            kee = mod1(k + 1, nz)
            keee = mod1(k + 2, nz)
        end

        u1 = u[i, j, kwww]
        u2 = u[i, j, kww]
        u3 = u[i, j, kw]
        u4 = u[i, j, ke]
        u5 = u[i, j, kee]
        u6 = u[i, j, keee]

        fl[i, j, k] = FiniteDiffWENO5.weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr[i, j, k] = FiniteDiffWENO5.weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

        if lim_ZS
            # left interface (from left stencil)
            u_avg = u3
            θ_fl = min(
                1.0,
                abs((u_max - u_avg) / (fl[I] - u_avg + ϵθ)),
                abs((u_avg - u_min) / (u_avg - fl[I] + ϵθ))
            )
            fl[I] = θ_fl * (fl[I] - u_avg) + u_avg

            # right interface (from right stencil)
            u_avg = u4
            θ_fr = min(
                1.0,
                abs((u_max - u_avg) / (fr[I] - u_avg + ϵθ)),
                abs((u_avg - u_min) / (u_avg - fr[I] + ϵθ))
            )
            fr[I] = θ_fr * (fr[I] - u_avg) + u_avg
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
