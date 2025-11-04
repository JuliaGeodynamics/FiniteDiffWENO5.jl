@kernel function WENO_flux_KA_2D_x(fl, fr, u, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, g, O)

    I = @index(Global, NTuple)
    I = I + O
    i, j = I[1], I[2]
    n, m = size(fl)

    if 1 <= i <= n && 1 <= j <= m

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

        u1 = u[iwww, j]
        u2 = u[iww, j]
        u3 = u[iw, j]
        u4 = u[ie, j]
        u5 = u[iee, j]
        u6 = u[ieee, j]

        fl[i, j] = FiniteDiffWENO5.weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr[i, j] = FiniteDiffWENO5.weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

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


@kernel function WENO_flux_KA_2D_y(fl, fr, u, boundary, ny, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, g, O)

    I = @index(Global, NTuple)
    I = I + O
    i, j = I[1], I[2]
    n, m = size(fl)

    if 1 <= i <= n && 1 <= j <= m

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

        u1 = u[i, jwww]
        u2 = u[i, jww]
        u3 = u[i, jw]
        u4 = u[i, je]
        u5 = u[i, jee]
        u6 = u[i, jeee]

        fl[i, j] = FiniteDiffWENO5.weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr[i, j] = FiniteDiffWENO5.weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

        if lim_ZS
            ϵθ = 1.0e-16 # small number to avoid division by zero

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
