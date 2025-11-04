@kernel inbounds = true function WENO_flux_KA_1D(fl, fr, u, boundary, nx, χ, γ, ζ, ϵ, lim_ZS, u_min, u_max, g, O)

    I = @index(Global, NTuple)
    I = I + O
    i = I[1]

    # Left boundary condition
    if boundary[1] == 0       # Dirichlet
        iwww = clamp(i - 3, 1, nx)
        iww = clamp(i - 2, 1, nx)
        iw = clamp(i - 1, 1, nx)
    elseif boundary[1] == 1   # Neumann
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

    u1 = u[iwww]
    u2 = u[iww]
    u3 = u[iw]
    u4 = u[ie]
    u5 = u[iee]
    u6 = u[ieee]

    fl[i] = FiniteDiffWENO5.weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
    fr[i] = FiniteDiffWENO5.weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

    if lim_ZS
        # --- Zhang-Shu positivity limiter ---
        # separate averages for left and right

        ϵθ = 1.0e-16 # small number to avoid division by zero

        u_avg = u3

        θ_fl = min(
            1.0,
            abs((u_max - u_avg) / (fl[i] - u_avg + ϵθ)),
            abs((u_avg - u_min) / (u_avg - fl[i] + ϵθ))
        )
        # apply limiter
        fl[i] = θ_fl * (fl[i] - u_avg) + u_avg

        # separate averages for left and right
        u_avg = u4

        θ_fr = min(
            1.0,
            abs((u_max - u_avg) / (fr[i] - u_avg + ϵθ)),
            abs((u_avg - u_min) / (u_avg - fr[i] + ϵθ))
        )
        fr[i] = θ_fr * (fr[i] - u_avg) + u_avg
    end
end

@kernel inbounds = true function WENO_semi_discretisation_weno5_KA_1D!(du, fl, fr, v, stag, Δx_, g, O)

    I = @index(Global, NTuple)
    I = I + O
    i = I[1]

    if stag
        du[i] = @muladd (
            max(v.x[i + 1], 0) * fl.x[i + 1] +
                min(v.x[i + 1], 0) * fr.x[i + 1] -
                max(v.x[i], 0) * fl.x[i] -
                min(v.x[i], 0) * fr.x[i]
        ) * Δx_
    else
        du[i] = @muladd max(v.x[i], 0) * (fl.x[i + 1] - fl.x[i]) * Δx_ + min(v.x[i], 0) * (fr.x[i + 1] - fr.x[i]) * Δx_
    end
end
