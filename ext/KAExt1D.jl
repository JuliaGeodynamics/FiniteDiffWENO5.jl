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

    fl[i] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
    fr[i] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

    if lim_ZS
        # --- Zhang-Shu positivity limiter ---
        ϵθ = 1.0e-16 # small number to avoid division by zero

        fl[i] = zhang_shu_limit(fl[i], u3, u_min, u_max, ϵθ)
        fr[i] = zhang_shu_limit(fr[i], u4, u_min, u_max, ϵθ)
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


@kernel inbounds = true function upwind_update_KA_1D!(
    u, v, nx, Δx_, Δt, stag, boundary, g, O
)
    I = @index(Global, NTuple)
    I = I + O
    i = I[1]

    # Left boundary condition
    if boundary[1] == 0       # Dirichlet
        iL = clamp(i-1, 1, nx)
    elseif boundary[1] == 1   # Neumann
        iL = max(i-1, 1)
    elseif boundary[1] == 2   # Periodic
        iL = mod1(i-1, nx)
    end

    # Right boundary condition
    if boundary[2] == 0
        iR = clamp(i+1, 1, nx)
    elseif boundary[2] == 1
        iR = min(i+1, nx)
    elseif boundary[2] == 2
        iR = mod1(i+1, nx)
    end

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