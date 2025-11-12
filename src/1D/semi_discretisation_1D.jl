
function WENO_flux!(fl, fr, u, weno, nx, u_min, u_max)
    @unpack boundary, χ, γ, ζ, ϵ, multithreading, lim_ZS = weno

    bL = Val(boundary[1])
    bR = Val(boundary[2])

    # small number to avoid division by zero
    ϵθ = 1.0e-18

    return @inbounds @maybe_threads multithreading for i in axes(fl.x, 1)
        iwww = left_index(i, 3, nx, bL)
        iww = left_index(i, 2, nx, bL)
        iw = left_index(i, 1, nx, bL)
        ie = right_index(i, 0, nx, bR)
        iee = right_index(i, 1, nx, bR)
        ieee = right_index(i, 2, nx, bR)

        u1 = u[iwww]
        u2 = u[iww]
        u3 = u[iw]
        u4 = u[ie]
        u5 = u[iee]
        u6 = u[ieee]

        fl.x[i] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr.x[i] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

        if lim_ZS
            # --- Zhang-Shu positivity limiter ---
            # separate averages for left and right
            fl.x[i] = zhang_shu_limit(fl.x[i], u3, u_min, u_max, ϵθ)
            fr.x[i] = zhang_shu_limit(fr.x[i], u4, u_min, u_max, ϵθ)
        end
    end
end


function semi_discretisation_weno5!(du::T, v, weno::WENOScheme, Δx_) where {T <: AbstractArray{<:Real, 1}}

    @unpack fl, fr, stag, multithreading = weno

    # use staggered grid or not for the velocities
    if stag
        @inbounds @maybe_threads multithreading for i in axes(du, 1)
            du[i] = @muladd (
                max(v.x[i + 1], 0) * fl.x[i + 1] +
                    min(v.x[i + 1], 0) * fr.x[i + 1] -
                    max(v.x[i], 0) * fl.x[i] -
                    min(v.x[i], 0) * fr.x[i]
            ) * Δx_
        end
    else
        @inbounds @maybe_threads multithreading for i in axes(du, 1)
            du[i] = @muladd max(v.x[i], 0) * (fl.x[i + 1] - fl.x[i]) * Δx_ + min(v.x[i], 0) * (fr.x[i + 1] - fr.x[i]) * Δx_
        end
    end

    return nothing
end


"""
    upwind_update!(u, v, weno, nx, Δx_, Δt; stag=false)

Perform a single explicit upwind advection update on field `u`
using velocity field `v` on a staggered or collocated grid.

- If `stag = true`, velocity is assumed to be defined at cell faces.
- If `stag = false`, velocity is defined at cell centers.
- Uses the boundary conditions stored in `weno.boundary`.
"""
function upwind_update_1D!(u, v, weno, nx, Δx_, Δt)
    @unpack boundary, stag, multithreading = weno

    bL = Val(boundary[1])
    bR = Val(boundary[2])

    @inbounds @maybe_threads multithreading for i in axes(u, 1)
        iL = left_index(i - 1, 0, nx, bL)
        iR = right_index(i + 1, 0, nx, bR)

        if stag
            u[i] -= Δt * (
                max(v.x[i], 0) * (u[i] - u[iL]) +
                min(v.x[iR], 0) * (u[iR] - u[i])
            ) * Δx_
        else
            u[i] -= Δt * (
                max(v.x[i], 0) * (u[i] - u[iL]) +
                min(v.x[i], 0) * (u[iR] - u[i])
            ) * Δx_
        end
    end

    return nothing
end