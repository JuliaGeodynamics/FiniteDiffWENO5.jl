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
            u_avg = u3

            θ_fl = min(
                1.0,
                abs((u_max - u_avg) / (fl.x[i] - u_avg + ϵθ)),
                abs((u_avg - u_min) / (u_avg - fl.x[i] + ϵθ))
            )
            # apply limiter
            fl.x[i] = θ_fl * (fl.x[i] - u_avg) + u_avg

            # separate averages for left and right
            u_avg = u4

            θ_fr = min(
                1.0,
                abs((u_max - u_avg) / (fr.x[i] - u_avg + ϵθ)),
                abs((u_avg - u_min) / (u_avg - fr.x[i] + ϵθ))
            )
            fr.x[i] = θ_fr * (fr.x[i] - u_avg) + u_avg
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
