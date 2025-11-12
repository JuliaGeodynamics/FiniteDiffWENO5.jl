function WENO_flux!(fl, fr, u, weno, nx, ny, u_max, u_min)
    @unpack boundary, χ, γ, ζ, ϵ, multithreading, lim_ZS = weno

    bLx = Val(boundary[1])
    bRx = Val(boundary[2])
    bLy = Val(boundary[3])
    bRy = Val(boundary[4])

    ϵθ = 1.0e-18  # small number to avoid division by zero for limiter

    @inbounds @maybe_threads multithreading for I in CartesianIndices(fl.x)
        i, j = Tuple(I)

        # --- x-direction reconstruction ---
        iwww = left_index(i, 3, nx, bLx)
        iww = left_index(i, 2, nx, bLx)
        iw = left_index(i, 1, nx, bLx)
        ie = right_index(i, 0, nx, bRx)
        iee = right_index(i, 1, nx, bRx)
        ieee = right_index(i, 2, nx, bRx)

        u1 = u[iwww, j]; u2 = u[iww, j]; u3 = u[iw, j]
        u4 = u[ie, j]; u5 = u[iee, j]; u6 = u[ieee, j]

        fl.x[I] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr.x[I] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

        if lim_ZS
            u_avg = u3

            θ_fl = min(
                1.0,
                abs((u_max - u_avg) / (fl.x[I] - u_avg + ϵθ)),
                abs((u_avg - u_min) / (u_avg - fl.x[I] + ϵθ))
            )
            fl.x[I] = @muladd θ_fl * (fl.x[I] - u_avg) + u_avg

            u_avg = u4

            θ_fr = min(
                1.0,
                abs((u_max - u_avg) / (fr.x[I] - u_avg + ϵθ)),
                abs((u_avg - u_min) / (u_avg - fr.x[I] + ϵθ))
            )
            fr.x[I] = @muladd θ_fr * (fr.x[I] - u_avg) + u_avg
        end

        # --- y-direction reconstruction ---
        if i <= nx  # avoid last column (handled separately)
            jwww = left_index(j, 3, ny, bLy)
            jww = left_index(j, 2, ny, bLy)
            jw = left_index(j, 1, ny, bLy)
            je = right_index(j, 0, ny, bRy)
            jee = right_index(j, 1, ny, bRy)
            jeee = right_index(j, 2, ny, bRy)

            u1 = u[i, jwww]; u2 = u[i, jww]; u3 = u[i, jw]
            u4 = u[i, je]; u5 = u[i, jee]; u6 = u[i, jeee]

            fl.y[I] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
            fr.y[I] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

            if lim_ZS
                fl.y[I] = zhang_shu_limit(fl.y[I], u3, u_min, u_max, ϵθ)
                fr.y[I] = zhang_shu_limit(fr.y[I], u4, u_min, u_max, ϵθ)
            end
        end
    end

    # Handle last row for y-direction (top boundary)
    return @inbounds @maybe_threads multithreading for i in axes(fl.y, 1)
        j = ny + 1

        jwww = left_index(j, 3, ny, bLy)
        jww = left_index(j, 2, ny, bLy)
        jw = left_index(j, 1, ny, bLy)
        je = right_index(j, 0, ny, bRy)
        jee = right_index(j, 1, ny, bRy)
        jeee = right_index(j, 2, ny, bRy)

        u1 = u[i, jwww]; u2 = u[i, jww]; u3 = u[i, jw]
        u4 = u[i, je]; u5 = u[i, jee]; u6 = u[i, jeee]

        fl.y[i, j] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr.y[i, j] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

        if lim_ZS
            fl.y[i, j] = zhang_shu_limit(fl.y[i, j], u3, u_min, u_max, ϵθ)
            fr.y[i, j] = zhang_shu_limit(fr.y[i, j], u4, u_min, u_max, ϵθ)
        end
    end
end

function semi_discretisation_weno5!(du::T, v, weno::WENOScheme, Δx_, Δy_) where {T <: AbstractArray{<:Real, 2}}

    @unpack fl, fr, stag, multithreading = weno

    # use staggered grid or not for the velocities
    if stag
        @inbounds @maybe_threads multithreading for I in CartesianIndices(du)

            i, j = Tuple(I)

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
        end
    else
        @inbounds @maybe_threads multithreading for I in CartesianIndices(du)

            i, j = Tuple(I)

            du[I] = @muladd max(v.x[I], 0) * (fl.x[i + 1, j] - fl.x[I]) * Δx_ +
                min(v.x[I], 0) * (fr.x[i + 1, j] - fr.x[I]) * Δx_ +
                max(v.y[I], 0) * (fl.y[i, j + 1] - fl.y[I]) * Δy_ +
                min(v.y[I], 0) * (fr.y[i, j + 1] - fr.y[I]) * Δy_
        end
    end

    return nothing
end

function upwind_update_2D!(
    u, v, weno, nx, ny, Δx_, Δy_, Δt
)
    @unpack boundary, stag, multithreading = weno

    bLx = Val(boundary[1])
    bRx = Val(boundary[2])
    bLy = Val(boundary[3])
    bRy = Val(boundary[4])

    @inbounds @maybe_threads multithreading for I in CartesianIndices(u)
        i, j = Tuple(I)

        iLx = left_index(i - 1, 0, nx, bLx)
        iRx = right_index(i + 1, 0, nx, bRx)
        jLy = left_index(j - 1, 0, ny, bLy)
        jRy = right_index(j + 1, 0, ny, bRy)

        if stag
            u[i, j] -= Δt * (
                (max(v.x[i, j], 0) * (u[i, j] - u[iLx, j]) +
                 min(v.x[iRx, j], 0) * (u[iRx, j] - u[i, j])) * Δx_ +
                (max(v.y[i, j], 0) * (u[i, j] - u[i, jLy]) +
                 min(v.y[i, jRy], 0) * (u[i, jRy] - u[i, j])) * Δy_
            )
        else
            u[i, j] -= Δt * (
                (max(v.x[i, j], 0) * (u[i, j] - u[iLx, j]) +
                 min(v.x[i, j], 0) * (u[iRx, j] - u[i, j])) * Δx_ +
                (max(v.y[i, j], 0) * (u[i, j] - u[i, jLy]) +
                 min(v.y[i, j], 0) * (u[i, jRy] - u[i, j])) * Δy_
            )
        end
    end

    return u
end
