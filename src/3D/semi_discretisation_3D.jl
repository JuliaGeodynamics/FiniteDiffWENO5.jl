function WENO_flux!(fl, fr, u, weno, nx, ny, nz, u_min, u_max)
    @unpack boundary, χ, γ, ζ, ϵ, multithreading, lim_ZS = weno

    bLx = Val(boundary[1])
    bRx = Val(boundary[2])
    bLy = Val(boundary[3])
    bRy = Val(boundary[4])
    bLz = Val(boundary[5])
    bRz = Val(boundary[6])

    ϵθ = 1.0e-18  # small number to avoid division by zero for limiter

    # fusion of loops for better performance
    @inbounds @maybe_threads multithreading for I in CartesianIndices(fl.x)
        i, j, k = Tuple(I)

        # --- x-direction reconstruction ---
        iwww = left_index(i, 3, nx, bLx)
        iww = left_index(i, 2, nx, bLx)
        iw = left_index(i, 1, nx, bLx)
        ie = right_index(i, 0, nx, bRx)
        iee = right_index(i, 1, nx, bRx)
        ieee = right_index(i, 2, nx, bRx)

        u1 = u[iwww, j, k]
        u2 = u[iww, j, k]
        u3 = u[iw, j, k]
        u4 = u[ie, j, k]
        u5 = u[iee, j, k]
        u6 = u[ieee, j, k]

        fl.x[I] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
        fr.x[I] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

        if lim_ZS
            fl.x[I] = zhang_shu_limit(fl.x[I], u3, u_min, u_max, ϵθ)
            fr.x[I] = zhang_shu_limit(fr.x[I], u4, u_min, u_max, ϵθ)
        end

        @inbounds if i < nx + 1
            # --- y-direction reconstruction ---
            jwww = left_index(j, 3, ny, bLy)
            jww = left_index(j, 2, ny, bLy)
            jw = left_index(j, 1, ny, bLy)
            je = right_index(j, 0, ny, bRy)
            jee = right_index(j, 1, ny, bRy)
            jeee = right_index(j, 2, ny, bRy)

            u1 = u[i, jwww, k]
            u2 = u[i, jww, k]
            u3 = u[i, jw, k]
            u4 = u[i, je, k]
            u5 = u[i, jee, k]
            u6 = u[i, jeee, k]

            fl.y[I] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
            fr.y[I] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

            if lim_ZS
                fl.y[I] = zhang_shu_limit(fl.y[I], u3, u_min, u_max, ϵθ)
                fr.y[I] = zhang_shu_limit(fr.y[I], u4, u_min, u_max, ϵθ)
            end

            # --- z-direction reconstruction ---
            kwww = left_index(k, 3, nz, bLz)
            kww = left_index(k, 2, nz, bLz)
            kw = left_index(k, 1, nz, bLz)
            ke = right_index(k, 0, nz, bRz)
            kee = right_index(k, 1, nz, bRz)
            keee = right_index(k, 2, nz, bRz)

            u1 = u[i, j, kwww]
            u2 = u[i, j, kww]
            u3 = u[i, j, kw]
            u4 = u[i, j, ke]
            u5 = u[i, j, kee]
            u6 = u[i, j, keee]

            fl.z[I] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
            fr.z[I] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

            if lim_ZS
                fl.z[I] = zhang_shu_limit(fl.z[I], u3, u_min, u_max, ϵθ)
                fr.z[I] = zhang_shu_limit(fr.z[I], u4, u_min, u_max, ϵθ)
            end
        end
    end

    # last column for y (top boundary in j)
    @inbounds @maybe_threads multithreading for i in axes(fr.y, 1)
        @inbounds for k in axes(fr.y, 3)
            j = ny + 1

            jwww = left_index(j, 3, ny, bLy)
            jww = left_index(j, 2, ny, bLy)
            jw = left_index(j, 1, ny, bLy)
            je = right_index(j, 0, ny, bRy)
            jee = right_index(j, 1, ny, bRy)
            jeee = right_index(j, 2, ny, bRy)

            u1 = u[i, jwww, k]
            u2 = u[i, jww, k]
            u3 = u[i, jw, k]
            u4 = u[i, je, k]
            u5 = u[i, jee, k]
            u6 = u[i, jeee, k]

            fl.y[i, j, k] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
            fr.y[i, j, k] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

            if lim_ZS
                fl.y[i, j, k] = zhang_shu_limit(fl.y[i, j, k], u3, u_min, u_max, ϵθ)
                fr.y[i, j, k] = zhang_shu_limit(fr.y[i, j, k], u4, u_min, u_max, ϵθ)
            end
        end
    end

    # last column for z (top boundary in k)
    @inbounds @maybe_threads multithreading for i in axes(fr.z, 1)
        @inbounds for j in axes(fr.z, 2)
            k = nz + 1

            kwww = left_index(k, 3, nz, bLz)
            kww = left_index(k, 2, nz, bLz)
            kw = left_index(k, 1, nz, bLz)
            ke = right_index(k, 0, nz, bRz)
            kee = right_index(k, 1, nz, bRz)
            keee = right_index(k, 2, nz, bRz)

            u1 = u[i, j, kwww]
            u2 = u[i, j, kww]
            u3 = u[i, j, kw]
            u4 = u[i, j, ke]
            u5 = u[i, j, kee]
            u6 = u[i, j, keee]

            fl.z[i, j, k] = weno5_reconstruction_upwind(u1, u2, u3, u4, u5, χ, γ, ζ, ϵ)
            fr.z[i, j, k] = weno5_reconstruction_downwind(u2, u3, u4, u5, u6, χ, γ, ζ, ϵ)

            if lim_ZS
                fl.z[i, j, k] = zhang_shu_limit(fl.z[i, j, k], u3, u_min, u_max, ϵθ)
                fr.z[i, j, k] = zhang_shu_limit(fr.z[i, j, k], u4, u_min, u_max, ϵθ)
            end
        end
    end

    return nothing
end

function semi_discretisation_weno5!(du::T, v, weno::WENOScheme, Δx_, Δy_, Δz_) where {T <: AbstractArray{<:Real, 3}}

    @unpack fl, fr, stag, multithreading = weno

    # use staggered grid or not for the velocities
    if stag
        @inbounds @maybe_threads multithreading for I in CartesianIndices(du)

            i, j, k = Tuple(I)

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
                (
                max(v.z[i, j, k + 1], 0) * fl.z[i, j, k + 1] +
                    min(v.z[i, j, k + 1], 0) * fr.z[i, j, k + 1] -
                    max(v.z[I], 0) * fl.z[I] -
                    min(v.z[I], 0) * fr.z[I]
            ) * Δz_
        end
    else
        @inbounds @maybe_threads multithreading for I in CartesianIndices(du)

            i, j, k = Tuple(I)

            du[I] = @muladd max(v.x[I], 0) * (fl.x[i + 1, j, k] - fl.x[I]) * Δx_ +
                min(v.x[I], 0) * (fr.x[i + 1, j, k] - fr.x[I]) * Δx_ +
                max(v.y[I], 0) * (fl.y[i, j + 1, k] - fl.y[I]) * Δy_ +
                min(v.y[I], 0) * (fr.y[i, j + 1, k] - fr.y[I]) * Δy_ +
                max(v.z[I], 0) * (fl.z[i, j, k + 1] - fl.z[I]) * Δz_ +
                min(v.z[I], 0) * (fr.z[i, j, k + 1] - fr.z[I]) * Δz_
        end
    end

    return nothing
end
