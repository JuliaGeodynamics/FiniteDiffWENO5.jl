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

@kernel function WENO_semi_discretisation_weno5_KA_2D!(du, fl, fr, v, Δx_, Δy_, g, O)

    I = @index(Global, Cartesian)
    I = I + O
    i, j = I[1], I[2]

    m, n = size(du)

    if 1 <= i <= m && 1 <= j <= n
        du[I] = @muladd max(v.x[I], 0) * (fl.x[i + 1, j] - fl.x[I]) * Δx_ +
            min(v.x[I], 0) * (fr.x[i + 1, j] - fr.x[I]) * Δx_ +
            max(v.y[I], 0) * (fl.y[i, j + 1] - fl.y[I]) * Δy_ +
            min(v.y[I], 0) * (fr.y[i, j + 1] - fr.y[I]) * Δy_
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

# ENO5 face-to-centre interpolation for one velocity component along its own
# staggered direction, matching `eno5_face_to_center_direction!` on the CPU.
@kernel inbounds = true function eno5_face_to_center_KA_2D_x!(center, face, nx, periodic, g, O)
    I = @index(Global, NTuple)
    I = I + O
    i, j = I[1], I[2]
    n, m = size(center)
    if 1 <= i <= n && 1 <= j <= m
        s = FiniteDiffWENO5.eno5_stencil_start(face, CartesianIndex(i, j), 1, i, nx, periodic)
        row = FiniteDiffWENO5.eno5_row(s, i)
        value = zero(eltype(face))
        for r in 0:4
            value += row[r + 1] * FiniteDiffWENO5.eno5_face_sample(face, CartesianIndex(i, j), 1, s + r, nx, periodic)
        end
        center[i, j] = value / oftype(value, 128)
    end
end

@kernel inbounds = true function eno5_face_to_center_KA_2D_y!(center, face, ny, periodic, g, O)
    I = @index(Global, NTuple)
    I = I + O
    i, j = I[1], I[2]
    n, m = size(center)
    if 1 <= i <= n && 1 <= j <= m
        s = FiniteDiffWENO5.eno5_stencil_start(face, CartesianIndex(i, j), 2, j, ny, periodic)
        row = FiniteDiffWENO5.eno5_row(s, j)
        value = zero(eltype(face))
        for r in 0:4
            value += row[r + 1] * FiniteDiffWENO5.eno5_face_sample(face, CartesianIndex(i, j), 2, s + r, ny, periodic)
        end
        center[i, j] = value / oftype(value, 128)
    end
end

@kernel inbounds = true function linear_face_to_center_KA_2D_x!(center, face, nx, periodic, g, O)
    I = @index(Global, NTuple)
    I = I + O
    i, j = I[1], I[2]
    n, m = size(center)
    if 1 <= i <= n && 1 <= j <= m
        lo = FiniteDiffWENO5.eno5_face_sample(face, CartesianIndex(i, j), 1, i, nx, periodic)
        hi = FiniteDiffWENO5.eno5_face_sample(face, CartesianIndex(i, j), 1, i + 1, nx, periodic)
        center[i, j] = 0.5 * (lo + hi)
    end
end

@kernel inbounds = true function linear_face_to_center_KA_2D_y!(center, face, ny, periodic, g, O)
    I = @index(Global, NTuple)
    I = I + O
    i, j = I[1], I[2]
    n, m = size(center)
    if 1 <= i <= n && 1 <= j <= m
        lo = FiniteDiffWENO5.eno5_face_sample(face, CartesianIndex(i, j), 2, j, ny, periodic)
        hi = FiniteDiffWENO5.eno5_face_sample(face, CartesianIndex(i, j), 2, j + 1, ny, periodic)
        center[i, j] = 0.5 * (lo + hi)
    end
end

# Conservative split-flux reconstruction along x and y independently, matching
# `conservative_semi_discretisation_weno5!` in 2D.
@kernel inbounds = true function WENO_conservative_flux_KA_2D_x!(fl, fr, u, v, α, boundary, nx, χ, γ, ζ, ϵ, g, O)
    I = @index(Global, NTuple)
    I = I + O
    i, j = I[1], I[2]
    bL, bR = boundary[1], boundary[2]
    iwww = left_index(i, 3, nx, bL); iww = left_index(i, 2, nx, bL); iw = left_index(i, 1, nx, bL)
    ie = right_index(i, 0, nx, bR); iee = right_index(i, 1, nx, bR); ieee = right_index(i, 2, nx, bR)
    fl[i, j] = weno5_reconstruction_upwind(
        FiniteDiffWENO5.lf_split_plus(u[iwww, j], v[iwww, j], α),
        FiniteDiffWENO5.lf_split_plus(u[iww, j], v[iww, j], α),
        FiniteDiffWENO5.lf_split_plus(u[iw, j], v[iw, j], α),
        FiniteDiffWENO5.lf_split_plus(u[ie, j], v[ie, j], α),
        FiniteDiffWENO5.lf_split_plus(u[iee, j], v[iee, j], α), χ, γ, ζ, ϵ,
    )
    fr[i, j] = weno5_reconstruction_downwind(
        FiniteDiffWENO5.lf_split_minus(u[iww, j], v[iww, j], α),
        FiniteDiffWENO5.lf_split_minus(u[iw, j], v[iw, j], α),
        FiniteDiffWENO5.lf_split_minus(u[ie, j], v[ie, j], α),
        FiniteDiffWENO5.lf_split_minus(u[iee, j], v[iee, j], α),
        FiniteDiffWENO5.lf_split_minus(u[ieee, j], v[ieee, j], α), χ, γ, ζ, ϵ,
    )

    if i == 1 && bL isa FiniteDiffWENO5.PrescribedInflowBC
        fl[i, j] = FiniteDiffWENO5.lf_split_plus(FiniteDiffWENO5.inflow_value(bL, j), v[begin, j], α)
        fr[i, j] = FiniteDiffWENO5.lf_split_minus(u[begin, j], v[begin, j], α)
    end
    if i == nx + 1 && bR isa FiniteDiffWENO5.PrescribedInflowBC
        fl[i, j] = FiniteDiffWENO5.lf_split_plus(u[end, j], v[end, j], α)
        fr[i, j] = FiniteDiffWENO5.lf_split_minus(FiniteDiffWENO5.inflow_value(bR, j), v[end, j], α)
    end
end

@kernel inbounds = true function WENO_conservative_flux_KA_2D_y!(fl, fr, u, v, α, boundary, ny, χ, γ, ζ, ϵ, g, O)
    I = @index(Global, NTuple)
    I = I + O
    i, j = I[1], I[2]
    bL, bR = boundary[3], boundary[4]
    jwww = left_index(j, 3, ny, bL); jww = left_index(j, 2, ny, bL); jw = left_index(j, 1, ny, bL)
    je = right_index(j, 0, ny, bR); jee = right_index(j, 1, ny, bR); jeee = right_index(j, 2, ny, bR)
    fl[i, j] = weno5_reconstruction_upwind(
        FiniteDiffWENO5.lf_split_plus(u[i, jwww], v[i, jwww], α),
        FiniteDiffWENO5.lf_split_plus(u[i, jww], v[i, jww], α),
        FiniteDiffWENO5.lf_split_plus(u[i, jw], v[i, jw], α),
        FiniteDiffWENO5.lf_split_plus(u[i, je], v[i, je], α),
        FiniteDiffWENO5.lf_split_plus(u[i, jee], v[i, jee], α), χ, γ, ζ, ϵ,
    )
    fr[i, j] = weno5_reconstruction_downwind(
        FiniteDiffWENO5.lf_split_minus(u[i, jww], v[i, jww], α),
        FiniteDiffWENO5.lf_split_minus(u[i, jw], v[i, jw], α),
        FiniteDiffWENO5.lf_split_minus(u[i, je], v[i, je], α),
        FiniteDiffWENO5.lf_split_minus(u[i, jee], v[i, jee], α),
        FiniteDiffWENO5.lf_split_minus(u[i, jeee], v[i, jeee], α), χ, γ, ζ, ϵ,
    )

    if j == 1 && bL isa FiniteDiffWENO5.PrescribedInflowBC
        fl[i, j] = FiniteDiffWENO5.lf_split_plus(FiniteDiffWENO5.inflow_value(bL, i), v[i, begin], α)
        fr[i, j] = FiniteDiffWENO5.lf_split_minus(u[i, begin], v[i, begin], α)
    end
    if j == ny + 1 && bR isa FiniteDiffWENO5.PrescribedInflowBC
        fl[i, j] = FiniteDiffWENO5.lf_split_plus(u[i, end], v[i, end], α)
        fr[i, j] = FiniteDiffWENO5.lf_split_minus(FiniteDiffWENO5.inflow_value(bR, i), v[i, end], α)
    end
end

@kernel inbounds = true function WENO_conservative_divergence_KA_2D!(du, fl, fr, Δx_, Δy_, g, O)
    I = @index(Global, Cartesian)
    I = I + O
    i, j = I[1], I[2]
    du[I] = @muladd ((fl.x[i + 1, j] + fr.x[i + 1, j]) - (fl.x[I] + fr.x[I])) * Δx_ +
        ((fl.y[i, j + 1] + fr.y[i, j + 1]) - (fl.y[I] + fr.y[I])) * Δy_
end

"""Device counterpart of `prepare_velocity!` in 2D."""
function prepare_velocity_KA_2D!(weno, v, nx, ny, backend)
    weno.stag || return v
    weno.vcenter === nothing && return v

    px, py = weno.vperiodic.x, weno.vperiodic.y
    FiniteDiffWENO5.validate_staggered_velocity!(weno.vcenter, v; periodic = weno.vperiodic)

    cx, cy = weno.vcenter.x, weno.vcenter.y
    kx = nx >= FiniteDiffWENO5.eno5_minimum_cells(px) ?
        eno5_face_to_center_KA_2D_x!(backend) : linear_face_to_center_KA_2D_x!(backend)
    ky = ny >= FiniteDiffWENO5.eno5_minimum_cells(py) ?
        eno5_face_to_center_KA_2D_y!(backend) : linear_face_to_center_KA_2D_y!(backend)
    kx(cx, v.x, nx, px, nothing, Offset0, ndrange = size(cx))
    ky(cy, v.y, ny, py, nothing, Offset0, ndrange = size(cy))
    synchronize(backend)
    return (; x = cx, y = cy)
end
