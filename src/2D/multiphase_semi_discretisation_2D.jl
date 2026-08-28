function multiphase_WENO_flux!(state, scheme::MultiphaseWENOScheme{T, NP}, nx, ny) where {T, NP}
    (; fl, fr, boundary, χ, γ, ζ, ϵ, multithreading) = scheme
    bLx, bRx, bLy, bRy = boundary
    valNP = Val(NP)

    @inbounds @maybe_threads multithreading for I in CartesianIndices(fl.x[1])
        i, j = Tuple(I)
        iwww = left_index(i, 3, nx, bLx)
        iww = left_index(i, 2, nx, bLx)
        iw = left_index(i, 1, nx, bLx)
        ie = right_index(i, 0, nx, bRx)
        iee = right_index(i, 1, nx, bRx)
        ieee = right_index(i, 2, nx, bRx)

        stencil_l = ntuple(
            k -> (
                state[k][iwww, j], state[k][iww, j], state[k][iw, j],
                state[k][ie, j], state[k][iee, j],
            ), valNP
        )
        stencil_r = ntuple(
            k -> (
                state[k][iww, j], state[k][iw, j], state[k][ie, j],
                state[k][iee, j], state[k][ieee, j],
            ), valNP
        )
        up = multiphase_reconstruction_upwind(stencil_l, χ, γ, ζ, ϵ)
        dn = multiphase_reconstruction_downwind(stencil_r, χ, γ, ζ, ϵ)
        up = limit_simplex(up, ntuple(k -> state[k][iw, j], valNP))
        dn = limit_simplex(dn, ntuple(k -> state[k][ie, j], valNP))
        for k in 1:NP
            fl.x[k][I] = up[k]
            fr.x[k][I] = dn[k]
        end
    end

    @inbounds @maybe_threads multithreading for I in CartesianIndices(fl.y[1])
        i, j = Tuple(I)
        jwww = left_index(j, 3, ny, bLy)
        jww = left_index(j, 2, ny, bLy)
        jw = left_index(j, 1, ny, bLy)
        je = right_index(j, 0, ny, bRy)
        jee = right_index(j, 1, ny, bRy)
        jeee = right_index(j, 2, ny, bRy)

        stencil_l = ntuple(
            k -> (
                state[k][i, jwww], state[k][i, jww], state[k][i, jw],
                state[k][i, je], state[k][i, jee],
            ), valNP
        )
        stencil_r = ntuple(
            k -> (
                state[k][i, jww], state[k][i, jw], state[k][i, je],
                state[k][i, jee], state[k][i, jeee],
            ), valNP
        )
        up = multiphase_reconstruction_upwind(stencil_l, χ, γ, ζ, ϵ)
        dn = multiphase_reconstruction_downwind(stencil_r, χ, γ, ζ, ϵ)
        up = limit_simplex(up, ntuple(k -> state[k][i, jw], valNP))
        dn = limit_simplex(dn, ntuple(k -> state[k][i, je], valNP))
        for k in 1:NP
            fl.y[k][I] = up[k]
            fr.y[k][I] = dn[k]
        end
    end

    apply_multiphase_inflow_boundaries!(fl, fr, boundary)
    return nothing
end

function multiphase_semi_discretisation!(
        du, state, v, scheme::MultiphaseWENOScheme{T, NP}, Δx_, Δy_
    ) where {T, NP}
    (; fl, fr, stag, divv, multithreading) = scheme

    if stag
        @inbounds @maybe_threads multithreading for I in CartesianIndices(du[1])
            i, j = Tuple(I)
            d = @muladd (v.x[i + 1, j] - v.x[i, j]) * Δx_ +
                (v.y[i, j + 1] - v.y[i, j]) * Δy_
            divv[I] = d
            for k in 1:NP
                fluxdiv = @muladd (
                    max(v.x[i + 1, j], 0) * fl.x[k][i + 1, j] +
                        min(v.x[i + 1, j], 0) * fr.x[k][i + 1, j] -
                        max(v.x[i, j], 0) * fl.x[k][i, j] -
                        min(v.x[i, j], 0) * fr.x[k][i, j]
                ) * Δx_ + (
                    max(v.y[i, j + 1], 0) * fl.y[k][i, j + 1] +
                        min(v.y[i, j + 1], 0) * fr.y[k][i, j + 1] -
                        max(v.y[i, j], 0) * fl.y[k][i, j] -
                        min(v.y[i, j], 0) * fr.y[k][i, j]
                ) * Δy_
                du[k][I] = @muladd fluxdiv - state[k][I] * d
            end
        end
    else
        @inbounds @maybe_threads multithreading for I in CartesianIndices(du[1])
            i, j = Tuple(I)
            for k in 1:NP
                du[k][I] = @muladd max(v.x[I], 0) *
                    (fl.x[k][i + 1, j] - fl.x[k][I]) * Δx_ +
                    min(v.x[I], 0) * (fr.x[k][i + 1, j] - fr.x[k][I]) * Δx_ +
                    max(v.y[I], 0) * (fl.y[k][i, j + 1] - fl.y[k][I]) * Δy_ +
                    min(v.y[I], 0) * (fr.y[k][i, j + 1] - fr.y[k][I]) * Δy_
            end
        end
    end
    return nothing
end
