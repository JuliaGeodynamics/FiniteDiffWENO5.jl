function multiphase_WENO_flux!(state, scheme::MultiphaseWENOScheme{T, NP}, nx, ny, nz) where {T, NP}
    (; fl, fr, boundary, χ, γ, ζ, ϵ, multithreading) = scheme
    bLx, bRx, bLy, bRy, bLz, bRz = boundary
    valNP = Val(NP)

    @inbounds @maybe_threads multithreading for I in CartesianIndices(fl.x[1])
        i, j, k = Tuple(I)
        iwww = left_index(i, 3, nx, bLx); iww = left_index(i, 2, nx, bLx)
        iw = left_index(i, 1, nx, bLx); ie = right_index(i, 0, nx, bRx)
        iee = right_index(i, 1, nx, bRx); ieee = right_index(i, 2, nx, bRx)
        sl = ntuple(q -> (state[q][iwww, j, k], state[q][iww, j, k],
            state[q][iw, j, k], state[q][ie, j, k], state[q][iee, j, k]), valNP)
        sr = ntuple(q -> (state[q][iww, j, k], state[q][iw, j, k],
            state[q][ie, j, k], state[q][iee, j, k], state[q][ieee, j, k]), valNP)
        up = limit_simplex(multiphase_reconstruction_upwind(sl, χ, γ, ζ, ϵ),
            ntuple(q -> state[q][iw, j, k], valNP))
        dn = limit_simplex(multiphase_reconstruction_downwind(sr, χ, γ, ζ, ϵ),
            ntuple(q -> state[q][ie, j, k], valNP))
        for q in 1:NP
            fl.x[q][I] = up[q]; fr.x[q][I] = dn[q]
        end
    end

    @inbounds @maybe_threads multithreading for I in CartesianIndices(fl.y[1])
        i, j, k = Tuple(I)
        jwww = left_index(j, 3, ny, bLy); jww = left_index(j, 2, ny, bLy)
        jw = left_index(j, 1, ny, bLy); je = right_index(j, 0, ny, bRy)
        jee = right_index(j, 1, ny, bRy); jeee = right_index(j, 2, ny, bRy)
        sl = ntuple(q -> (state[q][i, jwww, k], state[q][i, jww, k],
            state[q][i, jw, k], state[q][i, je, k], state[q][i, jee, k]), valNP)
        sr = ntuple(q -> (state[q][i, jww, k], state[q][i, jw, k],
            state[q][i, je, k], state[q][i, jee, k], state[q][i, jeee, k]), valNP)
        up = limit_simplex(multiphase_reconstruction_upwind(sl, χ, γ, ζ, ϵ),
            ntuple(q -> state[q][i, jw, k], valNP))
        dn = limit_simplex(multiphase_reconstruction_downwind(sr, χ, γ, ζ, ϵ),
            ntuple(q -> state[q][i, je, k], valNP))
        for q in 1:NP
            fl.y[q][I] = up[q]; fr.y[q][I] = dn[q]
        end
    end

    @inbounds @maybe_threads multithreading for I in CartesianIndices(fl.z[1])
        i, j, k = Tuple(I)
        kwww = left_index(k, 3, nz, bLz); kww = left_index(k, 2, nz, bLz)
        kw = left_index(k, 1, nz, bLz); ke = right_index(k, 0, nz, bRz)
        kee = right_index(k, 1, nz, bRz); keee = right_index(k, 2, nz, bRz)
        sl = ntuple(q -> (state[q][i, j, kwww], state[q][i, j, kww],
            state[q][i, j, kw], state[q][i, j, ke], state[q][i, j, kee]), valNP)
        sr = ntuple(q -> (state[q][i, j, kww], state[q][i, j, kw],
            state[q][i, j, ke], state[q][i, j, kee], state[q][i, j, keee]), valNP)
        up = limit_simplex(multiphase_reconstruction_upwind(sl, χ, γ, ζ, ϵ),
            ntuple(q -> state[q][i, j, kw], valNP))
        dn = limit_simplex(multiphase_reconstruction_downwind(sr, χ, γ, ζ, ϵ),
            ntuple(q -> state[q][i, j, ke], valNP))
        for q in 1:NP
            fl.z[q][I] = up[q]; fr.z[q][I] = dn[q]
        end
    end

    apply_multiphase_inflow_boundaries!(fl, fr, boundary)
    return nothing
end

function multiphase_semi_discretisation!(
        du, state, v, scheme::MultiphaseWENOScheme{T, NP}, Δx_, Δy_, Δz_) where {T, NP}
    (; fl, fr, stag, divv, multithreading) = scheme
    if stag
        @inbounds @maybe_threads multithreading for I in CartesianIndices(du[1])
            i, j, k = Tuple(I)
            d = @muladd (v.x[i + 1, j, k] - v.x[i, j, k]) * Δx_ +
                (v.y[i, j + 1, k] - v.y[i, j, k]) * Δy_ +
                (v.z[i, j, k + 1] - v.z[i, j, k]) * Δz_
            divv[I] = d
            for q in 1:NP
                fluxdiv = @muladd (
                    max(v.x[i + 1, j, k], 0) * fl.x[q][i + 1, j, k] +
                        min(v.x[i + 1, j, k], 0) * fr.x[q][i + 1, j, k] -
                        max(v.x[I], 0) * fl.x[q][I] - min(v.x[I], 0) * fr.x[q][I]
                ) * Δx_ + (
                    max(v.y[i, j + 1, k], 0) * fl.y[q][i, j + 1, k] +
                        min(v.y[i, j + 1, k], 0) * fr.y[q][i, j + 1, k] -
                        max(v.y[I], 0) * fl.y[q][I] - min(v.y[I], 0) * fr.y[q][I]
                ) * Δy_ + (
                    max(v.z[i, j, k + 1], 0) * fl.z[q][i, j, k + 1] +
                        min(v.z[i, j, k + 1], 0) * fr.z[q][i, j, k + 1] -
                        max(v.z[I], 0) * fl.z[q][I] - min(v.z[I], 0) * fr.z[q][I]
                ) * Δz_
                du[q][I] = @muladd fluxdiv - state[q][I] * d
            end
        end
    else
        @inbounds @maybe_threads multithreading for I in CartesianIndices(du[1])
            i, j, k = Tuple(I)
            for q in 1:NP
                du[q][I] = @muladd max(v.x[I], 0) *
                    (fl.x[q][i + 1, j, k] - fl.x[q][I]) * Δx_ +
                    min(v.x[I], 0) * (fr.x[q][i + 1, j, k] - fr.x[q][I]) * Δx_ +
                    max(v.y[I], 0) * (fl.y[q][i, j + 1, k] - fl.y[q][I]) * Δy_ +
                    min(v.y[I], 0) * (fr.y[q][i, j + 1, k] - fr.y[q][I]) * Δy_ +
                    max(v.z[I], 0) * (fl.z[q][i, j, k + 1] - fl.z[q][I]) * Δz_ +
                    min(v.z[I], 0) * (fr.z[q][i, j, k + 1] - fr.z[q][I]) * Δz_
            end
        end
    end
    return nothing
end
