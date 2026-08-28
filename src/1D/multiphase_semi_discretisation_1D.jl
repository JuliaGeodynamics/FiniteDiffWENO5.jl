"""
    multiphase_WENO_flux!(state, scheme, nx)

Reconstruct both one-sided face states for every phase from one shared set of WENO-Z
weights per face state, then apply one common simplex limiter coefficient to each
reconstructed composition.

The left face state uses stencil samples 1:5 and the right uses 2:6, exactly as the
scalar `WENO_flux!`. Each derives its own shared weights from its own five samples. The
limiter donors are the adjacent cell averages: the cell left of the face for `fl`, the
cell right of it for `fr`, matching the scalar Zhang-Shu convention.
"""
function multiphase_WENO_flux!(state, scheme::MultiphaseWENOScheme{T, NP}, nx) where {T, NP}
    (; fl, fr, boundary, χ, γ, ζ, ϵ, multithreading) = scheme

    bL = boundary[1]
    bR = boundary[2]
    valNP = Val(NP)

    @inbounds @maybe_threads multithreading for i in axes(fl.x[1], 1)
        iwww = left_index(i, 3, nx, bL)
        iww = left_index(i, 2, nx, bL)
        iw = left_index(i, 1, nx, bL)
        ie = right_index(i, 0, nx, bR)
        iee = right_index(i, 1, nx, bR)
        ieee = right_index(i, 2, nx, bR)

        # gather one five-point stencil per phase, sampled at identical positions
        stencil_l = ntuple(
            k -> (state[k][iwww], state[k][iww], state[k][iw], state[k][ie], state[k][iee]),
            valNP,
        )
        stencil_r = ntuple(
            k -> (state[k][iww], state[k][iw], state[k][ie], state[k][iee], state[k][ieee]),
            valNP,
        )

        # shared weights make the reconstructed composition sum to one
        up = multiphase_reconstruction_upwind(stencil_l, χ, γ, ζ, ϵ)
        dn = multiphase_reconstruction_downwind(stencil_r, χ, γ, ζ, ϵ)

        # one common coefficient per face state keeps that sum intact
        up = limit_simplex(up, ntuple(k -> state[k][iw], valNP))
        dn = limit_simplex(dn, ntuple(k -> state[k][ie], valNP))

        for k in 1:NP
            fl.x[k][i] = up[k]
            fr.x[k][i] = dn[k]
        end
    end

    apply_multiphase_inflow_boundaries!(fl, fr, boundary)
    return nothing
end

"""
    multiphase_semi_discretisation!(du, state, v, scheme, Δx_)

Material-fraction semi-discretisation in 1D.

For staggered velocity this discretises `∂tϕₖ + ∇·(vϕₖ) = ϕₖ∇·v`, using the *same*
two-point divergence for the source as the phase-flux sum telescopes into. That is what
makes the two cancel discretely: with `Σₖ fl[k][i] = Σₖ fr[k][i] = 1`,

    Σₖ fluxdivₖ = (v[i+1] − v[i]) · Δx⁻¹ = divv[i]

so `Σₖ duₖ = 0` exactly. A wider divergence stencil breaks the cancellation.

For collocated velocity the one-sided differences already telescope to zero over the
phases, so no source is formed and `scheme.divv` is `nothing`.
"""
function multiphase_semi_discretisation!(du, state, v, scheme::MultiphaseWENOScheme{T, NP}, Δx_) where {T, NP}
    (; fl, fr, stag, divv, multithreading) = scheme

    if stag
        @inbounds @maybe_threads multithreading for i in axes(du[1], 1)
            d = (v.x[i + 1] - v.x[i]) * Δx_
            divv[i] = d
            for k in 1:NP
                fluxdiv = (
                    max(v.x[i + 1], 0) * fl.x[k][i + 1] +
                        min(v.x[i + 1], 0) * fr.x[k][i + 1] -
                        max(v.x[i], 0) * fl.x[k][i] -
                        min(v.x[i], 0) * fr.x[k][i]
                ) * Δx_
                du[k][i] = @muladd fluxdiv - state[k][i] * d
            end
        end
    else
        @inbounds @maybe_threads multithreading for i in axes(du[1], 1)
            for k in 1:NP
                du[k][i] = @muladd max(v.x[i], 0) * (fl.x[k][i + 1] - fl.x[k][i]) * Δx_ +
                    min(v.x[i], 0) * (fr.x[k][i + 1] - fr.x[k][i]) * Δx_
            end
        end
    end

    return nothing
end
