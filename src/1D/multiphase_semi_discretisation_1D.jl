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

"""Evaluate the shared-weight multiphase material operator `v ∂ϕₖ/∂x`."""
function multiphase_material_semi_discretisation!(
        du, vcenter, scheme::MultiphaseWENOScheme{T, NP}, Δx_
    ) where {T, NP}
    (; fl, fr, multithreading) = scheme
    size(vcenter.x) == size(du[1]) || throw(
        DimensionMismatch(
            "prepared x velocity has size $(size(vcenter.x)), expected $(size(du[1]))",
        )
    )
    @inbounds @maybe_threads multithreading for i in eachindex(du[1])
        v = vcenter.x[i]
        for k in 1:NP
            du[k][i] = @muladd max(v, 0) * (fl.x[k][i + 1] - fl.x[k][i]) * Δx_ +
                min(v, 0) * (fr.x[k][i + 1] - fr.x[k][i]) * Δx_
        end
    end
    return nothing
end
