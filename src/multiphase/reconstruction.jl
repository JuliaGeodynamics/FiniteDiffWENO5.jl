# Simultaneous WENO5-Z reconstruction of a phase vector.
#
# Every phase at one face state shares a single set of nonlinear weights. Because each
# stencil candidate is exact for constants (its coefficients sum to one) and the weights
# are normalised, sharing them makes the reconstructed composition sum to one:
#
#     Σₖ ϕₖ,face = Σₖ Σᵣ ωᵣ sᵣ(ϕₖ) = Σᵣ ωᵣ Σₖ sᵣ(ϕₖ) = Σᵣ ωᵣ · 1 = 1
#
# Reconstructing each phase with its own weights makes ωᵣ depend on k, the sum over r no
# longer factors, and the identity fails. That is the entire reason these helpers exist
# rather than a loop over the scalar routines.

"""
    combined_weno_betas(stencils, χ)

Phase-averaged WENO smoothness indicators.

`stencils` is an `NTuple{NP}` of five-point stencils, one per phase, all sampled at the
same positions. The arithmetic mean is symmetric under phase permutation and keeps the
indicator magnitude independent of the phase count, so the WENO-Z `τ` and the `ϵ` floor
behave as they do in the scalar scheme.
"""
@inline function combined_weno_betas(stencils::NTuple{NP, <:NTuple{5, Any}}, χ) where {NP}
    β1 = zero(eltype(first(stencils)))
    β2 = β1
    β3 = β1
    for k in 1:NP
        b1, b2, b3 = weno_betas(stencils[k]..., χ)
        β1 += b1
        β2 += b2
        β3 += b3
    end
    scale = inv(oftype(β1, NP))
    return β1 * scale, β2 * scale, β3 * scale
end

"""
    shared_weights_upwind(β1, β2, β3, γ, ϵ)

Normalised WENO-Z weights for the upwind (left) face state, built from indicators shared
by every phase. Uses the same Borges et al. (2008) `α` formulation as the scalar scheme.
"""
@inline function shared_weights_upwind(β1, β2, β3, γ, ϵ)
    α1, α2, α3 = weno_alphas_upwind(β1, β2, β3, γ, ϵ)
    _αsum = inv(α1 + α2 + α3)
    return α1 * _αsum, α2 * _αsum, α3 * _αsum
end

"""
    shared_weights_downwind(β1, β2, β3, γ, ϵ)

Normalised WENO-Z weights for the downwind (right) face state.
"""
@inline function shared_weights_downwind(β1, β2, β3, γ, ϵ)
    α1, α2, α3 = weno_alphas_downwind(β1, β2, β3, γ, ϵ)
    _αsum = inv(α1 + α2 + α3)
    return α1 * _αsum, α2 * _αsum, α3 * _αsum
end

@inline function combine_candidates_upwind(stencil::NTuple{5, Any}, ω1, ω2, ω3, ζ)
    s1, s2, s3 = stencil_candidate_upwind(stencil..., ζ)
    return @muladd ω1 * s1 + ω2 * s2 + ω3 * s3
end

@inline function combine_candidates_downwind(stencil::NTuple{5, Any}, ω1, ω2, ω3, ζ)
    s1, s2, s3 = stencil_candidate_downwind(stencil..., ζ)
    return @muladd ω1 * s1 + ω2 * s2 + ω3 * s3
end

"""
    multiphase_reconstruction_upwind(stencils, χ, γ, ζ, ϵ)

Reconstruct every phase's upwind face state from one shared set of WENO-Z weights.
Returns an `NTuple{NP}` whose components sum to one whenever the stencil values do.
"""
@inline function multiphase_reconstruction_upwind(stencils::NTuple{NP, <:NTuple{5, Any}}, χ, γ, ζ, ϵ) where {NP}
    β1, β2, β3 = combined_weno_betas(stencils, χ)
    ω1, ω2, ω3 = shared_weights_upwind(β1, β2, β3, γ, ϵ)
    return ntuple(k -> combine_candidates_upwind(stencils[k], ω1, ω2, ω3, ζ), Val(NP))
end

"""
    multiphase_reconstruction_downwind(stencils, χ, γ, ζ, ϵ)

Downwind counterpart of [`multiphase_reconstruction_upwind`](@ref). The left and right
face states each derive their own shared weights from their own five-point stencils.
"""
@inline function multiphase_reconstruction_downwind(stencils::NTuple{NP, <:NTuple{5, Any}}, χ, γ, ζ, ϵ) where {NP}
    β1, β2, β3 = combined_weno_betas(stencils, χ)
    ω1, ω2, ω3 = shared_weights_downwind(β1, β2, β3, γ, ϵ)
    return ntuple(k -> combine_candidates_downwind(stencils[k], ω1, ω2, ω3, ζ), Val(NP))
end
