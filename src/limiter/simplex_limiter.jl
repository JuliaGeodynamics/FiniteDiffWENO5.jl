# Vector Zhang-Shu limiter for states constrained to the probability simplex.
#
# The scalar `zhang_shu_limit` scales each field independently. That is correct for
# unrelated fields but breaks `Σₖϕₖ = 1`, because independent coefficients turn a
# constrained vector into an unconstrained one. Here a single coefficient θ is shared by
# every component, so the limited state is a convex combination of two simplex points:
#
#     Σₖ limitedₖ = Σₖ donorₖ + θ(Σₖ highₖ − Σₖ donorₖ) = 1 + θ(1 − 1) = 1
#
# which holds for any θ whatsoever.

"""
    simplex_limiter_coefficient(high, donor)

Largest `θ ∈ [0,1]` such that `donor + θ*(high - donor)` has every component inside
`[0,1]`, given a `donor` that already lies in the simplex.

`high` is the unlimited high-order face state and `donor` the adjacent cell average.
Returns `one(T)` when `high` is already admissible.
"""
@inline function simplex_limiter_coefficient(high::Tuple{T, Vararg{T, M}}, donor::Tuple{T, Vararg{T, M}}) where {T, M}
    θ = one(T)
    for k in 1:(M + 1)
        hk = high[k]
        dk = donor[k]
        if hk < zero(T)
            # donor ≥ 0 and high < 0, so the denominator is strictly positive
            θ = min(θ, dk / (dk - hk))
        elseif hk > one(T)
            # donor ≤ 1 and high > 1, so the denominator is strictly positive
            θ = min(θ, (one(T) - dk) / (hk - dk))
        end
    end
    return clamp(θ, zero(T), one(T))
end

"""
    limit_simplex(high, donor, θ)

Blend `high` toward `donor` with the single shared coefficient `θ`.

The expression is written as `donor + θ*(high - donor)` rather than a fused
`muladd`, so the result is bitwise reproducible from the returned `θ`.
"""
@inline function limit_simplex(high::Tuple{T, Vararg{T, M}}, donor::Tuple{T, Vararg{T, M}}, θ) where {T, M}
    return ntuple(k -> donor[k] + θ * (high[k] - donor[k]), Val(M + 1))
end

"""
    limit_simplex(high, donor)

Convenience wrapper computing the shared coefficient and applying it in one call.
"""
@inline function limit_simplex(high::Tuple{T, Vararg{T, M}}, donor::Tuple{T, Vararg{T, M}}) where {T, M}
    return limit_simplex(high, donor, simplex_limiter_coefficient(high, donor))
end

"""
    simplex_rk_stage(initial, stage, du, a, b, Δt)

One SSP-RK3 sub-stage for a phase vector already on the simplex: `stage` is limited
toward the forward-Euler candidate exactly as `limit_simplex` does for a face state
(`stage` playing the role of `donor`, the candidate playing the role of `high`), then
the SSP convex combination `a*initial + b*limited` is taken.

This is the single place the RK-stage limiter formula is computed; every dimension's
`WENO_step!` (CPU and, via the mirrored GPU kernel, KA/Chmy) calls this instead of
re-deriving `θ` by hand at each stage, so the `clamp` in `simplex_limiter_coefficient`
protects every call site rather than only the ones that remembered to include it.
"""
@inline function simplex_rk_stage(
        initial::Tuple{T, Vararg{T, M}}, stage::Tuple{T, Vararg{T, M}},
        du::Tuple{T, Vararg{T, M}}, a, b, Δt,
    ) where {T, M}
    candidate = ntuple(k -> (@muladd stage[k] - Δt * du[k]), Val(M + 1))
    limited = limit_simplex(candidate, stage)
    return ntuple(k -> (@muladd a * initial[k] + b * limited[k]), Val(M + 1))
end
