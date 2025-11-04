@inline function zhang_shu_limit(u_val, u_avg, u_min, u_max, ϵθ)
    δ = u_val - u_avg
    if δ == zero(eltype(u_val))
        return u_val
    end
    θ1 = abs((u_max - u_avg) / (δ + ϵθ))
    θ2 = abs((u_avg - u_min) / (-δ + ϵθ))
    θ = min(one(eltype(u_val)), min(θ1, θ2))
    return @muladd θ * δ + u_avg
end
