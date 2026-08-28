@kernel inbounds = true function multiphase_WENO_flux_KA_2D_x!(
        fl, fr, state, boundary, nx, χ, γ, ζ, ϵ, phase_count::Val{NP}, g, O) where {NP}
    I = @index(Global, NTuple)
    i, j = I + O
    iwww = left_index(i, 3, nx, boundary[1]); iww = left_index(i, 2, nx, boundary[1])
    iw = left_index(i, 1, nx, boundary[1]); ie = right_index(i, 0, nx, boundary[2])
    iee = right_index(i, 1, nx, boundary[2]); ieee = right_index(i, 2, nx, boundary[2])
    sl = ntuple(q -> (state[q][iwww, j], state[q][iww, j], state[q][iw, j],
        state[q][ie, j], state[q][iee, j]), phase_count)
    sr = ntuple(q -> (state[q][iww, j], state[q][iw, j], state[q][ie, j],
        state[q][iee, j], state[q][ieee, j]), phase_count)
    up = limit_simplex(multiphase_reconstruction_upwind(sl, χ, γ, ζ, ϵ),
        ntuple(q -> state[q][iw, j], phase_count))
    dn = limit_simplex(multiphase_reconstruction_downwind(sr, χ, γ, ζ, ϵ),
        ntuple(q -> state[q][ie, j], phase_count))
    for q in 1:NP
        fl[q][i, j] = up[q]
        fr[q][i, j] = dn[q]
    end
    if i == 1 && 1 <= j <= size(state[1], 2) && boundary[1] isa PrescribedInflowBC
        for q in 1:NP
            fl[q][i, j] = multiphase_inflow_value(boundary[1], q, j)
        end
    end
    if i == nx + 1 && 1 <= j <= size(state[1], 2) && boundary[2] isa PrescribedInflowBC
        for q in 1:NP
            fr[q][i, j] = multiphase_inflow_value(boundary[2], q, j)
        end
    end
end

@kernel inbounds = true function multiphase_WENO_flux_KA_2D_y!(
        fl, fr, state, boundary, ny, χ, γ, ζ, ϵ, phase_count::Val{NP}, g, O) where {NP}
    I = @index(Global, NTuple)
    i, j = I + O
    jwww = left_index(j, 3, ny, boundary[3]); jww = left_index(j, 2, ny, boundary[3])
    jw = left_index(j, 1, ny, boundary[3]); je = right_index(j, 0, ny, boundary[4])
    jee = right_index(j, 1, ny, boundary[4]); jeee = right_index(j, 2, ny, boundary[4])
    sl = ntuple(q -> (state[q][i, jwww], state[q][i, jww], state[q][i, jw],
        state[q][i, je], state[q][i, jee]), phase_count)
    sr = ntuple(q -> (state[q][i, jww], state[q][i, jw], state[q][i, je],
        state[q][i, jee], state[q][i, jeee]), phase_count)
    up = limit_simplex(multiphase_reconstruction_upwind(sl, χ, γ, ζ, ϵ),
        ntuple(q -> state[q][i, jw], phase_count))
    dn = limit_simplex(multiphase_reconstruction_downwind(sr, χ, γ, ζ, ϵ),
        ntuple(q -> state[q][i, je], phase_count))
    for q in 1:NP
        fl[q][i, j] = up[q]
        fr[q][i, j] = dn[q]
    end
    if j == 1 && 1 <= i <= size(state[1], 1) && boundary[3] isa PrescribedInflowBC
        for q in 1:NP
            fl[q][i, j] = multiphase_inflow_value(boundary[3], q, i)
        end
    end
    if j == ny + 1 && 1 <= i <= size(state[1], 1) && boundary[4] isa PrescribedInflowBC
        for q in 1:NP
            fr[q][i, j] = multiphase_inflow_value(boundary[4], q, i)
        end
    end
end

@kernel inbounds = true function multiphase_semi_staggered_KA_2D!(
        du, state, fl, fr, v, divv, Δx_, Δy_, ::Val{NP}, g, O) where {NP}
    I = @index(Global, Cartesian)
    I = I + O
    i, j = I[1], I[2]
    d = @muladd (v.x[i + 1, j] - v.x[i, j]) * Δx_ +
        (v.y[i, j + 1] - v.y[i, j]) * Δy_
    divv[I] = d
    for q in 1:NP
        fluxdiv = @muladd (
            max(v.x[i + 1, j], 0) * fl.x[q][i + 1, j] +
                min(v.x[i + 1, j], 0) * fr.x[q][i + 1, j] -
                max(v.x[i, j], 0) * fl.x[q][i, j] -
                min(v.x[i, j], 0) * fr.x[q][i, j]
        ) * Δx_ + (
            max(v.y[i, j + 1], 0) * fl.y[q][i, j + 1] +
                min(v.y[i, j + 1], 0) * fr.y[q][i, j + 1] -
                max(v.y[i, j], 0) * fl.y[q][i, j] -
                min(v.y[i, j], 0) * fr.y[q][i, j]
        ) * Δy_
        du[q][I] = @muladd fluxdiv - state[q][I] * d
    end
end

@kernel inbounds = true function multiphase_semi_collocated_KA_2D!(
        du, fl, fr, v, Δx_, Δy_, ::Val{NP}, g, O) where {NP}
    I = @index(Global, Cartesian)
    I = I + O
    i, j = I[1], I[2]
    for q in 1:NP
        du[q][I] = @muladd max(v.x[I], 0) *
            (fl.x[q][i + 1, j] - fl.x[q][I]) * Δx_ +
            min(v.x[I], 0) * (fr.x[q][i + 1, j] - fr.x[q][I]) * Δx_ +
            max(v.y[I], 0) * (fl.y[q][i, j + 1] - fl.y[q][I]) * Δy_ +
            min(v.y[I], 0) * (fr.y[q][i, j + 1] - fr.y[q][I]) * Δy_
    end
end

function launch_multiphase_stage_KA_2D!(
        dest, initial, stage, du, fl, fr, v, divv, boundary, stag,
        nx, ny, χ, γ, ζ, ϵ, Δx_, Δy_, a, b, c, Δt, phase_count, backend,
        fx, fy, semi, update)
    fx(fl.x, fr.x, stage, boundary, nx, χ, γ, ζ, ϵ, phase_count, nothing, Offset0;
        ndrange = size(fl.x[1]))
    fy(fl.y, fr.y, stage, boundary, ny, χ, γ, ζ, ϵ, phase_count, nothing, Offset0;
        ndrange = size(fl.y[1]))
    synchronize(backend)
    if stag
        semi(du, stage, fl, fr, v, divv, Δx_, Δy_, phase_count, nothing, Offset0;
            ndrange = size(du[1]))
    else
        semi(du, fl, fr, v, Δx_, Δy_, phase_count, nothing, Offset0; ndrange = size(du[1]))
    end
    synchronize(backend)
    update(dest, initial, stage, du, a, b, c, Δt, phase_count, nothing, Offset0;
        ndrange = size(dest[1]))
    synchronize(backend)
    return nothing
end

if nameof(@__MODULE__) == :KAExt
function WENO_step!(
        phases::Tuple{A, Vararg{A, M}},
        v::NamedTuple{(:x, :y), <:Tuple{Vararg{AbstractArray{<:Real}, 2}}},
        scheme::MultiphaseWENOScheme{T, NP}, Δt, Δx, Δy, backend::Backend,
    ) where {M, A <: AbstractMatrix{<:Real}, T, NP}
    M + 1 == NP || throw(DimensionMismatch(
        "scheme was built for $NP phases but $(M + 1) were given"))
    for q in 1:NP
        @assert get_backend(phases[q]) == backend
    end
    @assert get_backend(v.x) == backend
    @assert get_backend(v.y) == backend

    (; fl, fr, ut, du, divv, boundary, stag, χ, γ, ζ, ϵ) = scheme
    nx, ny = size(phases[1])
    Δx_, Δy_ = inv(Δx), inv(Δy)
    phase_count = Val(NP)
    fx = multiphase_WENO_flux_KA_2D_x!(backend)
    fy = multiphase_WENO_flux_KA_2D_y!(backend)
    semi = stag ? multiphase_semi_staggered_KA_2D!(backend) :
        multiphase_semi_collocated_KA_2D!(backend)
    update = multiphase_RK_update_KA!(backend)
    launch_multiphase_stage_KA_2D!(ut, phases, phases, du, fl, fr, v, divv,
        boundary, stag, nx, ny, χ, γ, ζ, ϵ, Δx_, Δy_, 1.0, 0.0, 1.0, Δt,
        phase_count, backend, fx, fy, semi, update)
    launch_multiphase_stage_KA_2D!(ut, phases, ut, du, fl, fr, v, divv,
        boundary, stag, nx, ny, χ, γ, ζ, ϵ, Δx_, Δy_, 0.75, 0.25, 0.25, Δt,
        phase_count, backend, fx, fy, semi, update)
    launch_multiphase_stage_KA_2D!(phases, phases, ut, du, fl, fr, v, divv,
        boundary, stag, nx, ny, χ, γ, ζ, ϵ, Δx_, Δy_, 1.0 / 3.0, 2.0 / 3.0,
        2.0 / 3.0, Δt, phase_count, backend, fx, fy, semi, update)
    return nothing
end
end
