@kernel inbounds = true function multiphase_WENO_flux_KA_1D!(
        fl, fr, state, boundary, nx, χ, γ, ζ, ϵ, phase_count::Val{NP}, g, O
    ) where {NP}
    I = @index(Global, NTuple)
    I = I + O
    i = I[1]
    iwww = left_index(i, 3, nx, boundary[1])
    iww = left_index(i, 2, nx, boundary[1])
    iw = left_index(i, 1, nx, boundary[1])
    ie = right_index(i, 0, nx, boundary[2])
    iee = right_index(i, 1, nx, boundary[2])
    ieee = right_index(i, 2, nx, boundary[2])

    stencil_l = ntuple(
        q -> (
            state[q][iwww], state[q][iww], state[q][iw], state[q][ie], state[q][iee],
        ), phase_count
    )
    stencil_r = ntuple(
        q -> (
            state[q][iww], state[q][iw], state[q][ie], state[q][iee], state[q][ieee],
        ), phase_count
    )
    up = multiphase_reconstruction_upwind(stencil_l, χ, γ, ζ, ϵ)
    dn = multiphase_reconstruction_downwind(stencil_r, χ, γ, ζ, ϵ)
    up = limit_simplex(up, ntuple(q -> state[q][iw], phase_count))
    dn = limit_simplex(dn, ntuple(q -> state[q][ie], phase_count))

    for q in 1:NP
        fl[q][i] = up[q]
        fr[q][i] = dn[q]
    end
    if i == 1 && boundary[1] isa PrescribedInflowBC
        for q in 1:NP
            fl[q][i] = multiphase_inflow_value(boundary[1], q)
        end
    end
    if i == nx + 1 && boundary[2] isa PrescribedInflowBC
        for q in 1:NP
            fr[q][i] = multiphase_inflow_value(boundary[2], q)
        end
    end
end

@kernel inbounds = true function multiphase_semi_staggered_KA_1D!(
        du, state, fl, fr, v, divv, Δx_, ::Val{NP}, g, O
    ) where {NP}
    I = @index(Global, NTuple)
    I = I + O
    i = I[1]
    d = (v.x[i + 1] - v.x[i]) * Δx_
    divv[i] = d
    for q in 1:NP
        fluxdiv = (
            max(v.x[i + 1], 0) * fl.x[q][i + 1] +
                min(v.x[i + 1], 0) * fr.x[q][i + 1] -
                max(v.x[i], 0) * fl.x[q][i] -
                min(v.x[i], 0) * fr.x[q][i]
        ) * Δx_
        du[q][i] = @muladd fluxdiv - state[q][i] * d
    end
end

@kernel inbounds = true function multiphase_semi_collocated_KA_1D!(
        du, fl, fr, v, Δx_, ::Val{NP}, g, O
    ) where {NP}
    I = @index(Global, NTuple)
    I = I + O
    i = I[1]
    for q in 1:NP
        du[q][i] = @muladd max(v.x[i], 0) * (fl.x[q][i + 1] - fl.x[q][i]) * Δx_ +
            min(v.x[i], 0) * (fr.x[q][i + 1] - fr.x[q][i]) * Δx_
    end
end

@kernel inbounds = true function multiphase_RK_update_KA!(
        dest, initial, stage, du, a, b, c, Δt, ::Val{NP}, g, O
    ) where {NP}
    I = @index(Global, Cartesian)
    I = I + O
    for q in 1:NP
        dest[q][I] = @muladd a * initial[q][I] + b * stage[q][I] - c * Δt * du[q][I]
    end
end

if nameof(@__MODULE__) == :KAExt
    function WENO_step!(
            phases::Tuple{A, Vararg{A, M}},
            v::NamedTuple{(:x,), <:Tuple{<:AbstractVector{<:Real}}},
            scheme::MultiphaseWENOScheme{T, NP}, Δt, Δx, backend::Backend,
        ) where {M, A <: AbstractVector{<:Real}, T, NP}
        M + 1 == NP || throw(
            DimensionMismatch(
                "scheme was built for $NP phases but $(M + 1) were given"
            )
        )
        for q in 1:NP
            @assert get_backend(phases[q]) == backend
        end
        @assert get_backend(v.x) == backend

        (; fl, fr, ut, du, divv, boundary, stag, χ, γ, ζ, ϵ) = scheme
        nx = size(phases[1], 1)
        Δx_ = inv(Δx)
        phase_count = Val(NP)
        flux_kernel = multiphase_WENO_flux_KA_1D!(backend)
        update_kernel = multiphase_RK_update_KA!(backend)
        semi_kernel = stag ? multiphase_semi_staggered_KA_1D!(backend) :
            multiphase_semi_collocated_KA_1D!(backend)

        flux_kernel(
            fl.x, fr.x, phases, boundary, nx, χ, γ, ζ, ϵ, phase_count, nothing, Offset0;
            ndrange = size(fl.x[1])
        )
        synchronize(backend)
        if stag
            semi_kernel(du, phases, fl, fr, v, divv, Δx_, phase_count, nothing, Offset0; ndrange = size(du[1]))
        else
            semi_kernel(du, fl, fr, v, Δx_, phase_count, nothing, Offset0; ndrange = size(du[1]))
        end
        synchronize(backend)
        update_kernel(
            ut, phases, phases, du, 1.0, 0.0, 1.0, Δt, phase_count, nothing, Offset0;
            ndrange = size(ut[1])
        )
        synchronize(backend)

        flux_kernel(
            fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, phase_count, nothing, Offset0;
            ndrange = size(fl.x[1])
        )
        synchronize(backend)
        if stag
            semi_kernel(du, ut, fl, fr, v, divv, Δx_, phase_count, nothing, Offset0; ndrange = size(du[1]))
        else
            semi_kernel(du, fl, fr, v, Δx_, phase_count, nothing, Offset0; ndrange = size(du[1]))
        end
        synchronize(backend)
        update_kernel(
            ut, phases, ut, du, 0.75, 0.25, 0.25, Δt, phase_count, nothing, Offset0;
            ndrange = size(ut[1])
        )
        synchronize(backend)

        flux_kernel(
            fl.x, fr.x, ut, boundary, nx, χ, γ, ζ, ϵ, phase_count, nothing, Offset0;
            ndrange = size(fl.x[1])
        )
        synchronize(backend)
        if stag
            semi_kernel(du, ut, fl, fr, v, divv, Δx_, phase_count, nothing, Offset0; ndrange = size(du[1]))
        else
            semi_kernel(du, fl, fr, v, Δx_, phase_count, nothing, Offset0; ndrange = size(du[1]))
        end
        synchronize(backend)
        update_kernel(
            phases, phases, ut, du, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0,
            Δt, phase_count, nothing, Offset0; ndrange = size(phases[1])
        )
        synchronize(backend)
        return nothing
    end
end
