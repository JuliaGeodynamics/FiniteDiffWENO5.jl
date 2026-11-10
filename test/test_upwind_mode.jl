using Test
using KernelAbstractions
using FiniteDiffWENO5

# Out-of-place first-order upwind on staggered face velocities with unit spacing.
function upwind_reference(u, v, Δt, periodic)
    N = ndims(u)
    r = similar(u)
    for I in CartesianIndices(u)
        acc = zero(eltype(u))
        for d in 1:N
            e = CartesianIndex(ntuple(k -> Int(k == d), N))
            n = size(u, d)
            nb(J) = CartesianIndex(ntuple(k -> k == d ? (periodic ? mod1(J[k], n) : clamp(J[k], 1, n)) : J[k], N))
            acc += max(v[d][I], 0) * (u[I] - u[nb(I - e)]) + min(v[d][I + e], 0) * (u[nb(I + e)] - u[I])
        end
        r[I] = u[I] - Δt * acc
    end
    return r
end

@testset "upwind_mode is an out-of-place staggered update (#26)" begin
    for dims in ((12,), (10, 7), (8, 6, 5)), periodic in (true, false)
        N = length(dims)
        labels = (:x, :y, :z)[1:N]
        u0 = rand(dims...)
        v = NamedTuple{labels}(
            ntuple(N) do d
                vd = randn(ntuple(k -> dims[k] + (k == d), N)...)
                if periodic
                    selectdim(vd, d, dims[d] + 1) .= selectdim(vd, d, 1)
                else
                    selectdim(vd, d, 1) .= 0
                    selectdim(vd, d, dims[d] + 1) .= 0
                end
                vd
            end
        )
        Δt = 0.1
        Δs = ntuple(_ -> 1.0, N)
        bc = periodic ? PeriodicBC() : ExtrapolateBC()
        boundary = ntuple(_ -> bc, 2N)
        ref = upwind_reference(u0, Tuple(v), Δt, periodic)
        div = sum(d -> diff(v[d]; dims = d), 1:N)

        for mt in (false, true)
            u = copy(u0)
            weno = WENOScheme(u; boundary, form = :conservative, stag = true, upwind_mode = true, multithreading = mt)
            WENO_step!(u, v, weno, Δt, Δs...)
            @test u ≈ ref
            @test sum(u) - sum(u0) ≈ Δt * sum(u0 .* div)
        end

        u = copy(u0)
        weno = WENOScheme(u, CPU(); boundary, form = :conservative, stag = true, upwind_mode = true)
        WENO_step!(u, v, weno, Δt, Δs..., CPU())
        KernelAbstractions.synchronize(CPU())
        @test u ≈ ref
    end
end
