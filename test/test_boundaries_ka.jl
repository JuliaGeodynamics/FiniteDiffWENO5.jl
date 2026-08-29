using Test
using FiniteDiffWENO5
using KernelAbstractions

@testset "typed boundaries with KernelAbstractions" begin
    backend = CPU()
    nx, ny = 8, 6
    dx, dy = 1 / nx, 1 / ny
    west_temperature = collect(1.0:ny)

    u = KernelAbstractions.zeros(backend, Float64, nx, ny)
    velocity = (
        x = KernelAbstractions.ones(backend, Float64, nx + 1, ny),
        y = KernelAbstractions.zeros(backend, Float64, nx, ny + 1),
    )
    weno = WENOScheme(
        u,
        backend;
        boundary = (
            PrescribedInflowBC(west_temperature), ExtrapolateBC(),
            ExtrapolateBC(), ExtrapolateBC(),
        ),
        form = :nonconservative,
        stag = true,
    )
    WENO_step!(u, velocity, weno, 0.05dx, dx, dy, backend)

    @test collect(weno.fl.x[1, :]) == west_temperature
    @test all(>(0), Array(u)[1, :])
    @test Array(u)[1, end] > Array(u)[1, 1]

    @testset "staggered velocity geometry is validated before launch" begin
        n = 8
        scalar = KernelAbstractions.zeros(backend, Float64, n)
        scheme = WENOScheme(
            scalar, backend;
            form = :nonconservative,
            boundary = (ExtrapolateBC(), ExtrapolateBC()), stag = true,
        )
        malformed = (; x = KernelAbstractions.ones(backend, Float64, n))
        @test_throws DimensionMismatch WENO_step!(scalar, malformed, scheme, 0.01, 1 / n, backend)
    end
end
