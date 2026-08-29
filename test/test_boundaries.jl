using Test
using FiniteDiffWENO5

@testset "typed advection boundary conditions" begin
    @testset "construction and validation" begin
        @test FiniteDiffWENO5.is_conservative(FiniteDiffWENO5.ConservativeForm())
        @test !FiniteDiffWENO5.is_conservative(FiniteDiffWENO5.NonConservativeForm())
        @test FiniteDiffWENO5.velocity_periodicity(
            (PeriodicBC(), PeriodicBC()), (:x,),
        ) == (; x = true)
        @test_throws ArgumentError FiniteDiffWENO5.velocity_periodicity(
            (PeriodicBC(), ExtrapolateBC()), (:x,),
        )

        weno = WENOScheme(zeros(8); form = :nonconservative, stag = false)
        @test weno.boundary == (ExtrapolateBC(), ExtrapolateBC())

        bc = AdvectionBC(
            west = PrescribedInflowBC(3.0),
            east = ExtrapolateBC(),
            bot = PeriodicBC(),
            top = PeriodicBC(),
        )
        weno2 = WENOScheme(zeros(6, 5); boundary = bc, form = :conservative, stag = true)
        @test weno2.boundary == (
            PrescribedInflowBC(3.0), ExtrapolateBC(), PeriodicBC(), PeriodicBC(),
        )

        # Legacy integer boundary codes remain accepted.
        @test WENOScheme(zeros(8); boundary = (2, 2), form = :nonconservative, stag = false).boundary ==
            (PeriodicBC(), PeriodicBC())
        @test WENOScheme(zeros(8); boundary = (0, 1), form = :nonconservative, stag = false).boundary ==
            (ExtrapolateBC(), ExtrapolateBC())

        @test_throws DimensionMismatch WENOScheme(
            zeros(6, 5);
            boundary = (
                PrescribedInflowBC(ones(4)), ExtrapolateBC(),
                ExtrapolateBC(), ExtrapolateBC(),
            ),
            form = :nonconservative,
            stag = false,
        )
        @test_throws ArgumentError WENOScheme(
            zeros(6, 5);
            boundary = (
                PrescribedInflowBC([1.0, NaN, 1.0, 1.0, 1.0]),
                ExtrapolateBC(), ExtrapolateBC(), ExtrapolateBC(),
            ),
            form = :nonconservative,
            stag = false,
        )
        @test_throws ArgumentError WENOScheme(
            zeros(8);
            boundary = (PrescribedInflowBC(1.0), ExtrapolateBC()),
            form = :nonconservative,
            stag = false,
            upwind_mode = true,
        )

        # The conservative split-flux path has no bound-preserving limiter yet;
        # lim_ZS=true would silently do nothing there, so it is rejected outright,
        # An explicit form is required; the former stag=true default is gone.
        @test_throws ArgumentError WENOScheme(
            zeros(8); form = :conservative, stag = true, lim_ZS = true,
        )
        @test_throws UndefKeywordError WENOScheme(zeros(8); stag = true, lim_ZS = true)
        @test WENOScheme(
            zeros(8); form = :nonconservative, stag = true, lim_ZS = true,
        ) isa WENOScheme
    end

    @testset "1D inflow is sign-aware" begin
        nx = 16
        dx = 1 / nx

        u = zeros(nx)
        velocity = (; x = ones(nx + 1))
        weno = WENOScheme(
            u;
            boundary = (PrescribedInflowBC(2.0), ExtrapolateBC()),
            form = :conservative,
            stag = true,
            multithreading = false,
        )
        WENO_step!(u, velocity, weno, 0.1dx, dx)
        @test u[1] > 0
        # On the conservative path `fl`/`fr` hold split point fluxes f± = ½(vu ± αu),
        # not transported states, so the meaningful boundary invariant is that the
        # numerical flux entering the west face equals v·u_ghost.
        @test weno.fl.x[1] + weno.fr.x[1] ≈ 1.0 * 2.0 rtol = 0 atol = 8eps(Float64)

        # The west value is ignored when the west face is outflow.
        fill!(u, 1.0)
        velocity = (; x = -ones(nx + 1))
        weno = WENOScheme(
            u;
            boundary = (PrescribedInflowBC(99.0), ExtrapolateBC()),
            form = :conservative,
            stag = true,
            multithreading = false,
        )
        WENO_step!(u, velocity, weno, 0.1dx, dx)
        @test u ≈ ones(nx) rtol = 0 atol = 10eps(Float64)

        # A prescribed east state is selected for negative velocity.
        fill!(u, 0.0)
        weno = WENOScheme(
            u;
            boundary = (ExtrapolateBC(), PrescribedInflowBC(4.0)),
            form = :conservative,
            stag = true,
            multithreading = false,
        )
        WENO_step!(u, velocity, weno, 0.1dx, dx)
        @test u[end] > 0
        # Same flux semantics at the east face, where v = -1 makes the inward
        # numerical flux negative: v·u_ghost = -1 * 4.
        @test weno.fl.x[end] + weno.fr.x[end] ≈ -1.0 * 4.0 rtol = 0 atol = 8eps(Float64)
    end

    @testset "2D face profiles" begin
        nx, ny = 8, 6
        dx, dy = 1 / nx, 1 / ny
        west_temperature = collect(1.0:ny)
        u = zeros(nx, ny)
        velocity = (; x = ones(nx + 1, ny), y = zeros(nx, ny + 1))
        weno = WENOScheme(
            u;
            boundary = (
                PrescribedInflowBC(west_temperature), ExtrapolateBC(),
                ExtrapolateBC(), ExtrapolateBC(),
            ),
            # These exercise inflow *state* plumbing (per-face profiles landing on
            # the right faces), which is the non-conservative path's representation.
            # The conservative path stores split fluxes instead and is covered by
            # test_conservative_scalar.jl.
            form = :nonconservative,
            stag = true,
            multithreading = false,
        )
        WENO_step!(u, velocity, weno, 0.05dx, dx, dy)

        @test collect(weno.fl.x[1, :]) == west_temperature
        @test all(>(0), u[1, :])
        @test u[1, end] > u[1, 1]

        east_temperature = collect(11.0:(10.0 + ny))
        bot_temperature = collect(21.0:(20.0 + nx))
        top_temperature = collect(31.0:(30.0 + nx))
        weno = WENOScheme(
            u;
            boundary = (
                PrescribedInflowBC(west_temperature),
                PrescribedInflowBC(east_temperature),
                PrescribedInflowBC(bot_temperature),
                PrescribedInflowBC(top_temperature),
            ),
            # These exercise inflow *state* plumbing (per-face profiles landing on
            # the right faces), which is the non-conservative path's representation.
            # The conservative path stores split fluxes instead and is covered by
            # test_conservative_scalar.jl.
            form = :nonconservative,
            stag = true,
            multithreading = false,
        )
        WENO_step!(u, velocity, weno, 0.0, dx, dy)
        @test collect(weno.fl.x[1, :]) == west_temperature
        @test collect(weno.fr.x[end, :]) == east_temperature
        @test collect(weno.fl.y[:, 1]) == bot_temperature
        @test collect(weno.fr.y[:, end]) == top_temperature
    end

    @testset "3D face profiles" begin
        nx, ny, nz = 5, 4, 3
        u = zeros(nx, ny, nz)
        xlo = reshape(collect(1.0:(ny * nz)), ny, nz)
        xhi = xlo .+ 20
        ylo = reshape(collect(1.0:(nx * nz)), nx, nz) .+ 40
        yhi = ylo .+ 20
        zlo = reshape(collect(1.0:(nx * ny)), nx, ny) .+ 80
        zhi = zlo .+ 20
        velocity = (
            x = zeros(nx + 1, ny, nz),
            y = zeros(nx, ny + 1, nz),
            z = zeros(nx, ny, nz + 1),
        )
        weno = WENOScheme(
            u;
            boundary = (
                PrescribedInflowBC(xlo), PrescribedInflowBC(xhi),
                PrescribedInflowBC(ylo), PrescribedInflowBC(yhi),
                PrescribedInflowBC(zlo), PrescribedInflowBC(zhi),
            ),
            # These exercise inflow *state* plumbing (per-face profiles landing on
            # the right faces), which is the non-conservative path's representation.
            # The conservative path stores split fluxes instead and is covered by
            # test_conservative_scalar.jl.
            form = :nonconservative,
            stag = true,
            multithreading = false,
        )
        WENO_step!(u, velocity, weno, 0.0, 1 / nx, 1 / ny, 1 / nz)

        @test Array(weno.fl.x[1, :, :]) == xlo
        @test Array(weno.fr.x[end, :, :]) == xhi
        @test Array(weno.fl.y[:, 1, :]) == ylo
        @test Array(weno.fr.y[:, end, :]) == yhi
        @test Array(weno.fl.z[:, :, 1]) == zlo
        @test Array(weno.fr.z[:, :, end]) == zhi
    end

    @testset "CPU prescribed inflow does not allocate without multithreading" begin
        function allocated_cpu_step()
            nx, ny = 32, 24
            dx, dy = 1 / nx, 1 / ny
            u = zeros(nx, ny)
            velocity = (; x = ones(nx + 1, ny), y = zeros(nx, ny + 1))
            weno = WENOScheme(
                u;
                boundary = (
                    PrescribedInflowBC(300.0), ExtrapolateBC(),
                    ExtrapolateBC(), ExtrapolateBC(),
                ),
                form = :nonconservative,
                stag = true,
                multithreading = false,
            )

            WENO_step!(u, velocity, weno, 0.01dx, dx, dy)
            return @allocated WENO_step!(u, velocity, weno, 0.01dx, dx, dy)
        end

        allocated_cpu_step() # compile before measuring
        @test allocated_cpu_step() == 0
    end
end
