using Test
using FiniteDiffWENO5

@testset "typed advection boundary conditions" begin
    @testset "construction and validation" begin
        weno = WENOScheme(zeros(8))
        @test weno.boundary == (ExtrapolateBC(), ExtrapolateBC())

        bc = AdvectionBC(
            west = PrescribedInflowBC(3.0),
            east = ExtrapolateBC(),
            bot = PeriodicBC(),
            top = PeriodicBC(),
        )
        weno2 = WENOScheme(zeros(6, 5); boundary = bc, stag = true)
        @test weno2.boundary == (
            PrescribedInflowBC(3.0), ExtrapolateBC(), PeriodicBC(), PeriodicBC(),
        )

        # Legacy integer codes remain accepted for existing callers.
        @test WENOScheme(zeros(8); boundary = (2, 2)).boundary ==
            (PeriodicBC(), PeriodicBC())
        @test WENOScheme(zeros(8); boundary = (0, 1)).boundary ==
            (ExtrapolateBC(), ExtrapolateBC())

        @test_throws DimensionMismatch WENOScheme(
            zeros(6, 5);
            boundary = (
                PrescribedInflowBC(ones(4)), ExtrapolateBC(),
                ExtrapolateBC(), ExtrapolateBC(),
            ),
        )
        @test_throws ArgumentError WENOScheme(
            zeros(6, 5);
            boundary = (
                PrescribedInflowBC([1.0, NaN, 1.0, 1.0, 1.0]),
                ExtrapolateBC(), ExtrapolateBC(), ExtrapolateBC(),
            ),
        )
        @test_throws ArgumentError WENOScheme(
            zeros(8);
            boundary = (PrescribedInflowBC(1.0), ExtrapolateBC()),
            upwind_mode = true,
        )
    end

    @testset "1D inflow is sign-aware" begin
        nx = 16
        dx = 1 / nx

        u = zeros(nx)
        velocity = (; x = ones(nx + 1))
        weno = WENOScheme(
            u;
            boundary = (PrescribedInflowBC(2.0), ExtrapolateBC()),
            stag = true,
            multithreading = false,
        )
        WENO_step!(u, velocity, weno, 0.1dx, dx)
        @test u[1] > 0
        @test weno.fl.x[1] == 2.0

        # The west value is ignored when the west face is outflow.
        fill!(u, 1.0)
        velocity = (; x = -ones(nx + 1))
        weno = WENOScheme(
            u;
            boundary = (PrescribedInflowBC(99.0), ExtrapolateBC()),
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
            stag = true,
            multithreading = false,
        )
        WENO_step!(u, velocity, weno, 0.1dx, dx)
        @test u[end] > 0
        @test weno.fr.x[end] == 4.0
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
