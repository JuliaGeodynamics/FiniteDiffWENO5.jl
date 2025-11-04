@testset "1D limiter tests" begin
    @testset "1D Shu test" begin

        # Number of grid points
        nx = 200

        x_min = -1.0
        x_max = 1.0
        Lx = x_max - x_min

        x = range(x_min, stop = x_max, length = nx)

        # Courant number
        CFL = 0.4
        period = 4

        # Parameters for Shu test
        z = -0.7
        δ = 0.005
        β = log(2) / (36 * δ^2)
        a = 0.5
        α = 10

        # Functions
        G(x, β, z) = exp.(-β .* (x .- z) .^ 2)
        F(x, α, a) = sqrt.(max.(1 .- α^2 .* (x .- a) .^ 2, 0.0))

        # Grid x assumed defined
        u0_vec = zeros(length(x))

        # Gaussian-like smooth bump at x in [-0.8, -0.6]
        idx = (x .>= -0.8) .& (x .<= -0.6)
        u0_vec[idx] .= (1 / 6) .* (G(x[idx], β, z - δ) .+ 4 .* G(x[idx], β, z) .+ G(x[idx], β, z + δ))

        # Heaviside step at x in [-0.4, -0.2]
        idx = (x .>= -0.4) .& (x .<= -0.2)
        u0_vec[idx] .= 1.0

        # Piecewise linear ramp at x in [0, 0.2]
        # Triangular spike at x=0.1, base width 0.2
        idx = abs.(x .- 0.1) .<= 0.1
        u0_vec[idx] .= 1 .- 10 .* abs.(x[idx] .- 0.1)

        # Elliptic/smooth bell at x in [0.4, 0.6]
        idx = (x .>= 0.4) .& (x .<= 0.6)
        u0_vec[idx] .= (1 / 6) .* (F(x[idx], α, a - δ) .+ 4 .* F(x[idx], α, a) .+ F(x[idx], α, a + δ))


        u = copy(u0_vec)
        weno = WENOScheme(u; boundary = (2, 2), stag = true, lim_ZS = true)

        # advection velocity
        a = (; x = ones(nx + 1))

        # grid size
        Δx = x[2] - x[1]
        Δt = CFL * Δx^(5 / 3)

        tmax = period * (Lx + Δx) / maximum(abs.(a.x))

        t = 0

        while t < tmax
            WENO_step!(u, a, weno, Δt, Δx; u_min = 0.0, u_max = 1.0)

            t += Δt

            if t + Δt > tmax
                Δt = tmax - t
            end
        end

        @test sum(u) ≈ 51.92724276033528
        @test maximum(u) ≈ 0.9988671654518768 atol = 1.0e-10
    end

    @testset "1D Shu test KernelAbstractions CPU" begin #

        backend = CPU()
        nx = 200

        x_min = -1.0
        x_max = 1.0
        Lx = x_max - x_min

        x = range(x_min, stop = x_max, length = nx)

        # Courant number
        CFL = 0.7
        period = 4

        # Parameters
        z = -0.7
        δ = 0.005
        β = log(2) / (36 * δ^2)
        a = 0.5
        α = 10

        # Functions
        G(x, β, z) = exp.(-β .* (x .- z) .^ 2)
        F(x, α, a) = sqrt.(max.(1 .- α^2 .* (x .- a) .^ 2, 0.0))

        # Grid x assumed defined
        u0_vec = zeros(length(x))

        # Gaussian-like smooth bump at x in [-0.8, -0.6]
        idx = (x .>= -0.8) .& (x .<= -0.6)
        u0_vec[idx] .= (1 / 6) .* (G(x[idx], β, z - δ) .+ 4 .* G(x[idx], β, z) .+ G(x[idx], β, z + δ))

        # Heaviside step at x in [-0.4, -0.2]
        idx = (x .>= -0.4) .& (x .<= -0.2)
        u0_vec[idx] .= 1.0

        # Piecewise linear ramp at x in [0, 0.2]
        # Triangular spike at x=0.1, base width 0.2
        idx = abs.(x .- 0.1) .<= 0.1
        u0_vec[idx] .= 1 .- 10 .* abs.(x[idx] .- 0.1)

        # Elliptic/smooth bell at x in [0.4, 0.6]
        idx = (x .>= 0.4) .& (x .<= 0.6)
        u0_vec[idx] .= (1 / 6) .* (F(x[idx], α, a - δ) .+ 4 .* F(x[idx], α, a) .+ F(x[idx], α, a + δ))


        u = KernelAbstractions.zeros(backend, Float64, nx)
        copyto!(u, u0_vec)
        weno = WENOScheme(u, backend; boundary = (2, 2), stag = true, lim_ZS = true)

        # advection velocity
        a_vec = ones(nx + 1) .* -1
        a = (; x = KernelAbstractions.zeros(backend, Float64, nx + 1))
        copyto!(a.x, a_vec)

        # grid size
        Δx = x[2] - x[1]
        Δt = CFL * Δx^(5 / 3)

        tmax = period * (Lx + Δx) / maximum(abs.(a.x))

        t = 0

        while t < tmax
            WENO_step!(u, a, weno, Δt, Δx, backend, u_min = 0.0, u_max = 1.0)

            t += Δt

            if t + Δt > tmax
                Δt = tmax - t
            end
        end

        @test sum(u) ≈ 51.92724276038761
        @test maximum(u) ≈ 0.998866768379829 atol = 1.0e-10
    end


    @testset "1D Shu test Chmy CPU" begin #

        backend = CPU()
        nx = 200

        arch = Arch(backend)

        x_min = -1.0
        x_max = 1.0
        Lx = x_max - x_min

        x = range(x_min, stop = x_max, length = nx)

        grid = UniformGrid(arch; origin = (x_min,), extent = (Lx,), dims = (nx,))

        # Courant number
        CFL = 0.7
        period = 4

        # Parameters
        z = -0.7
        δ = 0.005
        β = log(2) / (36 * δ^2)
        a = 0.5
        α = 10

        # Functions
        G(x, β, z) = exp.(-β .* (x .- z) .^ 2)
        F(x, α, a) = sqrt.(max.(1 .- α^2 .* (x .- a) .^ 2, 0.0))

        # Grid x assumed defined
        u0_vec = zeros(length(x))

        # Gaussian-like smooth bump at x in [-0.8, -0.6]
        idx = (x .>= -0.8) .& (x .<= -0.6)
        u0_vec[idx] .= (1 / 6) .* (G(x[idx], β, z - δ) .+ 4 .* G(x[idx], β, z) .+ G(x[idx], β, z + δ))

        # Heaviside step at x in [-0.4, -0.2]
        idx = (x .>= -0.4) .& (x .<= -0.2)
        u0_vec[idx] .= 1.0

        # Piecewise linear ramp at x in [0, 0.2]
        # Triangular spike at x=0.1, base width 0.2
        idx = abs.(x .- 0.1) .<= 0.1
        u0_vec[idx] .= 1 .- 10 .* abs.(x[idx] .- 0.1)

        # Elliptic/smooth bell at x in [0.4, 0.6]
        idx = (x .>= 0.4) .& (x .<= 0.6)
        u0_vec[idx] .= (1 / 6) .* (F(x[idx], α, a - δ) .+ 4 .* F(x[idx], α, a) .+ F(x[idx], α, a + δ))


        u = Field(backend, grid, Center())
        set!(u, u0_vec)
        weno = WENOScheme(u, grid; boundary = (2, 2), stag = true, lim_ZS = true)

        # advection velocity
        a_vec = ones(nx + 1) .* -1
        a = VectorField(backend, grid)
        set!(a, a_vec)

        # grid size
        Δx = x[2] - x[1]
        Δt = CFL * Δx^(5 / 3)

        tmax = period * (Lx + Δx) / maximum(abs.(a.x))

        t = 0

        while t < tmax
            WENO_step!(u, a, weno, Δt, Δx, grid, arch, u_min = 0.0, u_max = 1.0)

            t += Δt

            if t + Δt > tmax
                Δt = tmax - t
            end
        end

        @test sum(u) ≈ 51.92724276038775
        @test maximum(u) ≈ 0.998866768379829 atol = 1.0e-10
    end
end
