using Test
using KernelAbstractions
using FiniteDiffWENO5

# The pre-existing KA/Chmy "matches the serial operator" tests all use a spatially
# uniform velocity. For a uniform v the conservative and material forms coincide
# (∇·v = 0 and v factors out of the flux difference), so those tests cannot detect
# a backend running a different PDE form from the CPU. Every case here therefore
# uses a velocity with non-zero divergence, which is exactly where the forms differ.

const BACKEND = KernelAbstractions.CPU()

# Absolute tolerances are unusable across backends because the flux difference is
# scaled by Δx⁻¹; compare relative to the magnitude of the reference instead.
function agree(a, b; rtol = 1.0e-12)
    scale = max(1.0, maximum(abs, b))
    return maximum(abs, a .- b) <= rtol * scale
end

@testset "CPU and KernelAbstractions agree under compressible velocity" begin
    @testset "scalar staggered transport" begin
        n = 64
        Δx = inv(float(n))
        x = ((1:n) .- 0.5) .* Δx
        faces = (0:n) .* Δx
        initial() = 1 .+ 0.2 .* sinpi.(2 .* x)
        # ∇·v ≠ 0: this is what separates the conservative and material forms.
        face_velocity = 1 .+ 0.3 .* sinpi.(2 .* faces)
        boundary = (PeriodicBC(), PeriodicBC())

        for form in (:conservative, :nonconservative)
            cpu_u = initial()
            cpu = WENOScheme(cpu_u; form, boundary, stag = true, multithreading = false)

            ka_u = initial()
            ka = WENOScheme(ka_u, BACKEND; form, boundary, stag = true)
            ka_v = (; x = copy(face_velocity))

            for _ in 1:5
                WENO_step!(cpu_u, (; x = face_velocity), cpu, 0.2Δx, Δx)
                WENO_step!(ka_u, ka_v, ka, 0.2Δx, Δx, BACKEND)
            end
            KernelAbstractions.synchronize(BACKEND)

            @test agree(ka_u, cpu_u)
        end
    end

    @testset "multiphase staggered material transport" begin
        n = 64
        Δx = inv(float(n))
        x = ((1:n) .- 0.5) .* Δx
        faces = (0:n) .* Δx
        function initial()
            p1 = 0.4 .+ 0.1 .* sinpi.(2 .* x)
            p2 = 0.35 .+ 0.08 .* cospi.(2 .* x)
            return (p1, p2, 1 .- p1 .- p2)
        end
        face_velocity = 0.7 .+ 0.25 .* sinpi.(2 .* faces)
        boundary = (PeriodicBC(), PeriodicBC())

        cpu_p = initial()
        cpu = MultiphaseWENOScheme(cpu_p; boundary, stag = true, multithreading = false)

        ka_p = initial()
        ka = MultiphaseWENOScheme(ka_p, BACKEND; boundary, stag = true)
        ka_v = (; x = copy(face_velocity))

        for _ in 1:5
            WENO_step!(cpu_p, (; x = face_velocity), cpu, 0.2Δx, Δx)
            WENO_step!(ka_p, ka_v, ka, 0.2Δx, Δx, BACKEND)
        end
        KernelAbstractions.synchronize(BACKEND)

        for k in 1:3
            @test agree(ka_p[k], cpu_p[k])
        end
        # Whatever else differs, the simplex invariant must hold on both.
        @test maximum(abs, ka_p[1] .+ ka_p[2] .+ ka_p[3] .- 1) < 1024eps(Float64)
    end
end

@testset "CPU and KA agree in 2D and 3D under compressible velocity" begin
    @testset "2D scalar staggered" begin
        n = 32
        Δ = inv(float(n))
        x = ((1:n) .- 0.5) .* Δ
        faces = (0:n) .* Δ
        initial() = [1 + 0.2 * sinpi(2xi) * cospi(2yj) for xi in x, yj in x]
        vx = [1 + 0.3 * sinpi(2xi) for xi in faces, _ in x]
        vy = [0.5 + 0.2 * cospi(2yj) for _ in x, yj in faces]
        boundary = ntuple(_ -> PeriodicBC(), 4)

        for form in (:conservative, :nonconservative)
            cu = initial()
            cpu = WENOScheme(cu; form, boundary, stag = true, multithreading = false)
            ku = initial()
            ka = WENOScheme(ku, BACKEND; form, boundary, stag = true)
            kv = (x = copy(vx), y = copy(vy))
            for _ in 1:3
                WENO_step!(cu, (x = vx, y = vy), cpu, 0.2Δ, Δ, Δ)
                WENO_step!(ku, kv, ka, 0.2Δ, Δ, Δ, BACKEND)
            end
            KernelAbstractions.synchronize(BACKEND)
            @test agree(ku, cu)
        end
    end

    @testset "2D multiphase staggered" begin
        n = 32
        Δ = inv(float(n))
        x = ((1:n) .- 0.5) .* Δ
        faces = (0:n) .* Δ
        function initial()
            p1 = [0.4 + 0.1 * sinpi(2xi) * cospi(2yj) for xi in x, yj in x]
            p2 = [0.35 + 0.08 * cospi(2xi) * sinpi(2yj) for xi in x, yj in x]
            return (p1, p2, 1 .- p1 .- p2)
        end
        vx = [0.7 + 0.25 * sinpi(2xi) for xi in faces, _ in x]
        vy = [0.4 + 0.15 * cospi(2yj) for _ in x, yj in faces]
        boundary = ntuple(_ -> PeriodicBC(), 4)

        cp = initial()
        cpu = MultiphaseWENOScheme(cp; boundary, stag = true, multithreading = false)
        kp = initial()
        ka = MultiphaseWENOScheme(kp, BACKEND; boundary, stag = true)
        kv = (x = copy(vx), y = copy(vy))
        for _ in 1:3
            WENO_step!(cp, (x = vx, y = vy), cpu, 0.2Δ, Δ, Δ)
            WENO_step!(kp, kv, ka, 0.2Δ, Δ, Δ, BACKEND)
        end
        KernelAbstractions.synchronize(BACKEND)
        for k in 1:3
            @test agree(kp[k], cp[k])
        end
    end

    @testset "3D multiphase staggered" begin
        n = 12
        Δ = inv(float(n))
        x = ((1:n) .- 0.5) .* Δ
        faces = (0:n) .* Δ
        function initial()
            p1 = [0.4 + 0.08 * sinpi(2xi) * cospi(2yj) * cospi(2zk) for xi in x, yj in x, zk in x]
            p2 = [0.35 + 0.06 * cospi(2xi) * sinpi(2yj) for xi in x, yj in x, _ in x]
            return (p1, p2, 1 .- p1 .- p2)
        end
        vx = [0.7 + 0.2 * sinpi(2xi) for xi in faces, _ in x, _ in x]
        vy = [0.4 + 0.15 * cospi(2yj) for _ in x, yj in faces, _ in x]
        vz = [0.3 + 0.1 * sinpi(2zk) for _ in x, _ in x, zk in faces]
        boundary = ntuple(_ -> PeriodicBC(), 6)

        cp = initial()
        cpu = MultiphaseWENOScheme(cp; boundary, stag = true, multithreading = false)
        kp = initial()
        ka = MultiphaseWENOScheme(kp, BACKEND; boundary, stag = true)
        kv = (x = copy(vx), y = copy(vy), z = copy(vz))
        for _ in 1:3
            WENO_step!(cp, (x = vx, y = vy, z = vz), cpu, 0.2Δ, Δ, Δ, Δ)
            WENO_step!(kp, kv, ka, 0.2Δ, Δ, Δ, Δ, BACKEND)
        end
        KernelAbstractions.synchronize(BACKEND)
        for k in 1:3
            @test agree(kp[k], cp[k])
        end
    end
end

@testset "CPU and Chmy agree under compressible velocity" begin
    backend_chmy = KernelAbstractions.CPU()
    arch = Chmy.Arch(backend_chmy)

    @testset "1D scalar staggered" begin
        n = 64
        Δx = inv(float(n))
        x = ((1:n) .- 0.5) .* Δx
        faces = (0:n) .* Δx
        initial() = 1 .+ 0.2 .* sinpi.(2 .* x)
        face_velocity = 1 .+ 0.3 .* sinpi.(2 .* faces)
        boundary = (PeriodicBC(), PeriodicBC())
        grid = Chmy.UniformGrid(arch; origin = (0.0,), extent = (1.0,), dims = (n,))

        for form in (:conservative, :nonconservative)
            cpu_u = initial()
            cpu = WENOScheme(cpu_u; form, boundary, stag = true, multithreading = false)

            chmy_u = Chmy.Field(backend_chmy, grid, Chmy.Center())
            Chmy.set!(chmy_u, initial())
            chmy_v = Chmy.VectorField(backend_chmy, grid)
            Chmy.set!(chmy_v.x, face_velocity)
            chmy = WENOScheme(chmy_u, grid; form, boundary, stag = true)

            for _ in 1:5
                WENO_step!(cpu_u, (; x = face_velocity), cpu, 0.2Δx, Δx)
                WENO_step!(chmy_u, chmy_v, chmy, 0.2Δx, Δx, grid, arch)
            end

            @test agree(Array(Chmy.interior(chmy_u)), cpu_u)
        end
    end

    @testset "1D multiphase staggered" begin
        n = 64
        Δx = inv(float(n))
        x = ((1:n) .- 0.5) .* Δx
        faces = (0:n) .* Δx
        function initial()
            p1 = 0.4 .+ 0.1 .* sinpi.(2 .* x)
            p2 = 0.35 .+ 0.08 .* cospi.(2 .* x)
            return (p1, p2, 1 .- p1 .- p2)
        end
        face_velocity = 0.7 .+ 0.25 .* sinpi.(2 .* faces)
        boundary = (PeriodicBC(), PeriodicBC())
        grid = Chmy.UniformGrid(arch; origin = (0.0,), extent = (1.0,), dims = (n,))

        cpu_p = initial()
        cpu = MultiphaseWENOScheme(cpu_p; boundary, stag = true, multithreading = false)

        init = initial()
        chmy_p = ntuple(3) do q
            f = Chmy.Field(backend_chmy, grid, Chmy.Center())
            Chmy.set!(f, init[q])
            f
        end
        chmy_v = Chmy.VectorField(backend_chmy, grid)
        Chmy.set!(chmy_v.x, face_velocity)
        chmy = MultiphaseWENOScheme(chmy_p, grid; boundary, stag = true)

        for _ in 1:5
            WENO_step!(cpu_p, (; x = face_velocity), cpu, 0.2Δx, Δx)
            WENO_step!(chmy_p, chmy_v, chmy, 0.2Δx, Δx, grid, arch)
        end

        for k in 1:3
            @test agree(Array(Chmy.interior(chmy_p[k])), cpu_p[k])
        end
    end

    @testset "tuple scalar upwind retains conservative face velocity" begin
        n = 32
        Δx = inv(float(n))
        x = ((1:n) .- 0.5) .* Δx
        faces = (0:n) .* Δx
        initial() = 1 .+ 0.2 .* sinpi.(2 .* x)
        face_velocity = 0.8 .+ 0.3 .* sinpi.(2 .* faces)
        boundary = (PeriodicBC(), PeriodicBC())
        grid = Chmy.UniformGrid(
            arch; origin = (0.0,), extent = (1.0,), dims = (n,),
        )
        velocity = Chmy.VectorField(backend_chmy, grid)
        Chmy.set!(velocity.x, face_velocity)

        single = Chmy.Field(backend_chmy, grid, Chmy.Center())
        Chmy.set!(single, initial())
        single_scheme = WENOScheme(
            single, grid; form = :conservative, boundary, stag = true,
            upwind_mode = true,
        )
        WENO_step!(single, velocity, single_scheme, 0.2Δx, Δx, grid, arch)

        first = Chmy.Field(backend_chmy, grid, Chmy.Center())
        second = Chmy.Field(backend_chmy, grid, Chmy.Center())
        Chmy.set!(first, initial())
        Chmy.set!(second, 0.5 .* initial())
        tuple_scheme = WENOScheme(
            first, grid; form = :conservative, boundary, stag = true,
            upwind_mode = true,
        )
        WENO_step!(
            (first, second), velocity, tuple_scheme, 0.2Δx, Δx, grid, arch;
            u_min = (0.0, 0.0), u_max = (2.0, 2.0),
        )

        @test agree(Array(Chmy.interior(first)), Array(Chmy.interior(single)))
    end

    @testset "2D scalar staggered" begin
        n = 24
        Δ = inv(float(n))
        x = ((1:n) .- 0.5) .* Δ
        faces = (0:n) .* Δ
        initial() = [1 + 0.2 * sinpi(2xi) * cospi(2yj) for xi in x, yj in x]
        vx = [1 + 0.3 * sinpi(2xi) for xi in faces, _ in x]
        vy = [0.5 + 0.2 * cospi(2yj) for _ in x, yj in faces]
        boundary = ntuple(_ -> PeriodicBC(), 4)
        grid = Chmy.UniformGrid(
            arch; origin = (0.0, 0.0), extent = (1.0, 1.0), dims = (n, n),
        )

        for form in (:conservative, :nonconservative)
            cpu_u = initial()
            cpu = WENOScheme(cpu_u; form, boundary, stag = true, multithreading = false)

            chmy_u = Chmy.Field(backend_chmy, grid, Chmy.Center())
            Chmy.set!(chmy_u, initial())
            chmy_v = Chmy.VectorField(backend_chmy, grid)
            Chmy.set!(chmy_v.x, vx)
            Chmy.set!(chmy_v.y, vy)
            chmy = WENOScheme(chmy_u, grid; form, boundary, stag = true)

            for _ in 1:3
                WENO_step!(cpu_u, (; x = vx, y = vy), cpu, 0.2Δ, Δ, Δ)
                WENO_step!(chmy_u, chmy_v, chmy, 0.2Δ, Δ, Δ, grid, arch)
            end

            @test agree(Array(Chmy.interior(chmy_u)), cpu_u)
        end
    end

    @testset "3D scalar staggered" begin
        n = 10
        Δ = inv(float(n))
        x = ((1:n) .- 0.5) .* Δ
        faces = (0:n) .* Δ
        initial() = [
            1 + 0.2 * sinpi(2xi) * cospi(2yj) * cospi(2zk)
                for xi in x, yj in x, zk in x
        ]
        vx = [1 + 0.2 * sinpi(2xi) for xi in faces, _ in x, _ in x]
        vy = [0.5 + 0.15 * cospi(2yj) for _ in x, yj in faces, _ in x]
        vz = [0.3 + 0.1 * sinpi(2zk) for _ in x, _ in x, zk in faces]
        boundary = ntuple(_ -> PeriodicBC(), 6)
        grid = Chmy.UniformGrid(
            arch; origin = (0.0, 0.0, 0.0), extent = (1.0, 1.0, 1.0), dims = (n, n, n),
        )

        for form in (:conservative, :nonconservative)
            cpu_u = initial()
            cpu = WENOScheme(cpu_u; form, boundary, stag = true, multithreading = false)

            chmy_u = Chmy.Field(backend_chmy, grid, Chmy.Center())
            Chmy.set!(chmy_u, initial())
            chmy_v = Chmy.VectorField(backend_chmy, grid)
            Chmy.set!(chmy_v.x, vx)
            Chmy.set!(chmy_v.y, vy)
            Chmy.set!(chmy_v.z, vz)
            chmy = WENOScheme(chmy_u, grid; form, boundary, stag = true)

            for _ in 1:3
                WENO_step!(cpu_u, (; x = vx, y = vy, z = vz), cpu, 0.2Δ, Δ, Δ, Δ)
                WENO_step!(chmy_u, chmy_v, chmy, 0.2Δ, Δ, Δ, Δ, grid, arch)
            end

            @test agree(Array(Chmy.interior(chmy_u)), cpu_u)
        end
    end
end
