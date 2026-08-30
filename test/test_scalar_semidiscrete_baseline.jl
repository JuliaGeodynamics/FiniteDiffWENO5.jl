using Test
using FiniteDiffWENO5

@testset "scalar semidiscrete baseline" begin
    @testset "ENO-prepared staggered velocity restores fifth-order material transport" begin
        errors = Float64[]
        for n in (32, 64, 128, 256)
            Δx = inv(float(n))
            x = ((1:n) .- 0.5) .* Δx
            u = 1 .+ 0.2 .* sinpi.(2 .* x) .+ 0.1 .* cospi.(4 .* x)
            v_face = 1 .+ 0.3 .* sinpi.(2 .* ((0:n) .* Δx))
            v_center = zeros(n)
            FiniteDiffWENO5.eno5_face_to_center!(v_center, v_face; periodic = true)
            weno = WENOScheme(u; boundary = (PeriodicBC(), PeriodicBC()), form = :nonconservative, stag = false, multithreading = false)

            FiniteDiffWENO5.WENO_flux!(weno.fl, weno.fr, u, weno, n, 0.0, 0.0)
            FiniteDiffWENO5.material_semi_discretisation_weno5!(weno.du, (; x = v_center), weno, inv(Δx))

            ux = 0.4π .* cospi.(2 .* x) .- 0.4π .* sinpi.(4 .* x)
            push!(errors, Δx * sum(abs, weno.du .- v_center .* ux))
        end
        @test all(>(4.5), log2.(errors[1:(end - 1)] ./ errors[2:end]))
    end

    @testset "2D ENO-prepared staggered velocity restores material transport" begin
        errors = Float64[]
        for n in (16, 32, 64, 128)
            Δ = inv(float(n))
            x = ((1:n) .- 0.5) .* Δ
            y = x
            u = [1 + 0.2sinpi(2xi) * cospi(2yj) for xi in x, yj in y]
            vx_face = [1 + 0.3sinpi(2xi) for xi in (0:n) .* Δ, _ in y]
            vy_face = [0.5 + 0.2cospi(2yj) for _ in x, yj in (0:n) .* Δ]
            vcenter = (; x = zeros(n, n), y = zeros(n, n))
            FiniteDiffWENO5.eno5_face_to_center!(
                vcenter, (; x = vx_face, y = vy_face); periodic = (; x = true, y = true),
            )
            weno = WENOScheme(u; boundary = ntuple(_ -> PeriodicBC(), 4), form = :nonconservative, stag = false, multithreading = false)

            FiniteDiffWENO5.WENO_flux!(weno.fl, weno.fr, u, weno, n, n, 0.0, 0.0)
            FiniteDiffWENO5.material_semi_discretisation_weno5!(weno.du, vcenter, weno, inv(Δ), inv(Δ))

            ux = [0.4π * cospi(2xi) * cospi(2yj) for xi in x, yj in y]
            uy = [-0.4π * sinpi(2xi) * sinpi(2yj) for xi in x, yj in y]
            push!(errors, Δ^2 * sum(abs, weno.du .- vcenter.x .* ux .- vcenter.y .* uy))
        end
        @test all(>(4.5), log2.(errors[1:(end - 1)] ./ errors[2:end]))
    end

    @testset "3D ENO-prepared staggered velocity restores material transport" begin
        errors = Float64[]
        for n in (8, 16, 32, 64)
            Δ = inv(float(n))
            x = ((1:n) .- 0.5) .* Δ
            u = [1 + 0.1sinpi(2xi) * cospi(2yj) * cospi(2zk) for xi in x, yj in x, zk in x]
            vx_face = [1 + 0.3sinpi(2xi) for xi in (0:n) .* Δ, _ in x, _ in x]
            vy_face = [0.5 + 0.2cospi(2yj) for _ in x, yj in (0:n) .* Δ, _ in x]
            vz_face = [0.4 + 0.1sinpi(2zk) for _ in x, _ in x, zk in (0:n) .* Δ]
            vcenter = (; x = zeros(n, n, n), y = zeros(n, n, n), z = zeros(n, n, n))
            FiniteDiffWENO5.eno5_face_to_center!(
                vcenter, (; x = vx_face, y = vy_face, z = vz_face);
                periodic = (; x = true, y = true, z = true),
            )
            weno = WENOScheme(u; boundary = ntuple(_ -> PeriodicBC(), 6), form = :nonconservative, stag = false, multithreading = false)

            FiniteDiffWENO5.WENO_flux!(weno.fl, weno.fr, u, weno, n, n, n, 0.0, 0.0)
            FiniteDiffWENO5.material_semi_discretisation_weno5!(weno.du, vcenter, weno, inv(Δ), inv(Δ), inv(Δ))

            ux = [0.2π * cospi(2xi) * cospi(2yj) * cospi(2zk) for xi in x, yj in x, zk in x]
            uy = [-0.2π * sinpi(2xi) * sinpi(2yj) * cospi(2zk) for xi in x, yj in x, zk in x]
            uz = [-0.2π * sinpi(2xi) * cospi(2yj) * sinpi(2zk) for xi in x, yj in x, zk in x]
            push!(errors, Δ^3 * sum(abs, weno.du .- vcenter.x .* ux .- vcenter.y .* uy .- vcenter.z .* uz))
        end
        @test all(>(4.5), log2.(errors[1:(end - 1)] ./ errors[2:end]))
    end

    @testset "non-conservative staggered API prepares velocity once per step" begin
        n = 32
        Δx = inv(float(n))
        x = ((1:n) .- 0.5) .* Δx
        initial = 1 .+ 0.2 .* sinpi.(2 .* x)
        face_velocity = 1 .+ 0.3 .* sinpi.(2 .* ((0:n) .* Δx))
        center_velocity = zeros(n)
        FiniteDiffWENO5.eno5_face_to_center!(center_velocity, face_velocity; periodic = true)
        boundary = (PeriodicBC(), PeriodicBC())

        from_faces = copy(initial)
        from_centers = copy(initial)
        staggered = WENOScheme(from_faces; form = :nonconservative, boundary, stag = true, multithreading = false)
        collocated = WENOScheme(from_centers; form = :nonconservative, boundary, stag = false, multithreading = false)
        WENO_step!(from_faces, (; x = face_velocity), staggered, 0.1Δx, Δx)
        WENO_step!(from_centers, (; x = center_velocity), collocated, 0.1Δx, Δx)

        @test from_faces ≈ from_centers rtol = 0 atol = 16eps(Float64)
    end
end
