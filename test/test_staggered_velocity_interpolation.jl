using Test
using FiniteDiffWENO5

@testset "ENO5 staggered velocity interpolation" begin
    @testset "affine nonperiodic face data interpolate at cell centers" begin
        n = 8
        face = collect(0.0:n)
        center = zeros(n)

        FiniteDiffWENO5.eno5_face_to_center!(center, face; periodic = false)

        @test center ≈ collect(0.5:1.0:(n - 0.5)) atol = 128eps(Float64)
    end

    @testset "2D direction-labelled velocities interpolate only normally" begin
        nx, ny = 8, 7
        xface = [Float64(i - 1) + 10j for i in 1:(nx + 1), j in 1:ny]
        yface = [Float64(i) + 10(j - 1) for i in 1:nx, j in 1:(ny + 1)]
        center = (; x = zeros(nx, ny), y = zeros(nx, ny))

        FiniteDiffWENO5.eno5_face_to_center!(
            center, (; x = xface, y = yface); periodic = (; x = false, y = false),
        )

        @test center.x ≈ [i - 0.5 + 10j for i in 1:nx, j in 1:ny] atol = 128eps(Float64)
        @test center.y ≈ [i + 10(j - 0.5) for i in 1:nx, j in 1:ny] atol = 128eps(Float64)
    end

    @testset "periodic duplicate is ignored and constants are exact" begin
        n = 8
        face = fill(Float32(3.5), n + 1)
        center = zeros(Float32, n)
        FiniteDiffWENO5.eno5_face_to_center!(center, face; periodic = true)
        @test center == fill(Float32(3.5), n)

        varying = sinpi.(2 .* (0:n) ./ n)
        reference = zeros(n)
        FiniteDiffWENO5.eno5_face_to_center!(reference, varying; periodic = true)
        varying[end] = 123.0
        duplicate_changed = zeros(n)
        FiniteDiffWENO5.eno5_face_to_center!(duplicate_changed, varying; periodic = true)
        @test duplicate_changed == reference
    end

    @testset "3D direction-labelled velocities retain tangential coordinates" begin
        nx, ny, nz = 6, 5, 4
        xface = [Float64(i - 1) + 10j + 100k for i in 1:(nx + 1), j in 1:ny, k in 1:nz]
        yface = [Float64(i) + 10(j - 1) + 100k for i in 1:nx, j in 1:(ny + 1), k in 1:nz]
        zface = [Float64(i) + 10j + 100(k - 1) for i in 1:nx, j in 1:ny, k in 1:(nz + 1)]
        center = (; x = zeros(nx, ny, nz), y = zeros(nx, ny, nz), z = zeros(nx, ny, nz))

        FiniteDiffWENO5.eno5_face_to_center!(
            center, (; x = xface, y = yface, z = zface);
            periodic = (; x = false, y = false, z = false),
        )

        @test center.x ≈ [i - 0.5 + 10j + 100k for i in 1:nx, j in 1:ny, k in 1:nz] atol = 128eps(Float64)
        @test center.y ≈ [i + 10(j - 0.5) + 100k for i in 1:nx, j in 1:ny, k in 1:nz] atol = 128eps(Float64)
        @test center.z ≈ [i + 10j + 100(k - 0.5) for i in 1:nx, j in 1:ny, k in 1:nz] atol = 128eps(Float64)
    end

    @testset "minimum face counts are explicit" begin
        @test_throws ArgumentError FiniteDiffWENO5.eno5_face_to_center!(zeros(4), zeros(5); periodic = true)
        @test_throws ArgumentError FiniteDiffWENO5.eno5_face_to_center!(zeros(3), zeros(4); periodic = false)
        @test_throws DimensionMismatch FiniteDiffWENO5.eno5_face_to_center!(zeros(4), zeros(4); periodic = false)
    end

    @testset "smooth periodic interpolation is fifth order" begin
        errors = Float64[]
        for n in (16, 32, 64, 128)
            Δx = inv(float(n))
            face = sinpi.(2 .* (0:n) .* Δx)
            center = zeros(n)
            FiniteDiffWENO5.eno5_face_to_center!(center, face; periodic = true)
            exact = sinpi.(2 .* ((1:n) .- 0.5) .* Δx)
            push!(errors, Δx * sum(abs, center .- exact))
        end
        @test all(>(4.5), log2.(errors[1:(end - 1)] ./ errors[2:end]))
    end
end
