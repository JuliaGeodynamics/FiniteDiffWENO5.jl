struct MultiphaseForeignBackend <: KernelAbstractions.Backend end

struct MultiphaseTaggedArray{T, N} <: AbstractArray{T, N}
    data::Array{T, N}
    foreign::Bool
end

Base.size(a::MultiphaseTaggedArray) = size(a.data)
Base.axes(a::MultiphaseTaggedArray) = axes(a.data)
Base.IndexStyle(::Type{<:MultiphaseTaggedArray}) = IndexCartesian()
Base.getindex(a::MultiphaseTaggedArray, I...) = getindex(a.data, I...)
Base.setindex!(a::MultiphaseTaggedArray, x, I...) = setindex!(a.data, x, I...)
KernelAbstractions.get_backend(a::MultiphaseTaggedArray) =
    a.foreign ? MultiphaseForeignBackend() : CPU()

@testset "multiphase KernelAbstractions CPU" begin
    backend = CPU()
    periodic_ka(N) = ntuple(_ -> PeriodicBC(), 2N)
    maxsumerr_ka(p) = maximum(abs, reduce(+, p) .- 1)

    function smooth_ka(dims)
        p1 = Array{Float64}(undef, dims)
        p2 = similar(p1)
        for I in CartesianIndices(p1)
            x = (I[1] - 0.5) / dims[1]
            y = length(dims) >= 2 ? (I[2] - 0.5) / dims[2] : 0.0
            z = length(dims) == 3 ? (I[3] - 0.5) / dims[3] : 0.0
            p1[I] = 0.30 + 0.06sinpi(2x) * cospi(2y)
            p2[I] = 0.30 + 0.06cospi(2x) * cospi(2z)
        end
        return (p1, p2, 1 .- p1 .- p2)
    end

    @testset "backend cache and mismatch validation" begin
        phases = (fill(0.4, 12), fill(0.6, 12))
        scheme = MultiphaseWENOScheme(
            phases, backend; boundary = periodic_ka(1), stag = true)
        @test all(a -> get_backend(a) == backend, scheme.du)
        @test all(a -> get_backend(a) == backend, scheme.ut)
        @test all(a -> get_backend(a) == backend, scheme.fl.x)
        @test get_backend(scheme.divv) == backend

        mixed = (
            MultiphaseTaggedArray(fill(0.4, 12), false),
            MultiphaseTaggedArray(fill(0.6, 12), true),
        )
        @test_throws AssertionError MultiphaseWENOScheme(
            mixed, backend; boundary = periodic_ka(1), stag = true)

        profile_boundary = (
            PrescribedInflowBC((
                MultiphaseTaggedArray(fill(0.4, 6), false),
                MultiphaseTaggedArray(fill(0.6, 6), true),
            )),
            ExtrapolateBC(), PeriodicBC(), PeriodicBC(),
        )
        phase2D = (fill(0.4, 8, 6), fill(0.6, 8, 6))
        @test_throws AssertionError MultiphaseWENOScheme(
            phase2D, backend; boundary = profile_boundary, stag = true)
    end

    @testset "1D matches the serial operator" begin
        nx = 36
        dx = 1 / nx
        dt = 0.12dx
        initial = smooth_ka((nx,))
        serial = map(copy, initial)
        ka = map(copy, initial)
        v = (; x = fill(0.7, nx + 1))

        ss = MultiphaseWENOScheme(
            serial; boundary = periodic_ka(1), stag = true, multithreading = false)
        sk = MultiphaseWENOScheme(
            ka, backend; boundary = periodic_ka(1), stag = true)
        for _ in 1:2
            WENO_step!(serial, v, ss, dt, dx)
            WENO_step!(ka, v, sk, dt, dx, backend)
        end
        for q in 1:3
            @test ka[q] ≈ serial[q] rtol = 128eps(Float64) atol = 128eps(Float64)
        end
        @test maxsumerr_ka(ka) <= 1024eps(Float64)
    end

    @testset "divergent flow and constant prescribed inflow" begin
        nx = 32
        dx = 1 / nx
        constant = (0.15, 0.35, 0.50)
        phases = ntuple(q -> fill(constant[q], nx), 3)
        vdiv = (; x = collect(range(0.3, 0.8, length = nx + 1)))
        scheme = MultiphaseWENOScheme(
            phases, backend; boundary = periodic_ka(1), stag = true)
        WENO_step!(phases, vdiv, scheme, 0.05dx, dx, backend)
        for q in 1:3
            @test maximum(abs, phases[q] .- constant[q]) <= 128eps(Float64)
        end

        boundary = (PrescribedInflowBC((0.7, 0.2, 0.1)), ExtrapolateBC())
        serial = ntuple(q -> fill(constant[q], nx), 3)
        ka = map(copy, serial)
        v = (; x = fill(0.6, nx + 1))
        ss = MultiphaseWENOScheme(
            serial; boundary = boundary, stag = true, multithreading = false)
        sk = MultiphaseWENOScheme(ka, backend; boundary = boundary, stag = true)
        WENO_step!(serial, v, ss, 0.1dx, dx)
        WENO_step!(ka, v, sk, 0.1dx, dx, backend)
        for q in 1:3
            @test ka[q] ≈ serial[q] rtol = 128eps(Float64) atol = 128eps(Float64)
        end
    end

    @testset "2D matches serial with tangential inflow" begin
        nx, ny = 16, 14
        dx, dy = 1 / nx, 1 / ny
        initial = smooth_ka((nx, ny))
        profile1 = collect(range(0.55, 0.70, length = ny))
        profile2 = collect(range(0.30, 0.20, length = ny))
        profile3 = 1 .- profile1 .- profile2
        boundary = (
            PrescribedInflowBC((profile1, profile2, profile3)), ExtrapolateBC(),
            PeriodicBC(), PeriodicBC(),
        )
        v = (; x = fill(0.5, nx + 1, ny), y = fill(-0.2, nx, ny + 1))
        serial = map(copy, initial)
        ka = map(copy, initial)
        ss = MultiphaseWENOScheme(
            serial; boundary = boundary, stag = true, multithreading = false)
        sk = MultiphaseWENOScheme(ka, backend; boundary = boundary, stag = true)
        for _ in 1:10
            WENO_step!(serial, v, ss, 0.08min(dx, dy), dx, dy)
            WENO_step!(ka, v, sk, 0.08min(dx, dy), dx, dy, backend)
        end
        for q in 1:3
            @test ka[q] ≈ serial[q] rtol = 128eps(Float64) atol = 128eps(Float64)
        end
        @test maxsumerr_ka(ka) <= 1024eps(Float64)

        collocated_serial = map(copy, initial)
        collocated_ka = map(copy, initial)
        vc = (; x = fill(0.4, nx, ny), y = fill(-0.1, nx, ny))
        cs = MultiphaseWENOScheme(collocated_serial;
            boundary = periodic_ka(2), stag = false, multithreading = false)
        ck = MultiphaseWENOScheme(
            collocated_ka, backend; boundary = periodic_ka(2), stag = false)
        for _ in 1:10
            WENO_step!(collocated_serial, vc, cs, 0.08min(dx, dy), dx, dy)
            WENO_step!(collocated_ka, vc, ck, 0.08min(dx, dy), dx, dy, backend)
        end
        for q in 1:3
            @test collocated_ka[q] ≈ collocated_serial[q] rtol = 128eps(Float64) atol = 128eps(Float64)
        end
    end

    @testset "3D matches the serial operator" begin
        nx, ny, nz = 8, 9, 10
        dx, dy, dz = 1 / nx, 1 / ny, 1 / nz
        initial = smooth_ka((nx, ny, nz))
        serial = map(copy, initial)
        ka = map(copy, initial)
        v = (;
            x = fill(0.3, nx + 1, ny, nz),
            y = fill(-0.2, nx, ny + 1, nz),
            z = fill(0.1, nx, ny, nz + 1),
        )
        ss = MultiphaseWENOScheme(
            serial; boundary = periodic_ka(3), stag = true, multithreading = false)
        sk = MultiphaseWENOScheme(
            ka, backend; boundary = periodic_ka(3), stag = true)
        for _ in 1:10
            WENO_step!(serial, v, ss, 0.05min(dx, dy, dz), dx, dy, dz)
            WENO_step!(ka, v, sk, 0.05min(dx, dy, dz), dx, dy, dz, backend)
        end
        for q in 1:3
            @test ka[q] ≈ serial[q] rtol = 128eps(Float64) atol = 128eps(Float64)
        end
        @test maxsumerr_ka(ka) <= 1024eps(Float64)
        @test all(p -> all(x -> -128eps(Float64) <= x <= 1 + 128eps(Float64), p), ka)
    end

    @testset "3D tangential inflow profiles on every face" begin
        nx, ny, nz = 7, 8, 9
        dx, dy, dz = 1 / nx, 1 / ny, 1 / nz
        initial = smooth_ka((nx, ny, nz))

        for face in 1:6
            direction = (face + 1) ÷ 2
            tangential = direction == 1 ? (ny, nz) :
                direction == 2 ? (nx, nz) : (nx, ny)
            profile1 = Array{Float64}(undef, tangential)
            profile2 = similar(profile1)
            for I in CartesianIndices(profile1)
                ξ = sum(Tuple(I)) / sum(tangential)
                profile1[I] = 0.45 + 0.05ξ
                profile2[I] = 0.30 - 0.02ξ
            end
            profile3 = 1 .- profile1 .- profile2
            inflow = PrescribedInflowBC((profile1, profile2, profile3))
            boundary = ntuple(f -> f == face ? inflow : ExtrapolateBC(), 6)

            speed = isodd(face) ? 0.4 : -0.4
            velocity = (;
                x = fill(direction == 1 ? speed : 0.0, nx + 1, ny, nz),
                y = fill(direction == 2 ? speed : 0.0, nx, ny + 1, nz),
                z = fill(direction == 3 ? speed : 0.0, nx, ny, nz + 1),
            )
            serial = map(copy, initial)
            ka = map(copy, initial)
            ss = MultiphaseWENOScheme(
                serial; boundary = boundary, stag = true, multithreading = false)
            sk = MultiphaseWENOScheme(
                ka, backend; boundary = boundary, stag = true)

            WENO_step!(serial, velocity, ss, 0.05min(dx, dy, dz), dx, dy, dz)
            WENO_step!(ka, velocity, sk, 0.05min(dx, dy, dz), dx, dy, dz, backend)

            for q in 1:3
                @test ka[q] ≈ serial[q] rtol = 128eps(Float64) atol = 128eps(Float64)
            end
        end
    end
end
