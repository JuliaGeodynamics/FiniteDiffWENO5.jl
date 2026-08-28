struct ChmyForeignBackend <: KernelAbstractions.Backend end

struct ChmyTaggedProfile{T, N} <: AbstractArray{T, N}
    data::Array{T, N}
end

Base.size(profile::ChmyTaggedProfile) = size(profile.data)
Base.axes(profile::ChmyTaggedProfile) = axes(profile.data)
Base.IndexStyle(::Type{<:ChmyTaggedProfile}) = IndexCartesian()
Base.getindex(profile::ChmyTaggedProfile, I...) = getindex(profile.data, I...)
KernelAbstractions.get_backend(::ChmyTaggedProfile) = ChmyForeignBackend()

@testset "multiphase Chmy CPU" begin
    backend = CPU()
    arch = Arch(backend)
    periodic_chmy(N) = ntuple(_ -> PeriodicBC(), 2N)

    function smooth_chmy(dims)
        p1 = Array{Float64}(undef, dims)
        p2 = similar(p1)
        for I in CartesianIndices(p1)
            x = (I[1] - 0.5) / dims[1]
            y = length(dims) >= 2 ? (I[2] - 0.5) / dims[2] : 0.0
            z = length(dims) == 3 ? (I[3] - 0.5) / dims[3] : 0.0
            p1[I] = 0.3 + 0.06sinpi(2x) * cospi(2y)
            p2[I] = 0.3 + 0.06cospi(2x) * cospi(2z)
        end
        return (p1, p2, 1 .- p1 .- p2)
    end

    function chmy_phases(grid, initial; sentinel = -7.0)
        fields = ntuple(length(initial)) do q
            f = Field(backend, grid, Center())
            fill!(parent(f), sentinel)
            set!(f, initial[q])
            f
        end
        return fields
    end

    function halos_equal(f, sentinel)
        data = parent(f)
        h = Chmy.halo(f)
        ranges = ntuple(d -> (2h + 1):(2h + size(f, d)), ndims(f))
        for I in CartesianIndices(data)
            inside = all(d -> I[d] in ranges[d], 1:ndims(f))
            !inside && data[I] != sentinel && return false
        end
        return true
    end

    @testset "inflow profiles are adapted to the field backend" begin
        nx, ny = 8, 6
        grid = UniformGrid(
            arch; origin = (0.0, 0.0), extent = (1.0, 1.0), dims = (nx, ny)
        )
        fields = chmy_phases(grid, (fill(0.4, nx, ny), fill(0.6, nx, ny)))
        profile1 = ChmyTaggedProfile(fill(0.4, ny))
        profile2 = ChmyTaggedProfile(fill(0.6, ny))
        boundary = (
            PrescribedInflowBC((profile1, profile2)), ExtrapolateBC(),
            PeriodicBC(), PeriodicBC(),
        )

        scheme = MultiphaseWENOScheme(fields, grid; boundary = boundary, stag = true)

        for component in scheme.boundary[1].value
            @test get_backend(component) == backend
        end
        @test scheme.boundary[1].value[1] == profile1.data
        @test scheme.boundary[1].value[2] == profile2.data
    end

    @testset "1D equivalence, divergent cancellation, and halos" begin
        nx = 24
        dx = 1 / nx
        grid = UniformGrid(arch; origin = (0.0,), extent = (1.0,), dims = (nx,))
        constant = (0.15, 0.35, 0.5)
        initial = ntuple(q -> fill(constant[q], nx), 3)
        ka = map(copy, initial)
        fields = chmy_phases(grid, initial)
        vhost = (; x = collect(range(0.3, 0.8, length = nx + 1)))
        velocity = VectorField(backend, grid)
        set!(velocity.x, vhost.x)

        ska = MultiphaseWENOScheme(
            ka, backend; boundary = periodic_chmy(1), stag = true
        )
        schmy = MultiphaseWENOScheme(
            fields, grid; boundary = periodic_chmy(1), stag = true
        )
        for _ in 1:20
            WENO_step!(ka, vhost, ska, 0.05dx, dx, backend)
            WENO_step!(fields, velocity, schmy, 0.05dx, dx, grid, arch)
        end

        for q in 1:3
            @test Array(interior(fields[q])) ≈ ka[q] rtol = 128eps(Float64) atol = 128eps(Float64)
            @test maximum(abs, Array(interior(fields[q])) .- constant[q]) <= 128eps(Float64)
            @test halos_equal(fields[q], -7.0)
        end
    end

    @testset "2D tangential inflow matches KA" begin
        nx, ny = 14, 12
        dx, dy = 1 / nx, 1 / ny
        grid = UniformGrid(
            arch; origin = (0.0, 0.0), extent = (1.0, 1.0), dims = (nx, ny)
        )
        initial = smooth_chmy((nx, ny))
        profile1 = collect(range(0.55, 0.7, length = ny))
        profile2 = collect(range(0.3, 0.2, length = ny))
        profile3 = 1 .- profile1 .- profile2
        boundary = (
            PrescribedInflowBC((profile1, profile2, profile3)), ExtrapolateBC(),
            PeriodicBC(), PeriodicBC(),
        )
        ka = map(copy, initial)
        fields = chmy_phases(grid, initial)
        vhost = (; x = fill(0.5, nx + 1, ny), y = fill(-0.2, nx, ny + 1))
        velocity = VectorField(backend, grid)
        set!(velocity.x, vhost.x)
        set!(velocity.y, vhost.y)

        ska = MultiphaseWENOScheme(ka, backend; boundary = boundary, stag = true)
        schmy = MultiphaseWENOScheme(fields, grid; boundary = boundary, stag = true)
        dt = 0.06min(dx, dy)
        for _ in 1:20
            WENO_step!(ka, vhost, ska, dt, dx, dy, backend)
            WENO_step!(fields, velocity, schmy, dt, dx, dy, grid, arch)
        end
        result = map(f -> Array(interior(f)), fields)
        for q in 1:3
            @test result[q] ≈ ka[q] rtol = 128eps(Float64) atol = 128eps(Float64)
        end
        @test minimum(minimum, result) >= -1024eps(Float64)
        @test maximum(maximum, result) <= 1 + 1024eps(Float64)
        @test maximum(abs, reduce(+, result) .- 1) <= 1024eps(Float64)

        collocated_ka = map(copy, initial)
        collocated_fields = chmy_phases(grid, initial)
        vcollocated_host = (; x = fill(0.4, nx, ny), y = fill(-0.1, nx, ny))
        vcollocated = (;
            x = Field(backend, grid, Center()),
            y = Field(backend, grid, Center()),
        )
        set!(vcollocated.x, vcollocated_host.x)
        set!(vcollocated.y, vcollocated_host.y)
        cka = MultiphaseWENOScheme(
            collocated_ka, backend; boundary = periodic_chmy(2), stag = false
        )
        cchmy = MultiphaseWENOScheme(
            collocated_fields, grid; boundary = periodic_chmy(2), stag = false
        )
        for _ in 1:20
            WENO_step!(collocated_ka, vcollocated_host, cka, dt, dx, dy, backend)
            WENO_step!(collocated_fields, vcollocated, cchmy, dt, dx, dy, grid, arch)
        end
        for q in 1:3
            @test Array(interior(collocated_fields[q])) ≈ collocated_ka[q] rtol = 128eps(Float64) atol = 128eps(Float64)
        end
    end

    @testset "3D equivalence smoke test" begin
        nx, ny, nz = 8, 8, 8
        dx, dy, dz = 1 / nx, 1 / ny, 1 / nz
        grid = UniformGrid(
            arch; origin = (0.0, 0.0, 0.0),
            extent = (1.0, 1.0, 1.0), dims = (nx, ny, nz)
        )
        initial = smooth_chmy((nx, ny, nz))
        ka = map(copy, initial)
        fields = chmy_phases(grid, initial)
        vhost = (;
            x = fill(0.3, nx + 1, ny, nz),
            y = fill(-0.2, nx, ny + 1, nz),
            z = fill(0.1, nx, ny, nz + 1),
        )
        velocity = VectorField(backend, grid)
        set!(velocity.x, vhost.x)
        set!(velocity.y, vhost.y)
        set!(velocity.z, vhost.z)

        ska = MultiphaseWENOScheme(
            ka, backend; boundary = periodic_chmy(3), stag = true
        )
        schmy = MultiphaseWENOScheme(
            fields, grid; boundary = periodic_chmy(3), stag = true
        )
        dt = 0.04min(dx, dy, dz)
        for _ in 1:10
            WENO_step!(ka, vhost, ska, dt, dx, dy, dz, backend)
            WENO_step!(fields, velocity, schmy, dt, dx, dy, dz, grid, arch)
        end
        result = map(f -> Array(interior(f)), fields)
        for q in 1:3
            @test result[q] ≈ ka[q] rtol = 128eps(Float64) atol = 128eps(Float64)
        end
        @test minimum(minimum, result) >= -1024eps(Float64)
        @test maximum(maximum, result) <= 1 + 1024eps(Float64)
        @test maximum(abs, reduce(+, result) .- 1) <= 1024eps(Float64)
    end
end
