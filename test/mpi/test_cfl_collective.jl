using Test
using MPI
using FiniteDiffWENO5
using FiniteDiffWENO5: weno_cartesian_topology, weno_cfl_dt, weno_substeps,
    weno_owned_size, weno_global_offset, owned_window, NoTopology,
    weno_allreduce_max, weno_allreduce_min

const comm = MPI.COMM_WORLD
const nprocs = MPI.Comm_size(comm)
const rank = MPI.Comm_rank(comm)

"""Rank-local CFL reference (2D):
`speed = maximum(abs, Vx) / dx + maximum(abs, Vz) / dz`."""
function reference_cfl_dt_2d(Vx, Vz, dx, dz, cfl)
    speed = maximum(abs, Vx) / dx + maximum(abs, Vz) / dz
    return iszero(speed) ? Inf : cfl / speed
end

@testset "CFL collective" begin

    @testset "weno_cfl_dt: bitwise-identical across ranks, rank-asymmetric velocity" begin
        halo = 3
        global_dims = (24, 16)
        topo = weno_cartesian_topology(global_dims; comm, halo)
        owned = weno_owned_size(topo)
        offset = weno_global_offset(topo)

        vx = zeros(owned .+ 2halo)
        vz = zeros(owned .+ 2halo)
        ow_vx = owned_window(vx, topo; stagger = 1)
        ow_vz = owned_window(vz, topo; stagger = 2)
        # a value that depends on this rank's own offset, so ranks genuinely differ
        fill!(ow_vx, 1.0 + offset[1] + offset[2])
        fill!(ow_vz, 0.5 * (1.0 + offset[1] + offset[2]))

        cfl = 0.4
        dt = weno_cfl_dt(topo, (; x = vx, y = vz), (0.1, 0.2), cfl)

        all_dt = MPI.Allgather(dt, comm)
        @test all(==(all_dt[1]), all_dt) # bitwise-identical on every rank
    end

    @testset "weno_substeps: identical Int on every rank in all three degenerate cases" begin
        halo = 3
        topo = weno_cartesian_topology((16, 16); comm, halo)

        # 1. zero velocity everywhere -> dt_cfl == Inf -> 0 substeps
        n1 = weno_substeps(topo, 1.0, Inf)
        @test all(==(0), MPI.Allgather(n1, comm))

        # 2. zero duration -> 0 substeps regardless of dt_cfl
        n2 = weno_substeps(topo, 0.0, 0.01)
        @test all(==(0), MPI.Allgather(n2, comm))

        # 3. velocity zero on some ranks but not others: this manifests as a
        # rank-ASYMMETRIC dt_cfl in a naive implementation, but `weno_substeps`
        # itself takes an already-collective `dt_cfl` as input, so feeding it
        # the SAME (collectively-agreed) dt_cfl on every rank must give the
        # same nsub regardless of which ranks originally had zero velocity.
        owned = weno_owned_size(topo)
        v = zeros(owned .+ 2halo)
        ow = owned_window(v, topo; stagger = 1)
        rank == 0 && fill!(ow, 2.0) # nonzero on exactly one rank
        v2 = zeros(owned .+ 2halo) # the y-component, zero everywhere
        dt_cfl = weno_cfl_dt(topo, (; x = v, y = v2), (0.1, 0.1), 0.5)
        n3 = weno_substeps(topo, 1.0, dt_cfl)
        results = MPI.Allgather(n3, comm)
        @test all(==(results[1]), results)
    end

    @testset "regression: fails against a rank-local implementation (velocity nonzero on exactly one rank)" begin
        if nprocs > 1
            halo = 3
            topo = weno_cartesian_topology((16, 16); comm, halo)
            owned = weno_owned_size(topo)

            v = zeros(owned .+ 2halo)
            ow = owned_window(v, topo; stagger = 1)
            rank == 0 && fill!(ow, 3.0) # nonzero on exactly rank 0
            v2 = zeros(owned .+ 2halo)

            # sanity: this scenario genuinely discriminates a rank-local
            # implementation — the LOCAL max differs across ranks.
            local_max = maximum(abs, owned_window(v, topo; stagger = 1))
            local_maxes = MPI.Allgather(local_max, comm)
            @test !all(==(local_maxes[1]), local_maxes) # confirms the scenario is discriminating

            # the real, collective implementation must still agree everywhere
            dt = weno_cfl_dt(topo, (; x = v, y = v2), (0.1, 0.1), 0.5)
            dts = MPI.Allgather(dt, comm)
            @test all(==(dts[1]), dts)
        end
    end

    @testset "NoTopology matches the rank-local CFL formula bit-for-bit" begin
        nx, ny = 20, 14
        Vx = [1.0 + 0.3sinpi(2i / nx) for i in 0:nx, j in 1:ny]
        Vz = [0.5 + 0.2cospi(2j / ny) for i in 1:nx, j in 0:ny]
        dx, dz = 0.05, 0.07
        cfl = 0.45

        expected = reference_cfl_dt_2d(Vx, Vz, dx, dz, cfl)
        got = weno_cfl_dt(NoTopology(), (; x = Vx, y = Vz), (dx, dz), cfl)
        @test got === expected # bit-for-bit, not just approximately

        # the zero-velocity early return also matches
        @test weno_cfl_dt(NoTopology(), (; x = zeros(nx, ny), y = zeros(nx, ny)), (dx, dz), cfl) == Inf
        @test weno_substeps(NoTopology(), 1.0, Inf) == 0
        @test weno_substeps(NoTopology(), 0.0, 0.01) == 0
        @test weno_substeps(NoTopology(), 1.0, 0.03) == max(1, ceil(Int, 1.0 / 0.03))
    end

    @testset "ghost entries set to NaN do not affect weno_cfl_dt (owned window only)" begin
        halo = 3
        topo = weno_cartesian_topology((16, 16); comm, halo)
        owned = weno_owned_size(topo)

        vx = fill(NaN, owned .+ 2halo)
        vz = fill(NaN, owned .+ 2halo)
        fill!(owned_window(vx, topo; stagger = 1), 1.0)
        fill!(owned_window(vz, topo; stagger = 2), 0.5)

        dt = weno_cfl_dt(topo, (; x = vx, y = vz), (0.1, 0.2), 0.4)
        @test isfinite(dt) # NaN ghosts must not poison the reduction
        dts = MPI.Allgather(dt, comm)
        @test all(==(dts[1]), dts)
    end

    @testset "vertex-lattice CFL: disjoint owned windows, no dependence on ghost values" begin
        halo = 3
        topo = weno_cartesian_topology((12, 8); comm, halo)
        owned_v = weno_owned_size(topo; geometry = :vertex)

        v1 = fill(NaN, owned_v .+ 2halo)
        v2 = fill(NaN, owned_v .+ 2halo)
        offset = weno_global_offset(topo; geometry = :vertex)
        ow1 = owned_window(v1, topo; geometry = :vertex)
        ow2 = owned_window(v2, topo; geometry = :vertex)
        for I in CartesianIndices(ow1)
            gi, gj = Tuple(I) .+ offset
            ow1[I] = 0.01 * gi # a globally-unique-ish value so the seam vertex is included exactly once
            ow2[I] = 0.01 * gj
        end

        dt = weno_cfl_dt(topo, (; x = v1, y = v2), (0.1, 0.1), 0.4; geometry = :vertex, staggered = false)
        @test isfinite(dt)
        dts = MPI.Allgather(dt, comm)
        @test all(==(dts[1]), dts)

        # The corner vertex is owned and must contribute to the global max;
        # ghost values are NaN here.
        speed = 0.01 * 13 / 0.1 + 0.01 * 9 / 0.1
        @test dt ≈ 0.4 / speed
    end

    # Tuple reductions must be elementwise. MPI.jl reduces a tuple with Julia's
    # `max`/`min`, which compare tuples lexicographically, so component 2 would
    # silently come from whichever rank has the largest component 1.
    @testset "weno_allreduce_max/min on an NTuple reduce each component independently" begin
        topo = weno_cartesian_topology((16, 16); comm, halo = 3)
        local_value = (Float64(rank), Float64(nprocs - 1 - rank))
        @test weno_allreduce_max(local_value, topo) == (Float64(nprocs - 1), Float64(nprocs - 1))
        @test weno_allreduce_min(local_value, topo) == (0.0, 0.0)
    end

    # The velocity components peak on different ranks, so a lexicographic
    # reduction takes the wrong y maximum and returns too large a timestep.
    @testset "weno_cfl_dt matches the global reference when components peak on different ranks" begin
        topo = weno_cartesian_topology((24, 16); comm, halo = 3)
        owned = weno_owned_size(topo)
        vx = zeros(owned .+ 6)
        vz = zeros(owned .+ 6)
        fill!(owned_window(vx, topo; stagger = 1), rank == 0 ? 2.0 : 0.1)
        fill!(owned_window(vz, topo; stagger = 2), rank == nprocs - 1 ? 5.0 : 0.1)

        dx, dz, cfl = 0.1, 0.2, 0.4
        dt = weno_cfl_dt(topo, (; x = vx, y = vz), (dx, dz), cfl)

        vx_max = MPI.Allreduce(maximum(abs, owned_window(vx, topo; stagger = 1)), max, comm)
        vz_max = MPI.Allreduce(maximum(abs, owned_window(vz, topo; stagger = 2)), max, comm)
        @test dt ≈ cfl / (vx_max / dx + vz_max / dz)
    end

    @testset "scheme_lf_speeds matches the global reference when components peak on different ranks" begin
        topo = weno_cartesian_topology((24, 16); comm, halo = 3)
        owned = weno_owned_size(topo)
        c0 = zeros(owned .+ 6)
        weno = WENOScheme(
            c0, topo; boundary = ntuple(_ -> ExtrapolateBC(), 4), form = :conservative,
            stag = false, multithreading = false,
        )
        vx = zeros(owned .+ 6)
        vy = zeros(owned .+ 6)
        fill!(owned_window(vx, topo), rank == 0 ? 2.0 : 0.1)
        fill!(owned_window(vy, topo), rank == nprocs - 1 ? 5.0 : 0.1)

        speeds = FiniteDiffWENO5.scheme_lf_speeds(weno, (; x = vx, y = vy))

        @test speeds.x == MPI.Allreduce(maximum(abs, owned_window(vx, topo)), max, comm)
        @test speeds.y == MPI.Allreduce(maximum(abs, owned_window(vy, topo)), max, comm)
    end

    # A rank that sees dt_cfl == Inf must still take part in the debug check;
    # returning early left the other ranks blocked in the reduction forever.
    @testset "weno_substeps(debug = true) throws on every rank when ranks disagree" begin
        if nprocs > 1
            topo = weno_cartesian_topology((16, 16); comm, halo = 3)
            dt_cfl = rank == 0 ? Inf : 1.0
            @test_throws ArgumentError weno_substeps(topo, 1.0, dt_cfl; debug = true)
        end
    end
end

@testset "CFL excludes an unfilled periodic duplicate before preparation" begin
    topo = weno_cartesian_topology((8nprocs, 8); comm, dims = (nprocs, 1), periodic = (false, true))
    vx = allocate_weno_field(topo; stagger = 1)
    vy = allocate_weno_field(topo; stagger = 2)
    fill!(vx, NaN)
    fill!(vy, NaN)
    owned_window(vx, topo; stagger = 1) .= 2.0
    owned_window(vy, topo; stagger = 2) .= 1.0
    # The unused high periodic duplicate is deliberately still NaN.
    @test all(isnan, vy[:, 12])
    @test weno_cfl_dt(topo, (vx, vy), (1.0, 1.0), 0.5) == 0.5 / 3
end
