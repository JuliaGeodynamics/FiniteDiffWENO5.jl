import FiniteDiffWENO5: validate_multiphase_boundary, validate_multiphase_inflow,
    normalize_boundary_faces, multiphase_inflow_value,
    apply_multiphase_inflow_boundaries!, nphases

@testset "multiphase boundary compositions" begin

    @testset "periodic and extrapolate are unchanged" begin
        phases = (zeros(8), zeros(8), zeros(8))
        for bc in (PeriodicBC(), ExtrapolateBC())
            scheme = MultiphaseWENOScheme(phases; boundary = (bc, bc))
            @test scheme.boundary == (bc, bc)
        end
        # legacy integer codes still normalize
        @test MultiphaseWENOScheme(phases; boundary = (2, 2)).boundary ==
            (PeriodicBC(), PeriodicBC())
        @test MultiphaseWENOScheme(phases; boundary = (0, 1)).boundary ==
            (ExtrapolateBC(), ExtrapolateBC())
        @test_throws ArgumentError MultiphaseWENOScheme(
            phases; boundary = (PeriodicBC(), ExtrapolateBC()), stag = true,
        )
    end

    @testset "constant inflow compositions" begin
        phases = (zeros(8), zeros(8), zeros(8))
        west = PrescribedInflowBC((0.2, 0.3, 0.5))
        scheme = MultiphaseWENOScheme(phases; boundary = (west, ExtrapolateBC()))
        @test scheme.boundary[1] === west

        @test multiphase_inflow_value(west, 1) == 0.2
        @test multiphase_inflow_value(west, 3) == 0.5

        # wrong phase count
        @test_throws ArgumentError MultiphaseWENOScheme(
            phases; boundary = (PrescribedInflowBC((0.5, 0.5)), ExtrapolateBC())
        )
        # sums that miss one
        @test_throws ArgumentError MultiphaseWENOScheme(
            phases; boundary = (PrescribedInflowBC((0.2, 0.3, 0.49)), ExtrapolateBC())
        )
        @test_throws ArgumentError MultiphaseWENOScheme(
            phases; boundary = (PrescribedInflowBC((0.2, 0.3, 0.51)), ExtrapolateBC())
        )
        # negative and out-of-range components
        @test_throws ArgumentError MultiphaseWENOScheme(
            phases; boundary = (PrescribedInflowBC((-0.1, 0.6, 0.5)), ExtrapolateBC())
        )
        @test_throws ArgumentError MultiphaseWENOScheme(
            phases; boundary = (PrescribedInflowBC((1.4, -0.2, -0.2)), ExtrapolateBC())
        )
        # NaN
        @test_throws ArgumentError MultiphaseWENOScheme(
            phases; boundary = (PrescribedInflowBC((NaN, 0.3, 0.5)), ExtrapolateBC())
        )
        # a scalar value instead of a composition
        @test_throws ArgumentError MultiphaseWENOScheme(
            phases; boundary = (PrescribedInflowBC(0.3), ExtrapolateBC())
        )
        # a non-numeric component
        @test_throws ArgumentError MultiphaseWENOScheme(
            phases; boundary = (PrescribedInflowBC((0.2, 0.3, "x")), ExtrapolateBC())
        )

        # a sum error just inside the tolerance is accepted
        ok = PrescribedInflowBC((0.2, 0.3, 0.5 + 8eps(Float64)))
        @test MultiphaseWENOScheme(phases; boundary = (ok, ExtrapolateBC())) isa
            MultiphaseWENOScheme
    end

    @testset "tangential profile compositions" begin
        ny = 6
        phases = (zeros(5, ny), zeros(5, ny), zeros(5, ny))
        p1 = fill(0.2, ny)
        p2 = collect(range(0.1, 0.3; length = ny))
        p3 = 1 .- p1 .- p2
        west = PrescribedInflowBC((p1, p2, p3))
        scheme = MultiphaseWENOScheme(
            phases;
            boundary = AdvectionBC(
                west = west, east = ExtrapolateBC(),
                bot = PeriodicBC(), top = PeriodicBC(),
            ),
            stag = true,
        )
        @test scheme.boundary[1] === west
        @test multiphase_inflow_value(west, 2, 3) == p2[3]

        # mixed scalar and profile components are allowed if they still sum to one
        mixed = PrescribedInflowBC((0.2, p2, 0.8 .- p2))
        @test MultiphaseWENOScheme(
            phases; boundary = AdvectionBC(west = mixed, east = ExtrapolateBC()),
        ) isa MultiphaseWENOScheme

        # wrong profile length
        bad_shape = PrescribedInflowBC((fill(0.2, ny + 1), p2, p3))
        @test_throws DimensionMismatch MultiphaseWENOScheme(
            phases; boundary = AdvectionBC(west = bad_shape, east = ExtrapolateBC())
        )

        # pointwise sum violated at a single node
        broken = copy(p3)
        broken[4] += 0.05
        @test_throws ArgumentError MultiphaseWENOScheme(
            phases;
            boundary = AdvectionBC(
                west = PrescribedInflowBC((p1, p2, broken)), east = ExtrapolateBC()
            ),
        )

        # a nonfinite entry inside a profile
        nan_profile = copy(p1)
        nan_profile[2] = NaN
        @test_throws ArgumentError MultiphaseWENOScheme(
            phases;
            boundary = AdvectionBC(
                west = PrescribedInflowBC((nan_profile, p2, p3)), east = ExtrapolateBC()
            ),
        )
    end

    @testset "scalar route still rejects tuple inflow" begin
        # this is what keeps a phase vector from being silently accepted by WENOScheme
        @test_throws ArgumentError WENOScheme(
            zeros(8);
            boundary = (PrescribedInflowBC((0.2, 0.3, 0.5)), ExtrapolateBC()),
            form = :nonconservative,
            stag = false,
        )

        # the refactor left face normalization behaviourally identical
        @test normalize_boundary_faces((PeriodicBC(), ExtrapolateBC()), 1) ==
            (PeriodicBC(), ExtrapolateBC())
        @test normalize_boundary_faces((2, 0), 1) == (PeriodicBC(), ExtrapolateBC())
        @test_throws ArgumentError normalize_boundary_faces((PeriodicBC(),), 1)
        @test_throws ArgumentError normalize_boundary_faces((PeriodicBC(), :nope), 1)
    end

    @testset "installation writes every phase component" begin
        phases = (zeros(8), zeros(8), zeros(8))
        west = PrescribedInflowBC((0.2, 0.3, 0.5))
        east = PrescribedInflowBC((0.6, 0.1, 0.3))
        scheme = MultiphaseWENOScheme(phases; boundary = (west, east), stag = true)

        apply_multiphase_inflow_boundaries!(scheme.fl, scheme.fr, scheme.boundary)
        @test [scheme.fl.x[k][begin] for k in 1:3] == [0.2, 0.3, 0.5]
        @test [scheme.fr.x[k][end] for k in 1:3] == [0.6, 0.1, 0.3]
        # interior face states are untouched
        @test all(iszero, scheme.fl.x[1][2:end])
        @test all(iszero, scheme.fr.x[1][1:(end - 1)])
        # the opposite one-sided state at each physical face is left to the interior
        # reconstruction, so velocity sign still decides whether inflow is used
        @test scheme.fr.x[1][begin] == 0.0
        @test scheme.fl.x[1][end] == 0.0

        # non-inflow faces install nothing
        plain = MultiphaseWENOScheme(phases; boundary = (PeriodicBC(), ExtrapolateBC()))
        apply_multiphase_inflow_boundaries!(plain.fl, plain.fr, plain.boundary)
        @test all(all(iszero, f) for f in plain.fl.x)
        @test all(all(iszero, f) for f in plain.fr.x)
    end

    @testset "wrong-shaped buffers raise instead of silently doing nothing" begin
        # the scalar `apply_*_inflow!` family falls back to a no-op for unmatched flux
        # types, which would discard a prescribed composition without any error. The
        # multiphase methods dispatch their no-op on the boundary type instead, so a
        # dimensional mismatch is a MethodError.
        bc = PrescribedInflowBC((0.2, 0.3, 0.5))
        matrices = ntuple(_ -> zeros(3, 3), 3)
        @test_throws MethodError FiniteDiffWENO5.apply_multiphase_lower_inflow!(matrices, bc)

        vectors = ntuple(_ -> zeros(3), 3)
        @test_throws MethodError FiniteDiffWENO5.apply_multiphase_x_lower_inflow!(vectors, bc)
    end

    @testset "2D and 3D installation" begin
        ny = 4
        phases2 = (zeros(5, ny), zeros(5, ny), zeros(5, ny))
        p2 = collect(range(0.1, 0.3; length = ny))
        west = PrescribedInflowBC((fill(0.2, ny), p2, 0.8 .- p2))
        s2 = MultiphaseWENOScheme(
            phases2; boundary = AdvectionBC(west = west, east = ExtrapolateBC()), stag = true
        )
        apply_multiphase_inflow_boundaries!(s2.fl, s2.fr, s2.boundary)
        for j in 1:ny
            @test s2.fl.x[1][begin, j] == 0.2
            @test s2.fl.x[2][begin, j] == p2[j]
            @test s2.fl.x[3][begin, j] ≈ 0.8 - p2[j]
            @test s2.fl.x[1][begin, j] + s2.fl.x[2][begin, j] + s2.fl.x[3][begin, j] ≈ 1.0
        end

        phases3 = (zeros(4, 4, 4), zeros(4, 4, 4))
        bot = PrescribedInflowBC((0.35, 0.65))
        s3 = MultiphaseWENOScheme(
            phases3;
            boundary = (
                ExtrapolateBC(), ExtrapolateBC(), ExtrapolateBC(),
                ExtrapolateBC(), bot, ExtrapolateBC(),
            ),
        )
        apply_multiphase_inflow_boundaries!(s3.fl, s3.fr, s3.boundary)
        @test all(s3.fl.z[1][:, :, begin] .== 0.35)
        @test all(s3.fl.z[2][:, :, begin] .== 0.65)
        @test all(iszero, s3.fl.x[1])
    end
end
