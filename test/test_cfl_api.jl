using Test, FiniteDiffWENO5
using FiniteDiffWENO5: SerialTopology, NoTopology, owned_window

@testset "CFL public API and periodic face ownership" begin
    for periodic in (false, true)
        topo = SerialTopology((8,); periodic)
        v = allocate_weno_field(topo; stagger = 1)
        fill!(v, NaN)
        v[4:11] .= 1.0
        if !periodic
            v[12] = 2.0
        end
        expected = periodic ? 0.5 : 0.25
        @test length(owned_window(v, topo; stagger = 1)) == (periodic ? 8 : 9)
        @test weno_cfl_dt(topo, (; x = v), (1.0,), 0.5) == expected
        @test weno_cfl_dt(topo, (v,), (1.0,), 0.5) == expected
    end
    v = (fill(2.0, 8), fill(3.0, 8))
    @test weno_cfl_dt(NoTopology(), v, (1.0, 2.0), 0.7) ==
          weno_cfl_dt(NoTopology(), (; x = v[1], y = v[2]), (1.0, 2.0), 0.7)
end
