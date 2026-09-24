using Test
using FiniteDiffWENO5
using FiniteDiffWENO5: SerialTopology

mp_detphases(n) = begin
    p1 = [0.3 + 0.15sinpi(2 * (i - 0.5) / n) for i in 1:n]
    p2 = [0.3 + 0.1cospi(4 * (i - 0.5) / n) for i in 1:n]
    (p1, p2, [1 - p1[i] - p2[i] for i in 1:n])
end
mp_detface(n) = [1.0 + 0.4sinpi(2 * i / n) for i in 0:n]

function mp_boundary(kind)
    kind === :extrapolate && return (ExtrapolateBC(), ExtrapolateBC())
    kind === :periodic && return (PeriodicBC(), PeriodicBC())
    error("unknown boundary kind $kind")
end

@testset "SerialTopology reproduces serial multiphase bit-for-bit" begin

    @testset "1D: $bk" for bk in (:extrapolate, :periodic)
        n = 20
        halo = 3
        dt, dx = 0.001, inv(n)
        boundary = mp_boundary(bk)
        p = mp_detphases(n)

        scheme_s = MultiphaseWENOScheme(p; boundary, stag = true, multithreading = false)
        v_s = (; x = mp_detface(n))
        p_s = map(copy, p)
        WENO_step!(p_s, v_s, scheme_s, dt, dx)

        topo = SerialTopology((n,); halo, periodic = bk === :periodic)
        c = ntuple(3) do q
            arr = zeros(n + 2halo)
            arr[(halo + 1):(halo + n)] .= p[q]
            arr
        end
        vf = zeros(n + 2halo + 1)
        vf[(halo + 1):(halo + n + 1)] .= v_s.x
        scheme_t = MultiphaseWENOScheme(c, topo; boundary, stag = true, multithreading = false)
        WENO_step!(c, (; x = vf), scheme_t, dt, dx)

        r = (halo + 1):(halo + n)
        for q in 1:3
            @test c[q][r] == p_s[q]
        end
    end

    @testset "2D: $bk" for bk in (:extrapolate, :periodic)
        nx, ny = 12, 10
        halo = 3
        dt, dx, dy = 0.001, inv(nx), inv(ny)
        boundary = ntuple(_ -> mp_boundary(bk)[1], 4)
        p1 = [0.3 + 0.1sinpi(2 * (i - 0.5) / nx) * cospi(2 * (j - 0.5) / ny) for i in 1:nx, j in 1:ny]
        p2 = [0.3 + 0.1cospi(2 * (i - 0.5) / nx) for i in 1:nx, j in 1:ny]
        p3 = [1 - p1[i, j] - p2[i, j] for i in 1:nx, j in 1:ny]
        p = (p1, p2, p3)

        scheme_s = MultiphaseWENOScheme(p; boundary, stag = true, multithreading = false)
        vx_s = [1.0 + 0.4sinpi(2 * i / nx) for i in 0:nx, j in 1:ny]
        vy_s = [0.5 + 0.2cospi(2 * (j - 0.5) / ny) for i in 1:nx, j in 0:ny]
        v_s = (x = vx_s, y = vy_s)
        p_s = map(copy, p)
        WENO_step!(p_s, v_s, scheme_s, dt, dx, dy)

        topo = SerialTopology((nx, ny); halo, periodic = bk === :periodic)
        c = ntuple(3) do q
            arr = zeros(nx + 2halo, ny + 2halo)
            arr[(halo + 1):(halo + nx), (halo + 1):(halo + ny)] .= p[q]
            arr
        end
        vx = zeros(nx + 2halo + 1, ny + 2halo)
        vx[(halo + 1):(halo + nx + 1), (halo + 1):(halo + ny)] .= vx_s
        vy = zeros(nx + 2halo, ny + 2halo + 1)
        vy[(halo + 1):(halo + nx), (halo + 1):(halo + ny + 1)] .= vy_s

        scheme_t = MultiphaseWENOScheme(c, topo; boundary, stag = true, multithreading = false)
        WENO_step!(c, (x = vx, y = vy), scheme_t, dt, dx, dy)

        r1 = (halo + 1):(halo + nx)
        r2 = (halo + 1):(halo + ny)
        for q in 1:3
            @test c[q][r1, r2] == p_s[q]
        end
    end
end
