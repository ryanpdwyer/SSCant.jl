import SSCant
using Base.Test

function test_amplitude()
    x = [0, 1, 0, -1.]
    v = [1, 0, -1, 0.]
    omega_c = 1
    exp_A = [1, 1, 1, 1.]
    A = SSCant.amplitude(x, v, omega_c=omega_c)
    @test_approx_eq(A, exp_A)
end

function test_phase()
    x = [0, 1, 0, -1.]
    v = [1, 0, -1, 0.]
    omega_c = 1
    exp_phi = [-pi/2, 0, pi/2, pi]
    phi = SSCant.phase(x, v, omega_c=omega_c)
    @test_approx_eq(phi, exp_phi)
end

function test_fit_phase()
    t = 0:3
    phi = [-pi/2, 0, pi/2, pi]
    dphi, df, f0 = SSCant.fit_phase(t, phi)
    @test_approx_eq_eps(dphi, [0., 0, 0, 0], 1e-14)
    @test_approx_eq_eps(df, [0., 0, 0, 0], 1e-14)
    @test_approx_eq(f0, 0.25)
end


function test_get_specified()
    x = 0:0.5:5
    y = map(x -> x^2, x)
    x_specified = 0:5
    y_specified_exp =  map(x -> x^2, x_specified)
    y_specified = SSCant.get_specified(x, y, x_specified)
    @test_approx_eq(y_specified, y_specified_exp)
end

function test_chunk_range()
    t0 = -10.
    tf = 14.
    @test_approx_eq(SSCant.chunk_range(t0, tf, 10.), [-10, 0.0, 10, 14])
end

test_amplitude()
test_phase()
test_fit_phase()
test_get_specified()
test_chunk_range()
