module SSCant

using PyCall

import ForwardDiff
import ODE
import CurveFit
import DSP
import Dierckx

@pyimport lockin

function unwrap(v, inplace=false)
  # currently assuming an array
  unwrapped = inplace ? v : copy(v)
  for i in 2:length(v)
    while unwrapped[i] - unwrapped[i-1] >= pi
      unwrapped[i] -= 2pi
    end
    while unwrapped[i] - unwrapped[i-1] <= -pi
      unwrapped[i] += 2pi
    end
  end
  return unwrapped
end

unwrap!(v) = unwrap(v, true)

@doc """Zero function""" ->
u(t) = 0

immutable CantileverParams
    f_c::Float64
    Q::Float64
    k::Float64
    C_z::Float64
    C_zz::Float64
    C_b::Float64
    R_t::Float64
    C_1::Float64
    R_1::Float64
    tau_s::Float64
    T::Float64
end

function CantileverParams(;f_c=0.05, Q=1000., k=1.,
                 C_z=1.6e-4, C_zz=-1.6e-5, C_b=0.1,
                 R_t=1e-4, C_1=600., R_1=3e-6,
                 tau_s=15e3, T=293.)
    CantileverParams(f_c, Q, k,
                 C_z, C_zz, C_b,
                 R_t, C_1, R_1,
                 tau_s, T)
end


"""
Params
------

- f_c: cantilever frequency (0.05 MHz)
- Q:  quality factor (1000)
- k: spring constant (1 µN/µm)
- C_z: first derivative of tip capacitance (1e-6 pF / µm)
- C_zz: second derivative of tip capacitance (-1e-5 pF / µm)
- C_b: cantilever body capacitance (0.1 pF)
- R_t: Cantilever tip capacitance (1e-4 Mohm)
- C_1: C1 capacitance (600 pF)
- R_1: R1 resistance (3e-6 Mohm)
- tau_s: sample response time (15e3 us)
- F: Force as a function of time (µN)
- V: Voltage as a function of time (V)
- U: percentange change in capacitance (unitless)

Returns
-------

"""
function ss_cant_1tc(f_c=0.05, Q=1000., k=1,
                 C_z=1e-6, C_zz=-1e-5, C_b=0.1,
                 R_t=1e-4, C_1=600., R_1=3e-6,
                 tau_s=15e3, F=u, V=u, U=u)

    omega_c = 2*pi*f_c
    omega_r = omega_c / Q
    m = k / omega_c ^ 2
    mC_b2 = m * C_b^2
    
    omega_11 = (C_1 * R_1)^-1
    omega_t1 = (R_t * C_1)^-1
    
    omega_tb = (R_t * C_b)^-1
    
    F_x(x) = [x[2], (
            -omega_c^2 * x[1] - omega_r * x[2]
            - C_z/(2mC_b2) * (1 + x[5]) * x[3]^2
            - C_zz/(2mC_b2)* (1 + x[5]) * x[1]*x[3]^2),
                
            (-omega_tb * x[3]
            + C_z/(R_t * C_b^2) * (1 + x[5]) * x[1] * x[3]
            + omega_t1 * x[4]),
                
            (omega_tb * x[3]
            - C_z/(R_t * C_b^2) * (1 + x[5]) * x[1] * x[3]
            - (omega_11 + omega_t1) * x[4]),
                
            -x[5] / tau_s
            ]
    
    function F_tx(t, x)
        return F_x(x) + [0., F(t) / m, 0,
                         V(t) / R_1, U(t) / tau_s]
    end
    
    F_x, F_tx
end

ss_cant_1tc(cant::CantileverParams, F::Function=u, V::Function=u, U::Function=u) = ss_cant_1tc(cant.f_c,
    cant.Q, cant.k, cant.C_z, cant.C_zz, cant.C_b, cant.R_t, cant.C_1,
    cant.R_1, cant.tau_s, F, V, U)


function ss_cant_xv(f_c=0.05, Q=1000., k=1., F=u)

    omega_c = 2*pi*f_c
    omega_r = omega_c / Q
    m = k / omega_c ^ 2
    
    F_x(x) = [x[2], -omega_c^2 * x[1] - omega_r * x[2]]
    
    function F_tx(t, x)
        return F_x(x) + [0., F(t) / m]
    end
    
    F_x, F_tx
end

ss_cant_xv(cant::CantileverParams, F::Function=u, V::Function=u, U::Function=u) = ss_cant_xv(
    cant.f_c, cant.Q, cant.k, F)


@doc """
Params
------
x: position
v: velocity
omega_c: angular frequency (optional)
f_c: frequency (optional)
""" ->
function amplitude(x, v; omega_c=nothing, f_c=nothing)
    if omega_c == nothing
        omega_c = 2 * pi * f_c
    end
    (x.^2 + (v ./ omega_c).^2).^0.5
end

function phase(x, v; omega_c=nothing, f_c=nothing)
    if omega_c == nothing
        omega_c = 2 * pi * f_c
    end
    unwrap(atan2(-v./omega_c, x))
end

function fit_phase(t, phi)
    mb = CurveFit.linear_fit(t, phi)
    dphi = phi - (t * mb[2] + mb[1])
    df = zeros(size(dphi))
    df[2:end] = diff(dphi)

    return dphi, df, mb[2] / (2*pi)
end

type TimeSeries
    cant::CantileverParams
    y0
    t
    x
    v
    q_t
    q_1
    x_5
    A
    phi
    dphi
    df
    f0
end


function TimeSeries(cant::CantileverParams, y0, t, y)
    x = y[:, 1]
    v = y[:, 2]
    q_t = y[:, 3]
    q_1 = y[:, 4]
    x_5 = y[:, 5]
    A = amplitude(x, v, f_c=cant.f_c)
    phi = phase(x, v, f_c=cant.f_c)
    dphi, df, f0 = fit_phase(t, phi)

    TimeSeries(cant, y0, t, x, v, q_t, q_1, x_5,
               A, phi, dphi, df, f0)
end

type ResampledTimeSeries
    cant::CantileverParams
    ts::TimeSeries
    dt
    m
    t
    x
    A
    phi
    dphi
    df
    f0
end

type RawTimeSeries
    cant::CantileverParams
    ts::TimeSeries
end

immutable DownsampledTimeSeries
    cant::CantileverParams
    y0::Array{Float64,1}
    dt0::Float64
    fs::Float64
    f0::Float64
    phi0::Float64
    t::Array{Float64,1}
    A::Array{Float64,1}
    phi::Array{Float64,1}
    df::Array{Float64,1}
end

immutable DownsampledTimeSeriesRaw
    cant::CantileverParams
    y0::Array{Float64,1}
    t_raw::Array{Float64,1}
    y_raw::Array{Float64,2}
    dt0::Float64
    fs::Float64
    f0::Float64
    phi0::Float64
    t::Array{Float64,1}
    A::Array{Float64,1}
    phi::Array{Float64,1}
    df::Array{Float64,1}
end

# function ResampledTimeSeries(ts::TimeSeries, T_pre, dt)
#     spline = Dierckx.Spline1D(ts.t, ts.x, k=3)
#     t = -T_pre:dt:T
#     x = A0 * pi(2*ts.f_c)
#     x = Dierckx.evaluate(spline, t)
# end


function squeeze_y(y)
    y_out = zeros(Float64, length(y), length(y[1]))
    for i in 1:length(y)
        y_out[i, :] = y[i]
    end
    y_out
end

function jacobian(dx)
    jac = ForwardDiff.jacobian(dx)

    jac_t(t, x) = jac(x)

    return jac_t
end

"""
Params
------

- cant: CantileverParams
- y0: array containing [x, v, q_t, q_1, x_5] at time zero
      The units are µm, µm / µs, pC, pC, unitless (percentage change)
"""
# function run_sscant(cant::CantileverParams, y0, t_range; F=u, V=u, U=u, args...)

#     F, F_tx = ss_cant_1tc(cant, F, V, U)
#     jac_t = jacobian(F)
#     t, _y = ODE.ode23s(F_tx, y0, t_range, jacobian=jac_t; args...)
#     y = squeeze_y(_y)

#     TimeSeries(cant, y0, t, y)
# end

# function resample_sscant(cant::CantileverParams, A0, phi0, T_pre, T, dt; V_t0=0, V_10=0, alpha_0=0, F=u, V=u, U=u, args...)
#     fs = 1/dt
#     N_pre = round(Int, T_pre / dt)
#     t_pre = collect((-N_pre:0) * dt)
#     x_pre = A0 * cospi(2*cant.f_c*t_pre + 2*phi0)

#     F, F_tx = ss_cant_1tc(cant, F, V, U)
#     jac_t = jacobian(F)
#     y0 = [x_pre[end], 2*pi*cant.f_c*A0*sinpi(2*phi0),
#           V_t0 * cant.C_b, V_10 * cant.C_1, alpha_0]

#     t, _y = ODE.ode23s(F_tx, y0, [0., T], jacobian=jac_t; maxstep=dt*0.5, args...)
#     y = squeeze_y(_y)

#     ts = TimeSeries(cant, y0, t, y)
#     N_post = trunc(Int, T / dt)
#     t_post = collect((1:N_post) * dt)
#     spline = Dierckx.Spline1D(ts.t, ts.x, k=3)
#     t = [t_pre; t_post]
#     x = [x_pre; Dierckx.evaluate(spline, t_post)]

#     li = lockin.LockIn(t, x, fs)

#     li[:lock](bw_ratio=0.25, coeff_ratio=5., window="blackmanharris")

#     ResampledTimeSeries(cant, ts, dt, li[:m], t, x, li[:A],
#                         li[:phi], li[:dphi], li[:df], li[:f0])

# end

function resample_sscant_sim(cant::CantileverParams, y0::Array{Float64,1},
                             T_pre::Float64,
                             T::Float64, dt::Float64;
                             F=u, V=u, U=u, coeff_ratio=8., args...)
    fs = 1/dt

    F_tx, jac_t = make_Ftx_jac(ss_cant_1tc, cant, F, V, U)

    t, _y = ODE.ode23s(F_tx, y0, [-T_pre, T], jacobian=jac_t; maxstep=dt*0.5, args...)
    y = squeeze_y(_y)

    ts = TimeSeries(cant, y0, t, y)

    spline = Dierckx.Spline1D(ts.t, ts.x, k=3)
    t = collect(-T_pre:dt:T)
    x = Dierckx.evaluate(spline, t)

    li = lockin.LockIn(t, x, fs)

    li[:lock2](fp_ratio=0.1, fc_ratio=0.4, coeff_ratio=coeff_ratio, window="blackman")
    li[:phase](tf=0.)

    ResampledTimeSeries(cant, ts, dt, li[:m], t, x, li[:A],
                        li[:phi], li[:dphi], li[:df], li[:f0])

end

function raw_sscant_sim(cant::CantileverParams, y0::Array{Float64,1},
                             T_pre::Float64,
                             T::Float64, dt::Float64;
                             F=u, V=u, U=u, coeff_ratio=8., args...)
    fs = 1/dt

    F_tx, jac_t = make_Ftx_jac(ss_cant_1tc, cant, F, V, U)


    t, _y = ODE.ode23s(F_tx, y0, [-T_pre, T], jacobian=jac_t; maxstep=dt*0.5, args...)
    y = squeeze_y(_y)

    ts = TimeSeries(cant, y0, t, y)

    RawTimeSeries(cant, ts)

end


function make_Ftx_jac(ss_cant::Function, cant::CantileverParams,
                      F::Function, V::Function, U::Function)
    
    Fx, F_tx = ss_cant(cant, F, V, U)
    jac = jacobian(Fx)

    return F_tx, jac
end

function chunk_sscant_sim(cant::CantileverParams, y0::Array{Float64,1},
                             T_pre::Float64,
                             T::Float64, dt::Float64;
                             F=u, V=u, U=u, coeff_ratio=8.,
                             Tchunk=1000., f_dec=nothing, args...)
    fs = 1/dt

    if f_dec == nothing
        f_dec = 2*cant.f_c
    end

    dec = trunc(Int, fs / f_dec)

    F_tx, jac = make_Ftx_jac(ss_cant_1tc, cant, F, V, U)

    Tendpts = linspace(-T_pre, T, ceil(Int, (T_pre+T)/Tchunk))
    for i = 1:length(Tendpts)-1
        t, _y = ODE.ode23s(F_tx, y0, [Tendpts[i], Tendpts[i+1]],
                           jacobian=jac; maxstep=dt*0.5)
        y = squeeze_y(_y)
        y0 = _y[end]
        spline = Dierckx.Spline1D(t, y[:, 1], k=3)
        t_equal = collect(Tendpts[i]:dt:Tendpts[i+1])
        x = Dierckx.evaluate(spline, t_equal)

        if i == 1
            global li = lockin.LockIn(t_equal, x, fs)
            li[:lock2](fp_ratio=0.1, fc_ratio=0.4,
                       coeff_ratio=coeff_ratio, window="blackman")
            li[:phase](tf=0.)
            global slock = lockin.FIRStateLock(li[:b], dec, li[:f0],
                                        -li[:mb][end], t0=-T_pre, fs=fs)
            slock[:filt](x)
        else
            slock[:filt](x[2:end])
        end

    end

    tout = slock[:get_t]()
    A = abs(slock[:output])
    phi = unwrap(angle(slock[:z]))
    df = zeros(phi)
    df[2:end] = diff(phi)

    DownsampledTimeSeries(cant, y0, dt, fs / dec,
                          slock[:f0], slock[:phi0][end], tout,
                          A, phi, df)
end



function downsample_sscant_sim(cant::CantileverParams, y0::Array{Float64,1},
                             T_pre::Float64,
                             T::Float64, dt::Float64;
                             F=u, V=u, U=u, coeff_ratio=8.,
                             f_resampled=nothing, args...)
    if f_resampled == nothing
        f_resampled = cant.f_c
    end

    fs = 1/dt

    F_tx, jac_t = make_Ftx_jac(ss_cant_1tc, cant, F, V, U)

    t, _y = ODE.ode23s(F_tx, y0, [-T_pre, T], jacobian=jac_t; maxstep=dt*0.5, args...)
    y = squeeze_y(_y)

    ts = TimeSeries(cant, y0, t, y)

    spline = Dierckx.Spline1D(ts.t, ts.x, k=3)
    t = collect(-T_pre:dt:T)
    x = Dierckx.evaluate(spline, t)

    li = lockin.LockIn(t, x, fs)

    li[:lock2](fp_ratio=0.1, fc_ratio=0.4, coeff_ratio=coeff_ratio, window="blackman")
    li[:phase](tf=0.)
    factor = trunc(fs / f_resampled)
    li[:decimate](factor)
    phi0 = li[:mb][2]

    DownsampledTimeSeries(cant, y0, dt, li[:dec_fs],
                          li[:dec_f0], li[:mb][2], li[:dec_t],
                          li[:dec_A], li[:dec_phi], li[:dec_df])
end

# type DownsampledTimeSeries
#     cant::CantileverParams
#     y0::Array{Float64,1}
#     dt0::Float64
#     fs::Float64
#     f0::Float64
#     t::Array{Float64,1}
#     A::Array{Float64,1}
#     phi::Array{Float64,1}
#     df::Array{Float64,1}
# end

# Since a discrete delta function has a magnitude $\delta[0] = 1/\Delta t$,
# we can model the thermal force on the cantilever as a random,
# Gaussian force with,

# $$\sigma = \sqrt{\frac{k_c k_B T}{\pi Q f_c \Delta t}}$$
# (see doi:10.1088/0034-4885/29/1/306).

function resample_sscant_noise(cant::CantileverParams, y0::Array{Float64,1},
                             T_pre::Float64,
                             T::Float64, dt::Float64;
                             V=u, U=u, coeff_ratio=8., seed=-1, args...)
    fs = 1/dt
    k_B = 1.38064852e-11

    dt_force = 0.05/cant.f_c
    sigma = sqrt(cant.k*k_B*cant.T/(pi*cant.Q*cant.f_c*dt_force))
    if seed > 0
        srand(seed)
    end
    N_force = round(Int, T/dt_force)+1


    F_spline = Dierckx.Spline1D(linspace(0, T, N_force), randn(N_force) * sigma,
                              k=1)

    function F(t)
        Dierckx.evaluate(F_spline, t)
    end


    F_tx, jac_t = make_Ftx_jac(ss_cant_1tc, cant, F, V, U)

    t, _y = ODE.ode23s(F_tx, y0, [-T_pre, T], jacobian=jac_t; maxstep=dt*0.5, args...)
    y = squeeze_y(_y)

    ts = TimeSeries(cant, y0, t, y)

    spline = Dierckx.Spline1D(ts.t, ts.x, k=3)
    t = collect(-T_pre:dt:T)
    x = Dierckx.evaluate(spline, t)

    li = lockin.LockIn(t, x, fs)

    li[:lock2](fp_ratio=0.1, fc_ratio=0.4,
               coeff_ratio=coeff_ratio, window="blackman")
    li[:autophase](tf=0.)

    ResampledTimeSeries(cant, ts, dt, li[:m], t, x, li[:A],
                        li[:phi], li[:dphi], li[:df], li[:f0])

end

# Another idea would be to "remember" the last value the 

function re_down_sscant_noise3(cant::CantileverParams, y0::Array{Float64,1},
                             T_pre::Float64,
                             T::Float64, dt::Float64;
                             V=u, U=u, coeff_ratio=8., seed=-1, args...)
    fs = 1/dt
    k_B = 1.38064852e-11

    dt_force = 0.05/cant.f_c
    sigma = sqrt(cant.k*k_B*cant.T/(pi*cant.Q*cant.f_c*dt_force))
    if seed > 0
        srand(seed)
    end
    N_force = round(Int, T/dt_force)+1
    N = round(Int, (T+T_pre)/dt_force)+1

    force(t) = t >= 0. ? randn() * sigma : 0.
    times = linspace(-T_pre, T, N)
    F_spline = Dierckx.Spline1D(times, map(force, times),
                              k=3)

    function F(t)
        Dierckx.evaluate(F_spline, t)
    end


    F_tx, jac_t = make_Ftx_jac(ss_cant_1tc, cant, F, V, U)

    t, _y = ODE.ode23s(F_tx, y0, [-T_pre, T], jacobian=jac_t; maxstep=dt*0.5, args...)
    y = squeeze_y(_y)

    ts = TimeSeries(cant, y0, t, y)

    spline = Dierckx.Spline1D(ts.t, ts.x, k=3)
    t = collect(-T_pre:dt:T)
    x = Dierckx.evaluate(spline, t)

    li = lockin.LockIn(t, x, fs)

    li[:lock2](fp_ratio=0.1, fc_ratio=0.4,
               coeff_ratio=coeff_ratio, window="blackman")
    li[:phase](tf=0.)
    factor = trunc(fs/cant.f_c)
    li[:decimate](factor)

    r = ResampledTimeSeries(cant, ts, dt, li[:m], t, x, li[:A],
                        li[:phi], li[:dphi], li[:df], li[:f0])

    down = DownsampledTimeSeries(cant, y0, dt, li[:dec_fs],
                          li[:dec_f0], li[:mb][2], li[:dec_t],
                          li[:dec_A], li[:dec_phi], li[:dec_df])

    return r, down

end

function down_sscant_noise3(cant::CantileverParams, y0::Array{Float64,1},
                             T_pre::Float64,
                             T::Float64, dt::Float64;
                             V=u, U=u, coeff_ratio=8., seed=-1, args...)
    fs = 1/dt
    k_B = 1.38064852e-11

    dt_force = 0.05/cant.f_c
    sigma = sqrt(cant.k*k_B*cant.T/(pi*cant.Q*cant.f_c*dt_force))
    if seed > 0
        srand(seed)
    end
    N_force = round(Int, T/dt_force)+1
    N = round(Int, (T+T_pre)/dt_force)+1

    force(t) = t >= 0. ? randn() * sigma : 0.
    times = linspace(-T_pre, T, N)
    F_spline = Dierckx.Spline1D(times, map(force, times),
                              k=3)

    function F(t)
        Dierckx.evaluate(F_spline, t)
    end


    F_tx, jac_t = make_Ftx_jac(ss_cant_1tc, cant, F, V, U)

    t, _y = ODE.ode23s(F_tx, y0, [-T_pre, T], jacobian=jac_t; maxstep=dt*0.5, args...)
    y = squeeze_y(_y)

    ts = TimeSeries(cant, y0, t, y)

    spline = Dierckx.Spline1D(ts.t, ts.x, k=3)
    t = collect(-T_pre:dt:T)
    x = Dierckx.evaluate(spline, t)

    li = lockin.LockIn(t, x, fs)

    li[:lock2](fp_ratio=0.1, fc_ratio=0.4,
               coeff_ratio=coeff_ratio, window="blackman")
    li[:phase](tf=0.)
    factor = trunc(fs/cant.f_c)
    li[:decimate](factor)

    down = DownsampledTimeSeries(cant, y0, dt, li[:dec_fs],
                          li[:dec_f0], li[:mb][2], li[:dec_t],
                          li[:dec_A], li[:dec_phi], li[:dec_df])

    return down

end

function sscant_noise3(cant::CantileverParams, y0::Array{Float64,1},
                             T_pre::Float64,
                             T::Float64, dt::Float64;
                             V=u, U=u, coeff_ratio=8., seed=-1, args...)
    fs = 1/dt
    k_B = 1.38064852e-11

    dt_force = 0.05/cant.f_c
    sigma = sqrt(cant.k*k_B*cant.T/(pi*cant.Q*cant.f_c*dt_force))
    if seed > 0
        srand(seed)
    end
    N_force = round(Int, T/dt_force)+1
    N = round(Int, (T+T_pre)/dt_force)+1

    force(t) = t >= 0. ? randn() * sigma : 0.
    times = linspace(-T_pre, T, N)
    F_spline = Dierckx.Spline1D(times, map(force, times),
                              k=3)

    function F(t)
        Dierckx.evaluate(F_spline, t)
    end


    Fx, F_tx = ss_cant_1tc(cant, F, V, U)
    jac_t = jacobian(Fx)

    t, _y = ODE.ode23s(F_tx, y0, [-T_pre, T], jacobian=jac_t; maxstep=dt*0.5, args...)
    y = squeeze_y(_y)

    ts = TimeSeries(cant, y0, t, y)

    spline = Dierckx.Spline1D(ts.t, ts.x, k=3)
    t = collect(-T_pre:dt:T)
    x = Dierckx.evaluate(spline, t)

    li = lockin.LockIn(t, x, fs)

    li[:lock2](fp_ratio=0.1, fc_ratio=0.4,
               coeff_ratio=coeff_ratio, window="blackman")
    li[:phase](tf=0.)
    factor = trunc(fs/cant.f_c)
    li[:decimate](factor)

    down = DownsampledTimeSeries(cant, y0, dt, li[:dec_fs],
                          li[:dec_f0], li[:mb][2], li[:dec_t],
                          li[:dec_A], li[:dec_phi], li[:dec_df])

    return down

end

function chunk_sscant_sim_noise(cant::CantileverParams, y0::Vector{Float64},
                             T_pre::Float64,
                             T::Float64, dt::Float64;
                              V=u, U=u, coeff_ratio=8.,
                             Tchunk=1000., f_dec=nothing, 
                             seed=-1, args...)
    fs = 1/dt
    k_B = 1.38064852e-11

    dt_force = 0.05/cant.f_c
    sigma = sqrt(cant.k*k_B*cant.T/(pi*cant.Q*cant.f_c*dt_force))

    if seed > 0
        srand(seed)
    end
    
    N_force = round(Int, T/dt_force)+1
    N = round(Int, (T+T_pre)/dt_force)+1

    force(t) = t >= 0. ? randn() * sigma : 0.
    times = linspace(-T_pre, T, N)
    F_spline = Dierckx.Spline1D(times, map(force, times),
                              k=3)

    function F(t)
        Dierckx.evaluate(F_spline, t)
    end

    if f_dec == nothing
        f_dec = 2*cant.f_c
    end

    dec = trunc(Int, fs / f_dec)

    F_tx, jac = make_Ftx_jac(ss_cant_1tc, cant, F, V, U)

    Tendpts = linspace(-T_pre, T, ceil(Int, (T_pre+T)/Tchunk))
    for i = 1:length(Tendpts)-1
        t, _y = ODE.ode23s(F_tx, y0, [Tendpts[i], Tendpts[i+1]],
                           jacobian=jac; maxstep=dt*0.5)
        y = squeeze_y(_y)

        y0 = _y[end]
        spline = Dierckx.Spline1D(t, y[:, 1], k=3)
        t_equal = collect(Tendpts[i]:dt:Tendpts[i+1])
        x = Dierckx.evaluate(spline, t_equal)

        if i == 1
            global li = lockin.LockIn(t_equal, x, fs)
            li[:lock2](fp_ratio=0.1, fc_ratio=0.4,
                       coeff_ratio=coeff_ratio, window="blackman")
            li[:phase](tf=0.)
            global slock = lockin.FIRStateLock(li[:b], dec, li[:f0],
                                        -li[:mb][end], t0=-T_pre, fs=fs)
            slock[:filt](x)
        else
            slock[:filt](x[2:end])
        end

    end

    tout = slock[:get_t]()
    A = abs(slock[:z_out])
    phi = unwrap(angle(slock[:z_out]))
    df = zeros(phi)
    df[2:end] = diff(phi) / (2pi)
    df[1] = df[2]

    DownsampledTimeSeries(cant, y0, dt, fs / dec,
                          slock[:f0], slock[:phi0], tout,
                          A, phi, df)
end

function chunk_all_sscant_sim_noise_outall(cant::CantileverParams, y0::Vector{Float64},
                             T_pre::Float64,
                             T::Float64, dt::Float64; xdot::Function=ss_cant_1tc,
                              V=u, U=u, coeff_ratio=8., V_append=0.01, U_append=0.005,
                             Tchunk=1000., f_dec::Float64=2*cant.f_c, 
                             seed=-1, force_ratio=20., S_x=0., args...)
    fs = 1/dt
    const k_B = 1.38064852e-11

    dt_force = 1./(cant.f_c * force_ratio)

    sigma = sqrt(cant.k*k_B*cant.T/(pi*cant.Q*cant.f_c*dt_force))

    sigma_x = S_x * sqrt(fs/2)

    if seed > 0
        srand(seed)
    end

    force(t) = t >= 0. ? randn() * sigma : 0.

    dec = trunc(Int, fs / f_dec)

    Tendpts = collect(-T_pre:Tchunk:T)
    if Tendpts[end] != T
        push!(Tendpts, T)
    end

    t_save = Vector{Float64}()
    y_save = Vector{Vector{Float64}}()
    delta_y_save = [cant.C_b*V_append, cant.C_1*V_append, U_append]

    for i = 1:length(Tendpts)-1
        N = round(Int, (Tendpts[i+1] - Tendpts[i])/dt_force) + 1
        t_force = linspace(Tendpts[i], Tendpts[i+1], N)
        F_spline = Dierckx.Spline1D(t_force, map(force, t_force),
                              k=3)
        
        F(t) = Dierckx.evaluate(F_spline, t)

        F_tx, jac = make_Ftx_jac(xdot, cant, F, V, U)

        t, _y = ODE.ode23s(F_tx, y0, [Tendpts[i], Tendpts[i+1]],
                           jacobian=jac; maxstep=dt*0.5)
        y = squeeze_y(_y)

        y0 = _y[end]

        spline = Dierckx.Spline1D(t, y[:, 1], k=3)
        t_equal = Tendpts[i]:dt:Tendpts[i+1]
        x = zeros(t_equal)
        for i = eachindex(x)
            x[i] = Dierckx.evaluate(spline, t_equal[i]) + randn()*sigma_x
        end

        if i == 1
            push!(t_save, t[1])
            push!(y_save, _y[1])
            global li = lockin.LockIn(collect(t_equal), x, fs)
            li[:lock2](fp_ratio=0.1, fc_ratio=0.4,
                       coeff_ratio=coeff_ratio, window="blackman")
            li[:phase](tf=0.)

            global slock = lockin.FIRStateLock(li[:b], dec, li[:f0],
                                        -li[:mb][end], t0=-T_pre, fs=fs)

            slock[:filt](x)
        else
            slock[:filt](x[2:end])
        end

        for i = eachindex(_y)
            if any(abs(_y[i][3:end] - y_save[end][3:end]) .> delta_y_save)
                push!(t_save, t[i])
                push!(y_save, _y[i])
            end
        end

    end

    y_save_sq = squeeze_y(y_save)

    tout = slock[:get_t]()
    A = abs(slock[:z_out])
    phi = unwrap(angle(slock[:z_out]))
    df = zeros(phi)
    df[2:end] = diff(phi) / (2pi)
    df[1] = df[2]

    DownsampledTimeSeriesRaw(cant, y0, t_save, y_save_sq, dt, fs / dec,
                          slock[:f0], slock[:phi0], tout,
                          A, phi, df)
end

function chunk_all_sscant_sim_noise(cant::CantileverParams, y0::Array{Float64,1},
                             T_pre::Float64,
                             T::Float64, dt::Float64; xdot::Function=ss_cant_1tc,
                              V=u, U=u, coeff_ratio=8.,
                             Tchunk=1000., f_dec=nothing, 
                             seed=-1, force_ratio=20., args...)
    fs = 1/dt
    k_B = 1.38064852e-11

    dt_force = 1./(cant.f_c * force_ratio)

    sigma = sqrt(cant.k*k_B*cant.T/(pi*cant.Q*cant.f_c*dt_force))

    if seed > 0
        srand(seed)
    end

    force(t) = t >= 0. ? randn() * sigma : 0.

    if f_dec == nothing
        f_dec = 2*cant.f_c
    end

    dec = trunc(Int, fs / f_dec)

    Tendpts = collect(-T_pre:Tchunk:T)
    if Tendpts[end] != T
        push!(Tendpts, T)
    end

    for i = 1:length(Tendpts)-1
        N = round(Int, (Tendpts[i+1] - Tendpts[i])/dt_force) + 1
        t_force = linspace(Tendpts[i], Tendpts[i+1], N)
        F_spline = Dierckx.Spline1D(t_force, map(force, t_force),
                              k=3)
        function F(t)
            Dierckx.evaluate(F_spline, t)
        end
        F_tx, jac = make_Ftx_jac(xdot, cant, F, V, U)

        t, _y = ODE.ode23s(F_tx, y0, [Tendpts[i], Tendpts[i+1]],
                           jacobian=jac; maxstep=dt*0.5)
        y = squeeze_y(_y)

        y0 = _y[end]
        spline = Dierckx.Spline1D(t, y[:, 1], k=3)
        t_equal = collect(Tendpts[i]:dt:Tendpts[i+1])
        x = Dierckx.evaluate(spline, t_equal)

        if i == 1
            global li = lockin.LockIn(t_equal, x, fs)
            li[:lock2](fp_ratio=0.1, fc_ratio=0.4,
                       coeff_ratio=coeff_ratio, window="blackman")
            li[:phase](tf=0.)
            global slock = lockin.FIRStateLock(li[:b], dec, li[:f0],
                                        -li[:mb][end], t0=-T_pre, fs=fs)
            slock[:filt](x)
        else
            slock[:filt](x[2:end])
        end

    end

    tout = slock[:get_t]()
    A = abs(slock[:z_out])
    phi = unwrap(angle(slock[:z_out]))
    df = zeros(phi)
    df[2:end] = diff(phi) / (2pi)
    df[1] = df[2]

    DownsampledTimeSeries(cant, y0, dt, fs / dec,
                          slock[:f0], slock[:phi0], tout,
                          A, phi, df)
end

function chunk_all_sscant_sim_brownian(cant::CantileverParams, y0::Array{Float64,1},
                             T_pre::Float64,
                             T::Float64, dt::Float64, dt_out::Float64;
                             xdot::Function=ss_cant_1tc,
                             V=u, U=u, coeff_ratio=8.,
                             Tchunk=1000., 
                             seed=-1, force_ratio=20., args...)
    fs = 1/dt
    k_B = 1.38064852e-11

    dt_force = 1./(cant.f_c * force_ratio)

    sigma = sqrt(cant.k*k_B*cant.T/(pi*cant.Q*cant.f_c*dt_force))

    if seed > 0
        srand(seed)
    end

    force(t) = t >= 0. ? randn() * sigma : 0.

    Tendpts = collect(-T_pre:Tchunk:T)
    if Tendpts[end] != T
        push!(Tendpts, T)
    end

    t_collected = Array{Float64,1}()
    x_collected = Array{Float64,1}()
    for i = 1:length(Tendpts)-1
        N = round(Int, (Tendpts[i+1] - Tendpts[i])/dt_force) + 1
        t_force = linspace(Tendpts[i], Tendpts[i+1], N)
        F_spline = Dierckx.Spline1D(t_force, map(force, t_force),
                              k=3)

        F(t) = Dierckx.evaluate(F_spline, t)

        F_tx, jac = make_Ftx_jac(xdot, cant, F, V, U)

        t, _y = ODE.ode23s(F_tx, y0, [Tendpts[i], Tendpts[i+1]],
                           jacobian=jac; maxstep=dt*0.5)
        y = squeeze_y(_y)

        y0 = _y[end]
        spline = Dierckx.Spline1D(t, y[:, 1], k=3)
        t_equal = collect(Tendpts[i]:dt_out:Tendpts[i+1])
        x = Dierckx.evaluate(spline, t_equal)
        
        append!(t_collected, t_equal[2:end])
        append!(x_collected, x[2:end])

    end

    t_collected, x_collected, y0
end



end # module
