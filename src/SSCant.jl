module SSCant

using PyCall
using ProgressMeter
import ForwardDiff
import ODE
import DSP
import Dierckx
@pyimport lockin

"Fits a straight line through a set of points, `y = a₁ + a₂ * x`

From CurveFit.jl."
linear_fit(x, y) = hcat(ones(x), x) \ y

"""Zero function"""
u(t) = 0


"""
get_specified(x, y, x_specified)
For sorted x, x_specified, get y data points corresponding to x_specified.
"""
function get_specified(x, y, x_specified)
    y_specified = Vector{eltype(y)}()
    j = 1
    for i = eachindex(x)
        if x[i] == x_specified[j]
            push!(y_specified, y[i])
            j += 1
        end
    end
    return y_specified
end

function chunk_range(t0, tf, Tchunk)
    tchunks = collect(t0:Tchunk:tf)
    if tchunks[end] != tf
        push!(tchunks, tf)
    end
    return tchunks
end


function make_spline(f, t0, tf, dt; k=3)
    T = tf - t0
    N = round(Int, T/dt) + 1
    t = linspace(t0, tf, N)
    ft = map(f, t)
    F_spline = Dierckx.Spline1D(t, ft, k=k)
    F(t) = Dierckx.evaluate(F_spline, t)
    return F
end


function make_spline(f, t0, tf, dt, prev; k=3)
    T = tf - t0
    N = round(Int, T/dt) + 1
    t = linspace(t0, tf, N)

    ft = Vector{typeof(prev)}(length(t))
    ft[1] = prev

    for i = 2:length(t)
        ft[i] = f(t[i])
    end

    F_spline = Dierckx.Spline1D(t, ft, k=k)
    F(t) = Dierckx.evaluate(F_spline, t)

    return F
end


"""
Params
------
x: position
v: velocity
omega_c: angular frequency (optional)
f_c: frequency (optional)
"""
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
    DSP.unwrap(atan2(-v./omega_c, x))
end

function fit_phase(t, phi)
    mb = linear_fit(t, phi)
    dphi = phi - (t * mb[2] + mb[1])
    df = zeros(size(dphi))
    df[2:end] = diff(dphi)

    return dphi, df, mb[2] / (2*pi)
end


"""
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
"""
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
    f_det::Float64
end


function CantileverParams(;f_c=0.05, Q=1000., k=1.,
                 C_z=1.6e-4, C_zz=-1.6e-5, C_b=0.1,
                 R_t=1e-4, C_1=600., R_1=3e-6,
                 tau_s=15e3, T=293., f_det=0.2)
    CantileverParams(f_c, Q, k,
                 C_z, C_zz, C_b,
                 R_t, C_1, R_1,
                 tau_s, T, f_det)
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
                 C_z=-1e-6, C_zz=1e-5, C_b=0.1,
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
            + C_z/(2mC_b2) * (1 + x[5]) * x[3]^2
            + (C_zz - 2*C_z^2/C_b)/(2mC_b2) * (1 + x[5]) * x[1]*x[3]^2),
                
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

ss_cant_1tc(cant::CantileverParams, F=u, V=u, U=u) = ss_cant_1tc(cant.f_c,
    cant.Q, cant.k, cant.C_z, cant.C_zz, cant.C_b, cant.R_t, cant.C_1,
    cant.R_1, cant.tau_s, F, V, U)


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
- Pn: Surface potential input (V/s)

Returns
-------

"""
function ss_cant_1tc_volt_noise(f_c=0.05, Q=1000., k=1,
                 C_z=1e-6, C_zz=-1e-5, C_b=0.1,
                 R_t=1e-4, C_1=600., R_1=3e-6,
                 tau_s=15e3, F=u, V=u, U=u, Pn=u)

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
            - C_zz/(2mC_b2) * (1 + x[5]) * x[1]*x[3]^2),
                
            (-omega_tb * x[3]
            + C_z/(R_t * C_b^2) * (1 + x[5]) * x[1] * x[3]
            + omega_t1 * x[4]),
                
            (omega_tb * x[3]
            - C_z/(R_t * C_b^2) * (1 + x[5]) * x[1] * x[3]
            - (omega_11 + omega_t1) * x[4]
            - x[6] / R_1
            ),
                
            -x[5] / tau_s,

            0.0
            ]
    
    function F_tx(t, x)
        return F_x(x) + [0., F(t) / m, 0,
                         V(t) / R_1, U(t) / tau_s, Pn(t)]
    end
    
    F_x, F_tx
end

ss_cant_1tc_volt_noise(
    cant::CantileverParams, F=u, V=u, U=u, Pn=u
    ) = ss_cant_1tc_volt_noise(cant.f_c,
    cant.Q, cant.k, cant.C_z, cant.C_zz, cant.C_b, cant.R_t, cant.C_1,
    cant.R_1, cant.tau_s, F, V, U, Pn)


function init_steady_state_1tc(cp::SSCant.CantileverParams, A0, phi0, t0, V, U)
    [A0*cos(phi0), -2*pi*cp.f_c*A0*sin(phi0),
     cp.C_b*V(t0), cp.C_1*V(t0), U(t0)]
end

function init_steady_state_1tc_volt_noise(cp::SSCant.CantileverParams, A0, phi0, t0, V, U, Pn)
    [A0*cos(phi0), -2*pi*cp.f_c*A0*sin(phi0),
     cp.C_b*V(t0), cp.C_1*V(t0), U(t0), Pn(t0)]
end


function ss_cant_det(;f_c=0.05, k=1, Q=10000,
                 C_z=1e-6, C_zz=-1e-5, C_b=0.1,
                 R_t=1e-4, C_1=600., R_1=3e-6,
                 tau_s=15e3, fdet=0.2, F=u, V=u, U=u, P=u)
    
    omega_det = 2*pi*fdet
    omega_c = 2*pi*f_c
    omega_r = omega_c / Q
    m = k / omega_c ^ 2
    mC_b2 = m * C_b^2
    
    omega_11 = (C_1 * R_1)^-1
    omega_t1 = (R_t * C_1)^-1
    
    omega_tb = (R_t * C_b)^-1
    
    Ftx(t, x) = [x[2], 
        (
        -omega_c^2 * x[1] - omega_r  * x[2]
            - C_z/(2mC_b2) * (1 + x[5]) * x[3]^2
            - (C_zz - 2*C_z^2/C_b)/(2mC_b2) * (1 + x[5]) * x[1]*x[3]^2
        
        + F(t) / m
        ),
                
            (-omega_tb * x[3]
            + C_z/(R_t * C_b^2) * (1 + x[5]) * x[1] * x[3]
            + omega_t1 * x[4]),
                
            (omega_tb * x[3]
            - C_z/(R_t * C_b^2) * (1 + x[5]) * x[1] * x[3]
            - (omega_11 + omega_t1) * x[4]
            - x[6] / R_1
            
            + V(t) / R_1
            ),
                
            -x[5] / tau_s + U(t) / tau_s,

            P(t),
            omega_det * (x[1] - x[7])
            ]

    Ftx
end


ss_cant_det(cant::SSCant.CantileverParams,
F=u, V=u, U=u, P=u) = ss_cant_det(f_c = cant.f_c, k=cant.k, Q=cant.Q, C_z=cant.C_z,
C_zz=cant.C_zz, C_b=cant.C_b, R_t=cant.R_t, C_1=cant.C_1, R_1=cant.R_1,
tau_s=cant.tau_s, F=F, V=V, U=U, P=P, fdet=cant.f_det)


function init_ss_cant_det(cant::SSCant.CantileverParams, A0, phi0, t0; V=u, U=u, P=u)
    resp = 1 / (1 + cant.f_c * im / cant.f_det)
    A_x = A0 / abs(resp)
    phi_x = phi0 - angle(resp)
    [A_x*cos(phi_x), -2*pi*cant.f_c*A_x*sin(phi_x), cant.C_b*V(t0),
     cant.C_1*V(t0), U(t0), P(t0), A0*cos(phi0)]
end


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

ss_cant_xv(cant::CantileverParams, F=u, V=u, U=u) = ss_cant_xv(
    cant.f_c, cant.Q, cant.k, F)


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
    y0::Vector{Float64}
    dt0::Float64
    fs::Float64
    f0::Float64
    phi0::Float64
    t::Vector{Float64}
    A::Vector{Float64}
    phi::Vector{Float64}
    df::Vector{Float64}
end

immutable DownsampledTimeSeriesRaw
    cant::CantileverParams
    y0::Vector{Float64}
    t_raw::Vector{Float64}
    y_raw::Array{Float64,2}
    dt0::Float64
    fs::Float64
    f0::Float64
    phi0::Float64
    t::Vector{Float64}
    A::Vector{Float64}
    phi::Vector{Float64}
    df::Vector{Float64}
end


function squeeze_y(y)
    y_out = zeros(Float64, length(y), length(y[1]))
    for i in 1:length(y)
        y_out[i, :] = y[i]
    end
    y_out
end

function jacobian(dx)
    jac_t(t, x) = ForwardDiff.jacobian(dx, x)

    return jac_t
end

"""
Params
------

- cant: CantileverParams
- y0: array containing [x, v, q_t, q_1, x_5] at time zero
      The units are µm, µm / µs, pC, pC, unitless (percentage change)
"""
function resample_sscant_sim(cant::CantileverParams, y0::Vector{Float64},
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

    li[:lock2](fp_ratio=0.1, fc_ratio=0.4,
               coeff_ratio=coeff_ratio, window="blackman")
    li[:phase](tf=0.)

    ResampledTimeSeries(cant, ts, dt, li[:m], t, x, li[:A],
                        li[:phi], li[:dphi], li[:df], li[:f0])

end

function raw_sscant_sim(cant::CantileverParams, y0::Vector{Float64},
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


function make_Ftx_jac(ss_cant, cant::CantileverParams, x...)
    
    Fx, F_tx = ss_cant(cant, x...)
    jac = jacobian(Fx)

    return F_tx, jac
end

function chunk_sscant_sim(cant::CantileverParams, y0::Vector{Float64},
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
            global slock = lockin.FIRStateLock(li[:fir], dec, li[:f0],
                                        -li[:mb][end], t0=-T_pre, fs=fs)
            slock[:filt](x)
        else
            slock[:filt](x[2:end])
        end

    end

    tout = slock[:get_t]()
    A = abs(slock[:output])
    phi = DSP.unwrap(angle(slock[:z]))
    df = zeros(phi)
    df[2:end] = diff(phi)

    DownsampledTimeSeries(cant, y0, dt, fs / dec,
                          slock[:f0], slock[:phi0][end], tout,
                          A, phi, df)
end



function downsample_sscant_sim(cant::CantileverParams, y0::Vector{Float64},
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

# Since a discrete delta function has a magnitude $\delta[0] = 1/\Delta t$,
# we can model the thermal force on the cantilever as a random,
# Gaussian force with,

# $$\sigma = \sqrt{\frac{k_c k_B T}{\pi Q f_c \Delta t}}$$
# (see doi:10.1088/0034-4885/29/1/306).

function resample_sscant_noise(cant::CantileverParams, y0::Vector{Float64},
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

function re_down_sscant_noise3(cant::CantileverParams, y0::Vector{Float64},
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

function down_sscant_noise3(cant::CantileverParams, y0::Vector{Float64},
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

    force(t) = randn() * sigma
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

# Eliminate Spline interpolation of x. Use tspan argument instead.
function down_sscant_noise4(cant::CantileverParams, y0::Vector{Float64},
                             T_pre::Float64,
                             T::Float64, dt::Float64;
                             V=u, U=u, coeff_ratio=8., maxstep=dt*0.5,
                             seed=-1, fs_force=20*cant.f_c, args...)
    fs = 1/dt
    k_B = 1.38064852e-11
    t0 = -T_pre
    tf = t0 + T

    if seed > 0
        srand(seed)
    end

    # Cut off force noise at 20•fc by default
    dt_force = 1/fs_force
    sigma = sqrt(cant.k*k_B*cant.T/(pi*cant.Q*cant.f_c*dt_force))

    force(t) = randn() * sigma

    times = t0:dt:tf

    F = make_spline(force, times[1], times[end], dt_force)

    F_tx, jac_t = make_Ftx_jac(ss_cant_1tc, cant, F, V, U)
    t, _y = ODE.ode23s(F_tx, y0, times, jacobian=jac_t; points=:specified,
                       maxstep=maxstep, args...)
    
    y_specified = squeeze_y(_y)

    return TimeSeries(cant, y0, times, y_specified)

end


# Eliminate Spline interpolation of x. Use tspan argument instead.
function chunk_sscant_noise4(cant::CantileverParams, y0::Vector{Float64},
                             T_pre::Float64,
                             T::Float64, dt::Float64, Tchunk;
                             V=u, U=u, coeff_ratio=8., maxstep=dt*0.5, seed=-1,
                             fs_force=20*cant.f_c, args...)
    fs = 1/dt
    k_B = 1.38064852e-11
    t0 = -T_pre
    tf = t0 + T

    if seed > 0
        srand(seed)
    end

    # Cut off force noise at 20•fc by default
    dt_force = 1/fs_force
    sigma = sqrt(cant.k*k_B*cant.T/(pi*cant.Q*cant.f_c*dt_force))
    force(t) = randn() * sigma


    times = t0:dt:tf
    N_total = length(times)
    N_chunk_pts = Int(Tchunk/dt)
    i_endpts = chunk_range(1, N_total, N_chunk_pts)

    y_specified = Matrix{Float64}(N_total, length(y0))
    y_specified[1, :] = y0

    y1 = y0
    F_prev_endpoint = 0.

    for i in eachindex(i_endpts[1 : end-1])
        j0 = i_endpts[i]
        jf = i_endpts[i+1]
        curr_times = times[j0 : jf]

        if i == 1
            F = make_spline(force, curr_times[1], curr_times[end], dt_force)
        else
            F = make_spline(force, curr_times[1], curr_times[end], dt_force,
                            F_prev_endpoint)
        end

        F_prev_endpoint = F(curr_times[end])

        F_tx, jac_t = make_Ftx_jac(ss_cant_1tc, cant, F, V, U)
        
        t, _y = ODE.ode23s(F_tx, y1, curr_times, jacobian=jac_t;
                           points=:specified,
                           maxstep=maxstep, args...)

        y1 = _y[end]
        
        _y_specified = squeeze_y(_y)
        y_specified[j0+1 : jf, :] = _y_specified[2 : end, :]
    end

    return TimeSeries(cant, y0, times, y_specified)

end

function N_chunks(T, Tchunk, dt, t0=0.)
    times = t0:dt:(t0+T)
    N_total = length(times)
    N_chunk_pts = Int(Tchunk/dt)
    i_endpts = chunk_range(1, N_total, N_chunk_pts)
    return length(i_endpts)-1
end

# Add voltage noise term.
function chunk_sscant_noise_volt_noise(cant::CantileverParams, y0::Vector{Float64},
                             T_pre::Float64,
                             T::Float64, dt::Float64, Tchunk;
                             V=u, U=u, maxstep=dt*0.5, seed=-1,
                             fs_force=20*cant.f_c, sigma_potential=1e-5,
                             dt_potential=10., progressbar=nothing,args...)
    fs = 1/dt
    k_B = 1.38064852e-11
    t0 = -T_pre
    tf = t0 + T

    if seed > 0
        srand(seed)
    end

    # Cut off force noise at 20•fc by default
    dt_force = 1/fs_force
    sigma = sqrt(cant.k * k_B * cant.T / (pi * cant.Q * cant.f_c * dt_force))
    force(t) = randn() * sigma
    potential_noise(t) = randn() * sigma_potential

    times = t0:dt:tf
    N_total = length(times)
    N_chunk_pts = Int(Tchunk/dt)
    i_endpts = chunk_range(1, N_total, N_chunk_pts)

    y_specified = Matrix{Float64}(N_total, length(y0))
    y_specified[1, :] = y0

    y1 = y0

    F_prev =0.
    Pn_prev = 0.

    if progressbar == true
        progressbar = Progress(length(i_endpts[1 : end-1]))
    end

    for i in eachindex(i_endpts[1 : end-1])
        j0 = i_endpts[i]
        jf = i_endpts[i+1]
        curr_times = times[j0 : jf]

        # Might be nice to properly handle endpoints, by using
        # F_prev_endpoint as first function value. Then evaluate the function
        # at t[2:end]
        if i == 1
            F = make_spline(force, curr_times[1], curr_times[end], dt_force)
            Pn = make_spline(potential_noise, curr_times[1], curr_times[end],
                         dt_potential)
        else
            F = make_spline(force, curr_times[1], curr_times[end], dt_force,
                            F_prev)
            Pn = make_spline(potential_noise, curr_times[1], curr_times[end],
                         dt_potential, Pn_prev)
        end
        F_prev = F(curr_times[end])
        Pn_prev = Pn(curr_times[end])

        F_tx, jac_t = make_Ftx_jac(ss_cant_1tc_volt_noise, cant, F, V, U, Pn)
        
        t, _y = ODE.ode23s(F_tx, y1, curr_times, jacobian=jac_t;
                           points=:specified,
                           maxstep=maxstep, args...)

        y1 = _y[end]

        _y_specified = squeeze_y(_y)
        y_specified[j0+1 : jf, :] = _y_specified[2 : end, :]

        if progressbar != nothing
            next!(progressbar)
        end
    end

    return times, y_specified

end

function make_jacobian(f)
    jac(t, x) = ForwardDiff.jacobian(x_ -> f(t, x_), x)
    return jac
end

# Add voltage noise term.
function chunk_sscant_noise_volt_noise_det(cant::CantileverParams, y0::Vector{Float64},
                             T_pre::Float64,
                             T::Float64, dt::Float64, Tchunk;
                             V=u, U=u, maxstep=dt*0.5, seed=-1,
                             fs_force=20*cant.f_c, sigma_potential=1e-5,
                             dt_potential=10., progressbar=nothing,args...)
    fs = 1/dt
    k_B = 1.38064852e-11
    t0 = -T_pre
    tf = t0 + T

    if seed > 0
        srand(seed)
    end

    # Cut off force noise at 20•fc by default
    dt_force = 1/fs_force
    sigma = sqrt(cant.k * k_B * cant.T / (pi * cant.Q * cant.f_c * dt_force))
    force(t) = randn() * sigma
    potential_noise(t) = randn() * sigma_potential

    times = t0:dt:tf
    N_total = length(times)
    N_chunk_pts = Int(Tchunk/dt)
    i_endpts = chunk_range(1, N_total, N_chunk_pts)

    y_specified = Matrix{Float64}(N_total, length(y0))
    y_specified[1, :] = y0

    y1 = y0

    F_prev =0.
    Pn_prev = 0.

    for i in eachindex(i_endpts[1 : end-1])
        j0 = i_endpts[i]
        jf = i_endpts[i+1]
        curr_times = times[j0 : jf]

        # Might be nice to properly handle endpoints, by using
        # F_prev_endpoint as first function value. Then evaluate the function
        # at t[2:end]
        if i == 1
            F = make_spline(force, curr_times[1], curr_times[end], dt_force)
            Pn = make_spline(potential_noise, curr_times[1], curr_times[end],
                         dt_potential)
        else
            F = make_spline(force, curr_times[1], curr_times[end], dt_force,
                            F_prev)
            Pn = make_spline(potential_noise, curr_times[1], curr_times[end],
                         dt_potential, Pn_prev)
        end
        F_prev = F(curr_times[end])
        Pn_prev = Pn(curr_times[end])

        F_tx = ss_cant_det(cant, u, V, U, Pn)
        jac_t = make_jacobian(F_tx)
        
        t, _y = ODE.ode23s(F_tx, y1, curr_times, jacobian=jac_t;
                           points=:specified,
                           maxstep=maxstep, args...)

        y1 = _y[end]

        _y_specified = squeeze_y(_y)
        y_specified[j0+1 : jf, :] = _y_specified[2 : end, :]

        if progressbar != nothing
            next!(progressbar)
        end
    end

    return times, y_specified

end

# # Add voltage noise term.
# function chunk_sscant_noise_volt_noise_sep_pulse(cant::CantileverParams, y0::Vector{Float64},
#                              T_pre::Float64,
#                              T::Float64, dt::Float64, Tchunk, tp;
#                              V=u, U=u, maxstep=dt*0.5, seed=-1,
#                              fs_force=20*cant.f_c, sigma_potential=1e-5,
#                              dt_potential=10., progressbar=nothing,args...)
#     fs = 1/dt
#     k_B = 1.38064852e-11
#     t0 = -T_pre
#     tf = t0 + T

#     if seed > 0
#         srand(seed)
#     end

#     # Cut off force noise at 20•fc by default
#     dt_force = 1/fs_force
#     sigma = sqrt(cant.k * k_B * cant.T / (pi * cant.Q * cant.f_c * dt_force))
#     force(t) = randn() * sigma
#     potential_noise(t) = randn() * sigma_potential

#     times = t0:dt:tf

#     N_total = length(times)
#     N_chunk_pts = Int(Tchunk/dt)
#     i_endpts = chunk_range(1, N_total, N_chunk_pts)

#     y_specified = Matrix{Float64}(N_total, length(y0))
#     y_specified[1, :] = y0

#     y1 = y0

#     F_prev =0.
#     Pn_prev = 0.

#     i=1
#     pulse_



#     while i < length(i_endpts[1 : end-1])
#         j0 = i_endpts[i]
#         jf = i_endpts[i+1]
#         curr_times = times[j0 : jf]
#         curr_ti = curr_times[1]
#         curr_tf = curr_times[end]

#         # Might be nice to properly handle endpoints, by using
#         # F_prev_endpoint as first function value. Then evaluate the function
#         # at t[2:end]
#         if i == 1
#             F = make_spline(force, curr_times[1], curr_times[end], dt_force)
#             Pn = make_spline(potential_noise, curr_times[1], curr_times[end],
#                          dt_potential)
#         else
#             F = make_spline(force, curr_times[1], curr_times[end], dt_force,
#                             F_prev)
#             Pn = make_spline(potential_noise, curr_times[1], curr_times[end],
#                          dt_potential, Pn_prev)
#         end
#         F_prev = F(curr_times[end])
#         Pn_prev = Pn(curr_times[end])

#         F_tx, jac_t = make_Ftx_jac(ss_cant_1tc_volt_noise, cant, F, V, U, Pn)
        
#         t, _y = ODE.ode23s(F_tx, y1, curr_times, jacobian=jac_t;
#                            points=:specified,
#                            maxstep=maxstep, args...)

#         y1 = _y[end]

#         _y_specified = squeeze_y(_y)
#         y_specified[j0+1 : jf, :] = _y_specified[2 : end, :]

#         if progressbar != nothing
#             next!(progressbar)
#         end
#     end

#     return times, y_specified

# end



function sscant_noise3(cant::CantileverParams, y0::Vector{Float64},
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
            global slock = lockin.FIRStateLock(li[:fir], dec, li[:f0],
                                        -li[:mb][end], t0=-T_pre, fs=fs)
            slock[:filt](x)
        else
            slock[:filt](x[2:end])
        end

    end

    tout = slock[:get_t]()
    A = abs(slock[:z_out])
    phi = DSP.unwrap(angle(slock[:z_out]))
    df = zeros(phi)
    df[2:end] = diff(phi) / (2pi)
    df[1] = df[2]

    DownsampledTimeSeries(cant, y0, dt, fs / dec,
                          slock[:f0], slock[:phi0], tout,
                          A, phi, df)
end

function chunk_all_sscant_sim_noise_outall(cant::CantileverParams, y0::Vector{Float64},
                             T_pre::Float64,
                             T::Float64, dt::Float64; xdot=ss_cant_1tc,
                              V=u, U=u, coeff_ratio=8., V_append=0.01, U_append=0.005,
                             Tchunk=1000., f_dec::Float64=2*cant.f_c, 
                             seed=-1, force_ratio=20.,
                             fp=nothing,
                             fc=nothing,
                             progressbar=nothing)
    fs = 1/dt
    const k_B = 1.38064852e-11

    dt_force = 1./(cant.f_c * force_ratio)

    sigma = sqrt(cant.k*k_B*cant.T/(pi*cant.Q*cant.f_c*dt_force))

    if seed > 0
        srand(seed)
    end


    force(t) = randn() * sigma

    Ndec = trunc(Int, fs / f_dec)

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

        # may need to revert these changes for some unknown reason.

        t_equal = collect(Tendpts[i]:dt:Tendpts[i+1])
        x = Dierckx.evaluate(spline, t_equal)

        if i == 1
            push!(t_save, t[1])
            push!(y_save, _y[1])
            global li = lockin.LockIn(t_equal, x, fs)
            if (fp != nothing) && (fc != nothing)
                li[:lock2](fp=fp, fc=fc,
                       coeff_ratio=coeff_ratio,
                       window="blackman")
            else
                li[:lock2](fp_ratio=0.1, fc_ratio=0.4,
                       coeff_ratio=coeff_ratio,
                       window="blackman")
            end
            li[:phase](tf=0.)

            global slock = lockin.FIRStateLock(li[:fir], Ndec, li[:f0corr],
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

        if progressbar != nothing
            next!(progressbar)
        end

    end

    y_save_sq = squeeze_y(y_save)

    tout = slock[:get_t]()
    A = abs(slock[:z_out])
    phi = DSP.unwrap(angle(slock[:z_out]))
    df = zeros(phi)
    df[2:end] = diff(phi) / (2pi)
    df[1] = df[2]

    DownsampledTimeSeriesRaw(cant, y0, t_save, y_save_sq, dt, fs / Ndec,
                          slock[:f0], slock[:phi0], tout,
                          A, phi, df)
end

function chunk_all_sscant_sim_noise(cant::CantileverParams, y0::Vector{Float64},
                             T_pre::Float64,
                             T::Float64, dt::Float64; xdot=ss_cant_1tc,
                              V=u, U=u, coeff_ratio=8.,
                             Tchunk=1000., f_dec=nothing, 
                             seed=-1, force_ratio=20., progressbar=nothing)
    fs = 1/dt
    k_B = 1.38064852e-11

    dt_force = 1./(cant.f_c * force_ratio)

    sigma = sqrt(cant.k*k_B*cant.T/(pi*cant.Q*cant.f_c*dt_force))

    if seed > 0
        srand(seed)
    end

    force(t) = t >= 0. ? randn() * sigma : 0.

    if f_dec == nothing
        f_dec = fs
    end

    dec = trunc(Int, fs / f_dec)

    Tendpts = collect(-T_pre:Tchunk:T)
    if Tendpts[end] != T
        push!(Tendpts, T)
    end


    N_Tendpts = length(Tendpts)-1

    if progressbar == true
        progressbar = Progress(N_Tendpts)
    end

    for i = 1:N_Tendpts
        N = round(Int, (Tendpts[i+1] - Tendpts[i])/dt_force) + 1
        t_force = linspace(Tendpts[i], Tendpts[i+1], N)
        F_spline = Dierckx.Spline1D(t_force, map(force, t_force),
                              k=3)
        function F(t)
            Dierckx.evaluate(F_spline, t)
        end
        F_tx, jac = make_Ftx_jac(xdot, cant, F, V, U)

        t, _y = ODE.ode23s(F_tx, y0, [Tendpts[i], Tendpts[i+1]],
                           jacobian=jac, maxstep=dt*0.5)
        y = squeeze_y(_y)

        y0 = _y[end]

        spline = Dierckx.Spline1D(t, y[:, 1], k=3)
        t_equal = collect(Tendpts[i]:dt:Tendpts[i+1])

        x = Dierckx.evaluate(spline, t_equal)

        if i == 1
            li = lockin.LockIn(t_equal, x, fs)
            li[:lock2](fp_ratio=0.1, fc_ratio=0.4,
                       coeff_ratio=coeff_ratio, window="blackman")

            li[:phase]()
            global slock = lockin.FIRStateLock(li[:fir], dec, li[:f0],
                                        -li[:mb][end], t0=-T_pre, fs=fs)
            slock[:filt](x)
        else
            slock[:filt](x[2:end])
        end
        

        if progressbar != nothing
            next!(progressbar)
        end

    end

    tout = slock[:get_t]()
    A = abs(slock[:z_out])
    phi = DSP.unwrap(angle(slock[:z_out]))
    df = zeros(phi)
    df[2:end] = diff(phi) / (2pi)
    df[1] = df[2]

    DownsampledTimeSeries(cant, y0, dt, fs / dec,
                          slock[:f0], slock[:phi0], tout,
                          A, phi, df)
end

function chunk_all_sscant_sim_brownian(cant::CantileverParams, y0::Vector{Float64},
                             T_pre::Float64,
                             T::Float64, dt::Float64, dt_out::Float64;
                             xdot=ss_cant_1tc,
                             V=u, U=u, coeff_ratio=8.,
                             Tchunk=1000., 
                             seed=-1, force_ratio=20., progressbar=nothing)
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

    N_Tendpts = length(Tendpts)-1

    if progressbar == true
        progressbar = Progress(N_Tendpts)
    end

    t_collected = Vector{Float64}()
    x_collected = Vector{Float64}()

    Npts_expected = round(Int, (T+t_pre)/dt)

    sizehint!(t_collected, Npts_expected)
    sizehint!(x_collected, Npts_expected)

    for i = 1:N_Tendpts

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

        if progressbar != nothing
            next!(progressbar)
        end

    end

    t_collected, x_collected, y0
end



end # module
