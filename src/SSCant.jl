module SSCant

import ForwardDiff
import ODE
import CurveFit
import DSP
import Dierckx

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

type CantileverParams
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
end

function CantileverParams(;f_c=0.05, Q=1000., k=1.,
                 C_z=1e-8, C_zz=1e-10, C_b=0.1,
                 R_t=1e-4, C_1=600., R_1=3e-6,
                 tau_s=15e3)
    CantileverParams(f_c, Q, k,
                 C_z, C_zz, C_b,
                 R_t, C_1, R_1,
                 tau_s)
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
function ss_cant(f_c=0.05, Q=1000, k=1,
                 C_z=1e-6, C_zz=-1e-5, C_b=0.1,
                 R_t=1e-4, C_1=600, R_1=3e-6,
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

ss_cant(cp::CantileverParams, F=u, V=u, U=u) = ss_cant(cp.f_c,
    cp.Q, cp.k, cp.C_z, cp.C_zz, cp.C_b, cp.R_t, cp.C_1,
    cp.R_1, cp.tau_s, F, V, U)


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
    cp::CantileverParams
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


function TimeSeries(cp::CantileverParams, t, y)
    x = y[:, 1]
    v = y[:, 2]
    q_t = y[:, 3]
    q_1 = y[:, 4]
    x_5 = y[:, 5]
    A = amplitude(x, v, f_c=cp.f_c)
    phi = phase(x, v, f_c=cp.f_c)
    dphi, df, f0 = fit_phase(t, phi)

    TimeSeries(cp, t, x, v, q_t, q_1, x_5,
               A, phi, dphi, df, f0)
end

type ResampledTimeSeries
    cp::CantileverParams
    _t
    _x
    _v
    _q_t
    _q_1
    _x_5
    dt
    t
    x
    v
    A
    phi
    dphi
    df
    f0
end

function ResampledTimeSeries(ts::TimeSeries, dt)
    # Resample x, v using 3rd order spline interpolation
    # Create z (fir hilbert filter)
    # Use z to calculate A, phi, dphi, df, f0
end


function squeeze_y(y)
    y_out = zeros(Float64, length(y), length(y[1]))
    for i in 1:length(y)
        y_out[i, :] = y[i]
    end
    y_out
end

function ss_jacobian(dx)
    jac = ForwardDiff.jacobian(dx)

    jac_t(t, x) = jac(x)

    return jac_t
end

"""
Params
------

- cp: CantileverParams
- y0: array containing [x, v, q_t, q_1, x_5] at time zero
      The units are µm, µm / µs, pC, pC, unitless (percentage change)
"""
function run_sscant(cp::CantileverParams, y0, t_range; F=u, V=u, U=u, args...)
    F, F_tx = ss_cant(cp, F, V, U)
    jac_t = ss_jacobian(F)
    t, _y = ODE.ode23s(F_tx, y0, t_range, jacobian=jac_t; args...)
    y = squeeze_y(_y)

    TimeSeries(cp, t, y)
end

end # module
