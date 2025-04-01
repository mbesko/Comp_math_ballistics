import numpy as np

def projectile_derivs(state, params):
    """
    Вычисляет производные для системы ОДУ.
    state = [x, y, vx, vy]
    """
    x, y, vx, vy = state
    m = params['mass']
    r = params['radius']
    A = np.pi * r ** 2  # площадь поперечного сечения
    drag_coef = 0.5 * params['Cd'] * params['rho'] * A
    wind = params['wind']
    v_rel = np.array([vx - wind[0], vy - wind[1]])
    speed_rel = np.linalg.norm(v_rel)
    if speed_rel != 0:
        drag_acc = - (drag_coef / m) * speed_rel * v_rel
    else:
        drag_acc = np.array([0.0, 0.0])
    ax = drag_acc[0]
    ay = -params['g'] + drag_acc[1]
    return np.array([vx, vy, ax, ay])


def rk4_step(state, dt, params):
    """Один шаг интегрирования методом Рунге–Кутты 4-го порядка."""
    k1 = projectile_derivs(state, params)
    k2 = projectile_derivs(state + dt / 2 * k1, params)
    k3 = projectile_derivs(state + dt / 2 * k2, params)
    k4 = projectile_derivs(state + dt * k3, params)
    new_state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return new_state


def dopri_step(state, dt, params):
    """
    Один шаг интегрирования методом Dormand–Prince (RKDP) вручную.
    Коэффициенты взяты из стандартной таблицы метода.
    """
    # Коэффициенты c (не используются напрямую, т.к. dt умножается на сумму с коэффициентами a)
    c2 = 1 / 5
    c3 = 3 / 10
    c4 = 4 / 5
    c5 = 8 / 9
    c6 = 1
    c7 = 1

    # Коэффициенты a_ij:
    a21 = 1 / 5

    a31 = 3 / 40
    a32 = 9 / 40

    a41 = 44 / 45
    a42 = -56 / 15
    a43 = 32 / 9

    a51 = 19372 / 6561
    a52 = -25360 / 2187
    a53 = 64448 / 6561
    a54 = -212 / 729

    a61 = 9017 / 3168
    a62 = -355 / 33
    a63 = 46732 / 5247
    a64 = 49 / 176
    a65 = -5103 / 18656

    a71 = 35 / 384
    a72 = 0
    a73 = 500 / 1113
    a74 = 125 / 192
    a75 = -2187 / 6784
    a76 = 11 / 84

    # b-коэффициенты для 5-го порядка:
    b1 = 35 / 384
    b2 = 0
    b3 = 500 / 1113
    b4 = 125 / 192
    b5 = -2187 / 6784
    b6 = 11 / 84
    b7 = 0

    k1 = projectile_derivs(state, params)
    k2 = projectile_derivs(state + dt * a21 * k1, params)
    k3 = projectile_derivs(state + dt * (a31 * k1 + a32 * k2), params)
    k4 = projectile_derivs(state + dt * (a41 * k1 + a42 * k2 + a43 * k3), params)
    k5 = projectile_derivs(state + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), params)
    k6 = projectile_derivs(state + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), params)
    k7 = projectile_derivs(state + dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6), params)

    new_state = state + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7)
    return new_state