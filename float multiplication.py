import pygame
import numpy as np

from pygame_rendering import pygame_rendering, sim_to_screen
from rk4_dorpi5 import rk4_step, dopri_step

def main():
    # ============================================
    # 1. ПАРАМЕТРЫ МОДЕЛИ
    # ============================================
    params = {
        # Начальное положение пушки
        'x0': 0.0,  # (м)
        'y0': 1.0,  # (м)

        # Характеристики выстрела
        'v0': 50.0,  # начальная скорость (м/с)
        'angle': 45.0,  # угол выстрела (градусы)

        # Физические параметры снаряда
        'mass': 3.0,  # масса (кг)
        'size': 0.1,  # диаметр (м)
        'radius': 0.1 / 2.0,  # радиус (м)

        # Гравитация
        'g': 9.81,  # (м/с²)

        # Параметры воздуха для расчёта сопротивления
        'rho': 1.225,  # плотность воздуха (кг/м³)
        'Cd': 0.47,  # коэффициент сопротивления для сферы

        # Параметры ветра
        'wind_speed': -20.0,  # (м/с)
        'wind_direction': 0.0,  # (градусы, 0° – вправо, 90° – вверх)

        # Коэффициент восстановления (отскока)
        'restitution': 0.8,  # от 0 (полное затухание) до 1 (идеальный упругий отскок)

        # Порог прекращения движения после отскока
        'min_vy': 0.5  # (м/с)
    }

    # Вычисляем вектор ветра
    wind_angle_rad = np.radians(params['wind_direction'])
    params['wind'] = np.array([
        params['wind_speed'] * np.cos(wind_angle_rad),
        params['wind_speed'] * np.sin(wind_angle_rad)
    ])

    # ============================================
    # 3. ИНИЦИАЛИЗАЦИЯ СИСТЕМ ДЛЯ ОБОИХ МЕТОДОВ
    # ============================================
    theta = np.radians(params['angle'])
    initial_state = np.array([
        params['x0'],
        params['y0'],
        params['v0'] * np.cos(theta),
        params['v0'] * np.sin(theta)
    ])
    rk4_state = initial_state.copy()
    dopri_state = initial_state.copy()
    sim_time = 0.0
    dt = 0.01  # шаг по времени (с)

    trajectory_rk4 = []  # для сохранения точек траектории (RK4)
    trajectory_dopri = []  # для траектории (Dormand–Prince)

    # ============================================
    # 4. НАСТРОЙКА PYGAME
    # ============================================
    pygame.init()
    WIDTH, HEIGHT = 1300, 600
    scale = 4
    error = 0
    ground_y = HEIGHT - 50
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Симуляция снаряда: RK4 (синий), Dormand–Prince (красный)")
    clock = pygame.time.Clock()

    rk4_running = True
    dopri_running = True

    # ============================================
    # 5. ГЛАВНЫЙ ЦИКЛ СИМУЛЯЦИИ
    # ============================================
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if rk4_running or dopri_running:
            # Выполняем шаг интегрирования для каждого метода
            rk4_state = rk4_step(rk4_state, dt, params)
            dopri_state = dopri_step(dopri_state, dt, params)
            sim_time += dt

            # Обработка отскока для метода RK4
            if rk4_state[1] < 0:
                rk4_state[1] = 0
                if rk4_state[3] < 0:
                    rk4_state[3] = -params['restitution'] * rk4_state[3]
                if abs(rk4_state[3]) < params['min_vy']:
                    rk4_running = False

            # Обработка отскока для метода Dormand–Prince
            if dopri_state[1] < 0:
                dopri_state[1] = 0
                if dopri_state[3] < 0:
                    dopri_state[3] = -params['restitution'] * dopri_state[3]
                if abs(dopri_state[3]) < params['min_vy']:
                    dopri_running = False

            # Сохраняем экранные координаты для отрисовки траекторий
            trajectory_rk4.append(sim_to_screen(rk4_state[0], rk4_state[1], scale, ground_y))
            trajectory_dopri.append(sim_to_screen(dopri_state[0], dopri_state[1], scale, ground_y))

        # Вычисляем разницу в положениях (ошибку) между методами
        error += np.linalg.norm(rk4_state[:2] - dopri_state[:2])

        pygame_rendering(screen, rk4_state, dopri_state, trajectory_rk4, trajectory_dopri, ground_y, scale, WIDTH, sim_time, error)
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    main()