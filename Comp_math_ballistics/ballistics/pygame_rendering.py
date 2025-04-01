import pygame

def sim_to_screen(x, y, scale, ground_y):
    """
    Преобразует координаты из метров в экранные пиксели.
    """
    screen_x = int(x * scale + 50)  # отступ слева 50 пикселей
    screen_y = int(ground_y - y * scale)  # инверсия оси y
    return screen_x, screen_y

def pygame_rendering(screen, rk4_state, dopri_state, trajectory_rk4, trajectory_dopri, ground_y, scale, WIDTH, sim_time, error):
    screen.fill((255, 255, 255))
    pygame.draw.line(screen, (0, 128, 0), (0, ground_y), (WIDTH, ground_y), 3)

    if len(trajectory_dopri) > 1:
        pygame.draw.lines(screen, (255, 165, 0), False, trajectory_dopri, 6)  # траектория Dormand–Prince (оранжевая)
    if len(trajectory_rk4) > 1:
        pygame.draw.lines(screen, (0, 0, 255), False, trajectory_rk4, 2)  # траектория RK4 (синяя)

    # Отображение текущих положений (снаряд: синий для RK4, красный для Dormand–Prince)
    pygame.draw.circle(screen, (0, 0, 255), sim_to_screen(rk4_state[0], rk4_state[1], scale, ground_y), 10)
    pygame.draw.circle(screen, (255, 0, 0), sim_to_screen(dopri_state[0], dopri_state[1], scale, ground_y), 5)

    # Вывод времени симуляции и ошибки интегрирования
    font = pygame.font.SysFont("Arial", 18)
    time_text = font.render(f"Время: {sim_time:.2f} с", True, (0, 0, 0))
    error_text = font.render(f"Ошибка (позиция): {error:.4f} м", True, (0, 0, 0))
    screen.blit(time_text, (10, 10))
    screen.blit(error_text, (10, 30))

    pygame.display.flip()