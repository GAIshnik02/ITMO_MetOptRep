import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings('ignore')


class PiyavskyOptimizer:
    def __init__(self, func, a, b, L=None, eps=0.01):
        self.func = func
        self.a = a
        self.b = b
        self.eps = eps
        self.L = L
        self.iterations = 0
        self.evaluations = []

    def estimate_lipschitz(self, n_samples=1000):
        x_samples = np.linspace(self.a, self.b, n_samples)
        f_samples = np.array([self.func(x) for x in x_samples])

        differences = np.abs(np.diff(f_samples) / np.diff(x_samples))
        L_est = np.max(differences) * 1.2

        return L_est if not np.isnan(L_est) and L_est > 0 else 10.0

    def find_next_point(self, points, values, L):
        sorted_indices = np.argsort(points)
        sorted_points = np.array(points)[sorted_indices]
        sorted_values = np.array(values)[sorted_indices]

        max_characteristic = -np.inf
        best_x = self.a

        for i in range(len(sorted_points) - 1):
            x1, x2 = sorted_points[i], sorted_points[i + 1]
            f1, f2 = sorted_values[i], sorted_values[i + 1]

            z = 0.5 * (f1 + f2 - L * (x2 - x1))
            characteristic = z + L * (x2 - x1) / 2

            x_candidate = (f1 - f2 + L * (x1 + x2)) / (2 * L)
            x_candidate = max(x1, min(x2, x_candidate))

            if characteristic > max_characteristic:
                max_characteristic = characteristic
                best_x = x_candidate

        return best_x, max_characteristic

    def optimize(self):
        start_time = time.time()

        if self.L is None:
            self.L = self.estimate_lipschitz()
        print(f"Используемая константа Липшица: {self.L:.4f}")

        points = [self.a, self.b]
        values = [self.func(self.a), self.func(self.b)]
        self.evaluations = list(zip(points, values))

        iteration = 0
        max_iterations = 50

        best_idx = np.argmin(values)
        best_x = points[best_idx]
        best_f = values[best_idx]

        print(f"Начальный минимум: f({best_x:.4f}) = {best_f:.4f}")

        while iteration < max_iterations:
            iteration += 1

            candidate_x, characteristic = self.find_next_point(points, values, self.L)

            current_min_f = min(values)

            lower_bound_candidate = max([values[i] - self.L * abs(candidate_x - points[i])
                                         for i in range(len(points))])

            gap = current_min_f - lower_bound_candidate

            print(f"Итерация {iteration:2d}: x={candidate_x:8.4f}, gap={gap:8.4f}")

            if gap <= self.eps:
                print(f"Достигнута точность {gap:.6f} <= {self.eps}")
                break

            candidate_f = self.func(candidate_x)

            points.append(candidate_x)
            values.append(candidate_f)
            self.evaluations.append((candidate_x, candidate_f))

            if candidate_f < best_f:
                best_x, best_f = candidate_x, candidate_f
                print(f"Улучшение: новый минимум f({best_x:.4f}) = {best_f:.4f}")

        self.iterations = iteration
        self.execution_time = time.time() - start_time
        self.optimum_x = best_x
        self.optimum_f = best_f

        return self.optimum_x, self.optimum_f

    def get_lower_bound(self, x, points, values, L):
        if len(points) == 0:
            return -np.inf

        bounds = [values[i] - L * abs(x - points[i]) for i in range(len(points))]
        return max(bounds)

    def plot_results(self, save_path=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        x_plot = np.linspace(self.a, self.b, 1000)
        f_plot = [self.func(x) for x in x_plot]

        eval_x = [p[0] for p in self.evaluations]
        eval_y = [p[1] for p in self.evaluations]

        ax1.plot(x_plot, f_plot, 'b-', linewidth=2, label='Целевая функция')
        ax1.plot(eval_x, eval_y, 'ro', markersize=6, alpha=0.7, label='Точки оценок')
        ax1.plot(self.optimum_x, self.optimum_f, 'g*',
                 markersize=15, label=f'Найденный минимум: f({self.optimum_x:.4f}) = {self.optimum_f:.4f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title('Метод Пиявского: поиск глобального минимума')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        lower_env = [self.get_lower_bound(x, eval_x, eval_y, self.L) for x in x_plot]

        ax2.plot(x_plot, f_plot, 'b-', linewidth=2, alpha=0.7, label='Целевая функция')
        ax2.plot(x_plot, lower_env, 'r--', linewidth=2, label='Нижняя огибающая')
        ax2.plot(eval_x, eval_y, 'ko', markersize=4, alpha=0.7, label='Точки оценок')
        ax2.plot(self.optimum_x, self.optimum_f, 'g*',
                 markersize=15, label=f'Минимум: f({self.optimum_x:.4f}) = {self.optimum_f:.4f}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('f(x)')
        ax2.set_title('Нижняя огибающая (вспомогательная функция)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен: {save_path}")

        plt.close()
        return fig


def custom_function(x):
    return x + np.sin(3.14159 * x)


def main():
    print("Программа для поиска глобального экстремума методом Пиявского")
    print("Целевая функция: f(x) = x + sin(3.14159*x)")
    print("=" * 60)

    a = -3.0
    b = 3.0
    eps = 0.01

    print(f"Интервал: [{a}, {b}]")
    print(f"Точность: {eps}")
    print("=" * 60)

    optimizer = PiyavskyOptimizer(
        func=custom_function,
        a=a,
        b=b,
        L=None,
        eps=eps
    )

    start_time = time.time()
    opt_x, opt_f = optimizer.optimize()
    execution_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("=" * 60)
    print(f"Найденный минимум:")
    print(f"  x* = {opt_x:.6f}")
    print(f"  f(x*) = {opt_f:.6f}")
    print(f"Количество итераций: {optimizer.iterations}")
    print(f"Количество оценок функции: {len(optimizer.evaluations)}")
    print(f"Время выполнения: {execution_time:.4f} секунд")

    x_check = np.linspace(a, b, 10000)
    f_check = [custom_function(x) for x in x_check]
    true_min_idx = np.argmin(f_check)
    true_min_x = x_check[true_min_idx]
    true_min_f = f_check[true_min_idx]

    print(f"\nПроверка (грубым перебором 10000 точек):")
    print(f"  Истинный минимум: x = {true_min_x:.6f}, f(x) = {true_min_f:.6f}")
    print(f"  Ошибка по x: {abs(opt_x - true_min_x):.6f}")
    print(f"  Ошибка по f(x): {abs(opt_f - true_min_f):.6f}")

    optimizer.plot_results(save_path='piyavsky_optimization_result.png')


if __name__ == "__main__":
    main()