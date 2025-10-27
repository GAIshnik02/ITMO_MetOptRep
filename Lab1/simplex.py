import numpy as np


def read_problem_from_file(filename):
    """
    Чтение задачи линейного программирования из файла.
    Формат файла:
    Строка 1: коэффициенты целевой функции (4 числа)
    Строки 2-4: матрица ограничений A (3 строки по 4 числа)
    Строка 5: правые части ограничений b (3 числа)
    Строка 6: типы ограничений (3 символа: <=, =, >=)
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Чтение целевой функции
        c = list(map(float, lines[0].split()))
        if len(c) != 4:
            raise ValueError("Целевая функция должна содержать 4 коэффициента")

        # Чтение матрицы ограничений
        A = []
        for i in range(1, 4):
            row = list(map(float, lines[i].split()))
            if len(row) != 4:
                raise ValueError(f"Строка {i + 1} должна содержать 4 коэффициента")
            A.append(row)

        # Чтение правых частей
        b = list(map(float, lines[4].split()))
        if len(b) != 3:
            raise ValueError("Должно быть 3 правых части ограничений")

        # Чтение типов ограничений
        constraint_types = lines[5].split()
        if len(constraint_types) != 3:
            raise ValueError("Должно быть 3 типа ограничений")

        # Проверка допустимости типов ограничений
        for ct in constraint_types:
            if ct not in ['<=', '=', '>=']:
                raise ValueError(f"Недопустимый тип ограничения: {ct}")

        return c, A, b, constraint_types

    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {filename} не найден")
    except Exception as e:
        raise ValueError(f"Ошибка чтения файла: {e}")


def simplex_method(c, A, b, constraint_types):
    """
    Реализация симплекс-метода для решения задач линейного программирования
    на максимизацию с ограничениями разных типов.
    """
    # Преобразование входных данных в массивы numpy
    c = np.array(c, dtype=float)
    A_orig = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # Определение размерности задачи
    m, n = A_orig.shape

    print("=== ЗАДАЧА ЛИНЕЙНОГО ПРОГРАММИРОВАНИЯ ===")
    print(f"Целевая функция: max {c}·x")
    print("Ограничения:")
    for i in range(m):
        print(f"{A_orig[i]}·x {constraint_types[i]} {b[i]}")
    print()

    # Подсчет количества дополнительных переменных
    num_slack = sum(1 for ct in constraint_types if ct == '<=')
    num_surplus = sum(1 for ct in constraint_types if ct == '>=')
    num_artificial = sum(1 for ct in constraint_types if ct in ['=', '>='])

    total_vars = n + num_slack + num_surplus + num_artificial

    print(f"Дополнительные переменные: {num_slack} slack, {num_surplus} surplus, {num_artificial} artificial")
    print(f"Общее количество переменных: {total_vars}")

    # Создание расширенной матрицы
    A_ext = np.zeros((m, total_vars))
    A_ext[:, :n] = A_orig

    # Индексы для дополнительных переменных
    slack_idx = n
    surplus_idx = n + num_slack
    artificial_idx = n + num_slack + num_surplus

    # Списки для отслеживания переменных
    slack_vars = []
    surplus_vars = []
    artificial_vars = []
    basis = []

    # Добавление дополнительных переменных
    current_slack = 0
    current_surplus = 0
    current_artificial = 0

    for i in range(m):
        if constraint_types[i] == '<=':
            # Добавляем slack переменную
            A_ext[i, slack_idx + current_slack] = 1
            slack_vars.append(slack_idx + current_slack)
            basis.append(slack_idx + current_slack)
            current_slack += 1

        elif constraint_types[i] == '>=':
            # Добавляем surplus и искусственную переменные
            A_ext[i, surplus_idx + current_surplus] = -1
            A_ext[i, artificial_idx + current_artificial] = 1
            surplus_vars.append(surplus_idx + current_surplus)
            artificial_vars.append(artificial_idx + current_artificial)
            basis.append(artificial_idx + current_artificial)
            current_surplus += 1
            current_artificial += 1

        elif constraint_types[i] == '=':
            # Добавляем искусственную переменную
            A_ext[i, artificial_idx + current_artificial] = 1
            artificial_vars.append(artificial_idx + current_artificial)
            basis.append(artificial_idx + current_artificial)
            current_artificial += 1

    b_ext = b.copy()

    print("1. Расширенная матрица системы ограничений:")
    header = "   " + " ".join([f"x{i + 1}" for i in range(total_vars)]) + "   B"
    print(header)
    for i in range(m):
        row = "   "
        for j in range(total_vars):
            row += f"{A_ext[i, j]:2.0f}  "
        row += f"{b_ext[i]:2.0f}"
        print(row)
    print(f"   Базисные переменные: {[f'x{i + 1}' for i in basis]}")
    print()

    # Двухфазный симплекс-метод
    if artificial_vars:
        print("2. Используем двухфазный симплекс-метод (есть искусственные переменные)")

        # Фаза 1: минимизация суммы искусственных переменных
        print("   Фаза 1: минимизация суммы искусственных переменных")

        # Целевая функция для фазы 1 (минимизация суммы искусственных переменных)
        c_phase1 = np.zeros(total_vars)
        for art_var in artificial_vars:
            c_phase1[art_var] = 1  # Минимизируем сумму искусственных переменных

        # Создание симплекс-таблицы для фазы 1
        table_phase1 = np.zeros((m + 1, total_vars + 1))
        table_phase1[:m, :total_vars] = A_ext
        table_phase1[:m, -1] = b_ext

        # Инициализация строки целевой функции фазы 1
        table_phase1[-1, :total_vars] = c_phase1

        # Приведение целевой строки к виду, согласованному с текущим базисом
        for i in range(m):
            basis_var = basis[i]
            coeff = c_phase1[basis_var]
            table_phase1[-1, :] -= coeff * table_phase1[i, :]

        print("   Начальная таблица фазы 1:")
        print_simplex_table(table_phase1, basis, m, total_vars)

        # Симплекс-метод для фазы 1
        phase1_result, phase1_basis = perform_simplex_iterations(
            table_phase1, basis.copy(), m, total_vars, "Фаза 1", artificial_vars
        )

        # Проверка результата фазы 1
        if abs(phase1_result[-1, -1]) > 1e-6:
            print("   ✗ Задача не имеет допустимого решения!")
            return {'success': False, 'message': "Задача не имеет допустимого решения"}

        print("   ✓ Фаза 1 завершена, найдено допустимое базисное решение")

        # Удаляем искусственные переменные из таблицы для фазы 2
        # Создаем новую таблицу без столбцов искусственных переменных
        non_artificial_cols = [j for j in range(total_vars) if j not in artificial_vars]
        new_total_vars = len(non_artificial_cols)

        # Создаем новую таблицу для фазы 2
        table_phase2 = np.zeros((m + 1, new_total_vars + 1))
        table_phase2[:m, :new_total_vars] = phase1_result[:m, non_artificial_cols]
        table_phase2[:m, -1] = phase1_result[:m, -1]

        # Обновляем базис - убираем искусственные переменные
        phase2_basis = []
        for basis_var in phase1_basis:
            if basis_var not in artificial_vars:
                # Находим новый индекс для базисной переменной
                new_idx = non_artificial_cols.index(basis_var)
                phase2_basis.append(new_idx)

        # Если в базисе осталось меньше m переменных, добавляем недостающие
        while len(phase2_basis) < m:
            # Находим свободный столбец для добавления в базис
            for j in range(new_total_vars):
                if j not in phase2_basis and abs(table_phase2[len(phase2_basis), j]) > 1e-6:
                    phase2_basis.append(j)
                    break

        # Фаза 2: исходная целевая функция
        print("   Фаза 2: максимизация исходной целевой функции")

        # Инициализация строки целевой функции для фазы 2
        for j in range(n):  # только исходные переменные
            if j < new_total_vars:
                table_phase2[-1, j] = -c[j]  # Для максимизации берем с отрицательным знаком

        # Приведение целевой строки к виду, согласованному с текущим базисом
        for i in range(m):
            basis_var = phase2_basis[i]
            if basis_var < n:  # если базисная переменная - исходная
                coeff = c[basis_var]
                table_phase2[-1, :] -= coeff * table_phase2[i, :]

        print("   Начальная таблица фазы 2:")
        print_simplex_table(table_phase2, phase2_basis, m, new_total_vars)

        # Симплекс-метод для фазы 2
        final_table, final_basis = perform_simplex_iterations(
            table_phase2, phase2_basis.copy(), m, new_total_vars, "Фаза 2"
        )

        # Формирование результата
        x_opt = np.zeros(n)
        for i, basis_var in enumerate(final_basis):
            if basis_var < n:
                x_opt[basis_var] = final_table[i, -1]

        optimal_value = final_table[-1, -1]  # Меняем знак, т.к. работали с -f(x)

    else:
        print("2. Используем обычный симплекс-метод (нет искусственных переменных)")

        # Создание симплекс-таблицы
        table = np.zeros((m + 1, total_vars + 1))
        table[:m, :total_vars] = A_ext
        table[:m, -1] = b_ext

        # Инициализация строки целевой функции
        for j in range(n):
            table[-1, j] = -c[j]

        # Приведение целевой строки к виду, согласованному с текущим базисом
        for i in range(m):
            basis_var = basis[i]
            if basis_var < n:  # если базисная переменная - исходная
                coeff = c[basis_var]
                table[-1, :] -= coeff * table[i, :]

        final_table, final_basis = perform_simplex_iterations(
            table, basis.copy(), m, total_vars, "Основная"
        )

        # Формирование результата
        x_opt = np.zeros(n)
        for i, basis_var in enumerate(final_basis):
            if basis_var < n:
                x_opt[basis_var] = final_table[i, -1]

        optimal_value = -final_table[-1, -1]  # Меняем знак, т.к. работали с -f(x)

    return {
        'success': True,
        'x': x_opt,
        'fun': optimal_value,
        'message': "Оптимальное решение найдено"
    }


def perform_simplex_iterations(table, basis, m, total_vars, phase_name, artificial_vars=None):
    """
    Выполнение итераций симплекс-метода.
    """
    iteration = 0
    max_iter = 20

    while iteration < max_iter:
        iteration += 1

        # Проверка условия оптимальности
        optimal = True
        entering = -1
        min_coeff = 0

        # Поиск переменной для ввода в базис (исключаем искусственные переменные в фазе 2)
        for j in range(total_vars):
            if artificial_vars is not None and phase_name == "Фаза 1" and j in artificial_vars:
                continue  # Пропускаем искусственные переменные при выборе входящей

            if table[-1, j] < -1e-6:
                optimal = False
                if table[-1, j] < min_coeff:
                    min_coeff = table[-1, j]
                    entering = j

        if optimal:
            print(f"   {phase_name}: достигнуто оптимальное решение на итерации {iteration}!")
            break

        if entering == -1:
            print(f"   {phase_name}: нет подходящей переменной для ввода в базис")
            break

        print(f"   {phase_name}: итерация {iteration}, вводим x{entering + 1}")

        # Определение переменной для вывода из базиса
        leaving = -1
        min_ratio = float('inf')

        for i in range(m):
            if table[i, entering] > 1e-6:
                ratio = table[i, -1] / table[i, entering]
                if ratio < min_ratio:
                    min_ratio = ratio
                    leaving = i

        if leaving == -1:
            print(f"   {phase_name}: задача неограничена!")
            break

        print(f"   Выводим x{basis[leaving] + 1}")

        # Пересчет симплекс-таблицы
        pivot = table[leaving, entering]
        table[leaving, :] /= pivot

        for i in range(m + 1):
            if i != leaving:
                factor = table[i, entering]
                table[i, :] -= factor * table[leaving, :]

        # Обновление базиса
        basis[leaving] = entering

        print_simplex_table(table, basis, m, total_vars)

    return table, basis


def print_simplex_table(table, basis, m, total_vars):
    """
    Вывод симплекс-таблицы.
    """
    print("   Базис     B    ", end="")
    for j in range(total_vars):
        print(f"x{j + 1:>5} ", end="")
    print()

    for i in range(m):
        row = f"   x{basis[i] + 1:2} {table[i, -1]:8.2f}"
        for j in range(total_vars):
            row += f" {table[i, j]:6.2f}"
        print(row)

    f_row = f"   F     {table[-1, -1]:8.2f}"
    for j in range(total_vars):
        f_row += f" {table[-1, j]:6.2f}"
    print(f_row)
    print()


# Основная программа
if __name__ == "__main__":
    try:
        # Чтение задачи из файла
        print("Чтение данных из файла input.txt...")
        c, A, b, constraint_types = read_problem_from_file("input.txt")

        # Решение задачи симплекс-методом
        result = simplex_method(c, A, b, constraint_types)

        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТ РЕШЕНИЯ ЗАДАЧИ ЛИНЕЙНОГО ПРОГРАММИРОВАНИЯ")
        print("=" * 60)

        if result['success']:
            x = result['x']
            print("✓ ОПТИМАЛЬНОЕ РЕШЕНИЕ НАЙДЕНО")
            print(f"Значения переменных:")
            for i, val in enumerate(x):
                print(f"  x{i + 1} = {val:.6f}")
            print(f"Максимальное значение целевой функции: {result['fun']:.6f}")

            # Проверка выполнения ограничений
            print("\nПроверка выполнения ограничений:")
            all_constraints_satisfied = True
            A_array = np.array(A)
            for i in range(len(A)):
                # Вычисление левой части ограничения
                lhs = np.dot(A_array[i], x)
                rhs = b[i]
                constraint = constraint_types[i]

                # Проверка выполнения ограничения
                satisfied = (
                        (constraint == '<=' and lhs <= rhs + 1e-6) or
                        (constraint == '=' and abs(lhs - rhs) < 1e-6) or
                        (constraint == '>=' and lhs >= rhs - 1e-6)
                )
                status = "✓" if satisfied else "✗"
                print(f"  {status} {A[i]}·x = {lhs:.6f} {constraint} {rhs}")

                if not satisfied:
                    all_constraints_satisfied = False

            if all_constraints_satisfied:
                print("\n✓ Все ограничения выполнены!")
            else:
                print("\n✗ Некоторые ограничения не выполнены!")

        else:
            print("✗ РЕШЕНИЕ НЕ НАЙДЕНО")
            print(f"Причина: {result['message']}")

    except Exception as e:
        print(f"Ошибка при выполнении программы: {e}")
        import traceback

        traceback.print_exc()