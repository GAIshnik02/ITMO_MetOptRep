import numpy as np
from scipy.optimize import linprog


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


def solve_with_scipy(c, A, b, constraint_types):
    """
    Решение задачи линейного программирования с помощью scipy
    """
    print("=== РЕШЕНИЕ С ПОМОЩЬЮ SCIPY ===")
    print(f"Целевая функция: max {c}·x")
    print("Ограничения:")
    for i in range(len(A)):
        print(f"  {A[i]}·x {constraint_types[i]} {b[i]}")
    print()

    # Scipy решает только задачу минимизации, поэтому меняем знак целевой функции
    # max c·x = min (-c)·x
    c_scipy = [-x for x in c]

    # Подготовка ограничений для scipy
    A_ub = []  # Неравенства <=
    b_ub = []  # Правые части для <=
    A_eq = []  # Равенства =
    b_eq = []  # Правые части для =

    for i in range(len(A)):
        if constraint_types[i] == '<=':
            A_ub.append(A[i])
            b_ub.append(b[i])
        elif constraint_types[i] == '=':
            A_eq.append(A[i])
            b_eq.append(b[i])
        elif constraint_types[i] == '>=':
            # Преобразуем >= в <=: A·x >= b => -A·x <= -b
            A_ub.append([-x for x in A[i]])
            b_ub.append(-b[i])

    print("Преобразованная задача для scipy:")
    print(f"Целевая функция: min {c_scipy}·x")
    if A_ub:
        print("Ограничения-неравенства (<=):")
        for i in range(len(A_ub)):
            print(f"  {A_ub[i]}·x <= {b_ub[i]}")
    if A_eq:
        print("Ограничения-равенства (=):")
        for i in range(len(A_eq)):
            print(f"  {A_eq[i]}·x = {b_eq[i]}")
    print()

    # Решение задачи
    result = linprog(
        c=c_scipy,
        A_ub=A_ub if A_ub else None,
        b_ub=b_ub if b_ub else None,
        A_eq=A_eq if A_eq else None,
        b_eq=b_eq if b_eq else None,
        bounds=(0, None),  # x_i >= 0
        method='highs'
    )

    return result


def main():
    """
    Основная функция
    """
    try:
        # Чтение данных из файла
        print("Чтение данных из файла input.txt...")
        c, A, b, constraint_types = read_problem_from_file("input.txt")

        # Решение с помощью scipy
        result = solve_with_scipy(c, A, b, constraint_types)

        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ РЕШЕНИЯ")
        print("=" * 60)

        if result.success:
            # Меняем знак обратно, так как решали min (-c)·x
            optimal_value = -result.fun
            x_opt = result.x

            print("✓ ОПТИМАЛЬНОЕ РЕШЕНИЕ НАЙДЕНО")
            print(f"Статус решения: {result.message}")
            print(f"Оптимальное значение целевой функции: {optimal_value:.6f}")
            print("\nОптимальные значения переменных:")
            for i, val in enumerate(x_opt):
                print(f"  x{i + 1} = {val:.6f}")

            # Проверка выполнения ограничений
            print("\nПроверка выполнения ограничений:")
            A_array = np.array(A)
            all_constraints_satisfied = True

            for i in range(len(A)):
                lhs = np.dot(A_array[i], x_opt)
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

            # Дополнительная информация
            print(f"\nДополнительная информация:")
            print(f"  Количество итераций: {result.nit}")
            print(f"  Время решения: {result.time if hasattr(result, 'time') else 'N/A'} сек")

        else:
            print("✗ РЕШЕНИЕ НЕ НАЙДЕНО")
            print(f"Причина: {result.message}")
            print(f"Статус: {result.status}")

    except FileNotFoundError:
        print("✗ Ошибка: Файл input.txt не найден")
        print("Убедитесь, что файл находится в той же папке, что и скрипт")
    except Exception as e:
        print(f"✗ Ошибка при выполнении программы: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()