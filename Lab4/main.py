import sys
from model import State, Decision, DataLoader, DPSolver


def main():
    print("ЗАДАЧА УПРАВЛЕНИЯ ИНВЕСТИЦИОННЫМ ПОРТФЕЛЕМ")
    print("Метод динамического программирования (Беллмана)")

    print("\n>> Загрузка данных...")
    default_path = "investment_data.xlsx"

    print(f"\nВведите путь к Excel-файлу")
    print(f"(Enter для '{default_path}'):")

    try:
        user_input = input().strip()
        filepath = user_input if user_input else default_path
    except (EOFError, KeyboardInterrupt):
        filepath = default_path
        print(f"Использую: {filepath}")

    data_loader = DataLoader(filepath)

    print("\n>> Начальное состояние портфеля:")
    initial_state = State(x1=100, x2=800, xD=400, cash=600)

    print(f"   ЦБ1 = {initial_state.x1}")
    print(f"   ЦБ2 = {initial_state.x2}")
    print(f"   Депозит = {initial_state.xD}")
    print(f"   Кэш = {initial_state.cash}")
    print(f"   Капитал = {initial_state.total_wealth()}")

    print("\n>> Параметры управления:")
    print(f"   Пакет ЦБ1 = {Decision.PACKAGE_SIZE_X1}")
    print(f"   Пакет ЦБ2 = {Decision.PACKAGE_SIZE_X2}")
    print(f"   Пакет Депозита = {Decision.PACKAGE_SIZE_XD}")
    print(f"   max_k_range = 1 (только ±1 пакет за шаг)")

    print("\n>> Создание решателя...")
    solver = DPSolver(
        data_loader=data_loader,
        initial_state=initial_state,
        num_stages=3,
        max_k_range=1,
        verbose=True  # False без промежуточных логов
    )

    print("\n>> Запуск алгоритма ДП...")
    solver.solve_backward()

    solver.print_solution()

    print("\n[OK] Задача решена!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Прервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERROR] Ошибка:")
        print(f"   {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)