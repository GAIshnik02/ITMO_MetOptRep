import numpy as np
from typing import Dict, List, Tuple
from .state import State
from .decision import Decision
from .data_loader import DataLoader


class DPSolver:
    def __init__(self, data_loader: DataLoader, initial_state: State,
                 num_stages: int = 3, max_k_range: int = 10, verbose: bool = True):
        self.data = data_loader
        self.initial_state = initial_state
        self.num_stages = num_stages
        self.max_k_range = max_k_range
        self.verbose = verbose

        self.V: Dict[int, Dict[State, float]] = {}
        self.optimal_decisions: Dict[int, Dict[State, Decision]] = {}
        self.reachable_states: Dict[int, set] = {}

        if self.verbose:
            print(f"\nИНИЦИАЛИЗАЦИЯ РЕШАТЕЛЯ ДП")
            print(f"Начальное состояние: {initial_state}")
            print(
                f"Этапов: {num_stages}, Размеры пакетов: ЦБ1={Decision.PACKAGE_SIZE_X1}, ЦБ2={Decision.PACKAGE_SIZE_X2}, Деп={Decision.PACKAGE_SIZE_XD}")

    def generate_feasible_decisions(self, state: State) -> List[Decision]:
        feasible = []

        max_sell_k1 = int(state.x1 / Decision.PACKAGE_SIZE_X1)
        max_sell_k2 = int(state.x2 / Decision.PACKAGE_SIZE_X2)
        max_sell_kD = int(state.xD / Decision.PACKAGE_SIZE_XD)

        max_buy_k1 = int(state.cash / Decision.PACKAGE_SIZE_X1)
        max_buy_k2 = int(state.cash / Decision.PACKAGE_SIZE_X2)
        max_buy_kD = int(state.cash / Decision.PACKAGE_SIZE_XD)

        k1_min = max(-max_sell_k1, -self.max_k_range)
        k1_max = min(max_buy_k1, self.max_k_range)

        k2_min = max(-max_sell_k2, -self.max_k_range)
        k2_max = min(max_buy_k2, self.max_k_range)

        kD_min = max(-max_sell_kD, -self.max_k_range)
        kD_max = min(max_buy_kD, self.max_k_range)

        for k1 in range(k1_min, k1_max + 1):
            for k2 in range(k2_min, k2_max + 1):
                for kD in range(kD_min, kD_max + 1):
                    decision = Decision(k1=k1, k2=k2, kD=kD)
                    if decision.is_feasible(state):
                        feasible.append(decision)

        return feasible

    def apply_scenario(self, state: State, stage: int, scenario: int) -> State:
        m1 = self.data.get_multiplier(stage, 'x1', scenario)
        m2 = self.data.get_multiplier(stage, 'x2', scenario)
        mD = self.data.get_multiplier(stage, 'xD', scenario)

        return State(
            x1=int(state.x1 * m1),
            x2=int(state.x2 * m2),
            xD=int(state.xD * mD),
            cash=state.cash
        )

    def solve_backward(self):
        if self.verbose:
            print(f"\nОБРАТНЫЙ ПРОХОД (BACKWARD INDUCTION)")

        stage = self.num_stages + 1
        self.V[stage] = {}

        self._collect_reachable_states()

        if self.verbose:
            print(f"\n--- Терминальный этап (t={stage}) ---")
            print(f"V_{stage}(x) = x1 + x2 + xD + cash")

        for state in self.reachable_states.get(stage, set()):
            self.V[stage][state] = state.total_wealth()

        if self.verbose:
            print(f"Вычислено {len(self.V[stage])} терминальных значений")

        for stage in range(self.num_stages, 0, -1):
            if self.verbose:
                print(f"\n--- Этап t={stage} ---")
            self.V[stage] = {}
            self.optimal_decisions[stage] = {}

            states_to_process = self.reachable_states.get(stage, set())
            if self.verbose:
                print(f"Обрабатывается состояний: {len(states_to_process)}")

            processed = 0
            for state in states_to_process:
                feasible_decisions = self.generate_feasible_decisions(state)

                if len(feasible_decisions) == 0:
                    continue

                best_value = -np.inf
                best_decision = None

                for decision in feasible_decisions:
                    state_after_decision = decision.apply_to_state(state)
                    expected_value = 0.0

                    for scenario in range(3):
                        next_state = self.apply_scenario(state_after_decision, stage, scenario)
                        if not next_state.is_valid():
                            continue

                        prob = self.data.get_probability(stage, scenario)
                        next_value = self.V[stage + 1].get(next_state, 0.0)

                        expected_value += prob * next_value

                    if expected_value > best_value:
                        best_value = expected_value
                        best_decision = decision

                self.V[stage][state] = best_value
                self.optimal_decisions[stage][state] = best_decision

                processed += 1
                if self.verbose and (processed % 100 == 0 or processed == len(states_to_process)):
                    print(f"  Обработано: {processed}/{len(states_to_process)}")

            if self.verbose:
                print(f"[OK] Этап {stage} завершён. Найдено решений: {len(self.optimal_decisions[stage])}")

        if self.verbose:
            print(f"\nОБРАТНЫЙ ПРОХОД ЗАВЕРШЁН")

    def _collect_reachable_states(self):
        if self.verbose:
            print(f"\n--- Сбор достижимых состояний ---")

        self.reachable_states[1] = {self.initial_state}

        for stage in range(1, self.num_stages + 1):
            self.reachable_states[stage + 1] = set()

            states_count = len(self.reachable_states[stage])
            processed = 0

            for state in self.reachable_states[stage]:
                feasible_decisions = self.generate_feasible_decisions(state)

                for decision in feasible_decisions:
                    state_after_decision = decision.apply_to_state(state)

                    for scenario in range(3):
                        next_state = self.apply_scenario(state_after_decision, stage, scenario)
                        if next_state.is_valid():
                            self.reachable_states[stage + 1].add(next_state)

                processed += 1
                if self.verbose and (processed % 50 == 0 or processed == states_count):
                    print(f"  Этап {stage}: обработано {processed}/{states_count} состояний...")

            if self.verbose:
                print(f"  Этап {stage}: {len(self.reachable_states[stage])} состояний -> "
                      f"Этап {stage + 1}: {len(self.reachable_states[stage + 1])} состояний")

    def _round_state(self, state: State, decimals: int = 2) -> State:
        return State(
            x1=int(state.x1),
            x2=int(state.x2),
            xD=int(state.xD),
            cash=int(state.cash)
        )

    def solve_forward(self) -> Tuple[List[State], List[Decision]]:
        if self.verbose:
            print(f"\nПРЯМОЙ ПРОХОД (ВОССТАНОВЛЕНИЕ СТРАТЕГИИ)")

        trajectory_states = [self.initial_state]
        trajectory_decisions = []

        current_state = self.initial_state

        for stage in range(1, self.num_stages + 1):
            if self.verbose:
                print(f"\n--- Этап t={stage} ---")
                print(f"Текущее состояние: {current_state}")
                print(f"Капитал: {current_state.total_wealth()}")

            optimal_decision = self.optimal_decisions[stage].get(current_state)

            if optimal_decision is None:
                if self.verbose:
                    print(f"[!] Решение не найдено для {current_state}")
                break

            if self.verbose:
                print(f"Оптимальное управление: {optimal_decision}")
            trajectory_decisions.append(optimal_decision)

            state_after_decision = optimal_decision.apply_to_state(current_state)
            if self.verbose:
                print(f"После управления: {state_after_decision}")

            # Используем наиболее вероятный сценарий вместо математического ожидания
            probs = [self.data.get_probability(stage, s) for s in range(3)]
            most_probable_scenario = probs.index(max(probs))

            if self.verbose:
                print(f"Переходы по сценариям:")
                scenario_names = ['Благоприятный', 'Нейтральный', 'Негативный']
                for scenario in range(3):
                    prob = self.data.get_probability(stage, scenario)
                    next_state_scenario = self.apply_scenario(state_after_decision, stage, scenario)
                    marker = " <-- наиболее вероятный" if scenario == most_probable_scenario else ""
                    print(f"  {scenario_names[scenario]} (p={prob:.2f}): {next_state_scenario}{marker}")

            # Берём состояние по наиболее вероятному сценарию
            next_state = self.apply_scenario(state_after_decision, stage, most_probable_scenario)

            if not next_state.is_valid():
                if self.verbose:
                    print(f"[!] Состояние после сценария невалидно: {next_state}")
                break

            if self.verbose:
                print(f"Следующее состояние (по наиболее вероятному сценарию): {next_state}")
                print(f"V_{stage}(s) = {self.V[stage][current_state]:.2f}")

            trajectory_states.append(next_state)
            current_state = next_state

        if self.verbose:
            print(f"\nФИНАЛЬНОЕ СОСТОЯНИЕ (t={self.num_stages + 1})")
            print(f"{current_state}")
            print(f"Капитал: {current_state.total_wealth()}")

        return trajectory_states, trajectory_decisions

    def get_optimal_expected_return(self) -> float:
        return self.V[1].get(self.initial_state, 0.0)

    def print_solution(self):
        print(f"\nИТОГОВОЕ РЕШЕНИЕ")

        print(f"\n>> Начальный портфель:")
        print(f"   ЦБ1 = {self.initial_state.x1}")
        print(f"   ЦБ2 = {self.initial_state.x2}")
        print(f"   Депозит = {self.initial_state.xD}")
        print(f"   Кэш = {self.initial_state.cash}")
        print(f"   Капитал = {self.initial_state.total_wealth()}")

        print(f"\n>> Оптимальные управления:")
        trajectory_states, trajectory_decisions = self.solve_forward()

        print(f"\n>> Траектория:")
        print(f"\n{'Этап':<8} {'ЦБ1':<10} {'ЦБ2':<10} {'Депозит':<10} {'Кэш':<10} {'Капитал':<12} {'Управление':<25}")
        print("-" * 100)

        for i, state in enumerate(trajectory_states):
            stage_label = f"t={i}" if i == 0 else f"-> t={i}"
            decision_str = ""
            if i < len(trajectory_decisions):
                dec = trajectory_decisions[i]
                decision_str = f"(k1={dec.k1}, k2={dec.k2}, kD={dec.kD})"

            print(f"{stage_label:<8} {state.x1:<10} {state.x2:<10} {state.xD:<10} "
                  f"{state.cash:<10} {state.total_wealth():<12} {decision_str:<25}")

        expected_return = self.get_optimal_expected_return()
        print(f"\n>> Максимальное мат. ожидание капитала:")
        print(f"   V_1(s_0) = {expected_return:.2f}")

        initial_capital = self.initial_state.total_wealth()
        expected_profit = expected_return - initial_capital
        expected_profit_pct = (expected_profit / initial_capital) * 100

        print(f"\n>> Ожидаемый прирост:")
        print(f"   Абсолютный: {expected_profit:.2f}")
        print(f"   Относительный: {expected_profit_pct:.2f}%")