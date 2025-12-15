import pandas as pd
import numpy as np
from typing import Dict


class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.num_stages = 3
        self.num_scenarios = 3
        self.probabilities: Dict[int, np.ndarray] = {}
        self.multipliers: Dict[int, Dict[str, np.ndarray]] = {}
        self._load_data()

    def _load_data(self):
        try:
            excel_data = pd.read_excel(self.filepath, sheet_name=None)
            print(f"Загружены листы: {list(excel_data.keys())}")

            if len(excel_data) == 0:
                raise ValueError("Excel-файл пуст")

            main_sheet_name = list(excel_data.keys())[0]
            df = excel_data[main_sheet_name]
            print(f"\nСтруктура данных (первый лист '{main_sheet_name}'):")
            print(df.head(20))

            self._parse_data(df)

        except FileNotFoundError:
            print(f"[!] Файл {self.filepath} не найден!")
            self._load_default_data()
        except Exception as e:
            print(f"[!] Ошибка: {e}")
            self._load_default_data()

    def _parse_data(self, df: pd.DataFrame):
        for stage in range(1, self.num_stages + 1):
            stage_data = df[df.iloc[:, 0] == stage]

            if len(stage_data) >= 3:
                probs = stage_data.iloc[:3, 2].values.astype(float)
                self.probabilities[stage] = probs

                m_x1 = stage_data.iloc[:3, 3].values.astype(float)
                m_x2 = stage_data.iloc[:3, 4].values.astype(float)
                m_xD = stage_data.iloc[:3, 5].values.astype(float)

                self.multipliers[stage] = {
                    'x1': m_x1,
                    'x2': m_x2,
                    'xD': m_xD
                }

    def _load_default_data(self):
        self.probabilities = {
            1: np.array([0.3, 0.5, 0.2]),
            2: np.array([0.3, 0.5, 0.2]),
            3: np.array([0.3, 0.5, 0.2])
        }

        self.multipliers = {
            1: {
                'x1': np.array([1.15, 1.00, 0.90]),
                'x2': np.array([1.20, 1.00, 0.85]),
                'xD': np.array([1.05, 1.03, 1.02])
            },
            2: {
                'x1': np.array([1.12, 1.00, 0.88]),
                'x2': np.array([1.18, 1.00, 0.82]),
                'xD': np.array([1.05, 1.03, 1.02])
            },
            3: {
                'x1': np.array([1.10, 1.00, 0.92]),
                'x2': np.array([1.15, 1.00, 0.87]),
                'xD': np.array([1.05, 1.03, 1.02])
            }
        }

        print("\n[OK] Загружены тестовые данные:")
        self.print_data()

    def get_probability(self, stage: int, scenario: int) -> float:
        return self.probabilities[stage][scenario]

    def get_multiplier(self, stage: int, instrument: str, scenario: int) -> float:
        return self.multipliers[stage][instrument][scenario]

    def print_data(self):
        scenario_names = ['Благоприятный', 'Нейтральный', 'Негативный']

        for stage in range(1, self.num_stages + 1):
            print(f"\n--- Этап {stage} ---")
            print(f"{'Сценарий':<15} {'Вероятность':<12} {'m_x1':<8} {'m_x2':<8} {'m_xD':<8}")
            print("-" * 60)

            for scen in range(self.num_scenarios):
                prob = self.probabilities[stage][scen]
                m_x1 = self.multipliers[stage]['x1'][scen]
                m_x2 = self.multipliers[stage]['x2'][scen]
                m_xD = self.multipliers[stage]['xD'][scen]

                print(f"{scenario_names[scen]:<15} {prob:<12.2f} {m_x1:<8.3f} {m_x2:<8.3f} {m_xD:<8.3f}")