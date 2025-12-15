from dataclasses import dataclass
from typing import Tuple
from .state import State


@dataclass(frozen=True)
class Decision:
    k1: int  # пакеты ЦБ1
    k2: int  # пакеты ЦБ2
    kD: int  # пакеты депозита

    PACKAGE_SIZE_X1 = 25
    PACKAGE_SIZE_X2 = 200
    PACKAGE_SIZE_XD = 100

    def __repr__(self) -> str:
        return f"Decision(k1={self.k1}, k2={self.k2}, kD={self.kD})"

    def apply_to_state(self, state: State) -> State:
        new_x1 = state.x1 + self.PACKAGE_SIZE_X1 * self.k1
        new_x2 = state.x2 + self.PACKAGE_SIZE_X2 * self.k2
        new_xD = state.xD + self.PACKAGE_SIZE_XD * self.kD

        cost = (self.PACKAGE_SIZE_X1 * self.k1 +
                self.PACKAGE_SIZE_X2 * self.k2 +
                self.PACKAGE_SIZE_XD * self.kD)

        new_cash = state.cash - cost
        return State(x1=new_x1, x2=new_x2, xD=new_xD, cash=new_cash)

    def is_feasible(self, state: State) -> bool:
        new_state = self.apply_to_state(state)
        return new_state.is_valid()

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.k1, self.k2, self.kD)