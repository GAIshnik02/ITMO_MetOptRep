from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class State:
    x1: int
    x2: int
    xD: int
    cash: int

    MAX_X1 = 600
    MAX_X2 = 2000
    MAX_XD = 1000
    MAX_CASH = 2000

    def __repr__(self) -> str:
        return f"State(x1={self.x1}, x2={self.x2}, xD={self.xD}, cash={self.cash})"

    def total_wealth(self) -> int:
        return self.x1 + self.x2 + self.xD + self.cash

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.x2, self.xD, self.cash)

    def is_valid(self) -> bool:
        return (self.x1 >= 0 and self.x2 >= 0 and self.xD >= 0 and self.cash >= 0 and
                self.x1 <= self.MAX_X1 and self.x2 <= self.MAX_X2 and
                self.xD <= self.MAX_XD and self.cash <= self.MAX_CASH)

    @staticmethod
    def from_tuple(t: Tuple[int, int, int, int]) -> 'State':
        return State(x1=t[0], x2=t[1], xD=t[2], cash=t[3])