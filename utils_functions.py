from random import randint


class Cell:
    def __init__(self, val: int, wormhole: bool):
        self.val = val
        self.wormhole = wormhole


def apply_wormholes_policy(row: list[str], enable_wormholes: bool = False, max_val: int = 0):
    out_row = []
    for el in row:
        if el == "*":
            if enable_wormholes:
                cell = Cell(0, True)
            elif not enable_wormholes and max_val > 0:
                cell = Cell(randint(0, max_val), False)
            else:
                cell = Cell(0, False)
        else:
            cell = Cell(int(el), False)
        out_row.append(cell)
    return out_row
