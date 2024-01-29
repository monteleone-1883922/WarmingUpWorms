from random import randint
from torch_geometric.data import Data
import torch


def build_field(table: list[list[str]], enable_wormholes: bool = False, max_val: int = 0):
    edge_begin = []
    edge_end = []
    nodes = []
    len_row = len(table[0])
    if enable_wormholes:
        near_wormholes = build_near_wormholes(table)
    for i in range(len(table)):
        for j in range(len_row):
            if table[i][j] == "*":
                if enable_wormholes:
                    pass
                else:
                    node = randint(0,max_val)
            else:
                node = int(table[i][j])
            nodes.append(node)
            if i + 1 < len(table):
                edge_begin.append(i * len_row + j)
                edge_end.append((i+1) * len_row + j)
            if i - 1 >= 0:
                edge_begin.append(i * len_row + j)
                edge_end.append((i - 1) * len_row + j)
            if j + 1 < len_row:
                edge_begin.append(i * len_row + j)
                edge_end.append(i * len_row + j+1)
            if j - 1 >= 0:
                edge_begin.append(i * len_row + j)
                edge_end.append(i * len_row + j-1)
    edge_begin = torch.tensor(edge_begin, dtype=torch.int)
    edge_end = torch.tensor(edge_end, dtype=torch.int)
    nodes = torch.tensor(nodes, dtype=torch.int)
    edge_index = torch.cat([edge_begin.unsqueeze(0), edge_end.unsqueeze(0)], dim=0)
    return Data(x=nodes, edge_index=edge_index)




def build_near_wormholes(table: list[list[str]]):
    near_wormholes = set()
    len_row = len(table[0])
    for i in range(len(table)):
        for j in range(len_row):
            if table[i][j] == "*":
                if i + 1 < len(table):
                    near_wormholes.add((i+1) * len_row + j)
                if i - 1 >= 0:
                    near_wormholes.add((i-1) * len_row + j)
                if j + 1 < len_row:
                    near_wormholes.add(i * len_row + j+1)
                if j - 1 >= 0:
                    near_wormholes.add(i * len_row + j-1)
    return list(near_wormholes)