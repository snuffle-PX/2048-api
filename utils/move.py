import numpy as np


def try_to_move(grid, direction):
    board_to_left = np.rot90(np.copy(grid), -direction)
    total_score = 0
    for row in range(grid.__len__()):
        core, score = merge(board_to_left[row])
        board_to_left[row, :len(core)] = core
        board_to_left[row, len(core):] = 0
        total_score += score

    # rotation to the original
    new_grid = np.rot90(board_to_left, direction)
    if np.equal(grid, new_grid).all():
        total_score = -1

    return new_grid, total_score


def merge(row):
    """merge the row, there may be some improvement"""
    non_zero = row[row != 0]  # remove zeros
    core = [None]
    score = 0
    for elem in non_zero:
        if core[-1] is None:
            core[-1] = elem
        elif core[-1] == elem:
            core[-1] = 2 * elem
            core.append(None)
            score += elem
        else:
            core.append(elem)
    if core[-1] is None:
        core.pop()
    return core, score
