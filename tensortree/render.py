from dataclasses import dataclass
from typing import Any, Callable, Generator, Optional, Union

from torch import Tensor

import tensortree


@dataclass
class Row:
    pre: str
    fill: str
    node: str


@dataclass
class Style:
    vertical: str
    cont: str
    end: str


@dataclass
class AsciiStyle(Style):
    vertical: str = '|   '
    cont: str = '|-- '
    end: str = '+-- '


@dataclass
class ContStyle(Style):
    vertical: str = '\u2502   '
    cont: str = '\u251c\u2500\u2500 '
    end: str = '\u2514\u2500\u2500 '


@dataclass
class ContRoundStyle(Style):
    vertical: str = '\u2502   '
    cont: str = '\u251c\u2500\u2500 '
    end: str = '\u2570\u2500\u2500 '


@dataclass
class DoubleStyle(Style):
    vertical: str = '\u2551   '
    cont: str = '\u2560\u2550\u2550 '
    end: str = '\u255a\u2550\u2550 '


def format_tree(
        tree, max_nodes: Optional[int] = None, node_renderer: Callable[[Any], str] = str,
        style: Union[Style] = ContRoundStyle,
) -> str:
    """
    Pretty prints a tree up to `max_nodes`. Define a node_renderer for custom node types (e.g. Dictionaries).
    :param max_nodes: Render up to this amount of nodes.
    :param node_renderer: A function that outputs a string.
    :param style: Style the tree.
    :return:
    """
    def format_nodes() -> Generator[Row, None, None]:
        incidences = tree.node_incidence_matrix()
        levels = tensortree.levels(incidences).tolist()
        active_levels = set()

        def make_row(token, node_idx, level, active_levels, fill, replace_token: bool = False):
            pre = "".join(style.vertical if l in active_levels else "    " for l in range(level - 1))

            if level == 0:
                fill = ""

            if isinstance(token, Tensor):
                token = token.item()

            if replace_token:
                token = "[...]"
            else:
                token = node_renderer(token)

            return Row(pre, fill, f"{node_idx}. {token}")

        for node_idx, (token, lev) in enumerate(zip(tree.node_data, levels), start=0):
            fill = style.end

            if tree.has_sibling(node_idx, check_left=False, check_right=True):
                fill = style.cont
                if not tree.is_leaf(node_idx):
                    active_levels.add((lev - 1))
            elif (lev - 1) in active_levels:
                active_levels.remove(lev - 1)

            row = make_row(token, node_idx, lev, active_levels, fill)

            yield row

            # code for stopping at max_nodes and finish open levels below
            # maybe we break before printing this row, to print a shorter version of this tree
            if max_nodes is not None and node_idx >= (max_nodes - len(active_levels) - 2):

                # this was the last node of this branch
                if fill == style.end:
                    next_node = tree.next_node_not_in_branch(node_idx)
                elif fill == style.cont:
                    next_node = tree.right_sibling(node_idx)
                active_levels.add(None)

                while active_levels and next_node:
                    yield make_row(None, next_node, levels[next_node], active_levels, style.end, replace_token=True)
                    next_node = tree.step_out(next_node)
                    some_lev = active_levels.pop()

                break

    return "\n".join(
        f"{row.pre}{row.fill}{row.node}" for row in format_nodes()
    )
