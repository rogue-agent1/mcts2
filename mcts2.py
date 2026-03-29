#!/usr/bin/env python3
"""mcts2 - Monte Carlo Tree Search for game playing."""
import sys, math, random

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = list(state.get_moves())

    def ucb1(self, c=1.414):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def select(self):
        return max(self.children, key=lambda c: c.ucb1())

    def expand(self):
        move = self.untried_moves.pop()
        new_state = self.state.apply_move(move)
        child = MCTSNode(new_state, self, move)
        self.children.append(child)
        return child

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(1 - result)

class SimpleGame:
    def __init__(self, values=None, turn=0):
        self.values = values or [random.random() for _ in range(4)]
        self.turn = turn
        self.chosen = []

    def get_moves(self):
        return list(range(len(self.values))) if len(self.chosen) < 2 else []

    def apply_move(self, move):
        g = SimpleGame(self.values, 1 - self.turn)
        g.chosen = self.chosen + [move]
        return g

    def is_terminal(self):
        return len(self.chosen) >= 2

    def result(self, player):
        if not self.is_terminal():
            return 0.5
        s = sum(self.values[c] for c in self.chosen)
        return 1 if s > 1 else 0

    def rollout(self):
        state = self
        while not state.is_terminal():
            moves = state.get_moves()
            state = state.apply_move(random.choice(moves))
        return state.result(0)

def mcts_search(root_state, iterations=100):
    root = MCTSNode(root_state)
    for _ in range(iterations):
        node = root
        while not node.untried_moves and node.children:
            node = node.select()
        if node.untried_moves:
            node = node.expand()
        result = node.state.rollout()
        node.backpropagate(result)
    if not root.children:
        return None
    return max(root.children, key=lambda c: c.visits).move

def test():
    random.seed(42)
    game = SimpleGame([0.9, 0.1, 0.8, 0.2])
    move = mcts_search(game, iterations=200)
    assert move is not None
    assert 0 <= move <= 3
    root = MCTSNode(game)
    assert root.visits == 0
    assert len(root.untried_moves) == 4
    child = root.expand()
    assert len(root.children) == 1
    assert len(root.untried_moves) == 3
    child.backpropagate(1)
    assert child.visits == 1
    assert root.visits == 1
    game2 = SimpleGame([0.0, 0.0, 0.0, 0.0])
    move2 = mcts_search(game2, iterations=50)
    assert move2 is not None
    print("All tests passed!")

if __name__ == "__main__":
    test() if "--test" in sys.argv else print("mcts2: Monte Carlo Tree Search. Use --test")
