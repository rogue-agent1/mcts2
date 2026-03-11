#!/usr/bin/env python3
"""Monte Carlo Tree Search with UCB1 selection."""
import math, random, sys, time

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state, self.parent, self.move = state, parent, move
        self.children = []; self.visits = 0; self.value = 0.0
        self.untried_moves = None
    def ucb1(self, c=1.414):
        if self.visits == 0: return float('inf')
        return self.value/self.visits + c * math.sqrt(math.log(self.parent.visits)/self.visits)

class MCTS:
    def __init__(self, game): self.game = game
    def search(self, state, iterations=1000, time_limit=None):
        root = MCTSNode(state); root.untried_moves = list(self.game.get_moves(state))
        start = time.monotonic()
        for _ in range(iterations):
            if time_limit and time.monotonic() - start > time_limit: break
            node = self._select(root)
            node = self._expand(node)
            result = self._simulate(node.state)
            self._backprop(node, result)
        best = max(root.children, key=lambda c: c.visits) if root.children else None
        return best.move if best else None, root.visits
    def _select(self, node):
        while node.untried_moves is not None and not node.untried_moves and node.children:
            node = max(node.children, key=lambda c: c.ucb1())
        return node
    def _expand(self, node):
        if node.untried_moves:
            move = random.choice(node.untried_moves); node.untried_moves.remove(move)
            new_state = self.game.apply(node.state, move)
            child = MCTSNode(new_state, node, move)
            child.untried_moves = list(self.game.get_moves(new_state))
            node.children.append(child); return child
        return node
    def _simulate(self, state):
        s = state
        while True:
            moves = self.game.get_moves(s)
            if not moves: return self.game.evaluate(s)
            s = self.game.apply(s, random.choice(moves))
    def _backprop(self, node, result):
        while node:
            node.visits += 1; node.value += result; node = node.parent

# Demo: simple Nim
class NimGame:
    def get_moves(self, n): return [i for i in [1,2,3] if i <= n]
    def apply(self, n, m): return n - m
    def evaluate(self, n): return 1 if n == 0 else 0  # whoever took last wins

if __name__ == "__main__":
    mcts = MCTS(NimGame())
    move, sims = mcts.search(15, iterations=5000)
    print(f"Nim(15): MCTS suggests take {move} ({sims} simulations)")
