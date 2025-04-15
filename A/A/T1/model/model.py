import numpy as np
import random
import heapq
from collections import defaultdict, deque
import multiprocessing as mp


def simulate_random_game_static(board, player):
    board_copy = board.copy()
    size = board_copy.shape[0]
    moves = [(i, j) for i in range(size) for j in range(size) if board_copy[i, j] == 0]
    random.shuffle(moves)
    cur_player = player
    for move in moves:
        board_copy[move] = cur_player
        cur_player = -cur_player
    temp = HexAI(size)
    temp.board = board_copy
    if temp.is_winner(1):
        return 1
    elif temp.is_winner(-1):
        return -1
    else:
        return 0


def simulate_for_move(args):
    move, board, player, simulations = args
    win_count = 0
    for _ in range(simulations):
        board_copy = board.copy()
        board_copy[move] = player
        result = simulate_random_game_static(board_copy, -player)
        if result == player:
            win_count += 1
    return move, win_count


class HexAI:
    def __init__(self, size=11):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.directions = [(1, 0), (0, 1), (1, -1), (-1, 1), (-1, 0), (0, -1)]

    def evaluate(self, player):
        score = 0
        connectivity_score = self.evaluate_connectivity(player)
        path_score = self.enhanced_shortest_path(player)
        opponent = -player
        opponent_threat = self.opponent_threat_analysis(opponent)
        bridge_score = self.evaluate_bridging_patterns(player)
        score = (
                connectivity_score * 15 +
                path_score * 100 +
                bridge_score * 20 -
                opponent_threat * 80
        )
        return score

    def evaluate_connectivity(self, player):
        size = self.size
        score = 0
        visited = np.zeros((size, size), dtype=bool)
        connected_regions = []

        for i in range(size):
            for j in range(size):
                if self.board[i, j] == player and not visited[i, j]:
                    region = []
                    queue = deque([(i, j)])
                    visited[i, j] = True
                    while queue:
                        x, y = queue.popleft()
                        region.append((x, y))
                        for dx, dy in self.directions:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < size and 0 <= ny < size and self.board[nx, ny] == player and not visited[
                                nx, ny]:
                                queue.append((nx, ny))
                                visited[nx, ny] = True
                    connected_regions.append(region)

        for region in connected_regions:
            region_size = len(region)
            edge_connections = 0
            if player == 1:
                top_connected = any(x == 0 for x, y in region)
                bottom_connected = any(x == size - 1 for x, y in region)
                edge_connections = 10 * (top_connected + bottom_connected)
            else:
                left_connected = any(y == 0 for x, y in region)
                right_connected = any(y == size - 1 for x, y in region)
                edge_connections = 10 * (left_connected + right_connected)

            edge_bonus = 0
            if player == 1:
                for x, y in region:
                    if x == 0:
                        edge_bonus += 2
                    if x == size - 1:
                        edge_bonus += 2
            else:
                for x, y in region:
                    if y == 0:
                        edge_bonus += 2
                    if y == size - 1:
                        edge_bonus += 2

            empty_neighbors = 0
            for x, y in region:
                for dx, dy in self.directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size and self.board[nx, ny] == 0:
                        empty_neighbors += 1

            region_score = region_size * region_size + edge_connections + edge_bonus + empty_neighbors * 0.5
            score += region_score

            if (player == 1 and top_connected and bottom_connected) or \
                    (player == -1 and left_connected and right_connected):
                score += 1000
        return score

    def enhanced_shortest_path(self, player):
        opponent = -player
        if player == 1:
            start_edge = [(0, j) for j in range(self.size)]
            end_edge = [(self.size - 1, j) for j in range(self.size)]
        else:
            start_edge = [(i, 0) for i in range(self.size)]
            end_edge = [(i, self.size - 1) for i in range(self.size)]

        dist = np.full((self.size, self.size), float('inf'))
        pq = []

        for x, y in start_edge:
            cost = 0
            if self.board[x, y] == 0:
                cost = 1
            elif self.board[x, y] == opponent:
                cost = 5
            elif self.board[x, y] == player:
                cost = 0

            if cost < float('inf'):
                heapq.heappush(pq, (cost, x, y))
                dist[x, y] = cost

        while pq:
            d, x, y = heapq.heappop(pq)
            if (player == 1 and x == self.size - 1) or (player == -1 and y == self.size - 1):
                return 100 / (1 + d)
            if d > dist[x, y]:
                continue
            for dx, dy in self.directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    new_cost = d
                    if self.board[nx, ny] == 0:
                        new_cost += 1
                    elif self.board[nx, ny] == opponent:
                        new_cost += 5
                    elif self.board[nx, ny] == player:
                        new_cost += 0
                    if new_cost < dist[nx, ny]:
                        dist[nx, ny] = new_cost
                        heapq.heappush(pq, (new_cost, nx, ny))
        return 0

    def opponent_threat_analysis(self, opponent):
        opponent_path_score = self.enhanced_shortest_path(opponent)
        if opponent_path_score > 50:
            return opponent_path_score * 1.5
        elif opponent_path_score > 25:
            return opponent_path_score * 1.2
        else:
            return opponent_path_score * 0.8

    def evaluate_bridging_patterns(self, player):
        score = 0
        size = self.size
        for i in range(size):
            for j in range(size):
                if self.board[i, j] == player:
                    score += self._check_bridges(i, j, player)
        return score

    def _check_bridges(self, x, y, player):
        bridge_score = 0
        size = self.size
        bridge_directions = [
            [(1, 0), (1, -1)],
            [(1, 0), (0, 1)],
            [(0, 1), (-1, 1)],
            [(0, 1), (1, -1)],
            [(-1, 0), (-1, 1)],
            [(-1, 0), (0, -1)]
        ]

        for dir1, dir2 in bridge_directions:
            dx1, dy1 = dir1
            dx2, dy2 = dir2
            x1, y1 = x + dx1, y + dy1
            if not (0 <= x1 < size and 0 <= y1 < size):
                continue
            x2, y2 = x + dx2, y + dy2
            if not (0 <= x2 < size and 0 <= y2 < size):
                continue
            x3, y3 = x + dx1 + dx2, y + dy1 + dy2
            if not (0 <= x3 < size and 0 <= y3 < size):
                continue

            if (self.board[x3, y3] == player and
                    self.board[x1, y1] == 0 and
                    self.board[x2, y2] == 0):
                bridge_score += 3
            elif (self.board[x3, y3] == 0 and
                  (self.board[x1, y1] == 0 or self.board[x2, y2] == 0)):
                if (self.board[x1, y1] == player or self.board[x2, y2] == player):
                    bridge_score += 2
                else:
                    bridge_score += 1

            x4, y4 = x + 2 * dx1, y + 2 * dy1
            x5, y5 = x + 2 * dx2, y + 2 * dy2
            if (0 <= x4 < size and 0 <= y4 < size and
                    0 <= x5 < size and 0 <= y5 < size):
                if (self.board[x1, y1] == 0 and self.board[x4, y4] == player and
                        self.board[x2, y2] == 0 and self.board[x5, y5] == player):
                    bridge_score += 4
        return bridge_score

    def bridge_potential(self, x, y, player):
        directions = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
        bridge_score = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            bx, by = x - dx, y - dy
            if 0 <= nx < self.size and 0 <= ny < self.size and 0 <= bx < self.size and 0 <= by < self.size:
                if self.board[nx, ny] == player and self.board[bx, by] == 0:
                    bridge_score += 1
        return bridge_score

    def virtual_connection(self, x, y, player):
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        count = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == player:
                count += 1
        return count

    def blocking_value(self, x, y, player):
        opponent = -player
        block_score = 0
        block_score += self.bridge_potential(x, y, opponent) * 0.5
        block_score += self.virtual_connection(x, y, opponent) * 0.5
        return block_score

    def shortest_path(self, player):
        pq = []
        dist = np.full((self.size, self.size), float('inf'))
        for i in range(self.size):
            if (player == 1 and self.board[0, i] == 1) or (player == -1 and self.board[i, 0] == -1):
                start_x, start_y = (0, i) if player == 1 else (i, 0)
                heapq.heappush(pq, (0, start_x, start_y))
                dist[start_x, start_y] = 0
        directions = [(1, 0), (0, 1), (1, -1), (-1, 1), (-1, 0), (0, -1)]
        while pq:
            d, x, y = heapq.heappop(pq)
            if (player == 1 and x == self.size - 1) or (player == -1 and y == self.size - 1):
                return d
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    new_cost = d + (0 if self.board[nx, ny] == player else 1)
                    if new_cost < dist[nx, ny]:
                        dist[nx, ny] = new_cost
                        heapq.heappush(pq, (new_cost, nx, ny))
        return float('inf')

    def minimax(self, depth, player, alpha, beta):
        if depth == 0 or self.is_winner(1) or self.is_winner(-1):
            return self.evaluate(player), None

        best_move = None
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i, j] == 0]
        empty_cells.sort(key=lambda move: self.evaluate_move(move, player), reverse=True)

        if player == 1:
            max_eval = -float('inf')
            for i, j in empty_cells:
                self.board[i, j] = player
                eval, _ = self.minimax(depth - 1, -player, alpha, beta)
                self.board[i, j] = 0
                if eval > max_eval:
                    max_eval = eval
                    best_move = (i, j)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for i, j in empty_cells:
                self.board[i, j] = player
                eval, _ = self.minimax(depth - 1, -player, alpha, beta)
                self.board[i, j] = 0
                if eval < min_eval:
                    min_eval = eval
                    best_move = (i, j)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def evaluate_move(self, move, player):
        i, j = move
        self.board[i, j] = player
        score = self.evaluate(player)
        self.board[i, j] = 0
        return score

    def mcts(self, player, simulations=400000, num_processes=mp.cpu_count()):
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i, j] == 0]
        if not empty_cells:
            return None
        total_moves = len(empty_cells)
        simulations_per_move = simulations // total_moves
        args_list = [(move, self.board.copy(), player, simulations_per_move) for move in empty_cells]

        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(simulate_for_move, args_list)

        move_scores = {move: wins / simulations_per_move for move, wins in results}
        return max(move_scores, key=move_scores.get)

    def get_best_move(self, player, depth=17, use_mcts=True):
        if use_mcts:
            return self.mcts(player)
        else:
            _, best_move = self.minimax(depth, player, -float('inf'), float('inf'))
            return best_move

    def is_winner(self, player):
        visited = set()

        def dfs(x, y):
            if (x, y) in visited:
                return False
            if player == 1 and x == self.size - 1:
                return True
            if player == -1 and y == self.size - 1:
                return True
            visited.add((x, y))
            directions = [(1, 0), (0, 1), (1, -1), (-1, 1), (-1, 0), (0, -1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == player:
                    if dfs(nx, ny):
                        return True
            return False

        for i in range(self.size):
            if player == 1 and self.board[0, i] == 1 and dfs(0, i):
                return True
            if player == -1 and self.board[i, 0] == -1 and dfs(i, 0):
                return True
        return False


def swap_move(move):
    """Swap a move by reflecting it across the diagonal of the board"""
    row, col = move
    # Swap row and column (reflect across the main diagonal)
    return (col, row)


def format_move(move):
    """Convert a move tuple (row, col) to a standard format like 'a1'"""
    row, col = move
    return f"{chr(col + ord('a'))}{row + 1}"


def parse_move(move_str, size):
    move_str = move_str.strip().lower()
    if len(move_str) < 2:
        return None
    col_char = move_str[0]
    row_str = move_str[1:]
    if not col_char.isalpha() or not row_str.isdigit():
        return None
    col = ord(col_char) - ord('a')
    row = int(row_str) - 1
    if 0 <= row < size and 0 <= col < size:
        return (row, col)
    return None


def main():
    size = 11
    game = HexAI(size)
    human_player = 1  # Default: human connects top to bottom (player 1)
    ai_player = -1  # Default: AI connects left to right (player -1)
    use_mcts = True
    first_move_made = False
    ai_first_move = None

    # Determine who goes first
    first_input = input()

    if first_input.lower() == "first":
        # AI goes first (plays as player 1, connects top to bottom)
        human_player = -1  # Human connects left to right
        ai_player = 1  # AI connects top to bottom

        # Make AI's first move
        ai_first_move = game.get_best_move(ai_player, depth=17, use_mcts=use_mcts)
        game.board[ai_first_move] = ai_player
        print(format_move(ai_first_move))
        first_move_made = True
        current_player = human_player  # Now it's human's turn

    elif first_input.lower() == "finish":
        print("游戏结束")
        return

    else:
        # Human goes first with a specific move
        first_move = parse_move(first_input, size)
        if first_move is None:
            print("无效输入，请重新输入。")
            return
        if game.board[first_move] != 0:
            print("该位置已被占用，请重新选择。")
            return

        # Place human's first move on the board
        game.board[first_move] = human_player
        first_move_made = True

        # AI (as second player) decides whether to use swap rule
        # Evaluate the position with and without swap

        # First, assess the position if AI keeps current roles
        normal_eval = game.evaluate(ai_player)

        # Then, assess what if AI takes human's position (swaps)
        # Reset the board
        game.board[first_move] = 0
        # Place AI's piece at human's position to evaluate
        game.board[first_move] = ai_player
        swap_eval = game.evaluate(ai_player)

        # Reset again to make the actual decision
        game.board[first_move] = 0

        # Compare evaluations to decide whether to swap
        if swap_eval > normal_eval * 1.2:  # If swapping is significantly better
            # Execute the swap rule
            swapped_move = swap_move(first_move)

            # Check if the swapped position is valid and empty
            if (0 <= swapped_move[0] < size and
                    0 <= swapped_move[1] < size and
                    game.board[swapped_move] == 0):

                # Remove human's piece from original position
                game.board[first_move] = 0

                # Place AI's piece at the swapped position
                game.board[swapped_move] = ai_player

                # Inform human that AI is using the swap rule
                print("change")

                # It's human's turn now
                current_player = human_player
            else:
                # If swapped position is invalid, don't use swap rule
                # Reset and make a normal move
                game.board[first_move] = human_player

                # Make AI's move
                move = game.get_best_move(ai_player, depth=17, use_mcts=use_mcts)
                game.board[move] = ai_player
                print(format_move(move))

                current_player = human_player
        else:
            # Don't use swap rule, just place human's move normally
            game.board[first_move] = human_player

            # Make AI's move
            move = game.get_best_move(ai_player, depth=17, use_mcts=use_mcts)
            game.board[move] = ai_player
            print(format_move(move))

            current_player = human_player

    # Main game loop
    while True:
        # Check for winner
        if game.is_winner(human_player):
            print("human win")
            break
        if game.is_winner(ai_player):
            print("AI win")
            break

        # Handle player's turn
        if current_player == human_player:
            move_input = input()

            # Check if the player wants to end the game
            if move_input.lower() == "finish":
                print("游戏结束")
                break

            # Check for swap rule when human is second player and this is their first move
            if first_move_made and ai_player == 1 and move_input.lower() == "change" and ai_first_move is not None:
                # Human wants to use the swap rule
                # Remove AI's first move
                game.board[ai_first_move] = 0

                # Swap the move
                swapped_move = swap_move(ai_first_move)

                # Place human's piece at the swapped position
                game.board[swapped_move] = human_player

                # Now it's AI's turn again
                current_player = ai_player
                continue

            # Regular move processing
            move = parse_move(move_input, size)
            if move is None:
                print("无效输入，请重新输入。")
                continue
            if game.board[move] != 0:
                print("该位置已被占用，请重新选择。")
                continue

            # Place human's move
            game.board[move] = human_player

            # Switch to AI's turn
            current_player = ai_player

        else:  # AI's turn
            move = game.get_best_move(ai_player, depth=17, use_mcts=use_mcts)
            if move is None:
                print("平局！")
                break

            # Place AI's move
            game.board[move] = ai_player
            print(format_move(move))

            # Switch to human's turn
            current_player = human_player


if __name__ == "__main__":
    main()