import pygame
import math
import random
import time
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA  # Updated import
from qiskit_algorithms.optimizers import COBYLA  # Add a classical optimizer
from qiskit.primitives import Sampler  # Use the updated Sampler
# Node class representing each point in the tree
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, obstacle_list, x_range, y_range, step_size=20, max_iter=500, radius=30):
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.obstacle_list = obstacle_list
        self.x_range = x_range
        self.y_range = y_range
        self.step_size = step_size
        self.max_iter = max_iter
        self.radius = radius
        self.nodes = [self.start]
        pygame.init()
        self.screen = pygame.display.set_mode((x_range[1], y_range[1]))
        pygame.display.set_caption("RRT* Pathfinding Visualization")
        self.clock = pygame.time.Clock()
        self.screen.fill((255, 255, 255))

    def draw_circle(self, x, y, radius, color):
        pygame.draw.circle(self.screen, color, (int(x), int(y)), int(radius))

    def draw_line(self, x1, y1, x2, y2, color):
        pygame.draw.line(self.screen, color, (int(x1), int(y1)), (int(x2), int(y2)))

    def distance(self, node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def get_random_node(self):
        x = random.uniform(self.x_range[0], self.x_range[1])
        y = random.uniform(self.y_range[0], self.y_range[1])
        return Node(x, y)

    def nearest_node(self, random_node):
        return min(self.nodes, key=lambda node: self.distance(node, random_node))

    def is_collision_free(self, node1, node2):
        for (ox, oy, size) in self.obstacle_list:
            steps = int(max(abs(node2.x - node1.x), abs(node2.y - node1.y)))
            for i in range(steps):
                x = node1.x + (node2.x - node1.x) * i / steps
                y = node1.y + (node2.y - node1.y) * i / steps
                if math.hypot(ox - x, oy - y) <= size:
                    return False
        return True

    def quantum_optimization(self, candidate_nodes):
        # Create a Quadratic Program to optimize node selection
        qp = QuadraticProgram(name="Node Selection")
        for i in range(len(candidate_nodes)):
            qp.binary_var(name=f'x{i}')  # Binary decision variable for each node

        # Objective: Minimize the total cost (distance to goal + cost to reach the node)
        objective = {}
        for i, node in enumerate(candidate_nodes):
            cost = node.cost + self.distance(node, self.goal)
            objective[f'x{i}'] = cost
        qp.minimize(linear=objective)

        # Constraints: Select exactly one node
        linear_constraint = {f'x{i}': 1 for i in range(len(candidate_nodes))}
        qp.linear_constraint(linear=linear_constraint, sense='==', rhs=1)

        # Solve the problem using QAOA
        optimizer = COBYLA()  # Use a classical optimizer
        qaoa = QAOA(sampler=Sampler(), optimizer=optimizer, reps=2)  # Use fewer layers
        minimum_eigen_optimizer = MinimumEigenOptimizer(qaoa)
        result = minimum_eigen_optimizer.solve(qp)

        # Get the selected node
        selected_index = int(result.variable_names[0][1:])  # Extract index from variable name (e.g., 'x0' -> 0)
        return candidate_nodes[selected_index]

    def generate_path(self):
        for _ in range(self.max_iter):
            random_node = self.get_random_node()
            nearest_node = self.nearest_node(random_node)
            theta = math.atan2(random_node.y - nearest_node.y, random_node.x - nearest_node.x)
            new_node = Node(nearest_node.x + self.step_size * math.cos(theta), nearest_node.y + self.step_size * math.sin(theta))
            new_node.cost = nearest_node.cost + self.distance(nearest_node, new_node)

            if self.is_collision_free(nearest_node, new_node):
                candidate_nodes = [new_node] + self.nodes[:5]  # Use fewer candidate nodes
                best_node = self.quantum_optimization(candidate_nodes)
                best_node.parent = nearest_node
                self.nodes.append(best_node)

                # Disable dynamic drawing during execution
                self.dynamic_draw(best_node)

                if self.distance(best_node, self.goal) <= self.step_size:
                    self.goal.parent = best_node
                    self.nodes.append(self.goal)
                    return self.extract_path()
        return None

    def extract_path(self):
        path = []
        node = self.goal
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]

    def dynamic_draw(self, node):
        self.screen.fill((255, 255, 255))
        for ox, oy, size in self.obstacle_list:
            self.draw_circle(ox, oy, size, (255, 0, 0))  # Obstacles in red
        for n in self.nodes:
            if n.parent:
                self.draw_line(n.x, n.y, n.parent.x, n.parent.y, (0, 255, 0))  # Tree edges in green
        self.draw_circle(self.start.x, self.start.y, 5, (0, 0, 255))  # Start in blue
        self.draw_circle(self.goal.x, self.goal.y, 5, (255, 0, 0))  # Goal in red
        pygame.display.update()
        self.clock.tick(30)  # Control frame rate

    def draw_final_path(self, path):
        for i in range(len(path) - 1):
            self.draw_line(path[i][0], path[i][1], path[i + 1][0], path[i + 1][1], (0, 0, 255))  # Final path in blue
        pygame.display.update()

if __name__ == '__main__':
    start = (10, 10)
    goal = (400, 400)
    obstacle_list = [(100, 100, 20), (200, 300, 30), (350, 150, 40)]  # Obstacles as (x, y, radius)
    x_range = (0, 500)
    y_range = (0, 500)

    rrt_star = RRTStar(start, goal, obstacle_list, x_range, y_range)
    path = rrt_star.generate_path()

    if path:
        print("Path found:", path)
        rrt_star.draw_final_path(path)
        time.sleep(5)
    else:
        print("No path found")
    pygame.quit()