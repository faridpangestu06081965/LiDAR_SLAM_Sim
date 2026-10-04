import pygame
import math
import sys
import heapq
import random

pygame.init()
pygame.font.init()

# Setup Split-Screen (600x600 World + 600x600 SLAM Map)
PANEL_SIZE = 600
SCREEN_WIDTH = PANEL_SIZE * 2
SCREEN_HEIGHT = PANEL_SIZE
FPS = 60

# Warna UI & Environment
COLOR_BG = (15, 18, 24)
COLOR_OBSTACLE = (180, 60, 60)
COLOR_OBSTACLE_BORDER = (255, 100, 100)
COLOR_ROBOT_BODY = (40, 48, 60)
COLOR_ROBOT_ACCENT = (0, 200, 255)
COLOR_WHEEL = (20, 22, 28)
COLOR_LASER = (0, 255, 120, 35)
COLOR_LASER_HIT = (255, 200, 0)

COLOR_SONAR_SAFE = (0, 255, 200)
COLOR_SONAR_WARN = (255, 190, 0)
COLOR_SONAR_ALERT = (255, 40, 40)

COLOR_TRAIL = (0, 140, 220)
COLOR_PATH = (0, 255, 255)
COLOR_TARGET = (255, 0, 120)

# SLAM Grid Config
GRID_CELL_SIZE = 6  # Pixels per cell
GRID_COLS = PANEL_SIZE // GRID_CELL_SIZE
GRID_ROWS = PANEL_SIZE // GRID_CELL_SIZE

VAL_UNKNOWN = 0
VAL_FREE = 1
VAL_OCCUPIED = 2

COLOR_CELL_UNKNOWN = (45, 50, 60)
COLOR_CELL_FREE = (235, 238, 242)
COLOR_CELL_OCCUPIED = (25, 25, 30)

class WorldObstacle:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)

    def draw(self, surface):
        pygame.draw.rect(surface, COLOR_OBSTACLE, self.rect, border_radius=4)
        pygame.draw.rect(surface, COLOR_OBSTACLE_BORDER, self.rect, width=2, border_radius=4)

class OccupancyGridMap:
    def __init__(self):
        self.grid = [[VAL_UNKNOWN for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

    def clear(self):
        self.grid = [[VAL_UNKNOWN for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

    def update_map(self, robot):
        robot_col = int(robot.x // GRID_CELL_SIZE)
        robot_row = int(robot.y // GRID_CELL_SIZE)

        for hit_x, hit_y, dist, is_hit, _ in robot.scan_data:
            end_col = int(hit_x // GRID_CELL_SIZE)
            end_row = int(hit_y // GRID_CELL_SIZE)

            steps = max(abs(end_col - robot_col), abs(end_row - robot_row))
            if steps > 0:
                dx = (end_col - robot_col) / steps
                dy = (end_row - robot_row) / steps

                for s in range(steps):
                    c = int(robot_col + s * dx)
                    r = int(robot_row + s * dy)
                    if 0 <= c < GRID_COLS and 0 <= r < GRID_ROWS:
                        if self.grid[r][c] != VAL_OCCUPIED:
                            self.grid[r][c] = VAL_FREE

            if is_hit and 0 <= end_col < GRID_COLS and 0 <= end_row < GRID_ROWS:
                self.grid[end_row][end_col] = VAL_OCCUPIED

    def get_exploration_percentage(self):
        mapped = sum(1 for r in range(GRID_ROWS) for c in range(GRID_COLS) if self.grid[r][c] != VAL_UNKNOWN)
        return (mapped / (GRID_COLS * GRID_ROWS)) * 100.0

    def is_near_obstacle(self, r, c, margin=3):
        if not (0 <= r < GRID_ROWS and 0 <= c < GRID_COLS):
            return True
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                    if self.grid[nr][nc] == VAL_OCCUPIED:
                        return True
        return False

    def has_line_of_sight(self, p1, p2, margin=3):
        x1, y1 = p1
        x2, y2 = p2
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist == 0:
            return True
        steps = max(1, int(dist / 4.0))
        dx = (x2 - x1) / steps
        dy = (y2 - y1) / steps
        
        for s in range(steps + 1):
            cx = x1 + s * dx
            cy = y1 + s * dy
            r = int(cy // GRID_CELL_SIZE)
            c = int(cx // GRID_CELL_SIZE)
            if self.is_near_obstacle(r, c, margin=margin):
                return False
        return True

    def smooth_path(self, grid_path, start_x, start_y):
        if not grid_path:
            return []
        
        pts = [((c + 0.5) * GRID_CELL_SIZE, (r + 0.5) * GRID_CELL_SIZE) for r, c in grid_path]
        pts.insert(0, (start_x, start_y))

        smoothed = []
        curr = 0
        while curr < len(pts) - 1:
            next_idx = curr + 1
            for j in range(len(pts) - 1, curr, -1):
                if self.has_line_of_sight(pts[curr], pts[j], margin=3):
                    next_idx = j
                    break
            smoothed.append(pts[next_idx])
            curr = next_idx

        return smoothed

    def find_frontier_goals(self, robot_r, robot_c, blocked_goals=None):
        candidates = []
        for r in range(4, GRID_ROWS - 4):
            for c in range(4, GRID_COLS - 4):
                if blocked_goals and (r, c) in blocked_goals:
                    continue
                if self.grid[r][c] == VAL_FREE and not self.is_near_obstacle(r, c, margin=3):
                    has_unknown = any(
                        self.grid[r + dr][c + dc] == VAL_UNKNOWN
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        if 0 <= r + dr < GRID_ROWS and 0 <= c + dc < GRID_COLS
                    )
                    if has_unknown:
                        dist = math.hypot(r - robot_r, c - robot_c)
                        if dist > 4:
                            candidates.append(((r, c), dist))
        
        candidates.sort(key=lambda x: x[1])
        return [c[0] for c in candidates]

    def get_random_free_node(self, robot_r, robot_c):
        free_nodes = []
        for r in range(6, GRID_ROWS - 6, 4):
            for c in range(6, GRID_COLS - 6, 4):
                if self.grid[r][c] == VAL_FREE and not self.is_near_obstacle(r, c, margin=3):
                    dist = math.hypot(r - robot_r, c - robot_c)
                    if dist > 15:
                        free_nodes.append((r, c))
        return random.choice(free_nodes) if free_nodes else None

    def get_astar_path(self, start, goal):
        if not goal:
            return []

        for margin in [3, 2, 1]:
            path = self._run_astar(start, goal, margin)
            if path:
                return path
        return []

    def _run_astar(self, start, goal, margin):
        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        max_iters = 900
        iters = 0

        while open_set and iters < max_iters:
            iters += 1
            current = heapq.heappop(open_set)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for dr, dc in neighbors:
                nr, nc = current[0] + dr, current[1] + dc
                if margin > 0 and self.is_near_obstacle(nr, nc, margin=margin):
                    continue

                step_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                tentative_g = g_score[current] + step_cost
                neighbor = (nr, nc)

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        return []

    def draw(self, surface):
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                val = self.grid[r][c]
                if val == VAL_UNKNOWN:
                    color = COLOR_CELL_UNKNOWN
                elif val == VAL_FREE:
                    color = COLOR_CELL_FREE
                else:
                    color = COLOR_CELL_OCCUPIED

                rect = (c * GRID_CELL_SIZE, r * GRID_CELL_SIZE, GRID_CELL_SIZE, GRID_CELL_SIZE)
                pygame.draw.rect(surface, color, rect)

class LidarRobot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0.0
        self.speed = 0.0
        self.max_speed = 2.6
        self.turn_speed = 5.0
        self.radius = 14
        
        # LiDAR
        self.num_rays = 180
        self.max_range = 230.0
        self.scan_data = []
        self.lidar_spin = 0.0
        
        # Sensors
        self.dist_front = 230.0
        self.dist_left = 230.0
        self.dist_right = 230.0
        self.dist_back = 230.0

        # Navigasi & Trail
        self.trail = []
        self.auto_mode = False
        self.show_rays = True
        self.total_distance = 0.0

        # Pathfinding berbasis Koordinat Lurus
        self.path_coords = []
        self.current_goal = None
        self.exploration_complete = False
        
        # State Machine Recovery & Anti-Stuck
        self.is_reversing = False
        self.reverse_timer = 0
        self.turn_dir = 1
        self.blocked_goals = set()

        self.stuck_ticks = 0
        self.last_pos = (x, y)
        self.escape_timer = 0
        self.escape_turn_dir = 1
        self.unblock_timer = 0

    def update_proximity_sensors(self):
        front_rays = [dist for _, _, dist, _, rel_a in self.scan_data if abs(rel_a) <= 22]
        left_rays = [dist for _, _, dist, _, rel_a in self.scan_data if 25 <= rel_a <= 65]
        right_rays = [dist for _, _, dist, _, rel_a in self.scan_data if -65 <= rel_a <= -25]
        back_rays = [dist for _, _, dist, _, rel_a in self.scan_data if abs(rel_a) >= 155]

        self.dist_front = min(front_rays, default=self.max_range)
        self.dist_left = min(left_rays, default=self.max_range)
        self.dist_right = min(right_rays, default=self.max_range)
        self.dist_back = min(back_rays, default=self.max_range)

    def trigger_u_turn_and_replan(self):
        self.is_reversing = True
        self.reverse_timer = 35
        self.turn_dir = 1 if self.dist_right > self.dist_left else -1
        if self.current_goal:
            self.blocked_goals.add(self.current_goal)
        self.path_coords.clear()
        self.current_goal = None

    def update(self, keys, obstacles, slam_map):
        prev_x, prev_y = self.x, self.y

        self.cast_rays(obstacles)
        self.update_proximity_sensors()

        self.unblock_timer += 1
        if self.unblock_timer > 200:
            self.unblock_timer = 0
            self.blocked_goals.clear()

        # DETEKSI STUCK
        if self.auto_mode:
            dist_moved_recently = math.hypot(self.x - self.last_pos[0], self.y - self.last_pos[1])
            if dist_moved_recently < 1.5 and not self.is_reversing:
                self.stuck_ticks += 1
            else:
                self.stuck_ticks = 0
                self.last_pos = (self.x, self.y)

            if self.stuck_ticks > 35:
                self.stuck_ticks = 0
                self.escape_timer = 35
                self.escape_turn_dir = 1 if self.dist_right > self.dist_left else -1
                if self.current_goal:
                    self.blocked_goals.add(self.current_goal)
                self.path_coords.clear()
                self.current_goal = None

        if self.auto_mode:
            if self.escape_timer > 0:
                self.escape_timer -= 1
                if self.dist_back > 20.0:
                    self.speed = -2.0
                else:
                    self.speed = 0.8
                self.angle += self.escape_turn_dir * 8.5
            
            elif self.is_reversing:
                self.reverse_timer -= 1

                if self.dist_back < 25.0 and self.reverse_timer > 20:
                    self.reverse_timer = 20

                if self.reverse_timer > 20:
                    self.speed = -2.2
                elif self.reverse_timer > 0:
                    self.speed = 0.5
                    self.angle += self.turn_dir * 7.5
                else:
                    self.is_reversing = False
                    self.path_coords.clear()
            else:
                if self.dist_front < 26.0 or self.dist_left < 12.0 or self.dist_right < 12.0:
                    self.trigger_u_turn_and_replan()
                else:
                    self.auto_pathfind(slam_map)
        else:
            self.path_coords.clear()
            self.current_goal = None
            self.is_reversing = False
            self.escape_timer = 0
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                self.speed = min(self.speed + 0.18, self.max_speed)
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                self.speed = max(self.speed - 0.18, -self.max_speed / 2)
            else:
                self.speed *= 0.88

            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                self.angle -= self.turn_speed
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                self.angle += self.turn_speed

        self.angle %= 360
        self.lidar_spin = (self.lidar_spin + 12) % 360

        rad = math.radians(self.angle)
        next_x = self.x + self.speed * math.cos(rad)
        next_y = self.y + self.speed * math.sin(rad)

        robot_rect = pygame.Rect(next_x - self.radius, next_y - self.radius, self.radius * 2, self.radius * 2)
        collided = any(robot_rect.colliderect(obs.rect) for obs in obstacles)

        if not collided:
            self.x = max(self.radius + 10, min(PANEL_SIZE - self.radius - 10, next_x))
            self.y = max(self.radius + 10, min(PANEL_SIZE - self.radius - 10, next_y))
            dist_moved = math.hypot(self.x - prev_x, self.y - prev_y)
            self.total_distance += dist_moved
        else:
            self.x -= math.cos(rad) * 4.0
            self.y -= math.sin(rad) * 4.0
            self.speed = 0
            if self.auto_mode and not self.is_reversing:
                self.trigger_u_turn_and_replan()

        if not self.trail or math.hypot(self.x - self.trail[-1][0], self.y - self.trail[-1][1]) > 6:
            self.trail.append((self.x, self.y))
            if len(self.trail) > 300:
                self.trail.pop(0)

    def auto_pathfind(self, slam_map):
        r_col = int(self.x // GRID_CELL_SIZE)
        r_row = int(self.y // GRID_CELL_SIZE)

        # 1. CEK ATAU HAPUS KOORDINAT WAYPOINT YANG SUDAH DICAPAI
        while self.path_coords:
            tx, ty = self.path_coords[0]
            if math.hypot(tx - self.x, ty - self.y) < 18:
                self.path_coords.pop(0)
            else:
                break

        # 2. CEK APAKAH GOAL UTAMA SUDAH SAMPAI
        if self.current_goal:
            gr, gc = self.current_goal
            gx = (gc + 0.5) * GRID_CELL_SIZE
            gy = (gr + 0.5) * GRID_CELL_SIZE
            if math.hypot(gx - self.x, gy - self.y) < 22:
                self.current_goal = None
                self.path_coords.clear()

        # 3. GOAL LOCK: JIKA TIDAK ADA GOAL, CARI TARGET BARU SECARA STABIL
        if self.current_goal is None:
            frontiers = slam_map.find_frontier_goals(r_row, r_col, self.blocked_goals)
            found = False

            for goal_cand in frontiers:
                raw_path = slam_map.get_astar_path((r_row, r_col), goal_cand)
                if raw_path:
                    self.current_goal = goal_cand
                    # Smoothing lintasan A* menjadi jalur koordinat lurus
                    self.path_coords = slam_map.smooth_path(raw_path, self.x, self.y)
                    found = True
                    self.exploration_complete = False
                    break
                else:
                    self.blocked_goals.add(goal_cand)

            if not found:
                all_raw = slam_map.find_frontier_goals(r_row, r_col, blocked_goals=None)
                if not all_raw:
                    self.exploration_complete = True

                patrol_target = slam_map.get_random_free_node(r_row, r_col)
                if patrol_target:
                    raw_path = slam_map.get_astar_path((r_row, r_col), patrol_target)
                    if raw_path:
                        self.current_goal = patrol_target
                        self.path_coords = slam_map.smooth_path(raw_path, self.x, self.y)

        # 4. JIKA GOAL ADA TAPI JALUR KOSONG, RE-PATH KE GOAL YANG SAMA (JANGAN GANTI TARGET)
        elif not self.path_coords and self.current_goal:
            raw_path = slam_map.get_astar_path((r_row, r_col), self.current_goal)
            if raw_path:
                self.path_coords = slam_map.smooth_path(raw_path, self.x, self.y)
            else:
                self.blocked_goals.add(self.current_goal)
                self.current_goal = None

        # 5. INTEGRASI PERGERAKAN MENUJU KOORDINAT BERIKUTNYA
        if self.path_coords:
            target_x, target_y = self.path_coords[0]

            desired_angle = math.degrees(math.atan2(target_y - self.y, target_x - self.x)) % 360
            angle_diff = (desired_angle - self.angle + 180) % 360 - 180

            if abs(angle_diff) > 2:
                if angle_diff > 0:
                    self.angle += min(self.turn_speed, angle_diff)
                else:
                    self.angle -= min(self.turn_speed, abs(angle_diff))

            # Putar diam di tempat jika beda sudut masih besar
            if abs(angle_diff) > 22:
                self.speed = max(0.0, self.speed - 0.3)
            else:
                self.speed = min(self.speed + 0.2, self.max_speed)
        else:
            self.speed *= 0.8

    def cast_rays(self, obstacles):
        self.scan_data.clear()
        step_angle = 360.0 / self.num_rays

        for i in range(self.num_rays):
            abs_angle_deg = self.angle + i * step_angle
            rel_angle = (i * step_angle)
            if rel_angle > 180:
                rel_angle -= 360

            ray_rad = math.radians(abs_angle_deg)
            cos_a = math.cos(ray_rad)
            sin_a = math.sin(ray_rad)

            hit_obstacle = False
            hit_x, hit_y = self.x + self.max_range * cos_a, self.y + self.max_range * sin_a
            hit_dist = self.max_range

            for d in range(0, int(self.max_range), 4):
                cx = self.x + d * cos_a
                cy = self.y + d * sin_a

                if cx <= 5 or cx >= PANEL_SIZE - 5 or cy <= 5 or cy >= PANEL_SIZE - 5:
                    hit_obstacle = True
                    hit_x, hit_y = cx, cy
                    hit_dist = d
                    break

                point_rect = pygame.Rect(cx - 1, cy - 1, 2, 2)
                if any(point_rect.colliderect(obs.rect) for obs in obstacles):
                    hit_obstacle = True
                    hit_x, hit_y = cx, cy
                    hit_dist = d
                    break

            self.scan_data.append((hit_x, hit_y, hit_dist, hit_obstacle, rel_angle))

    def draw_proximity_sensors(self, surface):
        angles = [0, 38, -38, 180]
        distances = [self.dist_front, self.dist_left, self.dist_right, self.dist_back]

        for ang, dist in zip(angles, distances):
            rad = math.radians(self.angle + ang)
            end_x = self.x + dist * math.cos(rad)
            end_y = self.y + dist * math.sin(rad)

            if dist < 26.0:
                color = COLOR_SONAR_ALERT
            elif dist < 42.0:
                color = COLOR_SONAR_WARN
            else:
                color = COLOR_SONAR_SAFE

            pygame.draw.line(surface, color, (self.x, self.y), (end_x, end_y), 3)
            pygame.draw.circle(surface, color, (int(end_x), int(end_y)), 4)

    def draw_chassis(self, surface, pos_x, pos_y, angle, is_slam=False):
        surf = pygame.Surface((44, 36), pygame.SRCALPHA)
        if is_slam:
            pygame.draw.circle(surf, (0, 200, 255), (22, 18), 7)
            pygame.draw.circle(surf, (255, 255, 255), (22, 18), 7, width=2)
            pygame.draw.line(surf, (255, 50, 50), (22, 18), (34, 18), 3)
        else:
            pygame.draw.rect(surf, COLOR_WHEEL, (4, 1, 10, 5), border_radius=2)
            pygame.draw.rect(surf, COLOR_WHEEL, (28, 1, 10, 5), border_radius=2)
            pygame.draw.rect(surf, COLOR_WHEEL, (4, 30, 10, 5), border_radius=2)
            pygame.draw.rect(surf, COLOR_WHEEL, (28, 30, 10, 5), border_radius=2)

            body_rect = pygame.Rect(7, 6, 30, 24)
            pygame.draw.rect(surf, COLOR_ROBOT_BODY, body_rect, border_radius=5)
            pygame.draw.rect(surf, COLOR_ROBOT_ACCENT, body_rect, width=2, border_radius=5)

            pygame.draw.circle(surf, (255, 230, 100), (35, 11), 3)
            pygame.draw.circle(surf, (255, 230, 100), (35, 25), 3)

            pygame.draw.circle(surf, (15, 20, 30), (22, 18), 8)
            pygame.draw.circle(surf, (0, 255, 150), (22, 18), 8, width=2)
            
            spin_rad = math.radians(self.lidar_spin)
            em_x = 22 + 5 * math.cos(spin_rad)
            em_y = 18 + 5 * math.sin(spin_rad)
            pygame.draw.circle(surf, (255, 50, 50), (int(em_x), int(em_y)), 2)

        rotated_surf = pygame.transform.rotate(surf, -angle)
        rect = rotated_surf.get_rect(center=(int(pos_x), int(pos_y)))
        surface.blit(rotated_surf, rect.topleft)

    def draw_path_and_target(self, surface):
        if len(self.path_coords) >= 2:
            pygame.draw.lines(surface, COLOR_PATH, False, self.path_coords, 3)
        elif len(self.path_coords) == 1:
            pygame.draw.line(surface, COLOR_PATH, (self.x, self.y), self.path_coords[0], 3)

        for pt in self.path_coords:
            pygame.draw.circle(surface, (0, 255, 200), (int(pt[0]), int(pt[1])), 4)

        if self.current_goal:
            gr, gc = self.current_goal
            gx = int((gc + 0.5) * GRID_CELL_SIZE)
            gy = int((gr + 0.5) * GRID_CELL_SIZE)
            pygame.draw.circle(surface, COLOR_TARGET, (gx, gy), 8, width=2)
            pygame.draw.line(surface, COLOR_TARGET, (gx - 10, gy), (gx + 10, gy), 2)
            pygame.draw.line(surface, COLOR_TARGET, (gx, gy - 10), (gx, gy + 10), 2)

    def draw(self, surface):
        if len(self.trail) >= 2:
            pygame.draw.lines(surface, COLOR_TRAIL, False, [(int(p[0]), int(p[1])) for p in self.trail], 2)

        if self.show_rays:
            laser_surf = pygame.Surface((PANEL_SIZE, PANEL_SIZE), pygame.SRCALPHA)
            for hit_x, hit_y, dist, is_hit, _ in self.scan_data:
                pygame.draw.line(laser_surf, COLOR_LASER, (self.x, self.y), (hit_x, hit_y), 1)
                if is_hit:
                    pygame.draw.circle(laser_surf, COLOR_LASER_HIT, (int(hit_x), int(hit_y)), 2)
            surface.blit(laser_surf, (0, 0))

        self.draw_proximity_sensors(surface)
        self.draw_path_and_target(surface)
        self.draw_chassis(surface, self.x, self.y, self.angle, is_slam=False)

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("2D LiDAR SLAM & Safe Differential Path Following")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Consolas", 13, bold=True)
    font_title = pygame.font.SysFont("Consolas", 15, bold=True)

    obstacles = [
        WorldObstacle(0, 0, PANEL_SIZE, 10),
        WorldObstacle(0, PANEL_SIZE - 10, PANEL_SIZE, 10),
        WorldObstacle(0, 0, 10, PANEL_SIZE),
        WorldObstacle(PANEL_SIZE - 10, 0, 10, PANEL_SIZE),

        WorldObstacle(120, 100, 150, 40),
        WorldObstacle(380, 80, 50, 200),
        WorldObstacle(100, 260, 40, 180),
        WorldObstacle(240, 360, 200, 40),
        WorldObstacle(440, 450, 100, 80)
    ]

    robot = LidarRobot(80, 80)
    slam_map = OccupancyGridMap()

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    robot.auto_mode = not robot.auto_mode
                elif event.key == pygame.K_l:
                    robot.show_rays = not robot.show_rays
                elif event.key == pygame.K_c:
                    slam_map.clear()
                    robot.trail.clear()
                    robot.path_coords.clear()
                    robot.current_goal = None
                    robot.total_distance = 0.0
                    robot.blocked_goals.clear()
                    robot.exploration_complete = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if mx < PANEL_SIZE:
                    obstacles.append(WorldObstacle(mx - 20, my - 20, 40, 40))

        keys = pygame.key.get_pressed()
        robot.update(keys, obstacles, slam_map)
        slam_map.update_map(robot)

        # RENDER WORLD
        world_surf = pygame.Surface((PANEL_SIZE, PANEL_SIZE))
        world_surf.fill(COLOR_BG)

        for obs in obstacles:
            obs.draw(world_surf)

        robot.draw(world_surf)

        # HUD World
        pygame.draw.rect(world_surf, (10, 12, 18, 220), (10, 10, 330, 90), border_radius=6)
        world_surf.blit(font_title.render("WORLD (GROUND TRUTH)", True, (255, 200, 0)), (18, 14))
        
        mode_str = "AUTO 360 NAV & REPLAN" if robot.auto_mode else "MANUAL DRIVE (WASD)"
        mode_color = (0, 255, 255) if robot.auto_mode else (255, 220, 0)
        world_surf.blit(font.render(f"MODE : {mode_str}", True, mode_color), (18, 33))
        
        if robot.escape_timer > 0:
            status_txt = "UNSTUCK ESCAPE ROUTINE!"
            status_col = (255, 50, 50)
        elif robot.is_reversing:
            status_txt = "U-TURN RECOVERING..."
            status_col = (255, 160, 0)
        elif robot.exploration_complete:
            status_txt = "ALL REACHABLE AREAS MAPPED!"
            status_col = (0, 255, 120)
        else:
            status_txt = f"Vector Nodes: {len(robot.path_coords)} waypoints"
            status_col = (200, 220, 255)

        world_surf.blit(font.render(f"Status     : {status_txt}", True, status_col), (18, 49))
        world_surf.blit(font.render(f"Odometer   : {int(robot.total_distance)} px", True, (220, 220, 220)), (18, 65))

        # RENDER SLAM MAP
        map_surf = pygame.Surface((PANEL_SIZE, PANEL_SIZE))
        slam_map.draw(map_surf)

        robot.draw_path_and_target(map_surf)
        robot.draw_chassis(map_surf, robot.x, robot.y, robot.angle, is_slam=True)

        # HUD SLAM
        expl_pct = slam_map.get_exploration_percentage()
        pygame.draw.rect(map_surf, (10, 12, 18, 220), (10, 10, 310, 95), border_radius=6)
        map_surf.blit(font_title.render("SLAM OCCUPANCY GRID MAP", True, (0, 200, 255)), (18, 14))
        
        pct_color = (0, 255, 120) if robot.exploration_complete else (255, 255, 255)
        pct_suffix = " (MAX REACHABLE)" if robot.exploration_complete else ""
        map_surf.blit(font.render(f"Explored Area : {expl_pct:.1f}%{pct_suffix}", True, pct_color), (18, 33))
        
        bar_w = int(2.7 * expl_pct)
        bar_col = (0, 255, 120) if robot.exploration_complete else (0, 200, 255)
        pygame.draw.rect(map_surf, (40, 45, 55), (18, 50, 270, 8), border_radius=4)
        pygame.draw.rect(map_surf, bar_col, (18, 50, bar_w, 8), border_radius=4)

        map_surf.blit(font.render("[SPACE] Auto Mode | [L] Rays | [C] Reset", True, (180, 190, 200)), (18, 65))
        map_surf.blit(font.render("[Left Click] Spawn Obstacle Wall", True, (180, 190, 200)), (18, 80))

        # Sensor Indicator Panel
        pygame.draw.rect(map_surf, (10, 12, 18, 220), (PANEL_SIZE - 280, PANEL_SIZE - 75, 270, 65), border_radius=6)
        map_surf.blit(font_title.render("4-WAY PROXIMITY SENSORS", True, (0, 255, 200)), (PANEL_SIZE - 270, PANEL_SIZE - 70))
        
        f_col = COLOR_SONAR_ALERT if robot.dist_front < 26 else (COLOR_SONAR_WARN if robot.dist_front < 42 else COLOR_SONAR_SAFE)
        l_col = COLOR_SONAR_ALERT if robot.dist_left < 26 else (COLOR_SONAR_WARN if robot.dist_left < 42 else COLOR_SONAR_SAFE)
        r_col = COLOR_SONAR_ALERT if robot.dist_right < 26 else (COLOR_SONAR_WARN if robot.dist_right < 42 else COLOR_SONAR_SAFE)
        b_col = COLOR_SONAR_ALERT if robot.dist_back < 26 else (COLOR_SONAR_WARN if robot.dist_back < 42 else COLOR_SONAR_SAFE)

        map_surf.blit(font.render(f"F:{int(robot.dist_front)}", True, f_col), (PANEL_SIZE - 270, PANEL_SIZE - 48))
        map_surf.blit(font.render(f"L:{int(robot.dist_left)}", True, l_col), (PANEL_SIZE - 205, PANEL_SIZE - 48))
        map_surf.blit(font.render(f"R:{int(robot.dist_right)}", True, r_col), (PANEL_SIZE - 140, PANEL_SIZE - 48))
        map_surf.blit(font.render(f"B:{int(robot.dist_back)}", True, b_col), (PANEL_SIZE - 75, PANEL_SIZE - 48))

        # Display Output
        screen.blit(world_surf, (0, 0))
        screen.blit(map_surf, (PANEL_SIZE, 0))

        pygame.draw.line(screen, (80, 90, 110), (PANEL_SIZE, 0), (PANEL_SIZE, PANEL_SIZE), 3)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
