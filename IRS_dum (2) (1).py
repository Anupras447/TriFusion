import pygame
import numpy as np
import random
import os
import csv
import collections
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import copy
import atexit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 480, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("IndianRoadSim - Multi-Agent Predictive Driving AI")
clock = pygame.time.Clock()

LOCAL_DIR = os.path.join(BASE_DIR, 'IndianRoadSim_Logs')
os.makedirs(LOCAL_DIR, exist_ok=True)
IRS_DATA_DIR = os.path.join(BASE_DIR, 'IRS data')
os.makedirs(IRS_DATA_DIR, exist_ok=True)
DATASET_FILE_PATH = os.path.join(IRS_DATA_DIR, 'clean_driving_dataset.csv')
MODEL_FILE_PATH = os.path.join(LOCAL_DIR, 'multi_agent_predictive_dqn.pt')
BUBBLE_CONFIG_PATH = os.path.join(LOCAL_DIR, 'learned_bubble_config.json')
BC_SAMPLE_LIMIT = 60000
REPLAY_DATASET_LIMIT = 20000
BC_EPOCHS = 1
FORCE_RETRAIN = os.environ.get("IRS_FORCE_RETRAIN", "0") == "1"
BRAKE_DECELERATION = 1.0
BUBBLE_SIDE_PADDING = 0.2
BUBBLE_LONGITUDINAL_PADDING = 0.35

# In-Memory Buffer for Efficient Disk I/O Logging
csv_write_buffer = []

if not os.path.exists(DATASET_FILE_PATH):
    with open(DATASET_FILE_PATH, 'w', newline='') as f:
        f.write("Y_Pos,Heading,Vx,Front_Dist,Left_Dist,Right_Dist,Control_Mode,Bot_Action,Human_Action,Is_Mistake,Collision_Occurred,Mistake_Reason\n")

def flush_csv_buffer():
    global csv_write_buffer
    if not csv_write_buffer:
        return
    with open(DATASET_FILE_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_write_buffer)
    csv_write_buffer.clear()

def log_sample_to_csv(y_pos, heading, vx, front_d, left_d, right_d, mode, bot_act, human_act, is_mistake, is_collision, reason):
    csv_write_buffer.append([
        round(float(y_pos), 2),
        round(float(heading), 2),
        round(float(vx), 2),
        round(float(front_d), 2),
        round(float(left_d), 2),
        round(float(right_d), 2),
        str(mode),
        int(bot_act),
        int(human_act),
        1 if is_mistake else 0,
        1 if is_collision else 0,
        str(reason)
    ])
    if len(csv_write_buffer) >= 100:
        flush_csv_buffer()

atexit.register(flush_csv_buffer)


def load_bubble_config():
    defaults = {
        "forward_rate": 5.0,
        "rear_rate": 1.5,
        "contraction_rate": 0.5,
    }
    try:
        with open(BUBBLE_CONFIG_PATH, 'r') as config_file:
            values = json.load(config_file)
        return {key: float(values.get(key, value)) for key, value in defaults.items()}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return defaults


def save_bubble_config(config):
    with open(BUBBLE_CONFIG_PATH, 'w') as config_file:
        json.dump(config, config_file, indent=2)


learned_bubble_config = load_bubble_config()


def check_oriented_box_collision(center1, dim1, angle1, center2, dim2, angle2=0.0):
    def get_corners(center, dim, angle_deg):
        cx, cy = center
        w, l = dim
        rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        
        hw, hl = w / 2.0, l / 2.0
        local_corners = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
        world_corners = []
        for lx, ly in local_corners:
            wx = cx + (lx * cos_a - ly * sin_a)
            wy = cy + (lx * sin_a + ly * cos_a)
            world_corners.append(np.array([wx, wy]))
        return world_corners

    box1 = get_corners(center1, dim1, angle1)
    box2 = get_corners(center2, dim2, angle2)

    boxes = [box1, box2]
    for b in range(2):
        for i in range(4):
            p1 = boxes[b][i]
            p2 = boxes[b][(i + 1) % 4]
            edge = p2 - p1
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm == 0:
                continue
            axis = axis / norm

            proj1 = [np.dot(corner, axis) for corner in box1]
            proj2 = [np.dot(corner, axis) for corner in box2]

            if max(proj1) < min(proj2) or max(proj2) < min(proj1):
                return False
    return True


class IndianRoadSimRL:
    def __init__(self, num_objects=320, oncoming_objects=200, rear_objects=150, num_hazards=45, road_length=600):
        self.road_length = road_length
        self.num_objects = num_objects            
        self.oncoming_count = oncoming_objects    
        self.rear_count = rear_objects
        self.road_half_width = 6.5 
        self.av_dim = (1.7, 4.2)
        self.angle_check_history = collections.deque(maxlen=3)

        self.merge_zones = [
            {"start_y": 80.0, "end_y": 180.0, "side": 1},
            {"start_y": 240.0, "end_y": 340.0, "side": -1},
            {"start_y": 400.0, "end_y": 500.0, "side": 1}
        ]
        
        self.road_segments = []
        curr_y = 0.0
        while curr_y < road_length + 100:
            c_type = np.random.choice([0, 1, 2], p=[0.65, 0.25, 0.10])
            if c_type == 0:
                amp, freq, seg_len = np.random.uniform(0.2, 0.8), np.random.uniform(100.0, 150.0), np.random.uniform(60.0, 100.0)
            elif c_type == 1:
                amp, freq, seg_len = np.random.uniform(1.2, 2.2), np.random.uniform(60.0, 90.0), np.random.uniform(40.0, 70.0)
            else: 
                amp, freq, seg_len = np.random.uniform(3.0, 4.5), np.random.uniform(35.0, 50.0), np.random.uniform(25.0, 45.0)
                
            self.road_segments.append((curr_y, curr_y + seg_len, amp, freq))
            curr_y += seg_len

        self.signals = [
            {"pos": 80.0, "state": "GREEN", "timer": 0},
            {"pos": 180.0, "state": "RED", "timer": 40},
            {"pos": 280.0, "state": "GREEN", "timer": 20},
            {"pos": 380.0, "state": "RED", "timer": 80},
            {"pos": 480.0, "state": "GREEN", "timer": 10},
            {"pos": 560.0, "state": "RED", "timer": 50}
        ]
        
        self.pedestrians = []
        self.pedestrian_spawn_timer = 0
        
        self.hazards = []
        for _ in range(num_hazards):
            h_y = np.random.uniform(15.0, road_length - 15.0)
            h_offset = np.random.uniform(-4.5, 4.5)
            h_type = np.random.choice(['pothole', 'speedbreaker'], p=[0.5, 0.5])
            self.hazards.append([h_y, h_offset, h_type])

        self.animals = []
        for _ in range(25):
            a_y = np.random.uniform(20.0, road_length - 20.0)
            a_offset = np.random.uniform(-4.5, 4.5)
            a_type = np.random.choice(['cow', 'dog'], p=[0.6, 0.4])
            self.animals.append([a_y, a_offset, a_type, 0])

        self.double_parked = []
        for _ in range(22):
            dp_y = np.random.uniform(20.0, road_length - 20.0)
            dp_offset = np.random.choice([-5.2, 5.2])
            self.double_parked.append([dp_y, dp_offset, 1.8, 4.2])

        self.rear_vehicles = []
        self.reset(soft_restart=False)

    def get_road_center(self, y_pos):
        base_val = 0.0
        for start_y, end_y, amp, freq in self.road_segments:
            if start_y <= y_pos <= end_y:
                local_y = y_pos - start_y
                base_val = amp * np.sin(local_y / freq) + (amp * 0.5) * np.cos(local_y / (freq * 0.5))
                break
        else:
            base_val = 1.0 * np.sin(y_pos / 80.0)
        return base_val

    def get_sensor_distances(self):
        ego_y, ego_x = self.av_state[0], self.av_state[1]
        front_dist = 60.0
        left_dist = 15.0
        right_dist = 15.0

        all_entities = []
        for obj in self.objects:
            all_entities.append((obj[0], obj[1], obj[5], obj[6]))
        for r_obj in self.rear_vehicles:
            all_entities.append((r_obj[0], r_obj[1], r_obj[5], r_obj[6]))
        for o_obj in self.oncoming_objects:
            all_entities.append((o_obj[0], o_obj[1], o_obj[3], o_obj[4]))
        for ped in self.pedestrians:
            all_entities.append((ped[0], ped[1], 0.6, 0.6))
        for dp in self.double_parked:
            dp_x = self.get_road_center(dp[0]) + dp[1]
            all_entities.append((dp[0], dp_x, dp[2], dp[3]))

        for ey, ex, ew, el in all_entities:
            dy = ey - ego_y
            dx = ex - ego_x
            if 0 < dy < front_dist and abs(dx) < 1.8:
                front_dist = dy
            if -10.0 < dy < 10.0:
                if -left_dist < dx < 0:
                    left_dist = abs(dx)
                if 0 < dx < right_dist:
                    right_dist = dx

        return round(float(front_dist), 2), round(float(left_dist), 2), round(float(right_dist), 2)

    def reset(self, soft_restart=False):
        self.angle_check_history.clear()
        if not soft_restart:
            road_c = self.get_road_center(0.0)
            self.av_state = [0.0, road_c + 2.0]
            self.av_heading = 0.0      
            self.av_vx = 0.0           
            self.objects = []
            self.oncoming_objects = []
            self.rear_vehicles = []
            self.pedestrians = []
            self.pedestrian_spawn_timer = 0
            
            for i in range(self.num_objects):
                obj_type = np.random.choice([0, 1, 2, 3], p=[0.4, 0.25, 0.1, 0.25])
                if obj_type == 0:
                    w, l, base_speed = np.random.uniform(1.6, 1.8), np.random.uniform(3.8, 4.4), np.random.uniform(0.30, 0.65)
                elif obj_type == 1:
                    w, l, base_speed = np.random.uniform(1.3, 1.5), np.random.uniform(2.8, 3.4), np.random.uniform(0.25, 0.60)
                elif obj_type == 2:
                    w, l, base_speed = np.random.uniform(2.2, 2.5), np.random.uniform(7.0, 9.5), np.random.uniform(0.20, 0.45)
                else: 
                    w, l, base_speed = np.random.uniform(0.6, 0.8), np.random.uniform(1.8, 2.2), np.random.uniform(0.35, 0.70)
                
                spawn_y = np.random.uniform(10.0, self.road_length)
                lateral_offset = np.random.uniform(-4.8, 5.2)
                aggression = np.random.uniform(0.7, 1.4)
                self.objects.append([spawn_y, self.get_road_center(spawn_y) + lateral_offset, base_speed, 0.0, obj_type, w, l, base_speed, 0, 0.0, 0, aggression])

            for i in range(self.rear_count):
                obj_type = np.random.choice([0, 1, 3], p=[0.6, 0.2, 0.2])
                w, l = (1.7, 4.2) if obj_type == 0 else ((1.4, 3.0) if obj_type == 1 else (0.8, 2.0))
                spawn_y = -np.random.uniform(20.0, 120.0)
                speed = np.random.uniform(0.40, 0.75)
                lane_offset = np.random.uniform(1.0, 4.5)
                aggression = np.random.uniform(0.75, 1.35)
                self.rear_vehicles.append([spawn_y, self.get_road_center(spawn_y) + lane_offset, speed, 0.0, obj_type, w, l, speed, 0, lane_offset, lane_offset, aggression])

            for i in range(self.oncoming_count):
                w, l = np.random.uniform(1.6, 1.8), np.random.uniform(3.8, 4.4)
                spawn_y = np.random.uniform(30.0, self.road_length)
                base_speed = np.random.uniform(0.25, 0.55)
                self.oncoming_objects.append([spawn_y, self.get_road_center(spawn_y) + np.random.uniform(-5.8, -1.8), base_speed, w, l, 0.0, 0, np.random.uniform(-5.8, -1.8), base_speed])

            for _ in range(80):
                p_y = np.random.uniform(10.0, self.road_length)
                side = np.random.choice([-1, 1])
                mode = np.random.choice([0, 1, 2], p=[0.20, 0.20, 0.60])
                lane_offset = np.random.uniform(-4.5, 4.5) if mode == 2 else side * (self.road_half_width - np.random.uniform(0.2, 0.8))
                p_x = self.get_road_center(p_y) + lane_offset
                p_speed = np.random.uniform(0.04, 0.10)
                p_dir = np.random.choice([-1, 1])
                self.pedestrians.append([p_y, p_x, p_dir, p_speed, p_speed, mode, side, 0.0])

            self.step_count = 0
        else:
            self.av_heading = 0.0
            self.av_vx = 0.0
            
        self.collision = False
        return self._get_obs()

    def _get_obs(self):
        road_c = self.get_road_center(self.av_state[0])
        ego_y, ego_x = self.av_state[0], self.av_state[1]
        
        obs_vector = [ego_x - road_c, self.av_heading, self.av_vx]

        all_veh = []
        for obj in self.objects:
            dy, dx = obj[0] - ego_y, obj[1] - ego_x
            dist = np.sqrt(dx**2 + dy**2)
            all_veh.append((dist, dx, dy, obj[3], obj[2]))
        for r_obj in self.rear_vehicles:
            dy, dx = r_obj[0] - ego_y, r_obj[1] - ego_x
            dist = np.sqrt(dx**2 + dy**2)
            all_veh.append((dist, dx, dy, r_obj[3], r_obj[2]))
            
        all_veh.sort(key=lambda item: item[0])
        for idx in range(4):
            if idx < len(all_veh):
                obs_vector.extend([all_veh[idx][1], all_veh[idx][2], all_veh[idx][3], all_veh[idx][4]])
            else:
                obs_vector.extend([0.0, 60.0, 0.0, 0.0])

        oncoming_sorted = []
        for o_obj in self.oncoming_objects:
            dy, dx = o_obj[0] - ego_y, o_obj[1] - ego_x
            dist = np.sqrt(dx**2 + dy**2)
            oncoming_sorted.append((dist, dx, dy, o_obj[5], -o_obj[2]))
            
        oncoming_sorted.sort(key=lambda item: item[0])
        for idx in range(2):
            if idx < len(oncoming_sorted):
                obs_vector.extend([oncoming_sorted[idx][1], oncoming_sorted[idx][2], oncoming_sorted[idx][3], oncoming_sorted[idx][4]])
            else:
                obs_vector.extend([0.0, 60.0, 0.0, 0.0])

        peds_sorted = []
        for ped in self.pedestrians:
            dy, dx = ped[0] - ego_y, ped[1] - ego_x
            dist = np.sqrt(dx**2 + dy**2)
            ped_vx = np.sign(ped[7] - ped[1]) * ped[3] if ped[5] == 1 else 0.0
            ped_vy = ped[2] * ped[3] if ped[5] in [0, 2] else 0.0
            peds_sorted.append((dist, dx, dy, ped_vx, ped_vy))

        peds_sorted.sort(key=lambda item: item[0])
        for idx in range(2):
            if idx < len(peds_sorted):
                obs_vector.extend([peds_sorted[idx][1], peds_sorted[idx][2], peds_sorted[idx][3], peds_sorted[idx][4]])
            else:
                obs_vector.extend([0.0, 60.0, 0.0, 0.0])

        return np.array(obs_vector, dtype=np.float32)

    def get_ground_truth_future_trajectories(self, horizon_steps=5):
        ego_y, ego_x = self.av_state[0], self.av_state[1]
        tracked_entities = []
        
        veh_candidates = []
        for obj in self.objects:
            d = np.sqrt((obj[0]-ego_y)**2 + (obj[1]-ego_x)**2)
            veh_candidates.append((d, obj[1]-ego_x, obj[0]-ego_y, obj[3], obj[2]))
        for r_obj in self.rear_vehicles:
            d = np.sqrt((r_obj[0]-ego_y)**2 + (r_obj[1]-ego_x)**2)
            veh_candidates.append((d, r_obj[1]-ego_x, r_obj[0]-ego_y, r_obj[3], r_obj[2]))
        veh_candidates.sort(key=lambda x: x[0])
        for i in range(4):
            if i < len(veh_candidates): tracked_entities.append(veh_candidates[i][1:])
            else: tracked_entities.append((0.0, 60.0, 0.0, 0.0))

        onc_candidates = []
        for o_obj in self.oncoming_objects:
            d = np.sqrt((o_obj[0]-ego_y)**2 + (o_obj[1]-ego_x)**2)
            onc_candidates.append((d, o_obj[1]-ego_x, o_obj[0]-ego_y, o_obj[5], -o_obj[2]))
        onc_candidates.sort(key=lambda x: x[0])
        for i in range(2):
            if i < len(onc_candidates): tracked_entities.append(onc_candidates[i][1:])
            else: tracked_entities.append((0.0, 60.0, 0.0, 0.0))

        ped_candidates = []
        for ped in self.pedestrians:
            d = np.sqrt((ped[0]-ego_y)**2 + (ped[1]-ego_x)**2)
            ped_vx = np.sign(ped[7] - ped[1]) * ped[3] if ped[5] == 1 else 0.0
            ped_vy = ped[2] * ped[3] if ped[5] in [0, 2] else 0.0
            ped_candidates.append((d, ped[1]-ego_x, ped[0]-ego_y, ped_vx, ped_vy))
        ped_candidates.sort(key=lambda x: x[0])
        for i in range(2):
            if i < len(ped_candidates): tracked_entities.append(ped_candidates[i][1:])
            else: tracked_entities.append((0.0, 60.0, 0.0, 0.0))

        dt_step = 1.0
        traj_matrix = []
        for dx, dy, vx, vy in tracked_entities:
            entity_future = []
            for k in range(1, horizon_steps + 1):
                future_dx = dx + vx * (k * dt_step * 2.0)
                future_dy = dy + vy * (k * dt_step * 2.0)
                entity_future.extend([future_dx, future_dy])
            traj_matrix.extend(entity_future)

        return np.array(traj_matrix, dtype=np.float32)

    def check_obstacle_in_path(self, detection_dist=18.0):
        ego_y, ego_x = self.av_state[0], self.av_state[1]
        ego_w, ego_l = self.av_dim
        road_c = self.get_road_center(ego_y)
        obstacle_detected = False

        for obj in self.objects:
            dy, dx = obj[0] - ego_y, abs(obj[1] - ego_x)
            if 0.2 < dy < detection_dist and dx < 1.6:
                obstacle_detected = True; break

        if not obstacle_detected:
            for dp in self.double_parked:
                dp_x = self.get_road_center(dp[0]) + dp[1]
                dy, dx = dp[0] - ego_y, abs(dp_x - ego_x)
                if 0.2 < dy < detection_dist and dx < 1.7:
                    obstacle_detected = True; break

        if not obstacle_detected:
            for animal in self.animals:
                a_x = self.get_road_center(animal[0]) + animal[1]
                dy, dx = animal[0] - ego_y, abs(a_x - ego_x)
                if 0.2 < dy < detection_dist and dx < 1.5:
                    obstacle_detected = True; break

        if not obstacle_detected:
            for ped in self.pedestrians:
                dy, dx = ped[0] - ego_y, abs(ped[1] - ego_x)
                if 0.2 < dy < detection_dist and dx < 1.5:
                    obstacle_detected = True; break

        if not obstacle_detected:
            for sig in self.signals:
                if sig["state"] in ["RED", "YELLOW"]:
                    dy = sig["pos"] - ego_y
                    if 0.0 < dy < detection_dist:
                        obstacle_detected = True; break

        if not obstacle_detected:
            for o_obj in self.oncoming_objects:
                dy, dx = o_obj[0] - ego_y, abs(o_obj[1] - ego_x)
                if 0.2 < dy < detection_dist and dx < 1.6:
                    obstacle_detected = True; break

        left_target_x = ego_x - 2.5
        right_target_x = ego_x + 2.5
        max_offset = (self.road_half_width - (ego_w / 2.0)) * 0.95

        left_clear = (left_target_x - road_c) >= -max_offset and self.check_lane_clear(ego_y + 3.0, left_target_x, ego_l)
        right_clear = (right_target_x - road_c) <= max_offset and self.check_lane_clear(ego_y + 3.0, right_target_x, ego_l)

        return obstacle_detected, left_clear, right_clear

    def get_dual_bubble_profile(self):
        ego_y, ego_x = self.av_state
        v = max(0.0, float(self.av_vx))

        nearby_count = 0
        for obj in self.objects:
            if abs(obj[0] - ego_y) < 50.0: nearby_count += 1
        for ped in self.pedestrians:
            if abs(ped[0] - ego_y) < 50.0: nearby_count += 1
        for o_obj in self.oncoming_objects:
            if abs(o_obj[0] - ego_y) < 50.0: nearby_count += 1
        for dp in self.double_parked:
            if abs(dp[0] - ego_y) < 50.0: nearby_count += 1
        for animal in self.animals:
            if abs(animal[0] - ego_y) < 50.0: nearby_count += 1

        density = min(2.0, nearby_count / 15.0)

        a = 0.30 + 0.5 * density
        b = 2.00 + 0.50 * density
        c = 5.00 + 0.2 * density
        d = 1.25 + 0.50 * density

        b1_dist = float(np.exp(a * v) - 1.0 + c)
        b1_width = self.av_dim[0] + (BUBBLE_SIDE_PADDING*2)

        b2_dist = float(b * v + d)
        b2_width = self.av_dim[0] + (BUBBLE_SIDE_PADDING*2)

        b1_triggered = False
        b2_triggered = False

        heading_rad = np.radians(self.av_heading)
        cos_h = np.cos(heading_rad)
        sin_h = np.sin(heading_rad)
        half_w = b1_width / 2.0

        max_scan = max(b1_dist, b2_dist)
        for s in np.arange(0.5, max_scan + 0.5, 0.5):
            proj_y = ego_y + s * cos_h
            proj_x = ego_x + s * sin_h
            road_c = self.get_road_center(proj_y)

            left_edge_x = proj_x - half_w * cos_h
            right_edge_x = proj_x + half_w * cos_h

            left_bound = road_c - self.road_half_width
            right_bound = road_c + self.road_half_width

            if left_edge_x <= left_bound or right_edge_x >= right_bound:
                if s <= b2_dist:
                    b2_triggered = True
                    b1_triggered = True
                    b2_dist = max(1.0, float(s))
                    b1_dist = min(b1_dist, b2_dist)
                    break
                elif s <= b1_dist:
                    b1_triggered = True
                    b1_dist = max(1.0, float(s))
                    break

        candidates = []
        for obj in self.objects:
            candidates.append((obj[0], obj[1], obj[6] / 2.0))
        for obj in self.oncoming_objects:
            candidates.append((obj[0], obj[1], obj[4] / 2.0))
        for ped in self.pedestrians:
            candidates.append((ped[0], ped[1], 0.3))
        for animal in self.animals:
            candidates.append((animal[0], self.get_road_center(animal[0]) + animal[1], 0.6))
        for dp in self.double_parked:
            candidates.append((dp[0], self.get_road_center(dp[0]) + dp[1], dp[3] / 2.0))
        for sig in self.signals:
            if sig["state"] in ["RED", "YELLOW"]:
                candidates.append((sig["pos"], self.get_road_center(sig["pos"]), 1.0))

        for entity_y, entity_x, half_length in candidates:
            relative_y = entity_y - ego_y
            lat_dist = abs(entity_x - ego_x)

            if 0.0 < relative_y <= b2_dist + half_length and lat_dist <= b2_width / 2.0:
                b2_triggered = True
                b1_triggered = True
                break
            elif 0.0 < relative_y <= b1_dist + half_length and lat_dist <= b1_width / 2.0:
                b1_triggered = True

        return {
            "b1_triggered": b1_triggered,
            "b2_triggered": b2_triggered,
            "b1_dist": b1_dist,
            "b1_width": b1_width,
            "b2_dist": b2_dist,
            "b2_width": b2_width,
        }

    def get_rear_bubble_profile(self):
        ego_y, ego_x = self.av_state
        ego_speed = max(0.0, self.av_vx)
        bubble_distance = float(np.clip(10.0 + ego_speed * 4.0, 12.0, 28.0))
        bubble_half_width = self.av_dim[0] / 2.0
        candidates = []

        for obj in self.rear_vehicles:
            candidates.append((obj[0], obj[1], obj[6] / 2.0, obj[2], "rear vehicle"))
        for obj in self.objects:
            candidates.append((obj[0], obj[1], obj[6] / 2.0, obj[2], "vehicle"))

        nearby = []
        for entity_y, entity_x, half_length, entity_speed, kind in candidates:
            relative_y = ego_y - entity_y
            lateral_distance = abs(entity_x - ego_x)
            closing_speed = max(0.0, entity_speed - ego_speed)
            closing_margin = closing_speed * 4.0
            if 0.0 < relative_y <= bubble_distance + closing_margin and lateral_distance <= bubble_half_width + 0.7:
                nearby.append({
                    "kind": kind,
                    "relative_y": relative_y,
                    "lateral_distance": lateral_distance,
                    "closing_speed": closing_speed,
                    "side": "left" if entity_x < ego_x else "right",
                })

        nearby.sort(key=lambda item: item["relative_y"])
        blocker = nearby[0] if nearby else None
        hard_stop = blocker is not None and (
            blocker["relative_y"] <= 5.0 or blocker["closing_speed"] >= 0.45
        )
        blocked_sides = {
            side for side in ("left", "right")
            if any(item["side"] == side for item in nearby)
        }
        return {
            "active": blocker is not None,
            "blocker": blocker,
            "hard_stop": hard_stop,
            "blocked_sides": blocked_sides,
            "distance": bubble_distance,
            "half_width": bubble_half_width,
        }

    def check_lane_clear(self, y_pos, target_x, obj_length, current_obj_idx=-1):
        for idx, other in enumerate(self.objects):
            if idx == current_obj_idx: continue
            if abs(other[0] - y_pos) < (obj_length + other[6] + 4.0) and abs(other[1] - target_x) < 1.6:
                return False
        for o_obj in self.oncoming_objects:
            if abs(o_obj[0] - y_pos) < (obj_length + o_obj[4] + 8.0) and abs(o_obj[1] - target_x) < 1.8:
                return False
        if abs(self.av_state[0] - y_pos) < (obj_length + self.av_dim[1] + 4.0) and abs(self.av_state[1] - target_x) < 1.6:
            return False
        return True

    def find_clear_bubble_angle(self):
        dual_prof = self.get_dual_bubble_profile()
        bubble_half_width = dual_prof["b1_width"] / 2.0
        max_scan_dist = max(40.0, dual_prof["b1_dist"] * 2.5)

        ego_y, ego_x = self.av_state[0], self.av_state[1]
        dy_step = 2.0
        c_x_curr = self.get_road_center(ego_y)
        c_x_ahead = self.get_road_center(ego_y + dy_step)
        road_angle_rad = np.arctan2(c_x_ahead - c_x_curr, dy_step)
        road_angle_deg = np.degrees(road_angle_rad)

        angles = list(range(0, 46, 3)) + list(range(-3, -46, -3))

        best_angle = None
        max_clear_distance = -1.0

        for offset_deg in angles:
            world_angle_deg = road_angle_deg + offset_deg
            angle_rad = np.radians(world_angle_deg)

            clear_dist = 0.0
            step_size = 1.5
            for longitudinal in np.arange(1.0, max_scan_dist + step_size, step_size):
                projected_y = ego_y + np.cos(angle_rad) * longitudinal
                projected_x = ego_x + np.sin(angle_rad) * longitudinal
                road_center = self.get_road_center(projected_y)

                if abs(projected_x - road_center) + bubble_half_width > self.road_half_width:
                    break

                blocked = False
                for entity_y, entity_x, entity_width, entity_length, _ in self._bubble_entities():
                    relative_y = entity_y - ego_y
                    if (0.0 < relative_y <= max_scan_dist and
                            abs(entity_x - projected_x) < bubble_half_width + entity_width / 2.0 and
                            abs(entity_y - projected_y) < entity_length / 2.0 + 1.0):
                        blocked = True
                        break

                if blocked:
                    break

                clear_dist = longitudinal

            if clear_dist > max_clear_distance:
                max_clear_distance = clear_dist
                best_angle = float(offset_deg)
            elif clear_dist == max_clear_distance and clear_dist > 0:
                if best_angle is None or abs(offset_deg) < abs(best_angle):
                    best_angle = float(offset_deg)

        if best_angle is not None and max_clear_distance >= 8.0:
            self.angle_check_history.append(best_angle)
            if len(self.angle_check_history) == 3:
                return float(np.mean(self.angle_check_history))
            return None

        self.angle_check_history.clear()
        return None

    def _bubble_entities(self):
        for obj in self.objects:
            yield obj[0], obj[1], obj[5], obj[6], "vehicle"
        for obj in self.oncoming_objects:
            yield obj[0], obj[1], obj[3], obj[4], "oncoming"
        for ped in self.pedestrians:
            yield ped[0], ped[1], 0.6, 0.6, "pedestrian"
        for animal in self.animals:
            yield animal[0], self.get_road_center(animal[0]) + animal[1], 1.2, 1.2, animal[2]
        for dp in self.double_parked:
            yield dp[0], self.get_road_center(dp[0]) + dp[1], dp[2], dp[3], "parked"

    def check_rear_overtake_clear(self, y_pos, target_x, r_idx):
        if abs(self.av_state[0] - y_pos) < 12.0 and abs(self.av_state[1] - target_x) < 2.4:
            return False
        for idx, r_other in enumerate(self.rear_vehicles):
            if idx == r_idx: continue
            if abs(r_other[0] - y_pos) < 12.0 and abs(r_other[1] - target_x) < 2.4:
                return False
        for f_obj in self.objects:
            if abs(f_obj[0] - y_pos) < 11.0 and abs(f_obj[1] - target_x) < 2.4:
                return False
        for o_obj in self.oncoming_objects:
            if abs(o_obj[0] - y_pos) < 20.0 and abs(o_obj[1] - target_x) < 2.5:
                return False
        return True

    def check_oncoming_route_clear(self, y_pos, candidate_x, current_o_idx):
        if abs(self.av_state[0] - y_pos) < 10.0 and abs(self.av_state[1] - candidate_x) < 1.8:
            return False
        for idx, other in enumerate(self.oncoming_objects):
            if idx == current_o_idx: continue
            if abs(other[0] - y_pos) < 6.0 and abs(other[1] - candidate_x) < 1.6:
                return False
        return True

    def step_with_inputs(self, steer_dir, throttle_change, action_code=None):
        self.step_count += 1
        max_speed, min_speed = 2.6, -1.0
        
        for sig in self.signals:
            sig["timer"] += 1
            if sig["timer"] > 140:
                sig["timer"] = 0
                sig["state"] = "YELLOW" if sig["state"] == "GREEN" else ("RED" if sig["state"] == "YELLOW" else "GREEN")

        self.pedestrian_spawn_timer += 1
        if self.pedestrian_spawn_timer > 8 and len(self.pedestrians) < 90:
            p_y = self.av_state[0] + np.random.uniform(30.0, 95.0)
            side = np.random.choice([-1, 1])
            mode = np.random.choice([0, 1, 2], p=[0.15, 0.25, 0.60])
            lane_offset = np.random.uniform(-4.5, 4.5) if mode == 2 else side * (self.road_half_width - np.random.uniform(0.2, 0.8))
            p_x = self.get_road_center(p_y) + lane_offset
            p_speed = np.random.uniform(0.04, 0.10)
            p_dir = np.random.choice([-1, 1])
            self.pedestrians.append([p_y, p_x, p_dir, p_speed, p_speed, mode, side, 0.0])
            self.pedestrian_spawn_timer = 0
        
        active_pedestrians = []
        for ped in self.pedestrians:
            c_x = self.get_road_center(ped[0])
            sidewalk_x = c_x + (ped[6] * (self.road_half_width - 0.4))

            if ped[5] == 0 and random.random() < 0.008 and abs(ped[0] - self.av_state[0]) > 15.0:
                ped[5] = 1 
                ped[7] = c_x - (ped[6] * (self.road_half_width - 0.4))

            if ped[5] == 1:
                step = np.sign(ped[7] - ped[1]) * ped[3]
                ped[1] += step
                if abs(ped[1] - ped[7]) < 0.3:
                    ped[5] = 0
                    ped[6] = -ped[6]
            elif ped[5] == 2:
                ped[0] += ped[2] * ped[3] * 0.9
                ped[1] += np.random.uniform(-0.04, 0.04)
            else:
                ped[0] += ped[2] * ped[3]
                ped[1] = 0.85 * ped[1] + 0.15 * sidewalk_x

            if abs(ped[0] - self.av_state[0]) < 120.0:
                active_pedestrians.append(ped)

        self.pedestrians = active_pedestrians

        if steer_dir != 0.0:
            self.av_heading += steer_dir * 2.2
            
        self.av_heading = float(np.clip(self.av_heading, -89.0, 89.0))
        
        hazard_speed_penalty = 0.0
        for h in self.hazards:
            h_center_x = self.get_road_center(h[0]) + h[1]
            if abs(self.av_state[0] - h[0]) < 0.7 and abs(self.av_state[1] - h_center_x) < 0.7:
                hazard_speed_penalty = 0.08 if h[2] == 'speedbreaker' else 0.05

        dual_prof = self.get_dual_bubble_profile()
        bubble_decel = 0.0
        if dual_prof["b2_triggered"]:
            bubble_decel = -BRAKE_DECELERATION
        elif dual_prof["b1_triggered"]:
            bubble_decel = -0.08

        if action_code == 3:
            if self.av_vx > 0.0:
                effective_throttle = 0.0
            else:
                effective_throttle = throttle_change
        else:
            effective_throttle = throttle_change

        total_change = effective_throttle + bubble_decel - hazard_speed_penalty

        if action_code == 0 or (self.av_vx > 0.0 and total_change < 0.0 and action_code != 3):
            self.av_vx = max(0.0, min(max_speed, self.av_vx + total_change))
        else:
            self.av_vx = max(min_speed, min(max_speed, self.av_vx + total_change))
        
        heading_rad = np.radians(self.av_heading)
        self.av_state[0] += self.av_vx * np.cos(heading_rad)
        self.av_state[1] += self.av_vx * np.sin(heading_rad)

        road_center = self.get_road_center(self.av_state[0])
        max_allowed_offset = (self.road_half_width - (self.av_dim[0] / 2.0)) * 0.95
        offset = self.av_state[1] - road_center

        obs = self._get_obs()
        
        own_lane_target = 2.0
        lane_dev = abs(offset - own_lane_target)
        opposite_lane_penalty = 6.0 * abs(offset) if offset < 0.0 else 0.0
        reward = self.av_vx * 0.5 - lane_dev * 0.2 - opposite_lane_penalty - abs(self.av_heading) * 0.01

        has_obstacle, left_clear, right_clear = self.check_obstacle_in_path(detection_dist=20.0)

        if not has_obstacle:
            if self.av_vx > 1.5:
                speed_ratio = self.av_vx / max_speed
                reward += 1.5 * speed_ratio
            if throttle_change > 0 and self.av_vx < max_speed:
                reward += 0.3
        else:
            if (left_clear or right_clear) and self.av_vx > 1.0:
                reward += 0.8
        
        if abs(offset) >= max_allowed_offset:
            return obs, -300.0, True, True, False, "Off-Road Deviation"

        for sig in self.signals:
            if 0.0 <= (self.av_state[0] - sig["pos"]) < 2.0 and sig["state"] == "RED":
                return obs, -600.0, True, True, False, "Red Light Signal Violation"

        ego_center = (self.av_state[1], self.av_state[0])
        ego_dim = self.av_dim

        for ped in self.pedestrians:
            if check_oriented_box_collision(ego_center, ego_dim, self.av_heading, (ped[1], ped[0]), (0.6, 0.6)):
                return obs, -500.0, True, True, False, "Pedestrian Collision"

        for animal in self.animals:
            a_center_x = self.get_road_center(animal[0]) + animal[1]
            if check_oriented_box_collision(ego_center, ego_dim, self.av_heading, (a_center_x, animal[0]), (1.2, 1.2) if animal[2] == 'cow' else (0.8, 0.8)):
                return obs, -500.0, True, True, False, "Animal Hazard Collision"

        for dp in self.double_parked:
            dp_center_x = self.get_road_center(dp[0]) + dp[1]
            if check_oriented_box_collision(ego_center, ego_dim, self.av_heading, (dp_center_x, dp[0]), (dp[2], dp[3])):
                return obs, -500.0, True, True, False, "Parked Vehicle Collision"

        for i, obj in enumerate(self.objects):
            aggression = obj[11]
            forced_stop = False

            for m_zone in self.merge_zones:
                if m_zone["start_y"] <= obj[0] <= m_zone["end_y"]:
                    prog = (obj[0] - m_zone["start_y"]) / (m_zone["end_y"] - m_zone["start_y"])
                    merge_target_x = self.get_road_center(obj[0]) + m_zone["side"] * (self.road_half_width - 1.5 * prog)
                    obj[1] = 0.92 * obj[1] + 0.08 * merge_target_x

            for sig in self.signals:
                if sig["state"] in ["RED", "YELLOW"] and 0 <= sig["pos"] - obj[0] < 10.0:
                    forced_stop = True

            dist_to_ahead, speed_ahead = 999.0, 0.0

            for j, other in enumerate(self.objects):
                if i == j: continue
                dx_other, dy_other = other[0] - obj[0], abs(other[1] - obj[1])
                if 0 < dx_other < dist_to_ahead and dy_other < 1.4:
                    dist_to_ahead, speed_ahead = dx_other, other[2]

            dx_ego, dy_ego = self.av_state[0] - obj[0], abs(self.av_state[1] - obj[1])
            is_ego_ahead = (0 < dx_ego < dist_to_ahead) and (dy_ego < 2.5)

            if is_ego_ahead:
                dist_to_ahead = dx_ego
                speed_ahead = max(0.0, self.av_vx)

            min_stop_gap = ((self.av_dim[1] + obj[6]) / 2.0) + 3.0
            safe_distance = (8.0 + (obj[2] * 4.0)) / aggression

            if random.random() < 0.015 * aggression and obj[8] == 0:
                c_x = self.get_road_center(obj[0])
                rand_offset = np.random.uniform(-4.5, 4.8)
                t_x = c_x + rand_offset
                if self.check_lane_clear(obj[0], t_x, obj[6], i):
                    obj[8] = 1
                    obj[9] = t_x

            if is_ego_ahead and (dx_ego < 25.0) and (self.av_vx < obj[7] * 0.8) and obj[8] == 0:
                c_x = self.get_road_center(obj[0])
                candidate_offsets = [self.av_state[1] - c_x - 2.6, self.av_state[1] - c_x + 2.6]
                for candidate in candidate_offsets:
                    t_x = c_x + candidate
                    if (c_x - self.road_half_width + 1.0 <= t_x <= c_x + self.road_half_width - 1.0):
                        if self.check_lane_clear(obj[0], t_x, obj[6], i):
                            obj[8] = 1         
                            obj[9] = t_x       
                            break

            if obj[8] == 1:
                lateral_diff = obj[9] - obj[1]
                obj[3] = np.clip(lateral_diff * 0.18, -0.18, 0.18)
                obj[2] = min(obj[7] * (1.1 * aggression), obj[2] + 0.02)
                if abs(obj[1] - obj[9]) < 0.25: obj[8] = 2

            elif obj[8] == 2:
                obj[2] = min(obj[7] * (1.15 * aggression), obj[2] + 0.02)
                obj[3] *= 0.80
                if obj[0] > self.av_state[0] + self.av_dim[1] + 10.0 or random.random() < 0.02:
                    obj[8] = 3
                    obj[9] = self.get_road_center(obj[0]) + np.random.uniform(-4.0, 4.8)

            elif obj[8] == 3:
                lateral_diff = obj[9] - obj[1]
                obj[3] = np.clip(lateral_diff * 0.14, -0.14, 0.14)
                if abs(obj[1] - obj[9]) < 0.25: obj[8] = 0

            else:
                if dist_to_ahead < min_stop_gap or forced_stop:
                    obj[2] = 0.0
                elif dist_to_ahead < safe_distance:
                    decel = 0.15 if dist_to_ahead < (safe_distance * 0.5) else 0.08
                    obj[2] = max(speed_ahead, obj[2] - decel)
                else:
                    obj[2] = min(obj[7], obj[2] + (0.015 * aggression))
                obj[3] *= 0.85

            obj[0] += obj[2]
            obj[1] += obj[3]
            
            obj_center_road = self.get_road_center(obj[0])
            min_offset, max_offset = 0.5, self.road_half_width - (obj[5] / 2.0) - 0.2
            obj[1] = np.clip(obj[1], obj_center_road - max_offset, obj_center_road + max_offset)

            if obj[0] < self.av_state[0] - 60.0:
                obj[0] = self.av_state[0] + np.random.uniform(40.0, self.road_length - self.av_state[0] + 10.0)
                obj[1] = self.get_road_center(obj[0]) + np.random.uniform(-4.5, 4.8)
                obj[2] = obj[7]
                obj[8] = 0

            if check_oriented_box_collision(ego_center, ego_dim, self.av_heading, (obj[1], obj[0]), (obj[5], obj[6])):
                return obs, -500.0, True, True, False, "Forward Vehicle Collision"

        for r_idx, r_obj in enumerate(self.rear_vehicles):
            c_road = self.get_road_center(r_obj[0])
            aggression = r_obj[11]

            side_obstacle_left, side_obstacle_right = False, False
            all_obstacles = self.objects + self.rear_vehicles + [[self.av_state[0], self.av_state[1], self.av_vx, 0.0, 0, self.av_dim[0], self.av_dim[1]]]

            for other in all_obstacles:
                if other is r_obj: continue
                dy_side = abs(other[0] - r_obj[0])
                dx_side = other[1] - r_obj[1]
                if dy_side < ((r_obj[6] + other[6]) / 2.0 + 3.0):
                    if -2.5 < dx_side < -0.3: side_obstacle_left = True
                    elif 0.3 < dx_side < 2.5: side_obstacle_right = True

            dist_ahead, speed_ahead, obstacle_y = 999.0, 0.0, 0.0
            dy_ego, dx_ego = self.av_state[0] - r_obj[0], abs(self.av_state[1] - r_obj[1])
            if 0 < dy_ego < 50.0 and dx_ego < 2.5:
                dist_ahead = dy_ego
                speed_ahead = max(0.0, self.av_vx)
                obstacle_y = self.av_state[0]

            for other in self.objects + self.rear_vehicles:
                if other is r_obj: continue
                dy_other, dx_other = other[0] - r_obj[0], abs(other[1] - r_obj[1])
                if 0 < dy_other < dist_ahead and dx_other < 1.8:
                    dist_ahead = dy_other
                    speed_ahead = max(0.0, other[2])
                    obstacle_y = other[0]

            min_stop_distance = ((self.av_dim[1] + r_obj[6]) / 2.0) + 3.5
            safe_cushion = min_stop_distance + 6.0 + (r_obj[2] * 4.0)

            for sig in self.signals:
                if sig["state"] in ["RED", "YELLOW"] and 0 <= sig["pos"] - r_obj[0] < 12.0:
                    dist_ahead = min(dist_ahead, sig["pos"] - r_obj[0])
                    speed_ahead = 0.0

            if r_obj[8] == 0:
                if dist_ahead < 18.0 and (speed_ahead < r_obj[7] * 0.85):
                    candidates = []
                    if not side_obstacle_left and (r_obj[10] - 2.5 >= 0.8): candidates.append(r_obj[10] - 2.5)
                    if not side_obstacle_right and (r_obj[10] + 2.5 <= self.road_half_width - 1.2): candidates.append(r_obj[10] + 2.5)

                    for cand_offset in candidates:
                        cand_x = c_road + cand_offset
                        if self.check_rear_overtake_clear(r_obj[0], cand_x, r_idx):
                            r_obj[8] = 1; r_obj[9] = cand_offset; break

            if r_obj[8] == 1:
                target_absolute_x = c_road + r_obj[9]
                lateral_error = target_absolute_x - r_obj[1]
                r_obj[3] = np.clip(lateral_error * 0.08, -0.06, 0.06)
                if abs(lateral_error) < 0.20: r_obj[8] = 2

            elif r_obj[8] == 2:
                r_obj[3] *= 0.80
                r_obj[2] = min(r_obj[7] * 1.15, r_obj[2] + 0.01)
                if r_obj[0] > obstacle_y + 11.0:
                    base_target_x = c_road + r_obj[10]
                    if self.check_rear_overtake_clear(r_obj[0], base_target_x, r_idx):
                        r_obj[8] = 3; r_obj[9] = r_obj[10]

            elif r_obj[8] == 3:
                target_absolute_x = c_road + r_obj[9]
                lateral_error = target_absolute_x - r_obj[1]
                if (lateral_error < 0 and side_obstacle_left) or (lateral_error > 0 and side_obstacle_right):
                    r_obj[3] = 0.0
                else:
                    r_obj[3] = np.clip(lateral_error * 0.08, -0.06, 0.06)
                if abs(lateral_error) < 0.20: r_obj[8] = 0

            else:
                target_absolute_x = c_road + r_obj[9]
                r_obj[3] = np.clip((target_absolute_x - r_obj[1]) * 0.06, -0.05, 0.05)

            if dist_ahead <= min_stop_distance:
                r_obj[2] = 0.0
                r_obj[3] = 0.0
            elif dist_ahead < safe_cushion:
                decel = 0.20 * (1.0 - (dist_ahead - min_stop_distance) / (safe_cushion - min_stop_distance))
                r_obj[2] = max(speed_ahead, r_obj[2] - decel)
            else:
                r_obj[2] = min(r_obj[7], r_obj[2] + 0.01)

            if r_obj[3] < 0 and side_obstacle_left: r_obj[3] = 0.0
            elif r_obj[3] > 0 and side_obstacle_right: r_obj[3] = 0.0

            r_obj[0] += r_obj[2]
            r_obj[1] += r_obj[3]
            r_obj[1] = np.clip(r_obj[1], c_road + 0.6, c_road + self.road_half_width - 0.8)

            if r_obj[0] > self.av_state[0] + 80.0:
                r_obj[0] = self.av_state[0] - np.random.uniform(40.0, 95.0)
                r_obj[10] = np.random.uniform(1.0, 4.2)
                r_obj[9] = r_obj[10]
                r_obj[1] = self.get_road_center(r_obj[0]) + r_obj[10]
                r_obj[2] = r_obj[7]
                r_obj[8] = 0

            if check_oriented_box_collision(ego_center, ego_dim, self.av_heading, (r_obj[1], r_obj[0]), (r_obj[5], r_obj[6])):
                return obs, -500.0, True, True, False, "Rear Vehicle Impact"

        for idx, o_obj in enumerate(self.oncoming_objects):
            dy_ego, dx_ego = o_obj[0] - self.av_state[0], abs(o_obj[1] - self.av_state[1])
            collision_risk = (0.0 < dy_ego < 35.0) and (dx_ego < 2.5)

            if collision_risk:
                if dy_ego < 8.0:
                    o_obj[2] = 0.0
                else:
                    o_obj[2] = max(0.0, o_obj[2] - 0.12)
                o_center = self.get_road_center(o_obj[0])
                for c_off in [-5.5, -4.0, -2.5]:
                    cand_x = o_center + c_off
                    if self.check_oncoming_route_clear(o_obj[0], cand_x, idx):
                        o_obj[7] = c_off; break
            else:
                o_obj[2] = min(o_obj[8], o_obj[2] + 0.01)

            o_obj[0] -= o_obj[2]
            o_center = self.get_road_center(o_obj[0])

            o_obj[6] += 1
            if o_obj[6] > 80 and not collision_risk and random.random() < 0.05:
                o_obj[6] = 0
                o_obj[7] = np.random.uniform(-5.8, -1.8)

            target_absolute_x = o_center + o_obj[7]
            o_obj[5] = (target_absolute_x - o_obj[1]) * 0.12
            o_obj[1] += o_obj[5]
            o_obj[1] = np.clip(o_obj[1], o_center - (self.road_half_width - 0.5), o_center - 0.3)
            
            if o_obj[0] < self.av_state[0] - 40.0:
                o_obj[0] = self.av_state[0] + np.random.uniform(50.0, 110.0)
                o_obj[7] = np.random.uniform(-5.8, -1.8)
                o_obj[1] = o_center + o_obj[7]
                o_obj[2] = o_obj[8]
            
            if check_oriented_box_collision(ego_center, ego_dim, self.av_heading, (o_obj[1], o_obj[0]), (o_obj[3], o_obj[4])):
                return obs, -500.0, True, True, False, "Oncoming Vehicle Collision"

        reached_finish = False
        if self.av_state[0] >= self.road_length: 
            reward += 300.0 
            done = True
            reached_finish = True
            
        return self._get_obs(), reward, False, False, reached_finish, "None"


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act2 = nn.SiLU()

    def forward(self, x):
        res = x
        out = self.act1(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        return self.act2(out + res)


class MultiAgentPredictiveDQN(nn.Module):
    def __init__(self, state_dim=35, action_dim=6, num_tracked=8, horizon=5):
        super(MultiAgentPredictiveDQN, self).__init__()
        self.horizon = horizon
        self.num_tracked = num_tracked
        
        self.input_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU()
        )
        self.res_block1 = ResBlock(256)
        self.res_block2 = ResBlock(256)

        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, action_dim)
        )

        self.trajectory_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, num_tracked * horizon * 2)
        )

    def forward(self, state):
        x = self.input_layer(state)
        x = self.res_block1(x)
        x = self.res_block2(x)

        values = self.value_stream(x)
        advantages = self.advantage_stream(x)

        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        predicted_trajectories = self.trajectory_head(x)

        return q_values, predicted_trajectories


class ReplayBuffer:
    def __init__(self, capacity=120000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        self.priorities = collections.deque(maxlen=capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done, future_traj):
        self.buffer.append((state, action, reward, next_state, done, future_traj))
        priority = self.max_priority * 2.0 if done > 0 else self.max_priority
        self.priorities.append(priority)

    def sample(self, batch_size):
        N = len(self.buffer)
        if N == 0: return None

        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha
        probs_sum = probs.sum()
        if probs_sum == 0: probs = np.ones(N, dtype=np.float32) / N
        else: probs /= probs_sum

        indices = np.random.choice(N, batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (N * probs[indices]) ** (-self.beta)
        weights /= (weights.max() + 1e-8)

        state, action, reward, next_state, done, future_traj = zip(*samples)
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32),
            np.array(future_traj, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            priority = (abs(err) + 1e-5) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim=35, action_dim=6, lr=0.0005, gamma=0.98, tau=0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.epsilon = 0.08
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_policy = MultiAgentPredictiveDQN(state_dim, action_dim).to(self.device)
        self.q_target = MultiAgentPredictiveDQN(state_dim, action_dim).to(self.device)
        self.q_target.load_state_dict(self.q_policy.state_dict())
        self.q_target.eval()

        self.optimizer = optim.AdamW(self.q_policy.parameters(), lr=lr, weight_decay=1e-4)
        self.huber_loss = nn.SmoothL1Loss(reduction='none')
        self.mse_loss = nn.MSELoss()
        self.memory = ReplayBuffer(capacity=120000)
        self.training_updates = 0

    def save_model(self):
        torch.save(self.q_policy.state_dict(), MODEL_FILE_PATH)

    def load_model(self):
        if not os.path.exists(MODEL_FILE_PATH):
            return False
        try:
            state_dict = torch.load(MODEL_FILE_PATH, map_location=self.device, weights_only=True)
            self.q_policy.load_state_dict(state_dict)
            self.q_target.load_state_dict(state_dict)
            self.q_target.eval()
            print(f"Loaded trained checkpoint: {MODEL_FILE_PATH}")
            return True
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            print(f"Checkpoint load failed; starting fresh: {error}")
            return False

    def get_action_and_predictions(self, state):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.q_policy.eval()
        with torch.no_grad():
            q_values, pred_trajs = self.q_policy(state_t)
            
        action = random.randint(0, self.action_dim - 1) if random.random() < self.epsilon else int(torch.argmax(q_values, dim=1).item())
        pred_trajs_np = pred_trajs.squeeze(0).cpu().numpy().reshape(8, 5, 2)
        return action, pred_trajs_np

    def update_target_network(self, soft=True):
        if soft:
            for target_param, policy_param in zip(self.q_target.parameters(), self.q_policy.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
        else:
            self.q_target.load_state_dict(self.q_policy.state_dict())

    def update_bubble_rates_from_batch(self, rewards):
        collision_rate = float(np.mean(rewards < -100.0))
        safe_rate = float(np.mean(rewards > 0.0))
        target_forward_rate = float(np.clip(5.0 + collision_rate * 3.0 - safe_rate * 0.5, 2.0, 9.0))
        target_rear_rate = float(np.clip(1.5 + collision_rate * 1.5 - safe_rate * 0.2, 0.5, 4.0))
        blend = 0.02
        learned_bubble_config["forward_rate"] = ((1.0 - blend) * learned_bubble_config["forward_rate"] +
                                                  blend * target_forward_rate)
        learned_bubble_config["rear_rate"] = ((1.0 - blend) * learned_bubble_config["rear_rate"] +
                                               blend * target_rear_rate)
        self.training_updates += 1
        if self.training_updates % 50 == 0:
            save_bubble_config(learned_bubble_config)

    def update_bubble_rates_from_collision(self):
        collision_step = 0.15
        learned_bubble_config["forward_rate"] = float(np.clip(
            learned_bubble_config["forward_rate"] + collision_step,
            2.0,
            9.0,
        ))
        learned_bubble_config["rear_rate"] = float(np.clip(
            learned_bubble_config["rear_rate"] + collision_step * 0.5,
            0.5,
            4.0,
        ))
        save_bubble_config(learned_bubble_config)

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size: return 0.0
        sample_batch = self.memory.sample(batch_size)
        if sample_batch is None: return 0.0

        states, actions, rewards, next_states, dones, future_trajs, indices, weights = sample_batch

        s_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        a_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        r_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        sn_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        d_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        traj_gt = torch.tensor(future_trajs, dtype=torch.float32).to(self.device)
        w_t = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.q_policy.train()
        q_eval, pred_trajs = self.q_policy(s_t)
        q_eval = q_eval.gather(1, a_t)

        with torch.no_grad():
            q_next_policy, _ = self.q_policy(sn_t)
            best_actions = q_next_policy.argmax(dim=1, keepdim=True)
            q_next_target, _ = self.q_target(sn_t)
            q_next = q_next_target.gather(1, best_actions)
            q_target = r_t + (1.0 - d_t) * self.gamma * q_next

        td_errors = (q_eval - q_target).detach().cpu().numpy().squeeze()
        q_loss = (w_t * self.huber_loss(q_eval, q_target)).mean()
        traj_loss = self.mse_loss(pred_trajs, traj_gt)
        total_loss = q_loss + 0.5 * traj_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.q_policy.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors)
        self.update_bubble_rates_from_batch(rewards)
        self.update_target_network(soft=True)
        return float(total_loss.item())

    def load_all_datasets_and_train(self, search_paths):
        collected_files = []
        seen_files = set()
        for path in search_paths:
            if os.path.isdir(path):
                for f in os.listdir(path):
                    if f.endswith('.csv') or f.endswith('.xls') or f.endswith('.xlsx'):
                        filepath = os.path.abspath(os.path.join(path, f))
                        if filepath not in seen_files:
                            collected_files.append(filepath)
                            seen_files.add(filepath)
            elif os.path.isfile(path):
                filepath = os.path.abspath(path)
                if filepath not in seen_files:
                    collected_files.append(filepath)
                    seen_files.add(filepath)

        bc_states, bc_actions, total_samples = [], [], 0

        for filepath in collected_files:
            try:
                df = pd.read_csv(filepath, on_bad_lines='skip') if filepath.endswith('.csv') else pd.read_excel(filepath)
                df.columns = [c.strip() for c in df.columns]
                
                action_col = 'Action' if 'Action' in df.columns else 'Human_Action'
                req_cols = ['Y_Pos', 'Heading', 'Vx', 'Front_Dist', 'Left_Dist', 'Right_Dist', action_col]
                if not all(col in df.columns for col in req_cols):
                    continue

                records = df.to_dict('records')
                for i in range(len(records) - 1):
                    if total_samples >= REPLAY_DATASET_LIMIT:
                        break
                    r_curr, r_next = records[i], records[i+1]

                    try:
                        s_curr = [
                            float(r_curr['Y_Pos']), float(r_curr['Heading']), float(r_curr['Vx']),
                            float(r_curr['Front_Dist']), float(r_curr['Left_Dist']), float(r_curr['Right_Dist'])
                        ] + [0.0] * 29
                        s_next = [
                            float(r_next['Y_Pos']), float(r_next['Heading']), float(r_next['Vx']),
                            float(r_next['Front_Dist']), float(r_next['Left_Dist']), float(r_next['Right_Dist'])
                        ] + [0.0] * 29
                        action = int(float(r_curr[action_col]))
                    except (TypeError, ValueError):
                        continue

                    if not 0 <= action < self.action_dim:
                        continue

                    if len(bc_states) < BC_SAMPLE_LIMIT:
                        bc_states.append(s_curr)
                        bc_actions.append(action)

                    reward = s_curr[2] * 0.5 - abs(s_curr[0]) * 0.1
                    done = 1.0 if (s_curr[3] < 2.5 or abs(s_curr[0]) > 5.5) else 0.0

                    dummy_traj = np.zeros((8 * 5 * 2,), dtype=np.float32)
                    self.memory.push(s_curr, action, reward, s_next, done, dummy_traj)
                    total_samples += 1

                if total_samples >= REPLAY_DATASET_LIMIT:
                    break

            except Exception as e:
                print(f"Skipping dataset parsing error for {filepath}: {e}")

        print(f"✅ Pre-loaded {total_samples} samples into Multi-Agent Prioritized Replay Buffer.")

        if len(bc_states) > 100:
            print("🎓 Phase 1: Pre-training Behavioral Cloning & Motion Predictor...")
            bc_s_tensor = torch.tensor(bc_states, dtype=torch.float32).to(self.device)
            bc_a_tensor = torch.tensor(bc_actions, dtype=torch.long).to(self.device)

            dataset = torch.utils.data.TensorDataset(bc_s_tensor, bc_a_tensor)
            loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

            ce_criterion = nn.CrossEntropyLoss()
            bc_optimizer = optim.AdamW(self.q_policy.parameters(), lr=0.001, weight_decay=1e-4)

            self.q_policy.train()
            for ep in range(BC_EPOCHS):
                total_bc_loss = 0.0
                for batch_s, batch_a in loader:
                    q_logits, _ = self.q_policy(batch_s)
                    bc_loss = ce_criterion(q_logits, batch_a)
                    bc_optimizer.zero_grad()
                    bc_loss.backward()
                    bc_optimizer.step()
                    total_bc_loss += bc_loss.item()

            print("✅ Behavioral Cloning pre-training completed successfully!")
            self.save_model()


def draw_vehicle_shape(w_px, h_px, color, v_type="car", heading=0.0):
    w_px, h_px = max(8, w_px), max(16, h_px)
    surface = pygame.Surface((w_px, h_px), pygame.SRCALPHA)
    
    if v_type == 0 or v_type == "ego":
        pygame.draw.rect(surface, color, (0, 0, w_px, h_px), border_radius=int(w_px*0.25))
        pygame.draw.rect(surface, (40, 45, 60), (int(w_px*0.12), int(h_px*0.22), int(w_px*0.76), int(h_px*0.15)), border_radius=2)
        pygame.draw.rect(surface, (40, 45, 60), (int(w_px*0.15), int(h_px*0.72), int(w_px*0.7), int(h_px*0.1)), border_radius=2)
        pygame.draw.rect(surface, [max(0, c-40) for c in color[:3]], (int(w_px*0.12), int(h_px*0.37), int(w_px*0.76), int(h_px*0.35)))
        pygame.draw.circle(surface, (255, 255, 200), (int(w_px*0.2), 3), 2)
        pygame.draw.circle(surface, (255, 255, 200), (int(w_px*0.8), 3), 2)
        pygame.draw.rect(surface, (220, 20, 20), (int(w_px*0.1), h_px - 3, int(w_px*0.25), 2))
        pygame.draw.rect(surface, (220, 20, 20), (int(w_px*0.65), h_px - 3, int(w_px*0.25), 2))

    elif v_type == 1:
        pts = [(w_px//2, 0), (w_px, int(h_px*0.3)), (w_px, h_px), (0, h_px), (0, int(h_px*0.3))]
        pygame.draw.polygon(surface, color, pts)
        pygame.draw.rect(surface, (30, 30, 35), (int(w_px*0.1), int(h_px*0.2), int(w_px*0.8), int(h_px*0.75)), border_radius=3)
        pygame.draw.rect(surface, (240, 200, 0), (int(w_px*0.15), int(h_px*0.35), int(w_px*0.7), int(h_px*0.25)))
        pygame.draw.circle(surface, (255, 255, 180), (w_px//2, 2), 3)

    elif v_type == 2:
        pygame.draw.rect(surface, (160, 160, 170), (0, int(h_px*0.2), w_px, int(h_px*0.8)), border_radius=2)
        pygame.draw.rect(surface, color, (0, 0, w_px, int(h_px*0.18)), border_radius=2)
        pygame.draw.rect(surface, (20, 30, 40), (int(w_px*0.1), int(h_px*0.04), int(w_px*0.8), int(h_px*0.06)))
        pygame.draw.rect(surface, (200, 200, 200), (0, int(h_px*0.04), 2, int(h_px*0.05)))
        pygame.draw.rect(surface, (200, 200, 200), (w_px-2, int(h_px*0.04), 2, int(h_px*0.05)))

    elif v_type == 3:
        pygame.draw.rect(surface, (10, 10, 10), (int(w_px*0.35), 0, int(w_px*0.3), h_px)) 
        pygame.draw.ellipse(surface, (40, 40, 40), (0, int(h_px*0.35), w_px, int(h_px*0.35))) 
        pygame.draw.circle(surface, color, (w_px//2, int(h_px*0.48)), int(w_px*0.4)) 
        pygame.draw.line(surface, (200, 200, 200), (0, int(h_px*0.2)), (w_px, int(h_px*0.2)), 2)

    elif v_type == "oncoming":
        pygame.draw.rect(surface, color, (0, 0, w_px, h_px), border_radius=int(w_px*0.25))
        pygame.draw.rect(surface, (40, 45, 60), (int(w_px*0.12), int(h_px*0.63), int(w_px*0.76), int(h_px*0.15)), border_radius=2)
        pygame.draw.circle(surface, (255, 255, 220), (int(w_px*0.2), h_px - 3), 3)
        pygame.draw.circle(surface, (255, 255, 220), (int(w_px*0.8), h_px - 3), 3)

    if heading != 0.0:
        surface = pygame.transform.rotate(surface, -heading)
        
    return surface


sim_env = IndianRoadSimRL(rear_objects=100)
agent = DQNAgent()
dataset_dir = os.path.join(BASE_DIR, 'IRS data')
if FORCE_RETRAIN or not agent.load_model():
    agent.load_all_datasets_and_train([dataset_dir, LOCAL_DIR])

current_state = sim_env.reset(soft_restart=False)
done = False
paused = False
is_bot_mode = True
status_msg = "BOT MODE Active - Press TAB to switch to Human Mode"

fps = 250

collision_timer = 0
collision_point_y = None
forced_human_override = False
state_history = collections.deque(maxlen=3 * fps)

def scale_x(x_val):
    return int((x_val + 9.0) / 18.0 * WIDTH)

def scale_y(y_val, av_y):
    rel_y = y_val - (av_y - 30)
    return int(HEIGHT - (rel_y / 60.0 * HEIGHT))

def meters_to_pixels_scale():
    return WIDTH / 18.0, HEIGHT / 60.0


def draw_dual_bubbles(surface, center_px, dual_profile, scale_x_px, scale_y_px, heading):
    cx, cy = center_px
    heading_rad = np.radians(heading)
    dir_x, dir_y = np.sin(heading_rad), -np.cos(heading_rad)
    side_x, side_y = np.cos(heading_rad), np.sin(heading_rad)

    def draw_single_bubble(dist, width, color_fill, color_outline):
        width*=1
        f_px = dist * scale_y_px
        r_px = (sim_env.av_dim[1] / 2.0 + 0.35) * scale_y_px
        half_w_px = (width / 2.0) * scale_x_px

        front = (cx + dir_x * f_px, cy + dir_y * f_px)
        rear = (cx - dir_x * r_px, cy - dir_y * r_px)
        pts = [
            (int(front[0] + side_x * half_w_px), int(front[1] + side_y * half_w_px)),
            (int(front[0] - side_x * half_w_px), int(front[1] - side_y * half_w_px)),
            (int(rear[0] - side_x * half_w_px), int(rear[1] - side_y * half_w_px)),
            (int(rear[0] + side_x * half_w_px), int(rear[1] + side_y * half_w_px)),
        ]
        b_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(b_surf, color_fill, pts)
        pygame.draw.lines(b_surf, color_outline, True, pts, 2)
        surface.blit(b_surf, (0, 0))

    b1_fill = (0, 120, 255, 45) if not dual_profile["b1_triggered"] else (0, 200, 255, 90)
    b1_outline = (0, 150, 255, 200) if not dual_profile["b1_triggered"] else (0, 220, 255, 255)
    draw_single_bubble(dual_profile["b1_dist"], dual_profile["b1_width"], b1_fill, b1_outline)

    b2_fill = (255, 0, 0, 40) if not dual_profile["b2_triggered"] else (255, 0, 0, 110)
    b2_outline = (255, 50, 50, 200) if not dual_profile["b2_triggered"] else (255, 0, 0, 255)
    draw_single_bubble(dual_profile["b2_dist"], dual_profile["b2_width"], b2_fill, b2_outline)


def choose_bot_action(proposed_action, pred_trajectories):
    dual_prof = sim_env.get_dual_bubble_profile()
    rear_profile = sim_env.get_rear_bubble_profile()

    if dual_prof["b2_triggered"]:
        bubble_angle = sim_env.find_clear_bubble_angle()
        if bubble_angle is not None and abs(bubble_angle) > 2.0:
            steer_act = 1 if bubble_angle < 0.0 else 4
            return steer_act, f"Bot: BUBBLE 2 EMERGENCY! MAX BRAKE & Steering ({bubble_angle:+.1f}°)"
        check_count = len(sim_env.angle_check_history)
        return 0, f"Bot: BUBBLE 2 EMERGENCY! MAX BRAKE (Verifying {check_count}/3...)"

    if dual_prof["b1_triggered"]:
        bubble_angle = sim_env.find_clear_bubble_angle()
        if bubble_angle is not None:
            if abs(bubble_angle) < 0.5:
                return 3, f"Bot: Bubble 1 Decelerating - Clear Ahead ({bubble_angle:+.1f}°)"
            steer_action = 1 if bubble_angle < 0.0 else 4
            return steer_action, f"Bot: Bubble 1 Decelerating & Rerouting ({bubble_angle:+.1f}°)"
        check_count = len(sim_env.angle_check_history)
        return 0, f"Bot: Bubble 1 Active - Scanning Longest Route ({check_count}/3)..."

    sim_env.angle_check_history.clear()

    if proposed_action in (1, 4):
        side = "left" if proposed_action == 1 else "right"
        if side in rear_profile["blocked_sides"]:
            return 2, f"Bot: steer {side} vetoed by rear bubble"
        target_x = sim_env.av_state[1] + (-2.5 if side == "left" else 2.5)
        road_c = sim_env.get_road_center(sim_env.av_state[0])
        if side == "left" and target_x < road_c:
            return 2, "Bot: Left steer vetoed - prioritizing own lane"
        max_offset = sim_env.road_half_width - sim_env.av_dim[0] / 2.0
        if abs(target_x - road_c) <= max_offset and sim_env.check_lane_clear(sim_env.av_state[0] + 4.0, target_x, sim_env.av_dim[1]):
            return proposed_action, f"Bot: DQN steer {side} validated"
        return 2, "Bot: DQN steer vetoed by lane occupancy"

    if proposed_action in (0, 3, 5):
        road_c = sim_env.get_road_center(sim_env.av_state[0])
        ego_offset = sim_env.av_state[1] - road_c
        if ego_offset < 1.2:
            return 4, "Bot: MOVE - re-centering right into own lane"
        elif ego_offset > 2.8:
            return 1, "Bot: MOVE - re-centering left into own lane"
        return 2, "Bot: MOVE - clear road ahead"

    return proposed_action, "Bot: DQN proposal validated"


action_map = {
    0: (0.0, -BRAKE_DECELERATION),
    1: (-1.0, 0.02),
    2: (0.0, 0.12),
    3: (0.0, -0.12),
    4: (1.0, 0.02),
    5: (0.0, 0.0)
}

running = True
hud_font = pygame.font.SysFont("Consolas", 15, bold=True)
alert_font = pygame.font.SysFont("Consolas", 24, bold=True)

while running:
    screen.fill((30, 30, 30))
    m_scale_x, m_scale_y = meters_to_pixels_scale()

    for y_coord in range(int(sim_env.av_state[0] - 10), int(sim_env.av_state[0] + 60), 2):
        c_x = sim_env.get_road_center(y_coord)
        hw = sim_env.road_half_width
        
        left_bound = scale_x(c_x - hw)
        right_bound = scale_x(c_x + hw)
        left_sidewalk = scale_x(c_x - hw - 1.2)
        right_sidewalk = scale_x(c_x + hw + 1.2)
        center_line = scale_x(c_x)
        scr_y = scale_y(y_coord, sim_env.av_state[0])
        
        if 0 <= scr_y <= HEIGHT:
            pygame.draw.line(screen, (50, 50, 55), (left_bound, scr_y), (right_bound, scr_y), int(m_scale_y * 2))
            pygame.draw.line(screen, (80, 80, 85), (left_sidewalk, scr_y), (left_bound, scr_y), int(m_scale_y * 2))
            pygame.draw.line(screen, (80, 80, 85), (right_bound, scr_y), (right_sidewalk, scr_y), int(m_scale_y * 2))

            for m_zone in sim_env.merge_zones:
                if m_zone["start_y"] <= y_coord <= m_zone["end_y"]:
                    prog = (y_coord - m_zone["start_y"]) / (m_zone["end_y"] - m_zone["start_y"])
                    merge_offset = m_zone["side"] * (hw + 3.5 * (1.0 - prog))
                    merge_scr_x = scale_x(c_x + merge_offset)
                    pygame.draw.circle(screen, (200, 150, 50), (merge_scr_x, scr_y), 3)

            pygame.draw.circle(screen, (255, 255, 255), (left_bound, scr_y), 2)
            pygame.draw.circle(screen, (255, 255, 255), (right_bound, scr_y), 2)
            if y_coord % 4 == 0:
                pygame.draw.circle(screen, (220, 220, 0), (center_line, scr_y), 2)

    for h in sim_env.hazards:
        h_scr_y = scale_y(h[0], sim_env.av_state[0])
        if 0 <= h_scr_y <= HEIGHT:
            h_scr_x = scale_x(sim_env.get_road_center(h[0]) + h[1])
            if h[2] == 'pothole':
                pygame.draw.circle(screen, (15, 15, 15), (h_scr_x, h_scr_y), 8)
            else:
                pygame.draw.rect(screen, (200, 180, 0), (h_scr_x - 15, h_scr_y - 3, 30, 6))

    for dp in sim_env.double_parked:
        dp_scr_y = scale_y(dp[0], sim_env.av_state[0])
        if 0 <= dp_scr_y <= HEIGHT:
            dp_scr_x = scale_x(sim_env.get_road_center(dp[0]) + dp[1])
            w_px, h_px = int(dp[2] * m_scale_x), int(dp[3] * m_scale_y)
            dp_surf = draw_vehicle_shape(w_px, h_px, (120, 120, 120), v_type=0)
            screen.blit(dp_surf, dp_surf.get_rect(center=(dp_scr_x, dp_scr_y)).topleft)

    for animal in sim_env.animals:
        a_scr_y = scale_y(animal[0], sim_env.av_state[0])
        if 0 <= a_scr_y <= HEIGHT:
            a_scr_x = scale_x(sim_env.get_road_center(animal[0]) + animal[1])
            color = (210, 180, 140) if animal[2] == 'cow' else (139, 69, 19)
            radius = 8 if animal[2] == 'cow' else 5
            pygame.draw.circle(screen, color, (a_scr_x, a_scr_y), radius)

    finish_screen_y = scale_y(sim_env.road_length, sim_env.av_state[0])
    if 0 <= finish_screen_y <= HEIGHT:
        c_x_fin = sim_env.get_road_center(sim_env.road_length)
        hw = sim_env.road_half_width
        pygame.draw.line(screen, (0, 255, 0), (scale_x(c_x_fin - hw), finish_screen_y), (scale_x(c_x_fin + hw), finish_screen_y), 6)

    for sig in sim_env.signals:
        sig_scr_y = scale_y(sig["pos"], sim_env.av_state[0])
        if 0 <= sig_scr_y <= HEIGHT:
            sig_color = (0, 255, 0) if sig["state"] == "GREEN" else ((255, 255, 0) if sig["state"] == "YELLOW" else (255, 0, 0))
            c_x_sig = sim_env.get_road_center(sig["pos"])
            pygame.draw.circle(screen, sig_color, (scale_x(c_x_sig - (sim_env.road_half_width + 0.6)), sig_scr_y), 9)

    for ped in sim_env.pedestrians:
        ped_scr_x, ped_scr_y = scale_x(ped[1]), scale_y(ped[0], sim_env.av_state[0])
        if 0 <= ped_scr_y <= HEIGHT:
            body_color = (255, 215, 0) if ped[5] == 1 else ((255, 100, 100) if ped[5] == 2 else (0, 220, 255))
            pygame.draw.circle(screen, body_color, (ped_scr_x, ped_scr_y), 5)

    for obj in sim_env.objects:
        obj_screen_x, obj_screen_y = scale_x(obj[1]), scale_y(obj[0], sim_env.av_state[0])
        if -50 <= obj_screen_y <= HEIGHT + 50:
            color = (255, 200, 0) if obj[4] == 0 else ((255, 60, 60) if obj[4] == 1 else ((180, 100, 255) if obj[4] == 2 else (0, 255, 128)))
            w_px, h_px = int(obj[5] * m_scale_x), int(obj[6] * m_scale_y)
            obj_surf = draw_vehicle_shape(w_px, h_px, color, v_type=obj[4])
            screen.blit(obj_surf, obj_surf.get_rect(center=(obj_screen_x, obj_screen_y)).topleft)

    for r_obj in sim_env.rear_vehicles:
        r_scr_x, r_scr_y = scale_x(r_obj[1]), scale_y(r_obj[0], sim_env.av_state[0])
        if -50 <= r_scr_y <= HEIGHT + 50:
            rear_color = (0, 200, 230) if r_obj[4] == 0 else ((30, 144, 255) if r_obj[4] == 1 else (72, 209, 204))
            w_px, h_px = int(r_obj[5] * m_scale_x), int(r_obj[6] * m_scale_y)
            r_surf = draw_vehicle_shape(w_px, h_px, rear_color, v_type=r_obj[4])
            screen.blit(r_surf, r_surf.get_rect(center=(r_scr_x, r_scr_y)).topleft)

    for o_obj in sim_env.oncoming_objects:
        o_screen_x, o_screen_y = scale_x(o_obj[1]), scale_y(o_obj[0], sim_env.av_state[0])
        if -50 <= o_screen_y <= HEIGHT + 50:
            w_px, h_px = int(o_obj[3] * m_scale_x), int(o_obj[4] * m_scale_y)
            o_surf = draw_vehicle_shape(w_px, h_px, (255, 140, 0), v_type="oncoming")
            screen.blit(o_surf, o_surf.get_rect(center=(o_screen_x, o_screen_y)).topleft)

    av_screen_x, av_screen_y = scale_x(sim_env.av_state[1]), scale_y(sim_env.av_state[0], sim_env.av_state[0])
    av_w_px, av_h_px = int(sim_env.av_dim[0] * m_scale_x), int(sim_env.av_dim[1] * m_scale_y)
    av_surf = draw_vehicle_shape(av_w_px, av_h_px, (0, 255, 255), v_type="ego", heading=sim_env.av_heading)
    
    dual_prof = sim_env.get_dual_bubble_profile()
    draw_dual_bubbles(
        screen,
        (av_screen_x, av_screen_y),
        dual_prof,
        m_scale_x,
        m_scale_y,
        sim_env.av_heading
    )
    screen.blit(av_surf, av_surf.get_rect(center=(av_screen_x, av_screen_y)).topleft)

    bot_action, pred_trajectories = agent.get_action_and_predictions(current_state)
    for entity_idx in range(8):
        trail_color = (255, 0, 255) if entity_idx >= 6 else (255, 255, 0)
        for k_step in range(5):
            dx_pred, dy_pred = pred_trajectories[entity_idx, k_step]
            target_px = scale_x(sim_env.av_state[1] + dx_pred)
            target_py = scale_y(sim_env.av_state[0] + dy_pred, sim_env.av_state[0])
            if 0 <= target_py <= HEIGHT:
                pygame.draw.circle(screen, trail_color, (target_px, target_py), 3 - (k_step // 2))

    speed_txt = hud_font.render(f"Velocity Vx: {sim_env.av_vx:+.2f} m/s", True, (0, 255, 255))
    screen.blit(speed_txt, (10, 10))

    angle_txt = hud_font.render(f"Steering Angle: {sim_env.av_heading:+.1f}°", True, (0, 255, 255))
    screen.blit(angle_txt, (10, 28))

    mode_str = "HUMAN (FORCED OVERRIDE)" if forced_human_override else ("BOT MODE" if is_bot_mode else "HUMAN MODE")
    mode_txt = hud_font.render(f"Control Mode: {mode_str}", True, (255, 255, 0) if forced_human_override else (0, 255, 0))
    screen.blit(mode_txt, (10, 46))

    b1_str = "TRIGGERED (Rerouting)" if dual_prof["b1_triggered"] else "Clear"
    b2_str = "CRITICAL (MAX BRAKE)" if dual_prof["b2_triggered"] else "Clear"
    b_txt = hud_font.render(f"Bubble 1 (Blue): {b1_str} | Bubble 2 (Red): {b2_str}", True, (255, 100, 100) if dual_prof["b2_triggered"] else (255, 255, 255))
    screen.blit(b_txt, (10, 64))

    status_txt = hud_font.render(f"Status: {status_msg}", True, (255, 255, 255))
    screen.blit(status_txt, (10, 82))

    if forced_human_override and collision_point_y is not None:
        prog_txt = hud_font.render(f"Override Active: Pass Y = {collision_point_y:.1f} (Curr Y: {sim_env.av_state[0]:.1f})", True, (255, 180, 0))
        screen.blit(prog_txt, (10, 100))

    if collision_timer > 0:
        time_left_sec = int(np.ceil(collision_timer / fps))
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((150, 0, 0, 120))
        screen.blit(overlay, (0, 0))

        alert_txt = alert_font.render("COLLISION DETECTED!", True, (255, 255, 255))
        timer_txt = alert_font.render(f"Human Takeover in {time_left_sec}s...", True, (255, 255, 0))
        screen.blit(alert_txt, alert_txt.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 20)))
        screen.blit(timer_txt, timer_txt.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 20)))

        collision_timer -= 1
        if collision_timer == 0:
            is_bot_mode = False
            forced_human_override = True
            status_msg = f"Human Control Active until Y > {collision_point_y:.1f}"

    human_action = 5
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            flush_csv_buffer()
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_TAB and not forced_human_override:
                is_bot_mode = not is_bot_mode
                status_msg = "BOT MODE Active - Press TAB to switch" if is_bot_mode else "HUMAN MODE Active - Press TAB to switch"
            elif event.key == pygame.K_p:
                paused = not paused
                status_msg = "Simulation Paused." if paused else "Simulation Resumed."
            elif done and event.key == pygame.K_RETURN:  
                current_state = sim_env.reset(soft_restart=False)
                state_history.clear()
                forced_human_override = False
                collision_point_y = None
                done = False
                status_msg = "New Run Started!"

    keys = pygame.key.get_pressed()
    if not is_bot_mode:
        if keys[pygame.K_SPACE]: human_action = 0      
        elif keys[pygame.K_DOWN]: human_action = 3     
        elif keys[pygame.K_UP]: human_action = 2       
        elif keys[pygame.K_LEFT]: human_action = 1     
        elif keys[pygame.K_RIGHT]: human_action = 4    
        else: human_action = 5

    if forced_human_override and collision_point_y is not None:
        if sim_env.av_state[0] > collision_point_y + 1.0:
            forced_human_override = False
            is_bot_mode = True
            collision_point_y = None
            status_msg = "Cleared collision point! Resuming BOT MODE."

    if not paused and not done and collision_timer == 0:
        state_history.append(copy.deepcopy(sim_env))

        if is_bot_mode:
            action_to_take, bot_status = choose_bot_action(bot_action, pred_trajectories)
            status_msg = bot_status
        else:
            action_to_take = human_action

        steer_dir, throttle_change = action_map.get(action_to_take, (0.0, 0.0))
        next_state, reward, is_done, is_collision, reached_finish, reason = sim_env.step_with_inputs(steer_dir, throttle_change, action_code=action_to_take)

        # Efficient Real-Time CSV Data Logging
        front_dist, left_dist, right_dist = sim_env.get_sensor_distances()
        log_sample_to_csv(
            y_pos=sim_env.av_state[0],
            heading=sim_env.av_heading,
            vx=sim_env.av_vx,
            front_d=front_dist,
            left_d=left_dist,
            right_d=right_dist,
            mode="BOT" if is_bot_mode else "HUMAN",
            bot_act=bot_action,
            human_act=human_action,
            is_mistake=is_collision,
            is_collision=is_collision,
            reason=reason
        )

        if is_collision:
            flush_csv_buffer()
            if is_bot_mode:
                collision_point_y = sim_env.av_state[0]
                if len(state_history) > 0:
                    sim_env = copy.deepcopy(state_history[0])
                state_history.clear()
                current_state = sim_env._get_obs()
                agent.update_bubble_rates_from_collision()
                collision_timer = 3 * fps
                status_msg = f"Collision Triggered ({reason})! Restored state to 3s earlier..."
            else:
                status_msg = f"Human Collision ({reason})! Press ENTER to restart."
                done = True
        else:
            dummy_traj = sim_env.get_ground_truth_future_trajectories()
            agent.memory.push(current_state, action_to_take, reward, next_state, float(is_done), dummy_traj)
            
            # Efficient RL Optimization Step
            agent.train_step(batch_size=64)
            current_state = next_state

            if reached_finish:
                flush_csv_buffer()

    pygame.display.flip()
    clock.tick(fps)

flush_csv_buffer()
pygame.quit()