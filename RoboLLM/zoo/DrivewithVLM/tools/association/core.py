"""Geometry-only matching. No cooperative annotation is consumed here."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

CLASSES = {**dict.fromkeys(['car', 'truck', 'van', 'bus'], 0),
           'pedestrian': 1, **dict.fromkeys(['bicycle', 'motorcycle', 'cyclist',
           'tricyclist', 'motorcyclist', 'barrowlist'], 2)}


def native_box(row):
    box = np.array([*[float(row['3d_location'][k]) for k in 'xyz'],
                    *[float(row['3d_dimensions'][k]) for k in 'lwh'],
                    float(row['rotation'])])
    if not np.isfinite(box).all() or (box[3:6] <= 0).any():
        raise ValueError('Invalid native box')
    return box


def transform_box(box, matrix):
    out = box.copy()
    out[:3] = matrix[:3, :3] @ box[:3] + matrix[:3, 3]
    direction = matrix[:3, :3] @ [np.cos(box[6]), np.sin(box[6]), 0.]
    out[6] = np.arctan2(direction[1], direction[0])
    return out


def extrapolate(current_world, previous_world, history_dt, age):
    """Return compensated box, world velocity, and validity (planar CV)."""
    out = current_world.copy()
    velocity = np.zeros(3)
    valid = (previous_world is not None and 0 < history_dt <= 1.
             and age >= 0 and np.isfinite([history_dt, age]).all())
    if valid:
        velocity[:2] = (current_world[:2] - previous_world[:2]) / history_dt
        valid = bool(np.isfinite(velocity).all())
    if valid:
        out[:2] += velocity[:2] * age
    return out, velocity, valid


def footprint(box):
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * box[3:5] / 2
    c, s = np.cos(box[6]), np.sin(box[6])
    return Polygon(corners @ np.array([[c, s], [-s, c]]) + box[:2])


def pair_features(vehicle, infra):
    """Yaw-aware footprints and upright-box 3D IoU in ego coordinates."""
    shape = (len(vehicle), len(infra))
    bev, iou3d, distance, size = [np.zeros(shape) for _ in range(4)]
    vp = [footprint(x['box']) for x in vehicle]
    ip = [footprint(x['box']) for x in infra]
    for a, v in enumerate(vehicle):
        x = v['box']
        for b, r in enumerate(infra):
            y = r['box']
            distance[a, b] = np.linalg.norm(x[:2] - y[:2])
            size[a, b] = np.abs(np.log(x[3:6] / y[3:6])).mean()
            if v['class'] != r['class']:
                continue
            intersection = vp[a].intersection(ip[b]).area
            bev[a, b] = intersection / (vp[a].area + ip[b].area - intersection)
            height = max(0., min(x[2]+x[5]/2, y[2]+y[5]/2)
                         - max(x[2]-x[5]/2, y[2]-y[5]/2))
            volume = intersection * height
            iou3d[a, b] = volume / (np.prod(x[3:6]) + np.prod(y[3:6]) - volume)
    return bev, iou3d, distance, size


def match(vehicle, infra, features, method, gates, threshold):
    if not vehicle or not infra:
        return []
    bev, iou3d, distance, size = features
    overlap = iou3d if method == 'C' else bev
    gate = np.array([gates[x['class']] for x in vehicle])[:, None]
    same = np.array([x['class'] for x in vehicle])[:, None] == np.array([x['class'] for x in infra])[None, :]
    allowed = same & (distance <= gate) & (overlap >= threshold)
    cost = 1 - overlap
    if method != 'A':
        cost = cost + distance / gate + size
    # Dummy columns let every vehicle remain unmatched. Big penalties maximize
    # valid matching cardinality first, then minimize geometric cost.
    maximum = float(cost[allowed].max()) if allowed.any() else 1.
    unmatched = (maximum + 1) * (min(len(vehicle), len(infra)) + 1)
    augmented = np.full((len(vehicle), len(infra)+len(vehicle)), unmatched)
    augmented[:, :len(infra)] = np.where(allowed, cost, unmatched*100)
    a, b = linear_sum_assignment(augmented)
    return [(int(i), int(j)) for i, j in zip(a, b)
            if j < len(infra) and allowed[i, j]]
