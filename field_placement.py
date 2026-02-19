"""
UNIT I - Informed Search: A* Algorithm for Cricket Field Placement
=================================================================
Models fielder positioning as a state-space search problem.
- State     : current fielder configuration on the oval
- Goal      : minimize the probability of ball reaching boundary
- Heuristic : weighted coverage of high-risk shot zones
"""

import math
import random

# ──────────────────────────────────────────────
# CRICKET FIELD ZONES (angle ranges in degrees)
# Centre = (0,0), boundary radius = 1.0
# ──────────────────────────────────────────────
ZONES = {
    "Fine Leg":       {"angle": (150, 190), "risk": 0.6},
    "Square Leg":     {"angle": (100, 150), "risk": 0.8},
    "Mid Wicket":     {"angle": (55,  100), "risk": 0.9},
    "Mid On":         {"angle": (25,   55), "risk": 0.7},
    "Mid Off":        {"angle": (-25,  25), "risk": 0.7},
    "Cover":          {"angle": (-55, -25), "risk": 0.95},
    "Point":          {"angle": (-100,-55), "risk": 0.85},
    "Third Man":      {"angle": (-190,-150),"risk": 0.6},
    "Long On":        {"angle": (10,   40), "risk": 0.5},
    "Long Off":       {"angle": (-40, -10), "risk": 0.5},
    "Deep Mid Wicket":{"angle": (70,  110), "risk": 0.55},
    "Slip":           {"angle": (-200,-160),"risk": 0.75},
}

DELIVERY_ZONE_PROBS = {
    "yorker":    {"Fine Leg":0.4,"Square Leg":0.3,"Mid Wicket":0.2,"Mid On":0.1,"Cover":0.1},
    "bouncer":   {"Square Leg":0.5,"Point":0.4,"Fine Leg":0.3,"Third Man":0.3},
    "full toss": {"Mid On":0.4,"Mid Off":0.4,"Cover":0.5,"Mid Wicket":0.3},
    "off spin":  {"Cover":0.6,"Point":0.5,"Mid Wicket":0.4,"Mid Off":0.3},
    "leg spin":  {"Fine Leg":0.5,"Square Leg":0.5,"Mid Wicket":0.4,"Cover":0.3},
    "outswing":  {"Slip":0.7,"Cover":0.5,"Point":0.4,"Third Man":0.3},
}

def angle_to_xy(angle_deg, radius):
    """Convert polar angle to x,y coordinates on the oval."""
    rad = math.radians(angle_deg)
    return round(math.cos(rad) * radius, 3), round(math.sin(rad) * radius, 3)

def zone_center(zone_name):
    """Get x,y center of a named zone at 65% of boundary."""
    z = ZONES[zone_name]
    mid_angle = (z["angle"][0] + z["angle"][1]) / 2
    return angle_to_xy(mid_angle, 0.65)

def heuristic(fielder_positions, shot_probs):
    """
    A* Heuristic: estimate uncovered run-scoring probability.
    Lower is better (we want to minimize this).
    h(n) = sum of (zone_risk × shot_prob) for uncovered zones
    """
    covered = set()
    for pos in fielder_positions:
        # A fielder covers the zone whose centre it's closest to
        best_zone = None
        best_dist = float('inf')
        for zname in shot_probs:
            cx, cy = zone_center(zname)
            d = math.dist(pos, (cx, cy))
            if d < best_dist:
                best_dist = d
                best_zone = zname
        if best_zone and best_dist < 0.35:
            covered.add(best_zone)

    uncovered_risk = sum(
        shot_probs.get(z, 0) * ZONES[z]["risk"]
        for z in shot_probs if z not in covered
    )
    return round(uncovered_risk, 4)

def astar_field_placement(delivery_type, n_fielders=9):
    """
    A* Search for optimal field placement given a delivery type.

    State   : frozenset of (zone_name) fielder assignments
    g(n)    : number of fielders used so far
    h(n)    : estimated uncovered scoring probability
    f(n)    : g(n) + h(n)  →  minimise

    Returns : list of dicts with fielder name, position, coverage score
    """
    shot_probs = DELIVERY_ZONE_PROBS.get(delivery_type, {})
    if not shot_probs:
        shot_probs = {z: ZONES[z]["risk"] for z in list(ZONES)[:6]}

    # Priority queue: (f_score, g_score, state)
    # state = frozenset of assigned zone names
    import heapq
    open_list = []
    initial = frozenset()
    g0 = 0
    h0 = heuristic([], shot_probs)
    heapq.heappush(open_list, (g0 + h0, g0, initial))
    visited = {}

    best_state = initial
    best_f = float('inf')

    iterations = 0
    while open_list and iterations < 500:
        iterations += 1
        f, g, state = heapq.heappop(open_list)

        if len(state) >= n_fielders:
            if f < best_f:
                best_f = f
                best_state = state
            continue

        if state in visited and visited[state] <= g:
            continue
        visited[state] = g

        # Expand: try adding each unassigned zone
        for zone_name in shot_probs:
            if zone_name in state:
                continue
            new_state = state | {zone_name}
            positions = [zone_center(z) for z in new_state]
            new_g = g + 1
            new_h = heuristic(positions, shot_probs)
            new_f = new_g + new_h
            heapq.heappush(open_list, (new_f, new_g, new_state))

        if f < best_f and len(state) > 0:
            best_f = f
            best_state = state

    # Fill remaining with low-risk zones if needed
    all_zones = list(ZONES.keys())
    assigned = set(best_state)
    for z in all_zones:
        if len(assigned) >= n_fielders:
            break
        if z not in assigned:
            assigned.add(z)

    # Build result
    fielders = []
    standard_names = [
        "Wicket Keeper","Slip","Gully","Point","Cover","Mid Off",
        "Mid On","Mid Wicket","Square Leg","Fine Leg","Long On","Long Off","Deep Mid Wicket"
    ]
    for i, zone in enumerate(list(assigned)[:n_fielders]):
        cx, cy = zone_center(zone)
        prob = shot_probs.get(zone, 0.1)
        fielders.append({
            "name": standard_names[i] if i < len(standard_names) else f"Fielder {i+1}",
            "zone": zone,
            "x": cx,
            "y": cy,
            "coverage": round(prob * ZONES[zone]["risk"], 3),
            "is_key": prob > 0.4
        })

    # Sort by coverage importance
    fielders.sort(key=lambda f: f["coverage"], reverse=True)

    total_risk = sum(shot_probs.get(z, 0) * ZONES[z]["risk"] for z in shot_probs)
    covered_risk = sum(f["coverage"] for f in fielders)
    efficiency = min(100, round((covered_risk / max(total_risk, 0.01)) * 100, 1))

    return {
        "delivery": delivery_type,
        "fielders": fielders,
        "efficiency": efficiency,
        "h_score": round(best_f, 3),
        "iterations": iterations,
        "algorithm": "A* Search",
        "g_cost": len(fielders),
        "h_cost": round(heuristic([zone_center(f["zone"]) for f in fielders], shot_probs), 3)
    }


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    for d in ["yorker", "bouncer", "off spin"]:
        result = astar_field_placement(d)
        print(f"\n[A* Field] Delivery: {d.upper()}")
        print(f"  Efficiency : {result['efficiency']}%")
        print(f"  f(n) score : {result['h_score']}  |  iterations: {result['iterations']}")
        for f in result["fielders"][:4]:
            print(f"  → {f['zone']:20s}  coverage={f['coverage']}  {'★ KEY' if f['is_key'] else ''}")
