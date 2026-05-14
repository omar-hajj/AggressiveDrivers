import random
from highway_env.vehicle.behavior import AggressiveVehicle


class VeryAggressiveVehicle(AggressiveVehicle):
    """
    Each instance gets independently randomized IDM + MOBIL parameters,
    producing a realistic spread of aggression levels across traffic.

    Parameter ranges are all more aggressive than highway-env defaults
    but vary enough to avoid a homogeneous fleet.
    """

    def __init__(self, road, position, heading=0, speed=0, target_lane_index=None,
                 target_speed=None, route=None, enable_lane_change=True, timer=None):

        super().__init__(
            road, position, heading, speed,
            target_lane_index=target_lane_index,
            target_speed=target_speed,
            route=route,
            enable_lane_change=enable_lane_change,
            timer=timer,
        )

        # ── IDM longitudinal params ───────────────────────────────────────────
        # Lower TIME_WANTED     = tailgates harder
        self.TIME_WANTED          = random.uniform(0.2, 0.6)
        # Lower DISTANCE_WANTED = accepts tighter gaps
        self.DISTANCE_WANTED      = random.uniform(1.0, 3.0)
        # Higher COMFORT_ACC_MAX = accelerates harder
        self.COMFORT_ACC_MAX      = random.uniform(5.0, 9.0)
        # Symmetric braking — more negative = brakes harder (less cautious)
        self.COMFORT_ACC_MIN      = -self.COMFORT_ACC_MAX

        # ── MOBIL lane-change params ──────────────────────────────────────────
        # Lower threshold = changes lanes more eagerly for tiny gains
        self.LANE_CHANGE_MIN_ACC_GAIN        = random.uniform(0.02, 0.15)
        # Higher = imposes more braking on others when cutting in
        self.LANE_CHANGE_MAX_BRAKING_IMPOSED = random.uniform(5.0, 9.0)

        # ── Target speed ──────────────────────────────────────────────────────
        # Aggressive drivers want to go faster than ego's comfort zone
        if target_speed is None:
            self.target_speed = random.uniform(25.0, 35.0)   # m/s (~90–126 km/h)

        # ── Force instance values onto the class lookup path ──────────────────
        # highway-env's IDM sometimes reads class-level attrs via type(self).X.
        # Shadowing them at instance level ensures per-vehicle randomization
        # is actually used during simulation steps.
        self.__class__ = type(
            "VeryAggressiveVehicle",
            (AggressiveVehicle,),
            {
                "TIME_WANTED":                     self.TIME_WANTED,
                "DISTANCE_WANTED":                 self.DISTANCE_WANTED,
                "COMFORT_ACC_MAX":                 self.COMFORT_ACC_MAX,
                "COMFORT_ACC_MIN":                 self.COMFORT_ACC_MIN,
                "LANE_CHANGE_MIN_ACC_GAIN":        self.LANE_CHANGE_MIN_ACC_GAIN,
                "LANE_CHANGE_MAX_BRAKING_IMPOSED": self.LANE_CHANGE_MAX_BRAKING_IMPOSED,
            },
        )