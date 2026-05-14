import gymnasium as gym
import highway_env
import random
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from very_aggressive import VeryAggressiveVehicle  # noqa: F401


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
POS_SCALE    = 100.0
SPEED_SCALE  =  40.0
TARGET_SPEED =  33.0        # ← raised from 29.0
LANES_COUNT  =     3
NORM_LANE    = 0.05

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "front_dist    (front d / POS_SCALE)",
    "front_closing (ego - front vx) / SPEED_SCALE",
    "left_clear    (left_front d / POS_SCALE)",
    "ego_speed     (v / SPEED_SCALE)",
    "right_clear   (right_front d / POS_SCALE)",
]

# ──────────────────────────────────────────────────────────────────────────────
# BINS — unchanged from previous calibration
# ──────────────────────────────────────────────────────────────────────────────
BINS_PER_FEATURE = [
    [0.04, 0.08, 0.13, 0.20, 0.95],   # front_dist
    [0.05, 0.45, 0.62, 0.76, 0.90],   # front_closing
    [0.15, 0.22, 0.30, 0.42, 0.95],   # left_clear
    [0.54, 0.64, 0.75, 0.86, 0.90],   # ego_speed — recalibrated from Stage 1 diagnostic (p10/p25/p50/p75/p90)
    [0.15, 0.22, 0.30, 0.42, 0.95],   # right_clear
]

# State space: 6 bins × 5 features × 3 lanes = 9375
MAX_POSSIBLE_STATES = (5 ** 5) * 3

TOP_N      = 6
EMPTY_DIST = 999.0

# ──────────────────────────────────────────────────────────────────────────────
# CURRICULUM SCHEDULE — each tuple: (max_episode, density, vehicles_count)
#
#   Stage 1 (ep    1–3000):  light traffic — agent learns basic survival
#   Stage 2 (ep 3001–8000):  moderate traffic — consolidate lane-change policy
#   Stage 3 (ep 8001–13000): dense traffic — stress-test
#   Stage 4 (ep 13001–20000): full aggression — final polish
# ──────────────────────────────────────────────────────────────────────────────
CURRICULUM = [
    (3000,  0.5,  5),
    (8000,  1.0,  8),
    (13000, 1.5, 12),
    (20000, 2.0, 15),
]

# Mid-traffic spawn only kicks in after this episode (agent needs basic survival first)
HARD_SPAWN_START = 5000


# ──────────────────────────────────────────────────────────────────────────────
# OBSERVATION PARSER
# ──────────────────────────────────────────────────────────────────────────────
def _compute_ttc(d, v_closing):
    if v_closing <= 0 or d <= 0:
        return float('inf')
    return d / v_closing


def parse_obs_with_filter(obs, ego_speed):
    candidates = []

    for row in obs[1:]:
        presence, x_n, y_n, vx_n, vy_n = row
        if presence < 0.5:
            continue

        vx = vx_n * SPEED_SCALE
        vy = vy_n * SPEED_SCALE
        x  = x_n  * POS_SCALE
        y  = y_n  * POS_SCALE
        d  = math.sqrt(x ** 2 + y ** 2)

        same_lane  = abs(y_n) <  NORM_LANE
        left_lane  = y_n      < -NORM_LANE
        right_lane = y_n      >  NORM_LANE
        ahead      = x_n > 0

        if same_lane:
            slot = "front" if ahead else "rear"
        elif left_lane:
            slot = "left_front" if ahead else "left_rear"
        elif right_lane:
            slot = "right_front" if ahead else "right_rear"
        else:
            continue

        v_closing = (ego_speed - vx) if ahead else (vx - ego_speed)
        ttc       = _compute_ttc(d, v_closing)

        candidates.append({
            "slot": slot,
            "x": x, "y": y, "vx": vx, "vy": vy, "d": d,
            "ttc": ttc,
        })

    candidates.sort(key=lambda c: (
        c["ttc"] if c["ttc"] != float('inf') else 1e9,
        c["d"],
    ))
    candidates = candidates[:TOP_N]

    slots = {s: {"x": 0.0, "y": 0.0, "vx": ego_speed, "vy": 0.0, "d": EMPTY_DIST}
             for s in ("front", "rear", "left_front", "left_rear",
                       "right_front", "right_rear")}
    filled = set()
    for c in candidates:
        slot = c["slot"]
        if slot not in filled:
            slots[slot] = {k: c[k] for k in ("x", "y", "vx", "vy", "d")}
            filled.add(slot)

    return slots


def _get_ego_lane(env) -> int:
    try:
        return int(env.unwrapped.vehicle.lane_index[2])
    except Exception:
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION CLASS
# ──────────────────────────────────────────────────────────────────────────────
class HighwaySimulation:
    def __init__(self, render=False, bins_per_feature=None):
        self._render                  = render
        self._current_curriculum_idx  = -1   # forces first _rebuild_env call
        self.env                      = None

        # Build env at Stage 1 config
        self._rebuild_env(episode=1)

        self.lr    = 0.1
        self.gamma = 0.99

        self.epsilon       = 1.0
        self.epsilon_min   = 0.05
        

        self.n_actions   = self.env.action_space.n
        self.q_table     = {}
        self.total_steps = 0

        self.ego_speed = 0.0
        self.ego_lane  = 0
        self.neighbors = {}

        self._ep_speeds      = []
        self._ep_ttc_values  = []
        self._ep_overtakes   = 0
        self._prev_ego_speed = 0.0
        self._prev_neighbors = {}
        self._prev_lane      = 0
        self._feature_log    = []

        self.bins_per_feature = bins_per_feature or BINS_PER_FEATURE


    # ──────────────────────────────────────────────────────────────────────────
    # CURRICULUM — ENV FACTORY
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _get_curriculum_idx(episode: int) -> int:
        for i, (upto, _, _) in enumerate(CURRICULUM):
            if episode <= upto:
                return i
        return len(CURRICULUM) - 1

    def _make_env(self, density: float, count: int):
        return gym.make(
            "highway-v0",
            render_mode="human" if self._render else None,
            config={
                "observation": {
                    "type":           "Kinematics",
                    "vehicles_count": 7,
                    "features":       ["presence", "x", "y", "vx", "vy"],
                    "absolute":       False,
                    "normalize":      True,
                },
                "action": {
                    "type":          "DiscreteMetaAction",
                    "target_speeds": [20, 25, 29, 33, 37],
                },
                "other_vehicles_type": "very_aggressive.VeryAggressiveVehicle",
                "speed_limit":          37,    # ← raised from 30 — was capping ego
                "target_speed":         33,    # ← raised from 28
                "vehicles_count":       count,
                "vehicles_density":     density,
                "lanes_count":          LANES_COUNT,
                "duration":             120,
                "simulation_frequency": 15,
                "policy_frequency":      5,
                "ego_spacing":           0.5,
                "initial_spacing":       0.5,
            },
        )

    def _rebuild_env(self, episode: int):
        """
        Close the current env and open a new one if the curriculum stage changed.
        Called at the start of every episode so transitions happen cleanly.
        """
        idx = self._get_curriculum_idx(episode)
        if idx == self._current_curriculum_idx:
            return   # same stage — no rebuild needed

        if self.env is not None:
            self.env.close()

        _, density, count = CURRICULUM[idx]
        self.env                     = self._make_env(density, count)
        self.n_actions               = self.env.action_space.n
        self._current_curriculum_idx = idx

        stage_label = f"Stage {idx + 1}/{len(CURRICULUM)}"
        print(f"\n  ╔══ [Curriculum] {stage_label}: "
              f"density={density}, vehicles={count} ══╗\n")


    # ──────────────────────────────────────────────────────────────────────────
    # MID-TRAFFIC SPAWN (only after HARD_SPAWN_START)
    # ──────────────────────────────────────────────────────────────────────────
    def _randomize_ego_position(self):
        road   = self.env.unwrapped.road
        ego    = self.env.unwrapped.vehicle
        others = [v for v in road.vehicles if v is not ego]

        if not others:
            return self.env.unwrapped.observation_type.observe()

        x_sorted = sorted(v.position[0] for v in others)
        n        = len(x_sorted)

        lo = x_sorted[max(0, n // 4)]
        hi = x_sorted[min(n - 1, 3 * n // 4)]
        if lo >= hi:
            lo, hi = x_sorted[0], x_sorted[-1]

        MIN_GAP   = 12.0
        SAME_LANE =  3.5
        target_x  = random.uniform(lo, hi)

        for _ in range(30):
            too_close = any(
                abs(v.position[0] - target_x) < MIN_GAP and
                abs(v.position[1] - ego.position[1]) < SAME_LANE
                for v in others
            )
            if not too_close:
                break
            target_x = random.uniform(lo, hi)

        ego.position = np.array([target_x, ego.position[1]])

        nearby = sorted(others, key=lambda v: abs(v.position[0] - target_x))[:5]
        if nearby:
            ego.speed = float(np.clip(
                np.mean([v.speed for v in nearby]), 20.0, 37.0
            ))

        return self.env.unwrapped.observation_type.observe()


    # ──────────────────────────────────────────────────────────────────────────
    # STATE UPDATE
    # ──────────────────────────────────────────────────────────────────────────
    def _update_state(self, obs, info):
        self.ego_speed = float(info.get("speed", self.ego_speed))
        self.ego_lane  = _get_ego_lane(self.env)
        self.neighbors = parse_obs_with_filter(obs, self.ego_speed)


    # ──────────────────────────────────────────────────────────────────────────
    # REWARD FUNCTION
    # ──────────────────────────────────────────────────────────────────────────
    def _compute_reward(self, action, done=False):
        nb = self.neighbors
        v  = self.ego_speed

        # ── r_speed ───────────────────────────────────────────────────────────
        r_speed = float(np.clip(
            (v - TARGET_SPEED) / max(TARGET_SPEED, 1.0), -1.0, 1.0
        ))

        # ── r_slow: extra penalty for crawling well below target ──────────────
        # Fires independently of r_speed so slow driving is doubly discouraged.
        r_slow = -0.8 if v < 24.0 else 0.0

        # ── r_ttc: zone-based (replaces unstable log formula) ─────────────────
        # Old log(min_ttc/2) produced -1.4 to -2.8 *every single step*, making
        # it indistinguishable from just existing in traffic.
        # Zones give a clear learnable gradient.
        def ttc(d, v_closing):
            if v_closing <= 0 or d <= 0:
                return float('inf')
            return d / v_closing

        ttc_values = [
            ttc(nb["front"]["d"],       v - nb["front"]["vx"]),
            ttc(nb["left_front"]["d"],  v - nb["left_front"]["vx"]),
            ttc(nb["right_front"]["d"], v - nb["right_front"]["vx"]),
        ]
        min_ttc = min(ttc_values)

        if min_ttc == float('inf'):
            r_ttc = +0.5      # clear lane — reward it
        elif min_ttc > 4.0:
            r_ttc = +0.2      # comfortable gap
        elif min_ttc > 2.0:
            r_ttc =  0.0      # acceptable — neutral
        elif min_ttc > 1.0:
            r_ttc = -0.5      # closing in — warning
        else:
            r_ttc = -1.5      # danger zone

     

        # ── r_idle ────────────────────────────────────────────────────────────
        r_idle = -1.5 if (action == 1 and r_speed < 0) else 0.0

       
       

        # ── r_coll ────────────────────────────────────────────────────────────
        r_coll = -20.0 if done else 0.0

        return (
            4.5 * r_speed          # ← weight raised from 3.0
            + 1.5 * r_ttc          # zone-based signal
            + r_slow               # NEW: explicit slow penalty
            + r_idle
            + r_coll
        )


    # ──────────────────────────────────────────────────────────────────────────
    # STATE DISCRETIZATION — 5 features × 6 bins × 3 lanes = 9375 states
    # ──────────────────────────────────────────────────────────────────────────
    def _discretize_obs(self):
        nb = self.neighbors
        v  = self.ego_speed

        features = [
            np.clip(nb["front"]["d"]         / POS_SCALE,   0.0, 1.0),
            np.clip((v - nb["front"]["vx"])  / SPEED_SCALE, 0.0, 1.0),
            np.clip(nb["left_front"]["d"]    / POS_SCALE,   0.0, 1.0),
            np.clip(v                         / SPEED_SCALE, 0.0, 1.0),
            np.clip(nb["right_front"]["d"]   / POS_SCALE,   0.0, 1.0),
        ]

        self._feature_log.append(features)

        feature_bins = tuple(
            int(np.digitize(f, self.bins_per_feature[i]))
            for i, f in enumerate(features)
        )

        # Lane appended as discrete dimension (0, 1, 2) — no binning needed
        lane_bin = min(self.ego_lane, LANES_COUNT - 1)

        return feature_bins + (lane_bin,)


    # ──────────────────────────────────────────────────────────────────────────
    # FEATURE DISTRIBUTION DIAGNOSIS
    # ──────────────────────────────────────────────────────────────────────────
    def diagnose_feature_distribution(self, filename="feature_distribution.png"):
        if not self._feature_log:
            print("No feature data logged. Run training first.")
            return None

        data = np.array(self._feature_log)

        print("\n" + "=" * 65)
        print("  FEATURE DISTRIBUTION ANALYSIS")
        print(f"  Total samples : {len(data):,}")
        print("=" * 65)

        suggested_bins = []
        for i, name in enumerate(FEATURE_NAMES):
            col = data[:, i]
            p   = np.percentile(col, [0, 10, 25, 50, 75, 90, 100])
            std = col.std()

            print(f"\n  [{i}] {name}")
            print(f"       min={p[0]:.3f}  p10={p[1]:.3f}  p25={p[2]:.3f}  "
                  f"median={p[3]:.3f}  p75={p[4]:.3f}  p90={p[5]:.3f}  max={p[6]:.3f}")
            print(f"       std={std:.4f}")

            edges = np.unique(np.round(p[[1, 2, 3, 4, 5, 6]], 4))
            edges = np.clip(edges, p[0], p[6])
            suggested_bins.append(edges.tolist())
            print(f"       suggested edges : {edges.tolist()}")
            print(f"       current edges   : {self.bins_per_feature[i]}")

        print("\n" + "=" * 65)

        n_features = len(FEATURE_NAMES)
        n_cols     = 3
        n_rows     = math.ceil(n_features / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        fig.suptitle("Feature Distributions vs Current Bin Edges\n"
                     "(red dashes = current edges, green dashes = suggested edges)",
                     fontsize=13, fontweight="bold")
        axes_flat = axes.flatten()

        for i, name in enumerate(FEATURE_NAMES):
            col = data[:, i]
            ax  = axes_flat[i]
            ax.hist(col, bins=80, color="steelblue", alpha=0.7, density=True)
            for b in self.bins_per_feature[i]:
                ax.axvline(b, color="red",   linestyle="--", linewidth=1.2, alpha=0.8)
            for b in suggested_bins[i]:
                ax.axvline(b, color="green", linestyle=":",  linewidth=1.2, alpha=0.9)
            ax.set_title(name, fontsize=9, fontweight="bold")
            ax.set_xlabel("Normalised value", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.tick_params(labelsize=7)

        for j in range(n_features, len(axes_flat)):
            axes_flat[j].set_visible(False)

        axes_flat[0].legend(handles=[
            Line2D([0], [0], color="red",   linestyle="--", label="Current edges"),
            Line2D([0], [0], color="green", linestyle=":",  label="Suggested edges"),
        ], fontsize=7)

        fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {filename}")
        return suggested_bins


    # ──────────────────────────────────────────────────────────────────────────
    # Q-TABLE OPERATIONS
    # ──────────────────────────────────────────────────────────────────────────
    def _get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def _choose_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self._get_q(state)))

    def _update_q(self, state, action, reward, next_state, done):
        current_q  = self._get_q(state)[action]
        max_next_q = 0.0 if done else np.max(self._get_q(next_state))
        target     = reward + self.gamma * max_next_q
        self.q_table[state][action] += self.lr * (target - current_q)

    def diagnose_q_table(self):
        n_states = len(self.q_table)
        if n_states == 0:
            print("Q-table is empty.")
            return
        revisit_ratio = self.total_steps / max(n_states, 1)
        coverage_pct  = (n_states / MAX_POSSIBLE_STATES) * 100.0

        all_q = np.array(list(self.q_table.values()))
        q_max = all_q.max(axis=1)

        print("\n" + "=" * 55)
        print("  Q-TABLE SATURATION DIAGNOSTIC")
        print("=" * 55)
        print(f"  Unique states visited : {n_states:>10,}")
        print(f"  Max possible states   : {MAX_POSSIBLE_STATES:>10,}  (6^5 × 3 lanes)")
        print(f"  State space coverage  : {coverage_pct:>9.1f}%")
        print(f"  Total steps taken     : {self.total_steps:>10,}")
        print(f"  Revisit ratio         : {revisit_ratio:>9.1f}x", end="  ")

        if revisit_ratio < 2:
            print("⚠ CRITICAL: almost no convergence")
        elif revisit_ratio < 10:
            print("△ LOW: partial learning")
        else:
            print("✓ GOOD: Q-values converging")

        print(f"\n  Q-value stats (best action per state):")
        print(f"    mean  : {q_max.mean():>8.3f}")
        print(f"    std   : {q_max.std():>8.3f}")
        print(f"    min   : {q_max.min():>8.3f}")
        print(f"    max   : {q_max.max():>8.3f}")
        print("=" * 55 + "\n")


    # ──────────────────────────────────────────────────────────────────────────
    # DEBUG
    # ──────────────────────────────────────────────────────────────────────────
    def debug_neighbor_params(self):
        print("\n── Neighbour IDM parameter check ──")
        for i, vehicle in enumerate(self.env.unwrapped.road.vehicles[1:5], start=1):
            tw  = getattr(vehicle, "TIME_WANTED",     "N/A")
            dw  = getattr(vehicle, "DISTANCE_WANTED", "N/A")
            acc = getattr(vehicle, "COMFORT_ACC_MAX", "N/A")
            ts  = getattr(vehicle, "target_speed",    "N/A")
            print(f"  Vehicle {i} ({type(vehicle).__name__}): "
                  f"TW={tw:.2f}  DW={dw:.2f}  ACC_MAX={acc:.2f}  "
                  f"v_target={ts:.1f}")
        print()


    # ──────────────────────────────────────────────────────────────────────────
    # METRICS TRACKING
    # ──────────────────────────────────────────────────────────────────────────
    def _reset_episode_metrics(self):
        self._ep_speeds      = []
        self._ep_ttc_values  = []
        self._ep_overtakes   = 0
        self._prev_ego_speed = self.ego_speed
        self._prev_neighbors = {}
        self._prev_lane      = self.ego_lane

    def _step_metrics(self, action):
        nb = self.neighbors
        v  = self.ego_speed

        self._ep_speeds.append(v)

        def ttc(d, v_closing):
            if v_closing <= 0 or d <= 0:
                return float('inf')
            return d / v_closing

        raw_ttc = [
            ttc(nb["front"]["d"],       v - nb["front"]["vx"]),
            ttc(nb["left_front"]["d"],  v - nb["left_front"]["vx"]),
            ttc(nb["right_front"]["d"], v - nb["right_front"]["vx"]),
        ]
        finite_ttc = [t for t in raw_ttc if t != float('inf')]
        if finite_ttc:
            self._ep_ttc_values.append(min(finite_ttc))

        self._prev_neighbors["front_d_before_change"] = nb["front"]["d"]
        self._prev_ego_speed = v
        self._prev_lane      = self.ego_lane

    def _finalize_episode_metrics(self, crashed):
        avg_speed = float(np.mean(self._ep_speeds))     if self._ep_speeds     else 0.0
        avg_ttc   = float(np.mean(self._ep_ttc_values)) if self._ep_ttc_values else float('inf')
        min_ttc   = float(np.min(self._ep_ttc_values))  if self._ep_ttc_values else float('inf')

        return {
            "crashed":   int(crashed),
            "avg_speed": avg_speed,
            "avg_ttc":   avg_ttc,
            "min_ttc":   min_ttc,
            "overtakes": self._ep_overtakes,
        }


    # ──────────────────────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ──────────────────────────────────────────────────────────────────────────
    def save_q_table(self, path="q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {path} ({len(self.q_table)} states)")

    def load_q_table(self, path="q_table.pkl"):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
        print(f"Q-table loaded from {path} ({len(self.q_table)} states)")


    # ──────────────────────────────────────────────────────────────────────────
    # EVALUATION — 100 episodes at ε=0, always at full-traffic config
    # ──────────────────────────────────────────────────────────────────────────
    def _run_evaluation(self, max_steps, total_episodes):
        print("\n=== FINAL EVALUATION: 100 episodes at ε=0 (full traffic) ===")

        # Force full-traffic env for evaluation regardless of current stage
        self._current_curriculum_idx = -1
        self._rebuild_env(episode=CURRICULUM[-1][0])

        saved_epsilon = self.epsilon
        self.epsilon  = 0.0

        survival_steps = []
        crashes        = 0

        for _ in range(100):
            obs, info = self.env.reset()
            obs       = self._randomize_ego_position()
            self._update_state(obs, info)
            state = self._discretize_obs()

            for s in range(max_steps):
                action                             = self._choose_action(state)
                next_obs, _, done, truncated, info = self.env.step(action)

                if info.get("crashed", False):
                    crashes += 1

                self._update_state(next_obs, info)
                state = self._discretize_obs()

                if done or truncated:
                    survival_steps.append(s + 1)
                    break
            else:
                survival_steps.append(max_steps)

        mean_survival = np.mean(survival_steps)
        crash_rate    = crashes / 100

        print(f"  Mean survival steps : {mean_survival:.1f} / {max_steps}")
        print(f"  Crash rate          : {crash_rate:.0%}")
        print(f"  Random baseline est : ~{max_steps * 0.3:.0f} steps")

        if mean_survival < max_steps * 0.4 or crash_rate > 0.7:
            print("  VERDICT: Agent not beating random. Move to DQN.")
        else:
            print("  VERDICT: Agent shows meaningful learning signal.")

        self.epsilon = saved_epsilon


    # ──────────────────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ──────────────────────────────────────────────────────────────────────────
    def train(self, num_episodes=1000, max_steps=600, start_episode=0, total_episodes=20000):
        EXPLORATION_PHASE_END = 3000

        print("=== Training started ===")
        print(f"Episodes: {num_episodes} | Max steps: {max_steps}")
        print(f"lr={self.lr} | gamma={self.gamma} | TARGET_SPEED={TARGET_SPEED}")
        print(f"  Phase 1 (ep 1–{EXPLORATION_PHASE_END})   : ε=1.0  (pure exploration)")
        print(f"  Phase 2 (ep {EXPLORATION_PHASE_END}–{total_episodes}): ε linear 1.0 → {self.epsilon_min}")
        print(f"State space: 6^5 × 3 lanes = {MAX_POSSIBLE_STATES} states")
        print(f"Features: front_dist | front_closing | left_clear | ego_speed | right_clear | lane")
        print(f"Hard spawn: enabled after episode {HARD_SPAWN_START}")
        print(f"Curriculum schedule:")
        for upto, density, count in CURRICULUM:
            print(f"  ep ≤ {upto:>5}: density={density}, vehicles={count}")
        print()

        rewards_log = []
        metrics_log = []

        for ep in range(num_episodes):
            actual_ep = start_episode + ep + 1

            # ── Curriculum env update ─────────────────────────────────────────
            self._rebuild_env(actual_ep)

            # ── Epsilon schedule ──────────────────────────────────────────────
            if actual_ep <= EXPLORATION_PHASE_END:
                self.epsilon = 1.0
            else:
                progress     = (actual_ep - EXPLORATION_PHASE_END) / max(total_episodes - EXPLORATION_PHASE_END, 1)
                self.epsilon = max(self.epsilon_min,
                                   1.0 - progress * (1.0 - self.epsilon_min))

            obs, info = self.env.reset()

            # ── Mid-traffic spawn: delayed until HARD_SPAWN_START ─────────────
            if actual_ep > HARD_SPAWN_START:
                obs = self._randomize_ego_position()

            self._reset_episode_metrics()
            self._update_state(obs, info)

            if ep == 0:
                self.debug_neighbor_params()

            if actual_ep % 100 == 0:
                print(f"Ep {actual_ep}/{total_episodes}  |  "
                      f"ε={self.epsilon:.4f}  |  "
                      f"states={len(self.q_table)}/{MAX_POSSIBLE_STATES}  |  "
                      f"curriculum_stage={self._current_curriculum_idx + 1}")

            state        = self._discretize_obs()
            total_reward = 0.0
            crashed      = False

            for step in range(max_steps):
                action                             = self._choose_action(state)
                next_obs, _, done, truncated, info = self.env.step(action)

                if info.get("crashed", False):
                    crashed = True

                self._update_state(next_obs, info)
                self._step_metrics(action)

                reward       = self._compute_reward(action, done=done or truncated)
                total_reward += reward
                next_state   = self._discretize_obs()

                self._update_q(state, action, reward, next_state, done or truncated)
                state = next_state
                self.total_steps += 1

                if done or truncated:
                    break

            rewards_log.append(total_reward)
            metrics_log.append(self._finalize_episode_metrics(crashed))

            if (ep + 1) % 1000 == 0:
                self.diagnose_q_table()
                self.save_q_table()
                print(f"  Checkpoint saved at episode {actual_ep}")

                if actual_ep == EXPLORATION_PHASE_END:
                    self.diagnose_feature_distribution(
                        filename=f"feature_distribution_{actual_ep}k.png"
                    )

                if actual_ep >= total_episodes:
                    self._run_evaluation(max_steps, total_episodes)

        print("\n=== Training complete ===")
        self.diagnose_q_table()
        self.save_q_table()
        self.env.close()
        return rewards_log, metrics_log


    # ──────────────────────────────────────────────────────────────────────────
    # RESUME TRAINING FROM CHECKPOINT
    # ──────────────────────────────────────────────────────────────────────────
    def resume_train(self, total_episodes=20000, max_steps=600, start_episode=0):
        EXPLORATION_PHASE_END = 3000

        self.load_q_table()
        self.total_steps = 0

        if start_episode <= EXPLORATION_PHASE_END:
            self.epsilon = 1.0
            print(f"Resuming from episode {start_episode} | "
                  f"Phase 1: ε=1.0 (pure exploration until ep {EXPLORATION_PHASE_END})")
        else:
            progress     = (start_episode - EXPLORATION_PHASE_END) / max(total_episodes - EXPLORATION_PHASE_END, 1)
            self.epsilon = max(self.epsilon_min,
                               1.0 - progress * (1.0 - self.epsilon_min))
            print(f"Resuming from episode {start_episode} | "
                  f"Phase 2: ε={self.epsilon:.4f}")

        return self.train(
            num_episodes   = total_episodes - start_episode,
            max_steps      = max_steps,
            start_episode  = start_episode,
            total_episodes = total_episodes,
        )


    # ──────────────────────────────────────────────────────────────────────────
    # TESTING LOOP — always runs at full-traffic config
    # ──────────────────────────────────────────────────────────────────────────
    def test(self, num_episodes=5, max_steps=600):
        print("\n=== Testing started ===")

        # Force full-traffic env regardless of what stage __init__ built
        self._current_curriculum_idx = -1
        self._rebuild_env(episode=CURRICULUM[-1][0])

        self.epsilon = 0.0
        metrics_log  = []

        for ep in range(num_episodes):
            print(f"\nTest Episode {ep + 1}/{num_episodes}")
            obs, info = self.env.reset()
            obs       = self._randomize_ego_position()   # always hard spawn in test
            self._reset_episode_metrics()
            self._update_state(obs, info)

            if ep == 0:
                self.debug_neighbor_params()

            state        = self._discretize_obs()
            total_reward = 0.0
            crashed      = False

            for step in range(max_steps):
                action                             = self._choose_action(state)
                next_obs, _, done, truncated, info = self.env.step(action)

                if info.get("crashed", False):
                    crashed = True

                self._update_state(next_obs, info)
                self._step_metrics(action)

                reward       = self._compute_reward(action, done=done or truncated)
                total_reward += reward
                state        = self._discretize_obs()

                if done or truncated:
                    break

            ep_metrics = self._finalize_episode_metrics(crashed)
            metrics_log.append(ep_metrics)

            print(f"  Total reward : {total_reward:.3f}")
            print(f"  Steps        : {step + 1}")
            print(f"  Crashed      : {bool(crashed)}")
            print(f"  Avg speed    : {ep_metrics['avg_speed']:.2f} m/s")
            print(f"  Avg TTC      : {ep_metrics['avg_ttc']:.2f} s")
            print(f"  Min TTC      : {ep_metrics['min_ttc']:.2f} s")

        self.diagnose_q_table()
        self.env.close()
        return metrics_log


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────
def _rolling_mean_std(data, window):
    data  = np.array(data, dtype=float)
    n     = len(data)
    means = np.full(n, np.nan)
    stds  = np.full(n, np.nan)
    for i in range(window - 1, n):
        chunk    = data[i - window + 1 : i + 1]
        means[i] = chunk.mean()
        stds[i]  = chunk.std()
    return means, stds


def _plot_metric(ax, raw, window, ylabel, title, color="royalblue", is_rate=False):
    x           = np.arange(len(raw))
    means, stds = _rolling_mean_std(raw, window)

    ax.plot(x, raw,   alpha=0.2, color=color, label="Per episode")
    ax.plot(x, means, color=color, linewidth=1.8, label=f"Rolling mean ({window} ep)")
    ax.fill_between(x, means - stds, means + stds,
                    alpha=0.2, color=color, label="±1 std")

    if is_rate:
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))

    # Draw vertical lines at curriculum stage transitions
    for upto, density, count in CURRICULUM[:-1]:
        ax.axvline(upto, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Episode", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)


def plot_all(rewards_log, metrics_log, window=50, filename="training_results.png"):
    CAP        = 30.0
    avg_speeds = [m["avg_speed"] for m in metrics_log]
    crashed    = [m["crashed"]   for m in metrics_log]
    avg_ttcs   = [min(m["avg_ttc"], CAP) for m in metrics_log]
    min_ttcs   = [min(m["min_ttc"], CAP) for m in metrics_log]
    x          = np.arange(len(rewards_log))

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Training Dashboard — Rewards & Metrics",
                 fontsize=18, fontweight="bold", y=1.01)
    ax = axes.flatten()

    _plot_metric(ax[0], rewards_log, window,
                 ylabel="Total Reward",
                 title="Episode Reward (Raw + Rolling Mean)",
                 color="steelblue")

    means, stds = _rolling_mean_std(rewards_log, window)
    ax[1].plot(x, means, color="darkorange", linewidth=2,
               label=f"Rolling mean ({window} ep)")
    ax[1].fill_between(x, means - stds, means + stds,
                       alpha=0.25, color="darkorange", label="±1 std")
    for upto, _, _ in CURRICULUM[:-1]:
        ax[1].axvline(upto, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax[1].set_title("Episode Reward (Smoothed)", fontsize=11, fontweight="bold", pad=8)
    ax[1].set_xlabel("Episode", fontsize=9)
    ax[1].set_ylabel("Avg Reward", fontsize=9)
    ax[1].tick_params(labelsize=8)
    ax[1].legend(fontsize=7, loc="best")
    ax[1].grid(True, alpha=0.3)

    metric_specs = [
        (avg_speeds, "Avg Speed (m/s)",          "Average Speed",         "teal",         False),
        (crashed,    "Collision Rate",            "Collision Rate (%)",    "crimson",      True ),
        (avg_ttcs,   f"Avg TTC (s, cap={CAP}s)", "Avg Time-to-Collision", "steelblue",    False),
        (min_ttcs,   f"Min TTC (s, cap={CAP}s)", "Min Time-to-Collision", "mediumpurple", False),
    ]
    for i, (raw, ylabel, title, color, is_rate) in enumerate(metric_specs, start=2):
        _plot_metric(ax[i], raw, window, ylabel=ylabel, title=title,
                     color=color, is_rate=is_rate)

    fig.tight_layout(rect=[0, 0, 1, 0.99], h_pad=3.5, w_pad=2.5)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── OPTION A: Re-run diagnostic if you change spawn logic or env config ───
    # sim_diag = HighwaySimulation(render=False)
    # sim_diag.epsilon = 1.0
    # sim_diag.train(num_episodes=500, max_steps=600, total_episodes=500)
    # sim_diag.diagnose_feature_distribution(filename="feature_distribution_new.png")

    # ── OPTION B: Fresh 20k training run ─────────────────────────────────────
    sim = HighwaySimulation(render=False)
    rewards, metrics = sim.train(
        num_episodes   = 20000,
        max_steps      = 600,
       start_episode  = 0,
        total_episodes = 20000,
    )
    plot_all(rewards, metrics, window=50, filename="training_results.png")

    # ── OPTION C: Resume from checkpoint ─────────────────────────────────────
    # sim = HighwaySimulation(render=False)
    # rewards, metrics = sim.resume_train(
    #     total_episodes = 20000,
    #     max_steps      = 600,
    #     start_episode  = 9000,
    # )
    # plot_all(rewards, metrics, window=50, filename="training_results_resumed.png")

    # ── OPTION D: Test trained policy ────────────────────────────────────────
    #sim_test = HighwaySimulation(render=True)
    #sim_test.load_q_table()
    # test_metrics = sim_test.test(num_episodes=20, max_steps=600)
