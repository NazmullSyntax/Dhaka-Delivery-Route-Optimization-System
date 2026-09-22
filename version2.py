import tkinter as tk
from tkinter import ttk
import random
import math
import time
import threading
from collections import deque
from datetime import datetime, timedelta


# ----------------------------------------------------------------------
# DHAKA LOCATION DATABASE
# ----------------------------------------------------------------------
DHAKA_LOCATIONS = {
    "Gulshan":         (72, 25, "residential", (9, 12)),
    "Banani":          (68, 22, "residential", (10, 13)),
    "Dhanmondi":       (35, 48, "residential", (8, 20)),
    "Mirpur":          (30, 15, "residential", (9, 18)),
    "Uttara":          (45, 5,  "residential", (9, 17)),
    "Mohammadpur":     (28, 45, "residential", (9, 19)),
    "Bashundhara":     (75, 18, "residential", (10, 20)),
    "Baridhara":       (75, 22, "residential", (10, 18)),
    "Badda":           (78, 32, "residential", (9, 21)),
    "Rampura":         (72, 35, "residential", (9, 20)),
    "Malibagh":        (62, 55, "residential", (9, 22)),
    "Mugda":           (65, 60, "residential", (9, 21)),
    "Jatrabari":       (70, 75, "residential", (10, 22)),
    "Bonosree":        (80, 40, "residential", (9, 20)),
    "Khilgaon":        (68, 52, "residential", (9, 21)),
    "Shyamoli":        (32, 42, "residential", (9, 19)),
    "Kallyanpur":      (30, 35, "residential", (9, 18)),

    "Motijheel":       (58, 72, "commercial",  (10, 18)),
    "Farmgate":        (48, 40, "commercial",  (9, 21)),
    "Karwan Bazar":    (50, 42, "commercial",  (9, 21)),
    "Paltan":          (58, 68, "commercial",  (10, 18)),
    "Banglamotor":     (52, 38, "commercial",  (9, 21)),
    "Shahbagh":        (50, 50, "commercial",  (9, 22)),
    "New Market":      (42, 52, "commercial",  (9, 22)),
    "Sadarghat":       (62, 82, "commercial",  (10, 20)),
    "Wari":            (62, 78, "commercial",  (10, 19)),
    "Lalbagh":         (45, 75, "commercial",  (9, 19)),
    "Agargaon":        (42, 32, "commercial",  (9, 20)),

    "Tejgaon":         (52, 35, "industrial",  (6, 22)),
    "Chittagong Road": (80, 85, "industrial",  (6, 22)),
    "Hazaribagh":      (38, 58, "industrial",  (7, 21)),
    "Kamrangirchar":   (40, 68, "industrial",  (7, 21)),
}

DEPOTS = [
    ("Tejgaon Hub",       52, 35),
    ("Motijheel Hub",     58, 72),
    ("Mirpur Hub",        30, 15),
]

VEHICLE_ICONS = ["🚚", "🚛", "🛻"]
VEHICLE_COLORS = ["#f59e0b", "#22d3ee", "#a855f7"]

FUEL_COST_PER_KM = 12.5     # BDT
DRIVER_COST_PER_HOUR = 250   # BDT
AVG_SPEED_KMH = 18           # Dhaka avg speed


# ----------------------------------------------------------------------
# HYBRID OPTIMIZER: Simulated Annealing + 2-Opt
# ----------------------------------------------------------------------
class HybridOptimizer:
    """
    Combines:
      - Greedy Nearest-Neighbor construction
      - Simulated Annealing with 2-opt and swap moves
      - Multi-start to escape local minima
    """

    def __init__(self, locations, depot):
        self.locations = locations
        self.depot = depot

    def _dist(self, a, b, mult):
        return math.hypot(a[0] - b[0], a[1] - b[1]) * mult

    def _route_length(self, order, mult):
        if not order:
            return 0.0
        total = 0.0
        prev = (self.depot[1], self.depot[2])
        for name in order:
            x, y, *_ = self.locations[name]
            total += self._dist(prev, (x, y), mult)
            prev = (x, y)
        total += self._dist(prev, (self.depot[1], self.depot[2]), mult)
        return total

    def _greedy(self, stops, mult):
        remaining = list(stops)
        order = []
        cur = (self.depot[1], self.depot[2])
        while remaining:
            best = min(remaining,
                       key=lambda n: self._dist(cur,
                                                (self.locations[n][0],
                                                 self.locations[n][1]), mult))
            order.append(best)
            cur = (self.locations[best][0], self.locations[best][1])
            remaining.remove(best)
        return order

    def _two_opt_move(self, order):
        n = len(order)
        if n < 4:
            return order
        i, j = sorted(random.sample(range(1, n), 2))
        return order[:i] + order[i:j][::-1] + order[j:]

    def _swap_move(self, order):
        n = len(order)
        if n < 2:
            return order
        i, j = random.sample(range(n), 2)
        new = order[:]
        new[i], new[j] = new[j], new[i]
        return new

    def _insert_move(self, order):
        n = len(order)
        if n < 3:
            return order
        i, j = random.sample(range(n), 2)
        item = order.pop(i)
        order.insert(j, item)
        return order

    def anneal(self, stops, mult, iterations=6000, t_start=50.0, t_end=0.05):
        """Simulated Annealing with mixed neighborhood moves."""
        current = self._greedy(stops, mult)
        best = current[:]
        best_len = self._route_length(current, mult)
        current_len = best_len

        alpha = (t_end / t_start) ** (1.0 / iterations)
        T = t_start

        for _ in range(iterations):
            move = random.random()
            if move < 0.5:
                candidate = self._two_opt_move(current)
            elif move < 0.85:
                candidate = self._swap_move(current)
            else:
                candidate = self._insert_move(current)

            cand_len = self._route_length(candidate, mult)
            delta = cand_len - current_len

            if delta < 0 or random.random() < math.exp(-delta / T):
                current = candidate
                current_len = cand_len
                if current_len < best_len:
                    best = current[:]
                    best_len = current_len

            T *= alpha

        return best, best_len

    def optimize(self, stops, mult, restarts=3):
        """Multi-start SA and return the best solution found."""
        t0 = time.time()
        greedy = self._greedy(stops, mult)
        greedy_len = self._route_length(greedy, mult)

        best_order = greedy[:]
        best_len = greedy_len

        for _ in range(restarts):
            order, length = self.anneal(stops, mult, iterations=4000)
            if length < best_len:
                best_order = order
                best_len = length

        elapsed = (time.time() - t0) * 1000
        improvement = ((greedy_len - best_len) / greedy_len * 100) if greedy_len else 0

        return {
            "order": best_order,
            "distance": best_len,
            "greedy_distance": greedy_len,
            "improvement": improvement,
            "time_ms": elapsed,
        }


# ----------------------------------------------------------------------
# MAIN APPLICATION
# ----------------------------------------------------------------------
class DhakaRouteOptimizerV2:
    def __init__(self, root):
        self.root = root
        self.root.title("Dhaka Route Optimizer — Multi-Vehicle Edition")
        self.root.geometry("1400x840")
        self.root.configure(bg="#0a0912")
        self.root.minsize(1200, 720)

        # State
        self.running = True
        self.num_vehicles = 3
        self.hour = 10
        self.selected_stops = []
        self.routes = []          # list of dicts: {stops, distance, color, icon}
        self.initial_distance = 0.0
        self.final_distance = 0.0
        self.improvement = 0.0
        self.compute_ms = 0.0
        self.total_fuel_cost = 0.0
        self.total_driver_cost = 0.0
        self.sim_running = False
        self.sim_progress = 0.0
        self.vehicle_positions = []

        self._setup_styles()
        self._build_ui()
        self._randomize_stops(15)

        self._animate()

    # ------------------------------------------------------------------
    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook", background="#0a0912", borderwidth=0)
        style.configure("TNotebook.Tab",
                        background="#1a1530", foreground="#94a3b8",
                        padding=[14, 7], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", "#2d2350")],
                  foreground=[("selected", "#f0abfc")])

    # ------------------------------------------------------------------
    def _build_ui(self):
        # HEADER
        header = tk.Frame(self.root, bg="#030014", height=64)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="⚡  DHAKA ROUTE OPTIMIZER  •  MULTI-VEHICLE EDITION",
                 font=("Segoe UI", 16, "bold"),
                 fg="#f0abfc", bg="#030014").pack(side=tk.LEFT, padx=20)

        self.clock = tk.Label(header, text="", font=("Consolas", 11),
                              fg="#94a3b8", bg="#030014")
        self.clock.pack(side=tk.RIGHT, padx=20)

        # MAIN
        main = tk.Frame(self.root, bg="#0a0912")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # LEFT: Map
        left = tk.Frame(main, bg="#0a0912")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        top_info = tk.Frame(left, bg="#0a0912")
        top_info.pack(fill=tk.X)
        tk.Label(top_info, text="LIVE DHAKA GRID MAP",
                 font=("Segoe UI", 10, "bold"),
                 fg="#94a3b8", bg="#0a0912").pack(side=tk.LEFT)
        self.sim_status = tk.Label(top_info, text="Ready",
                                   font=("Consolas", 9),
                                   fg="#f0abfc", bg="#0a0912")
        self.sim_status.pack(side=tk.RIGHT)

        self.map_canvas = tk.Canvas(left, bg="#0a0912", highlightthickness=0)
        self.map_canvas.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        # RIGHT: Control Panel
        right = tk.Frame(main, bg="#0a0912", width=440)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(12, 0))
        right.pack_propagate(False)

        notebook = ttk.Notebook(right)
        notebook.pack(fill=tk.BOTH, expand=True)

        t1 = tk.Frame(notebook, bg="#131024")
        notebook.add(t1, text="  ⚙ Config  ")
        self._build_config_tab(t1)

        t2 = tk.Frame(notebook, bg="#131024")
        notebook.add(t2, text="  🚚 Routes  ")
        self._build_routes_tab(t2)

        t3 = tk.Frame(notebook, bg="#131024")
        notebook.add(t3, text="  💰 Costs  ")
        self._build_costs_tab(t3)

        # FOOTER
        footer = tk.Frame(self.root, bg="#030014", height=26)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        self.footer = tk.Label(footer, text="Initialized.",
                               font=("Consolas", 9),
                               fg="#475569", bg="#030014")
        self.footer.pack(side=tk.LEFT, padx=12)
        tk.Label(footer, text="HybridOptimizer v2.0 | SA + 2-Opt | Multi-Vehicle",
                 font=("Consolas", 9), fg="#475569",
                 bg="#030014").pack(side=tk.RIGHT, padx=12)

    # ------------------------------------------------------------------
    def _build_config_tab(self, p):
        tk.Label(p, text="NUMBER OF DELIVERY STOPS",
                 font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#131024").pack(anchor=tk.W, padx=14, pady=(16, 4))

        self.stop_count = tk.IntVar(value=15)
        row = tk.Frame(p, bg="#131024")
        row.pack(fill=tk.X, padx=14)
        self.stop_count_lbl = tk.Label(row, text="15",
                                       font=("Consolas", 14, "bold"),
                                       fg="#f0abfc", bg="#131024")
        self.stop_count_lbl.pack(side=tk.RIGHT)
        tk.Scale(row, from_=6, to=30, orient=tk.HORIZONTAL,
                 variable=self.stop_count, showvalue=False,
                 bg="#131024", fg="#f0abfc", troughcolor="#2d2350",
                 highlightthickness=0,
                 command=lambda v: self.stop_count_lbl.config(text=str(int(float(v))))
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(p, text="NUMBER OF VEHICLES",
                 font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#131024").pack(anchor=tk.W, padx=14, pady=(18, 4))

        self.veh_count = tk.IntVar(value=3)
        row2 = tk.Frame(p, bg="#131024")
        row2.pack(fill=tk.X, padx=14)
        self.veh_count_lbl = tk.Label(row2, text="3",
                                      font=("Consolas", 14, "bold"),
                                      fg="#22d3ee", bg="#131024")
        self.veh_count_lbl.pack(side=tk.RIGHT)
        tk.Scale(row2, from_=1, to=3, orient=tk.HORIZONTAL,
                 variable=self.veh_count, showvalue=False,
                 bg="#131024", fg="#22d3ee", troughcolor="#2d2350",
                 highlightthickness=0,
                 command=lambda v: self.veh_count_lbl.config(text=str(int(float(v))))
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(p, text="DEPARTURE HOUR (0-23)",
                 font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#131024").pack(anchor=tk.W, padx=14, pady=(18, 4))

        self.hour_var = tk.IntVar(value=10)
        row3 = tk.Frame(p, bg="#131024")
        row3.pack(fill=tk.X, padx=14)
        self.hour_lbl = tk.Label(row3, text="10:00",
                                 font=("Consolas", 14, "bold"),
                                 fg="#fbbf24", bg="#131024")
        self.hour_lbl.pack(side=tk.RIGHT)
        tk.Scale(row3, from_=0, to=23, orient=tk.HORIZONTAL,
                 variable=self.hour_var, showvalue=False,
                 bg="#131024", fg="#fbbf24", troughcolor="#2d2350",
                 highlightthickness=0,
                 command=self._on_hour).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.traffic_lbl = tk.Label(p, text="Traffic: 1.40x",
                                    font=("Consolas", 10), fg="#94a3b8",
                                    bg="#131024")
        self.traffic_lbl.pack(anchor=tk.W, padx=14, pady=(8, 0))

        tk.Frame(p, bg="#2d2350", height=1).pack(fill=tk.X, padx=14, pady=18)

        self._btn(p, "🎲  RANDOMIZE STOPS", "#4c1d95",
                  lambda: self._randomize_stops(self.stop_count.get()))
        self._btn(p, "⚡  OPTIMIZE (SA + 2-OPT)", "#0e7490",
                  self._optimize)
        self._btn(p, "▶  START DELIVERY", "#7c3aed",
                  self._start_sim)
        self._btn(p, "⏹  RESET", "#334155", self._reset)

        tk.Frame(p, bg="#2d2350", height=1).pack(fill=tk.X, padx=14, pady=18)

        tk.Label(p, text="OPTIMIZATION SUMMARY",
                 font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#131024").pack(anchor=tk.W, padx=14)

        card = tk.Frame(p, bg="#1a1530")
        card.pack(fill=tk.X, padx=14, pady=(4, 14))

        self.summary = {}
        for key, label, color in [
            ("greedy", "Greedy Distance", "#fbbf24"),
            ("optimized", "Optimized Distance", "#22d3ee"),
            ("improve", "Improvement", "#22c55e"),
            ("time", "Compute Time", "#a855f7"),
        ]:
            r = tk.Frame(card, bg="#1a1530")
            r.pack(fill=tk.X, padx=10, pady=3)
            tk.Label(r, text=label, font=("Segoe UI", 9),
                     fg="#94a3b8", bg="#1a1530").pack(side=tk.LEFT)
            v = tk.Label(r, text="—", font=("Consolas", 10, "bold"),
                         fg=color, bg="#1a1530")
            v.pack(side=tk.RIGHT)
            self.summary[key] = v

    def _btn(self, parent, text, color, cmd):
        tk.Button(parent, text=text, font=("Segoe UI", 10, "bold"),
                  bg=color, fg="white", bd=0, pady=10,
                  activebackground=color, activeforeground="white",
                  cursor="hand2", command=cmd).pack(fill=tk.X, padx=14, pady=4)

    def _on_hour(self, val):
        h = int(float(val))
        self.hour = h
        self.hour_lbl.config(text=f"{h:02d}:00")
        self.traffic_lbl.config(text=f"Traffic: {self._traffic():.2f}x")

    def _traffic(self):
        h = self.hour
        if 7 <= h <= 10 or 16 <= h <= 20:
            return 2.2
        elif 11 <= h <= 15:
            return 1.4
        elif 21 <= h <= 23 or 5 <= h <= 6:
            return 0.85
        else:
            return 0.6

    # ------------------------------------------------------------------
    def _build_routes_tab(self, p):
        tk.Label(p, text="ASSIGNED ROUTES",
                 font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#131024").pack(anchor=tk.W, padx=14, pady=(14, 6))

        frame = tk.Frame(p, bg="#131024")
        frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))
        self.route_box = tk.Listbox(frame, bg="#1a1530", fg="#e2e8f0",
                                    font=("Consolas", 10), bd=0,
                                    highlightthickness=0,
                                    selectbackground="#2d2350")
        sb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.route_box.yview)
        self.route_box.configure(yscroll=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.route_box.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    def _build_costs_tab(self, p):
        tk.Label(p, text="OPERATIONAL COST ESTIMATE",
                 font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#131024").pack(anchor=tk.W, padx=14, pady=(14, 6))

        card = tk.Frame(p, bg="#1a1530")
        card.pack(fill=tk.X, padx=14)

        self.cost_widgets = {}
        for key, label, color in [
            ("distance", "Total Distance",  "#22d3ee"),
            ("fuel",     "Fuel Cost",        "#fbbf24"),
            ("driver",   "Driver Cost",      "#a855f7"),
            ("time",     "Est. Duration",    "#22c55e"),
            ("total",    "Total Cost (BDT)", "#f0abfc"),
        ]:
            r = tk.Frame(card, bg="#1a1530")
            r.pack(fill=tk.X, padx=12, pady=5)
            tk.Label(r, text=label, font=("Segoe UI", 9),
                     fg="#94a3b8", bg="#1a1530").pack(side=tk.LEFT)
            v = tk.Label(r, text="—", font=("Consolas", 11, "bold"),
                         fg=color, bg="#1a1530")
            v.pack(side=tk.RIGHT)
            self.cost_widgets[key] = v

        tk.Label(p, text="PER-VEHICLE BREAKDOWN",
                 font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#131024").pack(anchor=tk.W, padx=14, pady=(18, 6))

        self.per_vehicle_canvas = tk.Canvas(p, bg="#1a1530",
                                            highlightthickness=0, height=220)
        self.per_vehicle_canvas.pack(fill=tk.X, padx=14, pady=(0, 14))

    # ------------------------------------------------------------------
    # CONTROL ACTIONS
    # ------------------------------------------------------------------
    def _randomize_stops(self, n):
        n = max(6, min(30, int(n)))
        self.selected_stops = random.sample(list(DHAKA_LOCATIONS.keys()), n)
        self.routes = []
        self.initial_distance = 0.0
        self.final_distance = 0.0
        self.improvement = 0.0
        self.compute_ms = 0.0
        self.total_fuel_cost = 0.0
        self.total_driver_cost = 0.0
        self.sim_running = False
        self.sim_progress = 0.0
        self.vehicle_positions = []
        for v in self.summary.values():
            v.config(text="—")
        for v in self.cost_widgets.values():
            v.config(text="—")
        self._refresh_routes_tab()
        self._draw_map()
        self.footer.config(text=f"Randomized {n} stops.")

    def _split_among_vehicles(self, stops, n_vehicles):
        """Simple round-robin split — could be improved with clustering."""
        buckets = [[] for _ in range(n_vehicles)]
        for i, s in enumerate(stops):
            buckets[i % n_vehicles].append(s)
        return [b for b in buckets if b]

    def _optimize(self):
        if not self.selected_stops:
            return
        mult = self._traffic()
        n_veh = self.veh_count.get()
        buckets = self._split_among_vehicles(self.selected_stops, n_veh)

        self.routes = []
        total_initial = 0.0
        total_final = 0.0
        total_ms = 0.0

        for idx, bucket in enumerate(buckets):
            depot = DEPOTS[idx % len(DEPOTS)]
            optimizer = HybridOptimizer(DHAKA_LOCATIONS, depot)
            result = optimizer.optimize(bucket, mult, restarts=2)

            self.routes.append({
                "stops": result["order"],
                "distance": result["distance"],
                "greedy_distance": result["greedy_distance"],
                "depot": depot,
                "color": VEHICLE_COLORS[idx % len(VEHICLE_COLORS)],
                "icon": VEHICLE_ICONS[idx % len(VEHICLE_ICONS)],
                "improvement": result["improvement"],
            })
            total_initial += result["greedy_distance"]
            total_final += result["distance"]
            total_ms += result["time_ms"]

        self.initial_distance = total_initial
        self.final_distance = total_final
        self.improvement = ((total_initial - total_final) / total_initial * 100
                            ) if total_initial else 0
        self.compute_ms = total_ms

        # Costs
        self.total_fuel_cost = total_final * FUEL_COST_PER_KM
        est_hours = total_final / AVG_SPEED_KMH
        self.total_driver_cost = est_hours * DRIVER_COST_PER_HOUR * len(self.routes)

        self.summary["greedy"].config(text=f"{total_initial:.1f} km")
        self.summary["optimized"].config(text=f"{total_final:.1f} km")
        self.summary["improve"].config(text=f"{self.improvement:.1f}%")
        self.summary["time"].config(text=f"{total_ms:.0f} ms")

        # Cost tab
        self.cost_widgets["distance"].config(text=f"{total_final:.1f} km")
        self.cost_widgets["fuel"].config(text=f"৳ {self.total_fuel_cost:,.0f}")
        self.cost_widgets["driver"].config(text=f"৳ {self.total_driver_cost:,.0f}")
        self.cost_widgets["time"].config(text=f"{est_hours:.2f} hrs")
        self.cost_widgets["total"].config(
            text=f"৳ {self.total_fuel_cost + self.total_driver_cost:,.0f}")

        self._refresh_routes_tab()
        self._draw_map()
        self._draw_per_vehicle()
        self.footer.config(
            text=f"Optimized {len(self.selected_stops)} stops across "
                 f"{len(self.routes)} vehicles — saved {self.improvement:.1f}%"
        )

    def _refresh_routes_tab(self):
        self.route_box.delete(0, tk.END)
        if not self.routes:
            for i, s in enumerate(self.selected_stops, 1):
                self.route_box.insert(tk.END, f"  {i:2d}. {s}")
            return
        for idx, r in enumerate(self.routes):
            self.route_box.insert(
                tk.END,
                f"═══ VEHICLE {idx + 1} {r['icon']}  ({r['distance']:.1f} km) ═══"
            )
            self.route_box.insert(tk.END, f"  🏭 {r['depot'][0]}")
            for i, s in enumerate(r["stops"], 1):
                self.route_box.insert(tk.END, f"    {i:2d}. {s}")
            self.route_box.insert(tk.END, f"  🏭 {r['depot'][0]}")
            self.route_box.insert(tk.END, "")

    def _start_sim(self):
        if not self.routes:
            self._optimize()
        if not self.routes:
            return
        self.sim_running = True
        self.sim_progress = 0.0
        self.vehicle_positions = [
            (r["depot"][1], r["depot"][2]) for r in self.routes
        ]

    def _reset(self):
        self.sim_running = False
        self.sim_progress = 0.0
        self.vehicle_positions = []
        self._draw_map()
        self.footer.config(text="Reset.")

    # ------------------------------------------------------------------
    # MAP RENDERING
    # ------------------------------------------------------------------
    def _scale(self):
        self.map_canvas.update_idletasks()
        w = self.map_canvas.winfo_width()
        h = self.map_canvas.winfo_height()
        if w < 50 or h < 50:
            return None
        pad = 30
        return {"w": w, "h": h, "pad": pad,
                "sx": (w - 2 * pad) / 100.0,
                "sy": (h - 2 * pad) / 100.0}

    def _px(self, x, y, s):
        return (s["pad"] + x * s["sx"], s["pad"] + y * s["sy"])

    def _draw_map(self):
        c = self.map_canvas
        c.delete("all")
        s = self._scale()
        if not s:
            return

        w, h, pad = s["w"], s["h"], s["pad"]

        # Neon grid
        for i in range(0, 101, 5):
            x, _ = self._px(i, 0, s)
            _, y = self._px(0, i, s)
            color = "#1a1530" if i % 10 else "#251e42"
            c.create_line(x, pad, x, h - pad, fill=color)
            c.create_line(pad, y, w - pad, y, fill=color)

        # River
        river = [self._px(18, 92, s), self._px(35, 85, s),
                 self._px(55, 88, s), self._px(78, 80, s), self._px(97, 86, s)]
        c.create_line(*[v for p in river for v in p],
                      fill="#1e3a8a", width=16, smooth=True)
        c.create_line(*[v for p in river for v in p],
                      fill="#3b82f6", width=3, smooth=True)

        # Background stops
        for name, (x, y, zone, _) in DHAKA_LOCATIONS.items():
            if name in self.selected_stops:
                continue
            px, py = self._px(x, y, s)
            c.create_oval(px - 2, py - 2, px + 2, py + 2,
                          fill="#2d2350", outline="")

        # Routes
        for idx, r in enumerate(self.routes):
            depot = r["depot"]
            color = r["color"]
            depot_px = self._px(depot[1], depot[2], s)

            pts = [depot_px]
            for name in r["stops"]:
                x, y, *_ = DHAKA_LOCATIONS[name]
                pts.append(self._px(x, y, s))
            pts.append(depot_px)

            # Glow
            c.create_line(*[v for p in pts for v in p],
                          fill=color, width=8, smooth=True,
                          capstyle=tk.ROUND)
            c.create_line(*[v for p in pts for v in p],
                          fill="#ffffff", width=1, smooth=True,
                          capstyle=tk.ROUND)

            # Stops
            for i, name in enumerate(r["stops"], 1):
                x, y, zone, _ = DHAKA_LOCATIONS[name]
                px, py = self._px(x, y, s)
                c.create_oval(px - 7, py - 7, px + 7, py + 7,
                              fill=color, outline="#0a0912", width=2)
                c.create_text(px, py, text=str(i), fill="#0a0912",
                              font=("Segoe UI", 7, "bold"))

            # Depot
            c.create_oval(depot_px[0] - 12, depot_px[1] - 12,
                          depot_px[0] + 12, depot_px[1] + 12,
                          fill=color, outline="white", width=2)
            c.create_text(depot_px[0], depot_px[1],
                          text=r["icon"], font=("Segoe UI", 12))

        # Vehicle positions
        for idx, pos in enumerate(self.vehicle_positions):
            px, py = self._px(pos[0], pos[1], s)
            color = VEHICLE_COLORS[idx % len(VEHICLE_COLORS)]
            c.create_oval(px - 14, py - 14, px + 14, py + 14,
                          fill=color, outline="white", width=2)
            c.create_text(px, py, text=VEHICLE_ICONS[idx % len(VEHICLE_ICONS)],
                          font=("Segoe UI", 12))

        # Legend
        self._draw_legend(c, w, h)

    def _draw_legend(self, c, w, h):
        lx, ly = 15, h - 90
        c.create_rectangle(lx, ly, lx + 240, ly + 80,
                           fill="#0a0912", outline="#2d2350")
        c.create_text(lx + 10, ly + 12, text="LEGEND",
                      anchor=tk.W, fill="#64748b",
                      font=("Segoe UI", 8, "bold"))
        items = [
            ("#f59e0b", "Vehicle 1"),
            ("#22d3ee", "Vehicle 2"),
            ("#a855f7", "Vehicle 3"),
            ("#3b82f6", "Buriganga river"),
        ]
        for i, (col, lbl) in enumerate(items):
            r, cc = i % 2, i // 2
            x = lx + 12 + cc * 115
            y = ly + 32 + r * 20
            c.create_oval(x, y - 4, x + 8, y + 4, fill=col, outline="")
            c.create_text(x + 14, y, text=lbl, anchor=tk.W,
                          fill="#cbd5e1", font=("Segoe UI", 8))

    # ------------------------------------------------------------------
    def _draw_per_vehicle(self):
        c = self.per_vehicle_canvas
        c.delete("all")
        c.update_idletasks()
        w = c.winfo_width()
        if w < 50 or not self.routes:
            return
        bar_h = 22
        gap = 14
        max_d = max(r["distance"] for r in self.routes)
        y = 10
        for idx, r in enumerate(self.routes):
            c.create_text(10, y + bar_h / 2,
                          text=f"V{idx + 1} {r['icon']}",
                          anchor=tk.W, fill=r["color"],
                          font=("Consolas", 10, "bold"))
            bx = 60
            bw = w - bx - 90
            c.create_rectangle(bx, y, bx + bw, y + bar_h,
                               fill="#2d2350", outline="")
            ratio = r["distance"] / max_d if max_d else 0
            c.create_rectangle(bx, y, bx + bw * ratio, y + bar_h,
                               fill=r["color"], outline="")
            c.create_text(bx + bw + 8, y + bar_h / 2,
                          text=f"{r['distance']:.1f} km",
                          anchor=tk.W, fill="#e2e8f0",
                          font=("Consolas", 9))
            y += bar_h + gap

    # ------------------------------------------------------------------
    # ANIMATION LOOP
    # ------------------------------------------------------------------
    def _animate(self):
        if not self.running:
            return

        self.clock.config(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

        if self.sim_running and self.routes:
            self.sim_progress += 0.008
            if self.sim_progress >= 1.0:
                self.sim_running = False
                self.sim_progress = 1.0
                self.sim_status.config(text="✅ All deliveries complete")
                self.footer.config(text="✅ Simulation complete.")
            else:
                # Advance each vehicle independently
                self.vehicle_positions = []
                for r in self.routes:
                    stops = r["stops"]
                    n = len(stops)
                    segs = n + 1
                    local = min(1.0, self.sim_progress * segs / max(1, segs - 1))
                    seg = local * segs
                    si = int(seg)
                    st = seg - si

                    depot = (r["depot"][1], r["depot"][2])
                    if si == 0:
                        p1, p2 = depot, DHAKA_LOCATIONS[stops[0]][:2]
                    elif si >= n:
                        p1 = DHAKA_LOCATIONS[stops[-1]][:2]
                        p2 = depot
                    else:
                        p1 = DHAKA_LOCATIONS[stops[si - 1]][:2]
                        p2 = DHAKA_LOCATIONS[stops[si]][:2]

                    vx = p1[0] + (p2[0] - p1[0]) * st
                    vy = p1[1] + (p2[1] - p1[1]) * st
                    self.vehicle_positions.append((vx, vy))

                done = int(self.sim_progress * 100)
                self.sim_status.config(text=f"Delivering... {done}%")

        self._draw_map()
        self._draw_per_vehicle()

        self.root.after(40, self._animate)

    # ------------------------------------------------------------------
    def on_close(self):
        self.running = False
        self.root.destroy()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = DhakaRouteOptimizerV2(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()