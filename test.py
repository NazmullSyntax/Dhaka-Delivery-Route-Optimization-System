import tkinter as tk
from tkinter import ttk
import random
import math
import time
import threading
from collections import deque
from datetime import datetime


# ----------------------------------------------------------------------
# DHAKA LOCATION DATABASE (Real areas with approximate coordinates)
# Coordinates are normalized to a 100x100 virtual grid mapped to the canvas
# ----------------------------------------------------------------------
DHAKA_LOCATIONS = {
    # name: (x, y, zone_type)
    "Gulshan":         (72, 25, "residential"),
    "Banani":          (68, 22, "residential"),
    "Dhanmondi":       (35, 48, "residential"),
    "Mirpur":          (30, 15, "residential"),
    "Uttara":          (45, 5,  "residential"),
    "Mohammadpur":     (28, 45, "residential"),
    "Bashundhara":     (75, 18, "residential"),
    "Baridhara":       (75, 22, "residential"),
    "Motijheel":       (58, 72, "commercial"),
    "Farmgate":        (48, 40, "commercial"),
    "Karwan Bazar":    (50, 42, "commercial"),
    "Paltan":          (58, 68, "commercial"),
    "Banglamotor":     (52, 38, "commercial"),
    "Shahbagh":        (50, 50, "commercial"),
    "New Market":      (42, 52, "commercial"),
    "Sadarghat":       (62, 82, "commercial"),
    "Chittagong Road": (80, 85, "industrial"),
    "Tejgaon":         (52, 35, "industrial"),
    "Hazaribagh":      (38, 58, "industrial"),
    "Kamrangirchar":   (40, 68, "industrial"),
    "Badda":           (78, 32, "residential"),
    "Rampura":         (72, 35, "residential"),
    "Malibagh":        (62, 55, "residential"),
    "Mugda":           (65, 60, "residential"),
    "Jatrabari":       (70, 75, "residential"),
    "Bonosree":        (80, 40, "residential"),
    "Khilgaon":        (68, 52, "residential"),
    "Shyamoli":        (32, 42, "residential"),
    "Kallyanpur":      (30, 35, "residential"),
    "Agargaon":        (42, 32, "commercial"),
    "Wari":            (62, 78, "commercial"),
    "Lalbagh":         (45, 75, "commercial"),
}

# Depot (start / end point of every route)
DEPOT = ("Central Warehouse (Tejgaon)", 52, 35)

# Traffic multiplier by zone type and time-of-day
ZONE_TRAFFIC = {
    "residential": 1.0,
    "commercial":  1.6,
    "industrial":  1.2,
}


# ----------------------------------------------------------------------
# ROUTE OPTIMIZER
# ----------------------------------------------------------------------
class RouteOptimizer:
    """
    Two-stage optimizer:
      1. Nearest-neighbor to build an initial route
      2. 2-opt improvement to reduce total distance
    Distance is Euclidian * zone traffic multiplier.
    """

    def __init__(self):
        self.locations = DHAKA_LOCATIONS
        self.depot = DEPOT

    @staticmethod
    def _distance(a, b, traffic_mult=1.0):
        return math.hypot(a[0] - b[0], a[1] - b[1]) * traffic_mult

    def _route_distance(self, order, traffic_mult=1.0):
        if not order:
            return 0.0
        total = 0.0
        prev = (self.depot[1], self.depot[2])
        for name in order:
            x, y, _ = self.locations[name]
            total += self._distance(prev, (x, y), traffic_mult)
            prev = (x, y)
        total += self._distance(prev, (self.depot[1], self.depot[2]), traffic_mult)
        return total

    def nearest_neighbor(self, stops, traffic_mult=1.0):
        remaining = list(stops)
        order = []
        current = (self.depot[1], self.depot[2])
        while remaining:
            best = min(
                remaining,
                key=lambda n: self._distance(
                    current,
                    (self.locations[n][0], self.locations[n][1]),
                    traffic_mult
                )
            )
            order.append(best)
            current = (self.locations[best][0], self.locations[best][1])
            remaining.remove(best)
        return order

    def two_opt(self, order, traffic_mult=1.0, max_iter=100):
        best = order[:]
        best_dist = self._route_distance(best, traffic_mult)
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            for i in range(1, len(best) - 1):
                for j in range(i + 1, len(best)):
                    new_order = best[:i] + best[i:j][::-1] + best[j:]
                    new_dist = self._route_distance(new_order, traffic_mult)
                    if new_dist < best_dist - 1e-6:
                        best = new_order
                        best_dist = new_dist
                        improved = True
        return best, best_dist

    def optimize(self, stops, traffic_mult=1.0):
        t0 = time.time()
        nn = self.nearest_neighbor(stops, traffic_mult)
        nn_dist = self._route_distance(nn, traffic_mult)
        optimized, opt_dist = self.two_opt(nn, traffic_mult)
        elapsed = (time.time() - t0) * 1000
        improvement = ((nn_dist - opt_dist) / nn_dist * 100) if nn_dist > 0 else 0
        return {
            "order": optimized,
            "distance": opt_dist,
            "nn_distance": nn_dist,
            "improvement": improvement,
            "time_ms": elapsed,
        }


# ----------------------------------------------------------------------
# MAIN APPLICATION
# ----------------------------------------------------------------------
class DhakaDeliveryOptimizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Dhaka Delivery Route Optimization System")
        self.root.geometry("1350x800")
        self.root.configure(bg="#0a0f1e")
        self.root.minsize(1150, 700)

        self.optimizer = RouteOptimizer()
        self.running = True

        # State
        self.selected_stops = []
        self.current_route = []
        self.route_distance = 0.0
        self.initial_distance = 0.0
        self.improvement_pct = 0.0
        self.opt_time_ms = 0.0
        self.simulation_progress = 0.0
        self.simulating = False
        self.vehicle_pos = None
        self.zone_counts = {}
        self.time_of_day = 12  # 0-23
        self.delivery_stats = {
            "delivered": 0, "pending": 0, "total_distance": 0.0
        }

        self._setup_styles()
        self._build_ui()

        # Auto-select 12 random stops to start
        self._randomize_stops(12)

        # Animation loop
        self._animate()

    # ------------------------------------------------------------------
    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook", background="#0a0f1e", borderwidth=0)
        style.configure("TNotebook.Tab",
                        background="#1a2340", foreground="#94a3b8",
                        padding=[14, 7], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", "#2d3a5e")],
                  foreground=[("selected", "#e2e8f0")])
        style.configure("TCombobox",
                        fieldbackground="#1a2340", background="#1a2340",
                        foreground="#e2e8f0", arrowcolor="#94a3b8")

    # ------------------------------------------------------------------
    def _build_ui(self):
        # ===== HEADER =====
        header = tk.Frame(self.root, bg="#020617", height=62)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="🚚  DHAKA DELIVERY ROUTE OPTIMIZER",
                 font=("Segoe UI", 17, "bold"),
                 fg="#22d3ee", bg="#020617").pack(side=tk.LEFT, padx=20)

        self.clock = tk.Label(header, text="", font=("Consolas", 11),
                              fg="#94a3b8", bg="#020617")
        self.clock.pack(side=tk.RIGHT, padx=20)

        # ===== MAIN LAYOUT =====
        main = tk.Frame(self.root, bg="#0a0f1e")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ---- LEFT: Map Canvas ----
        left = tk.Frame(main, bg="#0a0f1e")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        map_header = tk.Frame(left, bg="#0a0f1e")
        map_header.pack(fill=tk.X)
        tk.Label(map_header, text="DHAKA CITY — LIVE ROUTE MAP",
                 font=("Segoe UI", 10, "bold"),
                 fg="#94a3b8", bg="#0a0f1e").pack(side=tk.LEFT)
        self.sim_label = tk.Label(map_header, text="",
                                  font=("Consolas", 9),
                                  fg="#22d3ee", bg="#0a0f1e")
        self.sim_label.pack(side=tk.RIGHT)

        self.map_canvas = tk.Canvas(left, bg="#0d1425", highlightthickness=0)
        self.map_canvas.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        # ---- RIGHT: Control Panel ----
        right = tk.Frame(main, bg="#0a0f1e", width=420)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(12, 0))
        right.pack_propagate(False)

        notebook = ttk.Notebook(right)
        notebook.pack(fill=tk.BOTH, expand=True)

        # --- Tab 1: Controls ---
        tab1 = tk.Frame(notebook, bg="#0d1425")
        notebook.add(tab1, text="  ⚙ Controls  ")
        self._build_controls_tab(tab1)

        # --- Tab 2: Delivery List ---
        tab2 = tk.Frame(notebook, bg="#0d1425")
        notebook.add(tab2, text="  📦 Stops  ")
        self._build_stops_tab(tab2)

        # --- Tab 3: Analytics ---
        tab3 = tk.Frame(notebook, bg="#0d1425")
        notebook.add(tab3, text="  📊 Analytics  ")
        self._build_analytics_tab(tab3)

        # ===== FOOTER =====
        footer = tk.Frame(self.root, bg="#020617", height=26)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        self.footer = tk.Label(footer, text="Ready.",
                               font=("Consolas", 9),
                               fg="#475569", bg="#020617")
        self.footer.pack(side=tk.LEFT, padx=12)
        tk.Label(footer,
                 text="RouteOptimizer v1.0 | Nearest-Neighbor + 2-Opt",
                 font=("Consolas", 9), fg="#475569",
                 bg="#020617").pack(side=tk.RIGHT, padx=12)

    # ------------------------------------------------------------------
    def _build_controls_tab(self, parent):
        # Number of stops
        tk.Label(parent, text="NUMBER OF STOPS",
                 font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#0d1425").pack(anchor=tk.W, padx=14, pady=(14, 4))

        self.stop_count = tk.IntVar(value=12)
        count_frame = tk.Frame(parent, bg="#0d1425")
        count_frame.pack(fill=tk.X, padx=14)
        self.count_label = tk.Label(count_frame, text="12",
                                    font=("Consolas", 14, "bold"),
                                    fg="#22d3ee", bg="#0d1425")
        self.count_label.pack(side=tk.RIGHT)
        scale = tk.Scale(count_frame, from_=4, to=25, orient=tk.HORIZONTAL,
                         variable=self.stop_count, showvalue=False,
                         bg="#0d1425", fg="#22d3ee",
                         troughcolor="#1a2340", highlightthickness=0,
                         command=self._on_count_change)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Time of day (traffic)
        tk.Label(parent, text="TIME OF DAY (TRAFFIC)",
                 font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#0d1425").pack(anchor=tk.W, padx=14, pady=(18, 4))

        self.hour_var = tk.IntVar(value=12)
        hour_frame = tk.Frame(parent, bg="#0d1425")
        hour_frame.pack(fill=tk.X, padx=14)
        self.hour_label = tk.Label(hour_frame, text="12:00",
                                   font=("Consolas", 14, "bold"),
                                   fg="#fbbf24", bg="#0d1425")
        self.hour_label.pack(side=tk.RIGHT)
        hour_scale = tk.Scale(hour_frame, from_=0, to=23, orient=tk.HORIZONTAL,
                              variable=self.hour_var, showvalue=False,
                              bg="#0d1425", fg="#fbbf24",
                              troughcolor="#1a2340", highlightthickness=0,
                              command=self._on_hour_change)
        hour_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Traffic multiplier display
        self.traffic_label = tk.Label(parent, text="Traffic Multiplier: 1.00x",
                                      font=("Consolas", 10),
                                      fg="#94a3b8", bg="#0d1425")
        self.traffic_label.pack(anchor=tk.W, padx=14, pady=(6, 0))

        # Buttons
        tk.Frame(parent, bg="#1e293b", height=1).pack(fill=tk.X, padx=14, pady=16)

        tk.Button(parent, text="🎲  RANDOMIZE STOPS",
                  font=("Segoe UI", 10, "bold"),
                  bg="#1e40af", fg="white", bd=0, pady=10,
                  activebackground="#1d4ed8", activeforeground="white",
                  cursor="hand2",
                  command=lambda: self._randomize_stops(self.stop_count.get())
                  ).pack(fill=tk.X, padx=14, pady=4)

        tk.Button(parent, text="⚡  OPTIMIZE ROUTE",
                  font=("Segoe UI", 10, "bold"),
                  bg="#059669", fg="white", bd=0, pady=10,
                  activebackground="#10b981", activeforeground="white",
                  cursor="hand2",
                  command=self._optimize_route
                  ).pack(fill=tk.X, padx=14, pady=4)

        tk.Button(parent, text="▶  START DELIVERY SIMULATION",
                  font=("Segoe UI", 10, "bold"),
                  bg="#7c3aed", fg="white", bd=0, pady=10,
                  activebackground="#8b5cf6", activeforeground="white",
                  cursor="hand2",
                  command=self._start_simulation
                  ).pack(fill=tk.X, padx=14, pady=4)

        tk.Button(parent, text="⏹  RESET",
                  font=("Segoe UI", 10, "bold"),
                  bg="#334155", fg="white", bd=0, pady=8,
                  activebackground="#475569", activeforeground="white",
                  cursor="hand2",
                  command=self._reset
                  ).pack(fill=tk.X, padx=14, pady=4)

        # Result box
        tk.Frame(parent, bg="#1e293b", height=1).pack(fill=tk.X, padx=14, pady=16)

        tk.Label(parent, text="OPTIMIZATION RESULT",
                 font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#0d1425").pack(anchor=tk.W, padx=14)

        result_card = tk.Frame(parent, bg="#131c33")
        result_card.pack(fill=tk.X, padx=14, pady=(4, 14))

        self.result_labels = {}
        for key, label, color in [
            ("stops", "Total Stops", "#e2e8f0"),
            ("nn_dist", "NN Distance", "#fbbf24"),
            ("opt_dist", "Optimized Distance", "#22d3ee"),
            ("improve", "Improvement", "#22c55e"),
            ("time", "Compute Time", "#a855f7"),
        ]:
            row = tk.Frame(result_card, bg="#131c33")
            row.pack(fill=tk.X, padx=10, pady=3)
            tk.Label(row, text=label, font=("Segoe UI", 9),
                     fg="#94a3b8", bg="#131c33").pack(side=tk.LEFT)
            val = tk.Label(row, text="—", font=("Consolas", 10, "bold"),
                           fg=color, bg="#131c33")
            val.pack(side=tk.RIGHT)
            self.result_labels[key] = val

    # ------------------------------------------------------------------
    def _build_stops_tab(self, parent):
        tk.Label(parent, text="DELIVERY STOPS (OPTIMIZED ORDER)",
                 font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#0d1425").pack(anchor=tk.W, padx=14, pady=(14, 6))

        frame = tk.Frame(parent, bg="#0d1425")
        frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))

        self.stop_listbox = tk.Listbox(frame,
                                       bg="#131c33", fg="#e2e8f0",
                                       font=("Consolas", 10), bd=0,
                                       highlightthickness=0,
                                       selectbackground="#334155",
                                       activestyle="none")
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL,
                               command=self.stop_listbox.yview)
        self.stop_listbox.configure(yscroll=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.stop_listbox.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    def _build_analytics_tab(self, parent):
        # Stats cards
        stats_frame = tk.Frame(parent, bg="#0d1425")
        stats_frame.pack(fill=tk.X, padx=14, pady=(14, 6))

        self.stat_widgets = {}
        for key, label, color in [
            ("delivered", "DELIVERED", "#22c55e"),
            ("pending",   "PENDING",   "#fbbf24"),
            ("distance",  "TOTAL KM",  "#22d3ee"),
            ("zones",     "ZONES",     "#a855f7"),
        ]:
            card = tk.Frame(stats_frame, bg="#131c33")
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
            tk.Label(card, text=label, font=("Segoe UI", 8, "bold"),
                     fg="#64748b", bg="#131c33").pack(anchor=tk.W, padx=10, pady=(8, 0))
            v = tk.Label(card, text="0", font=("Segoe UI", 16, "bold"),
                         fg=color, bg="#131c33")
            v.pack(anchor=tk.W, padx=10, pady=(0, 8))
            self.stat_widgets[key] = v

        tk.Label(parent, text="ZONE BREAKDOWN",
                 font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#0d1425").pack(anchor=tk.W, padx=14, pady=(14, 4))

        self.zone_canvas = tk.Canvas(parent, bg="#131c33",
                                     highlightthickness=0, height=180)
        self.zone_canvas.pack(fill=tk.X, padx=14, pady=(0, 14))

        tk.Label(parent, text="ROUTE LEG DISTANCES",
                 font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#0d1425").pack(anchor=tk.W, padx=14, pady=(4, 4))

        self.leg_canvas = tk.Canvas(parent, bg="#131c33",
                                    highlightthickness=0, height=150)
        self.leg_canvas.pack(fill=tk.X, padx=14, pady=(0, 14))

    # ------------------------------------------------------------------
    # CONTROL CALLBACKS
    # ------------------------------------------------------------------
    def _on_count_change(self, val):
        self.count_label.config(text=str(int(float(val))))

    def _on_hour_change(self, val):
        h = int(float(val))
        self.time_of_day = h
        self.hour_label.config(text=f"{h:02d}:00")
        mult = self._traffic_multiplier()
        self.traffic_label.config(text=f"Traffic Multiplier: {mult:.2f}x")

    def _traffic_multiplier(self):
        """Rush-hour traffic simulation for Dhaka."""
        h = self.time_of_day
        if 7 <= h <= 10 or 16 <= h <= 20:
            return 2.2   # peak rush hour
        elif 11 <= h <= 15:
            return 1.4   # midday
        elif 21 <= h <= 23 or 5 <= h <= 6:
            return 0.8   # late night / early morning
        else:
            return 0.6   # deep night

    def _randomize_stops(self, n):
        n = max(4, min(25, int(n)))
        self.selected_stops = random.sample(list(DHAKA_LOCATIONS.keys()), n)
        self.current_route = []
        self.route_distance = 0.0
        self.initial_distance = 0.0
        self.improvement_pct = 0.0
        self.opt_time_ms = 0.0
        self.delivery_stats = {"delivered": 0, "pending": n, "total_distance": 0.0}
        self.simulation_progress = 0.0
        self.simulating = False
        self.vehicle_pos = None
        self._clear_result_labels()
        self._refresh_stop_list()
        self._draw_map()
        self.footer.config(text=f"Randomized {n} delivery stops.")

    def _clear_result_labels(self):
        for v in self.result_labels.values():
            v.config(text="—")

    def _optimize_route(self):
        if not self.selected_stops:
            return
        mult = self._traffic_multiplier()
        result = self.optimizer.optimize(self.selected_stops, traffic_mult=mult)

        self.current_route = result["order"]
        self.route_distance = result["distance"]
        self.initial_distance = result["nn_distance"]
        self.improvement_pct = result["improvement"]
        self.opt_time_ms = result["time_ms"]

        self.result_labels["stops"].config(text=str(len(self.selected_stops)))
        self.result_labels["nn_dist"].config(text=f"{result['nn_distance']:.1f} km")
        self.result_labels["opt_dist"].config(text=f"{result['distance']:.1f} km")
        self.result_labels["improve"].config(text=f"{result['improvement']:.1f}%")
        self.result_labels["time"].config(text=f"{result['time_ms']:.1f} ms")

        self.delivery_stats["pending"] = len(self.selected_stops)
        self.delivery_stats["delivered"] = 0
        self.delivery_stats["total_distance"] = result["distance"]

        self._refresh_stop_list()
        self._draw_map()
        self.footer.config(
            text=f"Optimized {len(self.selected_stops)} stops | "
                 f"{result['distance']:.1f} km (saved {result['improvement']:.1f}%)"
        )

    def _start_simulation(self):
        if not self.current_route:
            self._optimize_route()
        if not self.current_route:
            return
        self.simulating = True
        self.simulation_progress = 0.0
        self.delivery_stats["delivered"] = 0
        self.delivery_stats["pending"] = len(self.current_route)
        self.footer.config(text="Simulation started...")

    def _reset(self):
        self.simulating = False
        self.simulation_progress = 0.0
        self.vehicle_pos = None
        self.delivery_stats = {
            "delivered": 0,
            "pending": len(self.selected_stops),
            "total_distance": self.route_distance,
        }
        self._draw_map()
        self.footer.config(text="Reset.")

    # ------------------------------------------------------------------
    def _refresh_stop_list(self):
        self.stop_listbox.delete(0, tk.END)
        if not self.current_route:
            for i, name in enumerate(self.selected_stops, 1):
                x, y, zone = DHAKA_LOCATIONS[name]
                self.stop_listbox.insert(tk.END, f"  {i:2d}. {name}  [{zone}]")
            return

        # Depot start
        self.stop_listbox.insert(tk.END, f"  🏭 START: {DEPOT[0]}")
        for i, name in enumerate(self.current_route, 1):
            x, y, zone = DHAKA_LOCATIONS[name]
            delivered = i <= self.delivery_stats["delivered"]
            status = "✅" if delivered else "⬜"
            self.stop_listbox.insert(
                tk.END, f"  {status} {i:2d}. {name}  [{zone}]  ({x},{y})"
            )
        self.stop_listbox.insert(tk.END, f"  🏭 END: {DEPOT[0]}")

    # ------------------------------------------------------------------
    # MAP DRAWING
    # ------------------------------------------------------------------
    def _map_scale(self):
        """Convert virtual 0-100 coords to canvas pixels."""
        self.map_canvas.update_idletasks()
        w = self.map_canvas.winfo_width()
        h = self.map_canvas.winfo_height()
        if w < 50 or h < 50:
            return None
        pad = 30
        return {
            "w": w, "h": h, "pad": pad,
            "sx": (w - 2 * pad) / 100.0,
            "sy": (h - 2 * pad) / 100.0,
        }

    def _to_px(self, x, y, scale):
        return (
            scale["pad"] + x * scale["sx"],
            scale["pad"] + y * scale["sy"],
        )

    def _draw_map(self):
        c = self.map_canvas
        c.delete("all")
        scale = self._map_scale()
        if not scale:
            return

        w, h = scale["w"], scale["h"]
        pad = scale["pad"]

        # Background grid
        for i in range(0, 101, 10):
            x1, _ = self._to_px(i, 0, scale)
            _, y1 = self._to_px(0, i, scale)
            c.create_line(x1, pad, x1, h - pad, fill="#111a30")
            c.create_line(pad, y1, w - pad, y1, fill="#111a30")

        # "River" (Buriganga - decorative)
        river_pts = [
            self._to_px(20, 90, scale),
            self._to_px(35, 82, scale),
            self._to_px(55, 85, scale),
            self._to_px(75, 78, scale),
            self._to_px(95, 82, scale),
        ]
        c.create_line(*[p for pt in river_pts for p in pt],
                      fill="#1e3a8a", width=14, smooth=True)

        # Draw all Dhaka locations (faded background)
        for name, (x, y, zone) in DHAKA_LOCATIONS.items():
            px, py = self._to_px(x, y, scale)
            if name in self.selected_stops:
                continue
            color = {
                "residential": "#334155",
                "commercial":  "#3f3f46",
                "industrial":  "#3f2d20",
            }[zone]
            c.create_oval(px - 3, py - 3, px + 3, py + 3,
                          fill=color, outline="")
            c.create_text(px, py - 8, text=name, fill="#2d3748",
                          font=("Segoe UI", 6), anchor=tk.S)

        # Draw optimized route
        if self.current_route:
            depot_px = self._to_px(DEPOT[1], DEPOT[2], scale)

            # Route polyline
            pts = [depot_px]
            for name in self.current_route:
                x, y, _ = DHAKA_LOCATIONS[name]
                pts.append(self._to_px(x, y, scale))
            pts.append(depot_px)

            # Draw with glow effect
            c.create_line(*[p for pt in pts for p in pt],
                          fill="#0ea5e9", width=6, smooth=True,
                          capstyle=tk.ROUND, joinstyle=tk.ROUND)
            c.create_line(*[p for pt in pts for p in pt],
                          fill="#22d3ee", width=2, smooth=True,
                          capstyle=tk.ROUND, joinstyle=tk.ROUND)

            # Draw arrows (direction indicators)
            for i in range(len(pts) - 1):
                if i % 2 == 0:
                    self._draw_arrow(c, pts[i], pts[i + 1])

        # Draw depot
        depot_px = self._to_px(DEPOT[1], DEPOT[2], scale)
        c.create_oval(depot_px[0] - 10, depot_px[1] - 10,
                      depot_px[0] + 10, depot_px[1] + 10,
                      fill="#a855f7", outline="white", width=2)
        c.create_text(depot_px[0], depot_px[1], text="🏭",
                      font=("Segoe UI", 10))

        # Draw stops
        delivered_count = self.delivery_stats.get("delivered", 0)
        for i, name in enumerate(self.current_route if self.current_route
                                 else self.selected_stops):
            x, y, zone = DHAKA_LOCATIONS[name]
            px, py = self._to_px(x, y, scale)

            if self.current_route:
                delivered = i < delivered_count
                is_current = i == delivered_count and self.simulating
            else:
                delivered = False
                is_current = False

            if delivered:
                color = "#22c55e"
            elif is_current:
                color = "#fbbf24"
            else:
                color = {
                    "residential": "#38bdf8",
                    "commercial":  "#f472b6",
                    "industrial":  "#fb923c",
                }[zone]

            r = 9 if is_current else 7
            # Pulse for current stop
            if is_current:
                c.create_oval(px - r - 6, py - r - 6,
                              px + r + 6, py + r + 6,
                              outline=color, width=2)

            c.create_oval(px - r, py - r, px + r, py + r,
                          fill=color, outline="white", width=2)
            c.create_text(px, py, text=str(i + 1),
                          fill="#0a0f1e",
                          font=("Segoe UI", 7, "bold"))
            c.create_text(px, py + r + 8, text=name,
                          fill="#e2e8f0",
                          font=("Segoe UI", 8, "bold"))

        # Draw vehicle during simulation
        if self.vehicle_pos:
            vx, vy = self.vehicle_pos
            px, py = self._to_px(vx, vy, scale)
            c.create_oval(px - 12, py - 12, px + 12, py + 12,
                          fill="#fbbf24", outline="white", width=2)
            c.create_text(px, py, text="🚚",
                          font=("Segoe UI", 12))

        # Legend
        self._draw_map_legend(c, w, h)

    def _draw_arrow(self, canvas, p1, p2):
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        size = 6
        a1 = (mx - size * math.cos(angle - 0.5),
              my - size * math.sin(angle - 0.5))
        a2 = (mx - size * math.cos(angle + 0.5),
              my - size * math.sin(angle + 0.5))
        canvas.create_polygon(mx, my, *a1, *a2, fill="#22d3ee", outline="")

    def _draw_map_legend(self, canvas, w, h):
        lx = 15
        ly = h - 95
        # Legend box
        canvas.create_rectangle(lx, ly, lx + 230, ly + 85,
                                fill="#0a0f1e", outline="#1e293b")
        canvas.create_text(lx + 10, ly + 12, text="LEGEND",
                           fill="#64748b", anchor=tk.W,
                           font=("Segoe UI", 8, "bold"))

        items = [
            ("#a855f7", "Depot / Warehouse"),
            ("#38bdf8", "Residential stop"),
            ("#f472b6", "Commercial stop"),
            ("#fb923c", "Industrial stop"),
            ("#22c55e", "Delivered"),
            ("#fbbf24", "Current / Vehicle"),
        ]
        for i, (color, label) in enumerate(items):
            row = i % 3
            col = i // 3
            x = lx + 12 + col * 115
            y = ly + 30 + row * 18
            canvas.create_oval(x, y - 4, x + 8, y + 4, fill=color, outline="")
            canvas.create_text(x + 14, y, text=label, fill="#cbd5e1",
                               anchor=tk.W, font=("Segoe UI", 7))

    # ------------------------------------------------------------------
    # ANIMATION LOOP
    # ------------------------------------------------------------------
    def _animate(self):
        if not self.running:
            return

        self.clock.config(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

        # Advance delivery simulation
        if self.simulating and self.current_route:
            self.simulation_progress += 0.012
            n = len(self.current_route)

            # Map progress 0..1 → full route traversal
            total_segments = n + 1  # depot → stop1 → ... → stopN → depot
            seg = self.simulation_progress * total_segments
            seg_idx = int(seg)
            seg_t = seg - seg_idx

            if seg_idx >= total_segments:
                # Finished
                self.simulating = False
                self.delivery_stats["delivered"] = n
                self.delivery_stats["pending"] = 0
                self.vehicle_pos = None
                self.footer.config(text="✅ Delivery route completed!")
            else:
                # Interpolate vehicle position
                if seg_idx == 0:
                    p1 = (DEPOT[1], DEPOT[2])
                    p2 = DHAKA_LOCATIONS[self.current_route[0]][:2]
                elif seg_idx >= n:
                    p1 = DHAKA_LOCATIONS[self.current_route[-1]][:2]
                    p2 = (DEPOT[1], DEPOT[2])
                else:
                    p1 = DHAKA_LOCATIONS[self.current_route[seg_idx - 1]][:2]
                    p2 = DHAKA_LOCATIONS[self.current_route[seg_idx]][:2]

                vx = p1[0] + (p2[0] - p1[0]) * seg_t
                vy = p1[1] + (p2[1] - p1[1]) * seg_t
                self.vehicle_pos = (vx, vy)

                # Update delivered count
                self.delivery_stats["delivered"] = min(seg_idx, n)
                self.delivery_stats["pending"] = n - self.delivery_stats["delivered"]
                self.sim_label.config(
                    text=f"Progress: {self.delivery_stats['delivered']}/{n}"
                )

            # Refresh list occasionally
            if int(self.simulation_progress * 100) % 4 == 0:
                self._refresh_stop_list()

        # Redraw map
        self._draw_map()

        # Update stats
        self._update_stats()

        # Update analytics
        self._draw_zone_breakdown()
        self._draw_leg_distances()

        self.root.after(50, self._animate)

    # ------------------------------------------------------------------
    def _update_stats(self):
        self.stat_widgets["delivered"].config(
            text=str(self.delivery_stats.get("delivered", 0)))
        self.stat_widgets["pending"].config(
            text=str(self.delivery_stats.get("pending", 0)))
        self.stat_widgets["distance"].config(
            text=f"{self.delivery_stats.get('total_distance', 0):.1f}")
        self.stat_widgets["zones"].config(
            text=str(len(set(
                DHAKA_LOCATIONS[n][2] for n in self.current_route
            )) if self.current_route else 0))

    # ------------------------------------------------------------------
    def _draw_zone_breakdown(self):
        c = self.zone_canvas
        c.delete("all")
        c.update_idletasks()
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 50 or h < 50:
            return

        stops = self.current_route if self.current_route else self.selected_stops
        counts = {"residential": 0, "commercial": 0, "industrial": 0}
        for n in stops:
            counts[DHAKA_LOCATIONS[n][2]] += 1

        total = max(1, sum(counts.values()))
        colors = {
            "residential": "#38bdf8",
            "commercial":  "#f472b6",
            "industrial":  "#fb923c",
        }

        # Horizontal stacked bar
        bar_x = 20
        bar_y = 30
        bar_w = w - 40
        bar_h = 30
        cx = bar_x
        for zone, cnt in counts.items():
            if cnt == 0:
                continue
            seg_w = bar_w * cnt / total
            c.create_rectangle(cx, bar_y, cx + seg_w, bar_y + bar_h,
                               fill=colors[zone], outline="")
            if seg_w > 30:
                c.create_text(cx + seg_w / 2, bar_y + bar_h / 2,
                              text=str(cnt), fill="white",
                              font=("Segoe UI", 10, "bold"))
            cx += seg_w

        # Legend
        lx = 20
        ly = 80
        for zone, cnt in counts.items():
            c.create_rectangle(lx, ly, lx + 12, ly + 12,
                               fill=colors[zone], outline="")
            pct = cnt / total * 100
            c.create_text(lx + 20, ly + 6,
                          text=f"{zone.title()}: {cnt} ({pct:.0f}%)",
                          anchor=tk.W, fill="#cbd5e1",
                          font=("Segoe UI", 9))
            ly += 24

    # ------------------------------------------------------------------
    def _draw_leg_distances(self):
        c = self.leg_canvas
        c.delete("all")
        c.update_idletasks()
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 50 or h < 50 or not self.current_route:
            return

        # Compute leg distances
        mult = self._traffic_multiplier()
        legs = []
        prev = (DEPOT[1], DEPOT[2])
        prev_name = "Depot"
        for name in self.current_route:
            x, y, _ = DHAKA_LOCATIONS[name]
            d = math.hypot(prev[0] - x, prev[1] - y) * mult
            legs.append((prev_name, name, d))
            prev = (x, y)
            prev_name = name

        max_d = max((l[2] for l in legs), default=1)

        bar_h = 14
        gap = 6
        n_show = min(10, len(legs))
        start_y = 10

        c.create_text(w / 2, 8, text=f"Leg distances (x{mult:.2f} traffic)",
                      fill="#64748b", font=("Segoe UI", 8))

        for i in range(n_show):
            from_n, to_n, d = legs[i]
            y = start_y + i * (bar_h + gap)
            label = f"{from_n[:8]} → {to_n[:8]}"
            c.create_text(10, y + bar_h / 2, text=label,
                          anchor=tk.W, fill="#94a3b8",
                          font=("Consolas", 7))

            bar_x = 145
            bar_w = w - bar_x - 55
            c.create_rectangle(bar_x, y, bar_x + bar_w, y + bar_h,
                               fill="#1a2340", outline="")
            ratio = d / max_d if max_d > 0 else 0
            c.create_rectangle(bar_x, y, bar_x + bar_w * ratio, y + bar_h,
                               fill="#22d3ee", outline="")
            c.create_text(bar_x + bar_w + 6, y + bar_h / 2,
                          text=f"{d:.1f}", anchor=tk.W,
                          fill="#e2e8f0", font=("Consolas", 8))

    # ------------------------------------------------------------------
    def on_close(self):
        self.running = False
        self.root.destroy()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = DhakaDeliveryOptimizer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()