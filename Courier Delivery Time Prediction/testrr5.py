import tkinter as tk
from tkinter import ttk
import random
import math
import time
import threading
from collections import deque
from datetime import datetime, timedelta


# ----------------------------------------------------------------------
# DHAKA DELIVERY ZONES (with difficulty multipliers)
# ----------------------------------------------------------------------
DHAKA_ZONES = {
    "Gulshan":         (72, 25, 1.0,  "premium"),
    "Banani":          (68, 22, 1.0,  "premium"),
    "Baridhara":       (75, 22, 1.0,  "premium"),
    "Bashundhara":     (75, 18, 1.1,  "premium"),
    "Dhanmondi":       (35, 48, 1.3,  "residential"),
    "Mohammadpur":     (28, 45, 1.35, "residential"),
    "Shyamoli":        (32, 42, 1.3,  "residential"),
    "Mirpur":          (30, 15, 1.4,  "residential"),
    "Kallyanpur":      (30, 35, 1.35, "residential"),
    "Uttara":          (45, 5,  1.2,  "residential"),
    "Badda":           (78, 32, 1.25, "residential"),
    "Rampura":         (72, 35, 1.25, "residential"),
    "Malibagh":        (62, 55, 1.35, "residential"),
    "Mugda":           (65, 60, 1.4,  "residential"),
    "Jatrabari":       (70, 75, 1.5,  "residential"),
    "Bonosree":        (80, 40, 1.3,  "residential"),
    "Khilgaon":        (68, 52, 1.35, "residential"),
    "Motijheel":       (58, 72, 1.55, "commercial"),
    "Farmgate":        (48, 40, 1.6,  "commercial"),
    "Karwan Bazar":    (50, 42, 1.6,  "commercial"),
    "Paltan":          (58, 68, 1.55, "commercial"),
    "Banglamotor":     (52, 38, 1.5,  "commercial"),
    "Shahbagh":        (50, 50, 1.45, "commercial"),
    "New Market":      (42, 52, 1.4,  "commercial"),
    "Sadarghat":       (62, 82, 1.7,  "commercial"),
    "Wari":            (62, 78, 1.65, "commercial"),
    "Lalbagh":         (45, 75, 1.6,  "commercial"),
    "Agargaon":        (42, 32, 1.2,  "commercial"),
    "Tejgaon":         (52, 35, 1.4,  "industrial"),
    "Chittagong Road": (80, 85, 1.55, "industrial"),
    "Hazaribagh":      (38, 58, 1.5,  "industrial"),
    "Kamrangirchar":   (40, 68, 1.55, "industrial"),
}

HUBS = [
    ("Tejgaon Central Hub", 52, 35),
    ("Motijheel South Hub", 58, 72),
    ("Mirpur North Hub",    30, 15),
]

PRIORITIES = {
    "Standard":  {"mult": 1.00, "color": "#94a3b8", "sla_hours": 48},
    "Express":   {"mult": 0.75, "color": "#38bdf8", "sla_hours": 24},
    "Same-Day":  {"mult": 0.50, "color": "#fbbf24", "sla_hours": 12},
    "Urgent":    {"mult": 0.30, "color": "#f43f5e", "sla_hours": 4},
}

WEATHER = {
    "Clear":     {"mult": 1.00, "color": "#22c55e"},
    "Cloudy":    {"mult": 1.05, "color": "#94a3b8"},
    "Rain":      {"mult": 1.35, "color": "#38bdf8"},
    "Heavy Rain":{"mult": 1.70, "color": "#3b82f6"},
    "Storm":     {"mult": 2.10, "color": "#a855f7"},
}

COURIERS = [
    "Rahim Uddin", "Karim Ahmed", "Jamal Hossain", "Sumaiya Akter",
    "Tanvir Islam", "Nusrat Jahan", "Sabbir Rahman", "Mehedi Hasan",
    "Farhana Yasmin", "Arif Chowdhury",
]


# ----------------------------------------------------------------------
# PREDICTION ENGINE
# ----------------------------------------------------------------------
class DeliveryTimePredictor:
    """
    Rule-based + statistical prediction engine.

    Predicts ETA by combining:
      - Base travel time (distance / avg speed)
      - Traffic multiplier (hour of day)
      - Weather multiplier
      - Zone difficulty multiplier (narrow roads, congestion)
      - Package priority modifier
      - Courier experience & current load
      - Small stochastic noise

    Also produces a confidence interval using a heuristic variance model.
    A real system would use a trained regression model (XGBoost / LightGBM)
    with these same features.
    """

    def __init__(self):
        self.base_speed = 20.0          # km/h free-flow
        self.history = deque(maxlen=500)
        self.accuracy_tracker = deque(maxlen=200)

    # ------------------------------------------------------------------
    def _traffic_mult(self, hour):
        if 7 <= hour <= 10 or 16 <= hour <= 20:
            return 2.30
        elif 11 <= hour <= 15:
            return 1.50
        elif 21 <= hour <= 23 or 5 <= hour <= 6:
            return 0.85
        else:
            return 0.60

    def _distance(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1]) * 1.2  # ~1.2km per unit

    # ------------------------------------------------------------------
    def predict(self, package):
        """
        package = {
          origin_hub, dest_zone, priority, weather, hour,
          weight_kg, cod_amount, courier, courier_experience,
          courier_active_deliveries, road_factor
        }
        Returns dict with eta_minutes, eta_confidence, sla_status, etc.
        """
        hub = next(h for h in HUBS if h[0] == package["origin_hub"])
        zone_name = package["dest_zone"]
        zx, zy, zone_diff, zone_type = DHAKA_ZONES[zone_name]

        distance_km = self._distance((hub[1], hub[2]), (zx, zy))

        traffic = self._traffic_mult(package["hour"])
        weather = WEATHER[package["weather"]]["mult"]
        priority = PRIORITIES[package["priority"]]["mult"]

        # Courier experience factor (0..1 → more experience = faster)
        exp = max(0.0, min(1.0, package["courier_experience"]))
        courier_skill = 1.15 - 0.30 * exp

        # Courier current workload
        load = package["courier_active_deliveries"]
        load_penalty = 1.0 + max(0, load - 3) * 0.04

        # Package weight penalty (heavy items slow handling)
        weight = package["weight_kg"]
        weight_penalty = 1.0 + max(0, weight - 2.0) * 0.03

        # COD handling adds a few minutes
        cod_penalty_min = 4 if package["cod_amount"] > 0 else 0

        # Road factor (0.8 good, 1.4 bad)
        road = package["road_factor"]

        # Effective speed
        eff_speed = (self.base_speed
                     / traffic
                     / weather
                     / zone_diff
                     / courier_skill
                     / load_penalty
                     / weight_penalty
                     / road)

        # Base travel time in minutes
        travel_min = (distance_km / max(1.0, eff_speed)) * 60.0
        travel_min *= priority
        travel_min += cod_penalty_min

        # Handling time at hub (pickup, scan)
        handling = random.uniform(4, 9)
        # Last-mile walk / building time
        last_mile = random.uniform(3, 8) * zone_diff

        # Stochastic variance for realism
        noise = random.gauss(0, travel_min * 0.08)
        eta = max(5.0, travel_min + handling + last_mile + noise)

        # Confidence interval width (heuristic)
        variance = travel_min * 0.15 + 3.0
        ci_low = max(2.0, eta - variance)
        ci_high = eta + variance

        # SLA check
        sla_hours = PRIORITIES[package["priority"]]["sla_hours"]
        sla_minutes = sla_hours * 60
        sla_status = "ON TIME" if eta <= sla_minutes else "AT RISK"

        # Confidence score (0-100)
        conf = max(35, min(99,
                           100 - variance * 1.5
                           - (traffic - 1) * 5
                           - (weather - 1) * 8
                           - (zone_diff - 1) * 15
                           ))
        conf = round(conf, 1)

        result = {
            "eta_minutes": round(eta, 1),
            "ci_low": round(ci_low, 1),
            "ci_high": round(ci_high, 1),
            "distance_km": round(distance_km, 2),
            "traffic_mult": round(traffic, 2),
            "weather_mult": round(weather, 2),
            "zone_diff": round(zone_diff, 2),
            "eff_speed": round(eff_speed, 2),
            "sla_status": sla_status,
            "confidence": conf,
            "zone_type": zone_type,
            "hub": hub[0],
        }
        self.history.append(result)
        return result


# ----------------------------------------------------------------------
# MAIN APPLICATION
# ----------------------------------------------------------------------
class CourierTimePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Courier Delivery Time Prediction — Dhaka")
        self.root.geometry("1400x820")
        self.root.configure(bg="#0b1120")
        self.root.minsize(1200, 700)

        self.predictor = DeliveryTimePredictor()
        self.running = True

        # State
        self.predictions = deque(maxlen=80)
        self.prediction_log = deque(maxlen=200)
        self.accuracy_samples = deque(maxlen=200)
        self.total_predicted = 0
        self.on_time_count = 0
        self.at_risk_count = 0
        self.avg_eta = 0.0
        self.avg_confidence = 0.0
        self.zone_stats = {}
        self.priority_stats = {k: 0 for k in PRIORITIES}
        self.auto_running = False

        # Current form defaults
        self.hour = 10

        self._setup_styles()
        self._build_ui()
        self._predict_one()      # initial prediction
        self._animate()

    # ------------------------------------------------------------------
    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook", background="#0b1120", borderwidth=0)
        style.configure("TNotebook.Tab",
                        background="#152036", foreground="#94a3b8",
                        padding=[14, 7], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", "#1e3a5f")],
                  foreground=[("selected", "#38bdf8")])
        style.configure("TCombobox",
                        fieldbackground="#152036", background="#152036",
                        foreground="#e2e8f0", arrowcolor="#94a3b8")

    # ------------------------------------------------------------------
    def _build_ui(self):
        # HEADER
        header = tk.Frame(self.root, bg="#020617", height=62)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="📦  COURIER DELIVERY TIME PREDICTION  •  DHAKA",
                 font=("Segoe UI", 16, "bold"),
                 fg="#38bdf8", bg="#020617").pack(side=tk.LEFT, padx=20)

        self.clock = tk.Label(header, text="", font=("Consolas", 11),
                              fg="#94a3b8", bg="#020617")
        self.clock.pack(side=tk.RIGHT, padx=20)

        # KPI STRIP
        kpi = tk.Frame(self.root, bg="#0b1120", height=86)
        kpi.pack(fill=tk.X, padx=10, pady=(10, 0))
        kpi.pack_propagate(False)

        self.kpi_widgets = {}
        kpi_defs = [
            ("TOTAL PREDICTIONS", "0",     "#38bdf8"),
            ("ON-TIME SLA",       "0.0%",  "#22c55e"),
            ("AT-RISK SLA",       "0.0%",  "#f43f5e"),
            ("AVG ETA (min)",     "0",     "#fbbf24"),
            ("AVG CONFIDENCE",    "0.0%",  "#a855f7"),
        ]
        for i, (title, init, color) in enumerate(kpi_defs):
            card = tk.Frame(kpi, bg="#152036")
            card.grid(row=0, column=i, sticky="nsew", padx=4, pady=4)
            kpi.grid_columnconfigure(i, weight=1)
            tk.Label(card, text=title, font=("Segoe UI", 8, "bold"),
                     fg="#64748b", bg="#152036").pack(anchor=tk.W, padx=12, pady=(8, 0))
            v = tk.Label(card, text=init, font=("Segoe UI", 20, "bold"),
                         fg=color, bg="#152036")
            v.pack(anchor=tk.W, padx=12, pady=(0, 8))
            self.kpi_widgets[title] = v

        # MAIN
        main = tk.Frame(self.root, bg="#0b1120")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # LEFT: form + result
        left = tk.Frame(main, bg="#0b1120", width=420)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        nb = ttk.Notebook(left)
        nb.pack(fill=tk.BOTH, expand=True)

        t1 = tk.Frame(nb, bg="#131c33")
        nb.add(t1, text="  📝 New Prediction  ")
        self._build_form_tab(t1)

        t2 = tk.Frame(nb, bg="#131c33")
        nb.add(t2, text="  📜 History  ")
        self._build_history_tab(t2)

        # RIGHT: analytics
        right = tk.Frame(main, bg="#0b1120")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        nb2 = ttk.Notebook(right)
        nb2.pack(fill=tk.BOTH, expand=True)

        t3 = tk.Frame(nb2, bg="#0b1120")
        nb2.add(t3, text="  📈 ETA Trend  ")
        self.trend_canvas = tk.Canvas(t3, bg="#020617", highlightthickness=0)
        self.trend_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        t4 = tk.Frame(nb2, bg="#0b1120")
        nb2.add(t4, text="  🗺 Zone Map  ")
        self.zone_canvas = tk.Canvas(t4, bg="#020617", highlightthickness=0)
        self.zone_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        t5 = tk.Frame(nb2, bg="#0b1120")
        nb2.add(t5, text="  🧩 Factor Impact  ")
        self.factor_canvas = tk.Canvas(t5, bg="#020617", highlightthickness=0)
        self.factor_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        t6 = tk.Frame(nb2, bg="#0b1120")
        nb2.add(t6, text="  📊 Zone Stats  ")
        self.stats_canvas = tk.Canvas(t6, bg="#020617", highlightthickness=0)
        self.stats_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # FOOTER
        footer = tk.Frame(self.root, bg="#020617", height=26)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        self.footer = tk.Label(footer, text="Ready.",
                               font=("Consolas", 9),
                               fg="#475569", bg="#020617")
        self.footer.pack(side=tk.LEFT, padx=12)
        tk.Label(footer, text="DeliveryPredictor v1.0 | Rule + Statistical Model",
                 font=("Consolas", 9), fg="#475569",
                 bg="#020617").pack(side=tk.RIGHT, padx=12)

    # ------------------------------------------------------------------
    def _build_form_tab(self, p):
        # Destination
        tk.Label(p, text="DESTINATION ZONE", font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#131c33").pack(anchor=tk.W, padx=14, pady=(14, 4))
        self.dest_var = tk.StringVar(value="Dhanmondi")
        ttk.Combobox(p, textvariable=self.dest_var,
                     values=sorted(DHAKA_ZONES.keys()),
                     state="readonly").pack(fill=tk.X, padx=14)

        # Hub
        tk.Label(p, text="ORIGIN HUB", font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#131c33").pack(anchor=tk.W, padx=14, pady=(12, 4))
        self.hub_var = tk.StringVar(value=HUBS[0][0])
        ttk.Combobox(p, textvariable=self.hub_var,
                     values=[h[0] for h in HUBS],
                     state="readonly").pack(fill=tk.X, padx=14)

        # Priority
        tk.Label(p, text="PRIORITY", font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#131c33").pack(anchor=tk.W, padx=14, pady=(12, 4))
        self.prio_var = tk.StringVar(value="Express")
        row = tk.Frame(p, bg="#131c33")
        row.pack(fill=tk.X, padx=14)
        for pr in PRIORITIES:
            rb = tk.Radiobutton(row, text=pr, variable=self.prio_var,
                                value=pr, bg="#131c33", fg="#e2e8f0",
                                selectcolor="#152036",
                                activebackground="#131c33",
                                activeforeground="#38bdf8",
                                font=("Segoe UI", 9))
            rb.pack(side=tk.LEFT, padx=2)

        # Weather
        tk.Label(p, text="WEATHER", font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#131c33").pack(anchor=tk.W, padx=14, pady=(12, 4))
        self.weather_var = tk.StringVar(value="Clear")
        ttk.Combobox(p, textvariable=self.weather_var,
                     values=list(WEATHER.keys()),
                     state="readonly").pack(fill=tk.X, padx=14)

        # Hour
        tk.Label(p, text="DEPARTURE HOUR", font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#131c33").pack(anchor=tk.W, padx=14, pady=(12, 4))
        row3 = tk.Frame(p, bg="#131c33")
        row3.pack(fill=tk.X, padx=14)
        self.hour_lbl = tk.Label(row3, text="10:00", font=("Consolas", 12, "bold"),
                                 fg="#fbbf24", bg="#131c33")
        self.hour_lbl.pack(side=tk.RIGHT)
        self.hour_var = tk.IntVar(value=10)
        tk.Scale(row3, from_=0, to=23, orient=tk.HORIZONTAL,
                 variable=self.hour_var, showvalue=False,
                 bg="#131c33", fg="#fbbf24", troughcolor="#1e3a5f",
                 highlightthickness=0,
                 command=self._on_hour).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Weight + COD
        row4 = tk.Frame(p, bg="#131c33")
        row4.pack(fill=tk.X, padx=14, pady=(12, 0))

        tk.Label(row4, text="WEIGHT (kg)", font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#131c33").grid(row=0, column=0, sticky=tk.W)
        tk.Label(row4, text="COD (৳)", font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#131c33").grid(row=0, column=1,
                                                   sticky=tk.W, padx=(10, 0))

        self.weight_var = tk.DoubleVar(value=1.5)
        self.cod_var = tk.DoubleVar(value=0)

        tk.Spinbox(row4, from_=0.1, to=20.0, increment=0.1, width=8,
                   textvariable=self.weight_var, bg="#152036", fg="#e2e8f0",
                   buttonbackground="#1e3a5f", bd=0,
                   insertbackground="#e2e8f0").grid(row=1, column=0, sticky=tk.W)
        tk.Spinbox(row4, from_=0, to=50000, increment=100, width=10,
                   textvariable=self.cod_var, bg="#152036", fg="#e2e8f0",
                   buttonbackground="#1e3a5f", bd=0,
                   insertbackground="#e2e8f0").grid(row=1, column=1,
                                                     sticky=tk.W, padx=(10, 0))

        # Courier
        tk.Label(p, text="COURIER", font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#131c33").pack(anchor=tk.W, padx=14, pady=(12, 4))
        self.courier_var = tk.StringVar(value=COURIERS[0])
        ttk.Combobox(p, textvariable=self.courier_var,
                     values=COURIERS,
                     state="readonly").pack(fill=tk.X, padx=14)

        # Active deliveries + experience
        row5 = tk.Frame(p, bg="#131c33")
        row5.pack(fill=tk.X, padx=14, pady=(12, 0))
        tk.Label(row5, text="ACTIVE DELIVERIES", font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#131c33").grid(row=0, column=0, sticky=tk.W)
        tk.Label(row5, text="EXPERIENCE (0-1)", font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#131c33").grid(row=0, column=1,
                                                  sticky=tk.W, padx=(10, 0))

        self.active_var = tk.IntVar(value=3)
        self.exp_var = tk.DoubleVar(value=0.7)
        tk.Spinbox(row5, from_=0, to=15, width=6, textvariable=self.active_var,
                   bg="#152036", fg="#e2e8f0", buttonbackground="#1e3a5f",
                   bd=0, insertbackground="#e2e8f0").grid(row=1, column=0, sticky=tk.W)
        tk.Spinbox(row5, from_=0.0, to=1.0, increment=0.1, width=6,
                   textvariable=self.exp_var, bg="#152036", fg="#e2e8f0",
                   buttonbackground="#1e3a5f", bd=0,
                   insertbackground="#e2e8f0").grid(row=1, column=1,
                                                     sticky=tk.W, padx=(10, 0))

        # Buttons
        tk.Frame(p, bg="#1e3a5f", height=1).pack(fill=tk.X, padx=14, pady=14)

        tk.Button(p, text="🔮  PREDICT DELIVERY TIME",
                  font=("Segoe UI", 10, "bold"),
                  bg="#0369a1", fg="white", bd=0, pady=10,
                  activebackground="#0284c7", activeforeground="white",
                  cursor="hand2", command=self._predict_one).pack(fill=tk.X, padx=14, pady=4)

        self.auto_btn = tk.Button(p, text="▶  AUTO-PREDICT STREAM",
                                  font=("Segoe UI", 10, "bold"),
                                  bg="#7c3aed", fg="white", bd=0, pady=10,
                                  activebackground="#8b5cf6", activeforeground="white",
                                  cursor="hand2", command=self._toggle_auto)
        self.auto_btn.pack(fill=tk.X, padx=14, pady=4)

        # RESULT CARD
        tk.Label(p, text="PREDICTION RESULT", font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#131c33").pack(anchor=tk.W, padx=14, pady=(14, 4))

        self.result_card = tk.Frame(p, bg="#152036")
        self.result_card.pack(fill=tk.X, padx=14, pady=(0, 14))

        self.eta_big = tk.Label(self.result_card, text="— min",
                                font=("Segoe UI", 30, "bold"),
                                fg="#38bdf8", bg="#152036")
        self.eta_big.pack(anchor=tk.W, padx=14, pady=(10, 0))

        self.eta_ci = tk.Label(self.result_card, text="Confidence interval: —",
                               font=("Consolas", 10),
                               fg="#94a3b8", bg="#152036")
        self.eta_ci.pack(anchor=tk.W, padx=14)

        self.sla_lbl = tk.Label(self.result_card, text="SLA: —",
                                font=("Segoe UI", 11, "bold"),
                                fg="#22c55e", bg="#152036")
        self.sla_lbl.pack(anchor=tk.W, padx=14, pady=(6, 0))

        self.conf_lbl = tk.Label(self.result_card, text="Confidence: —",
                                 font=("Consolas", 10),
                                 fg="#a855f7", bg="#152036")
        self.conf_lbl.pack(anchor=tk.W, padx=14, pady=(0, 10))

        # Details row
        self.detail_lbl = tk.Label(p, text="", font=("Consolas", 9),
                                   fg="#94a3b8", bg="#131c33",
                                   justify=tk.LEFT)
        self.detail_lbl.pack(anchor=tk.W, padx=14, pady=(0, 14))

    def _on_hour(self, val):
        self.hour = int(float(val))
        self.hour_lbl.config(text=f"{self.hour:02d}:00")

    # ------------------------------------------------------------------
    def _build_history_tab(self, p):
        tk.Label(p, text="PREDICTION HISTORY", font=("Segoe UI", 9, "bold"),
                 fg="#64748b", bg="#131c33").pack(anchor=tk.W, padx=14, pady=(14, 6))

        frame = tk.Frame(p, bg="#131c33")
        frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))

        self.history_box = tk.Listbox(frame, bg="#152036", fg="#cbd5e1",
                                      font=("Consolas", 9), bd=0,
                                      highlightthickness=0,
                                      selectbackground="#1e3a5f")
        sb = ttk.Scrollbar(frame, orient=tk.VERTICAL,
                           command=self.history_box.yview)
        self.history_box.configure(yscroll=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_box.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------
    def _predict_one(self):
        package = {
            "origin_hub": self.hub_var.get(),
            "dest_zone": self.dest_var.get(),
            "priority": self.prio_var.get(),
            "weather": self.weather_var.get(),
            "hour": self.hour,
            "weight_kg": float(self.weight_var.get()),
            "cod_amount": float(self.cod_var.get()),
            "courier": self.courier_var.get(),
            "courier_experience": float(self.exp_var.get()),
            "courier_active_deliveries": int(self.active_var.get()),
            "road_factor": random.uniform(0.9, 1.3),
        }
        result = self.predictor.predict(package)
        self.predictions.append(result)
        self.total_predicted += 1

        if result["sla_status"] == "ON TIME":
            self.on_time_count += 1
        else:
            self.at_risk_count += 1

        self.avg_eta = (self.avg_eta * (self.total_predicted - 1)
                        + result["eta_minutes"]) / self.total_predicted
        self.avg_confidence = (self.avg_confidence * (self.total_predicted - 1)
                               + result["confidence"]) / self.total_predicted

        # Update log
        ts = datetime.now().strftime("%H:%M:%S")
        msg = (f"[{ts}] {package['dest_zone']:<14} | "
               f"{package['priority']:<9} | "
               f"ETA {result['eta_minutes']:>5.1f} min | "
               f"CI ±{result['ci_high'] - result['eta_minutes']:>4.1f} | "
               f"{result['sla_status']}")
        self.prediction_log.appendleft(msg)

        # Zone stats
        z = package["dest_zone"]
        if z not in self.zone_stats:
            self.zone_stats[z] = {"count": 0, "eta_sum": 0.0, "at_risk": 0}
        self.zone_stats[z]["count"] += 1
        self.zone_stats[z]["eta_sum"] += result["eta_minutes"]
        if result["sla_status"] == "AT RISK":
            self.zone_stats[z]["at_risk"] += 1

        self.priority_stats[package["priority"]] = \
            self.priority_stats.get(package["priority"], 0) + 1

        # Update result card
        self.eta_big.config(text=f"{result['eta_minutes']:.1f} min")
        self.eta_ci.config(
            text=f"Confidence interval: {result['ci_low']:.1f} – "
                 f"{result['ci_high']:.1f} min"
        )
        color = "#22c55e" if result["sla_status"] == "ON TIME" else "#f43f5e"
        self.sla_lbl.config(text=f"SLA: {result['sla_status']}", fg=color)
        self.conf_lbl.config(text=f"Confidence: {result['confidence']:.1f}%")

        self.detail_lbl.config(
            text=(f"Distance:      {result['distance_km']:.2f} km\n"
                  f"Traffic mult:  {result['traffic_mult']:.2f}x\n"
                  f"Weather mult:  {result['weather_mult']:.2f}x\n"
                  f"Zone diff:     {result['zone_diff']:.2f}x\n"
                  f"Eff. speed:    {result['eff_speed']:.2f} km/h\n"
                  f"Hub:           {result['hub']}")
        )

        self._refresh_history()
        self.footer.config(
            text=f"Predicted {result['eta_minutes']:.1f} min to "
                 f"{package['dest_zone']} via {package['priority']}"
        )

    def _refresh_history(self):
        self.history_box.delete(0, tk.END)
        for msg in list(self.prediction_log)[:100]:
            self.history_box.insert(tk.END, msg)

    # ------------------------------------------------------------------
    def _toggle_auto(self):
        self.auto_running = not self.auto_running
        if self.auto_running:
            self.auto_btn.config(text="⏹  STOP AUTO-PREDICT", bg="#dc2626")
        else:
            self.auto_btn.config(text="▶  AUTO-PREDICT STREAM", bg="#7c3aed")

    # ------------------------------------------------------------------
    # ANALYTICS CANVASES
    # ------------------------------------------------------------------
    def _draw_trend(self):
        c = self.trend_canvas
        c.delete("all")
        c.update_idletasks()
        w, h = c.winfo_width(), c.winfo_height()
        if w < 100 or h < 100:
            return

        c.create_text(w / 2, 16, text="DELIVERY ETA TREND",
                      fill="#94a3b8", font=("Segoe UI", 10, "bold"))

        pad_l, pad_r, pad_t, pad_b = 55, 25, 35, 35
        cw, ch = w - pad_l - pad_r, h - pad_t - pad_b

        preds = list(self.predictions)
        if len(preds) < 2:
            c.create_text(w / 2, h / 2, text="Run predictions to see trend",
                          fill="#475569", font=("Segoe UI", 11))
            return

        etas = [p["eta_minutes"] for p in preds]
        max_e = max(etas) * 1.15
        min_e = 0

        # Grid
        for i in range(6):
            y = pad_t + ch * i / 5
            c.create_line(pad_l, y, pad_l + cw, y, fill="#152036")
            v = max_e - (max_e - min_e) * i / 5
            c.create_text(pad_l - 8, y, text=f"{v:.0f}",
                          anchor=tk.E, fill="#64748b",
                          font=("Consolas", 8))

        # CI band
        n = len(preds)
        lo_pts, hi_pts = [], []
        for i, p in enumerate(preds):
            x = pad_l + cw * i / (n - 1)
            y_lo = pad_t + ch - (p["ci_low"] / max_e) * ch
            y_hi = pad_t + ch - (p["ci_high"] / max_e) * ch
            lo_pts.append((x, y_lo))
            hi_pts.append((x, y_hi))

        # Shaded band
        polygon = lo_pts + list(reversed(hi_pts))
        flat = [v for pt in polygon for v in pt]
        c.create_polygon(*flat, fill="#1e3a5f", outline="")

        # ETA line
        pts = []
        for i, p in enumerate(preds):
            x = pad_l + cw * i / (n - 1)
            y = pad_t + ch - (p["eta_minutes"] / max_e) * ch
            pts.append((x, y))

        for i in range(len(pts) - 1):
            color = "#f43f5e" if preds[i]["sla_status"] == "AT RISK" else "#22c55e"
            c.create_line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1],
                          fill=color, width=2)

        if pts:
            x, y = pts[-1]
            c.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#38bdf8",
                          outline="white", width=2)

        c.create_text(pad_l + cw / 2, h - 12, text="Prediction # →",
                      fill="#64748b", font=("Consolas", 8))
        c.create_text(pad_l - 8, pad_t - 8, text="min",
                      anchor=tk.W, fill="#64748b", font=("Consolas", 8))

    # ------------------------------------------------------------------
    def _draw_zone_map(self):
        c = self.zone_canvas
        c.delete("all")
        c.update_idletasks()
        w, h = c.winfo_width(), c.winfo_height()
        if w < 100 or h < 100:
            return

        c.create_text(w / 2, 16, text="DHAKA ZONE MAP — LATEST PREDICTION",
                      fill="#94a3b8", font=("Segoe UI", 10, "bold"))

        pad = 40
        sx = (w - 2 * pad) / 100.0
        sy = (h - 2 * pad - 20) / 100.0

        def px(x, y):
            return (pad + x * sx, pad + 20 + y * sy)

        # Grid
        for i in range(0, 101, 10):
            x1, _ = px(i, 0)
            _, y1 = px(0, i)
            c.create_line(x1, pad + 20, x1, h - pad, fill="#0f1a2e")
            c.create_line(pad, y1, w - pad, y1, fill="#0f1a2e")

        # Latest prediction
        latest = self.predictions[-1] if self.predictions else None

        # Draw all zones
        for name, (x, y, diff, ztype) in DHAKA_ZONES.items():
            pxx, pyy = px(x, y)

            if latest and name == self.dest_var.get():
                color = "#f43f5e" if latest["sla_status"] == "AT RISK" else "#22c55e"
                r = 12
                c.create_oval(pxx - r - 5, pyy - r - 5, pxx + r + 5, pyy + r + 5,
                              outline=color, width=2)
            else:
                color = {
                    "premium":     "#a855f7",
                    "residential": "#38bdf8",
                    "commercial":  "#fbbf24",
                    "industrial":  "#fb923c",
                }[ztype]
                r = 6

            c.create_oval(pxx - r, pyy - r, pxx + r, pyy + r,
                          fill=color, outline="white", width=1)
            c.create_text(pxx, pyy + r + 9, text=name,
                          fill="#94a3b8", font=("Segoe UI", 7))

        # Hub markers
        for name, hx, hy in HUBS:
            pxx, pyy = px(hx, hy)
            c.create_rectangle(pxx - 9, pyy - 9, pxx + 9, pyy + 9,
                               fill="#0ea5e9", outline="white", width=2)
            c.create_text(pxx, pyy, text="🏭", font=("Segoe UI", 10))

        # Legend
        items = [
            ("#0ea5e9", "Hub"),
            ("#a855f7", "Premium"),
            ("#38bdf8", "Residential"),
            ("#fbbf24", "Commercial"),
            ("#fb923c", "Industrial"),
        ]
        lx, ly = 15, h - 25
        for col, lbl in items:
            c.create_oval(lx, ly - 4, lx + 8, ly + 4, fill=col, outline="")
            c.create_text(lx + 12, ly, text=lbl, anchor=tk.W,
                          fill="#94a3b8", font=("Segoe UI", 8))
            lx += 90

    # ------------------------------------------------------------------
    def _draw_factors(self):
        c = self.factor_canvas
        c.delete("all")
        c.update_idletasks()
        w, h = c.winfo_width(), c.winfo_height()
        if w < 100 or h < 100:
            return

        c.create_text(w / 2, 16, text="FACTOR IMPACT ON LATEST PREDICTION",
                      fill="#94a3b8", font=("Segoe UI", 10, "bold"))

        if not self.predictions:
            c.create_text(w / 2, h / 2, text="No prediction yet",
                          fill="#475569", font=("Segoe UI", 11))
            return

        p = self.predictions[-1]
        factors = [
            ("Traffic",   p["traffic_mult"],  "#f43f5e", 0.5, 2.5),
            ("Weather",   p["weather_mult"],  "#38bdf8", 0.9, 2.2),
            ("Zone",      p["zone_diff"],     "#fbbf24", 0.9, 1.8),
            ("Priority",  PRIORITIES[self.prio_var.get()]["mult"], "#22c55e", 0.2, 1.1),
        ]

        bar_h = 34
        gap = 18
        total_h = len(factors) * (bar_h + gap)
        start_y = (h - total_h) / 2 + 10
        bar_x = 130
        bar_w = w - bar_x - 90

        for i, (label, val, color, lo, hi) in enumerate(factors):
            y = start_y + i * (bar_h + gap)
            c.create_text(bar_x - 12, y + bar_h / 2, text=label,
                          anchor=tk.E, fill="#cbd5e1",
                          font=("Segoe UI", 11, "bold"))

            c.create_rectangle(bar_x, y, bar_x + bar_w, y + bar_h,
                               fill="#152036", outline="")

            # Normalize val in [lo, hi] → width
            ratio = max(0.0, min(1.0, (val - lo) / (hi - lo)))
            fill_w = bar_w * ratio
            c.create_rectangle(bar_x, y, bar_x + fill_w, y + bar_h,
                               fill=color, outline="")

            c.create_text(bar_x + bar_w + 10, y + bar_h / 2,
                          text=f"{val:.2f}x", anchor=tk.W,
                          fill=color, font=("Consolas", 11, "bold"))

        # Explanation
        c.create_text(w / 2, h - 15,
                      text="Values > 1.0 increase ETA. Priority multiplier < 1.0 reduces ETA.",
                      fill="#64748b", font=("Segoe UI", 8))

    # ------------------------------------------------------------------
    def _draw_stats(self):
        c = self.stats_canvas
        c.delete("all")
        c.update_idletasks()
        w, h = c.winfo_width(), c.winfo_height()
        if w < 100 or h < 100:
            return

        c.create_text(w / 2, 16, text="ZONE PERFORMANCE STATISTICS",
                      fill="#94a3b8", font=("Segoe UI", 10, "bold"))

        if not self.zone_stats:
            c.create_text(w / 2, h / 2, text="No data yet",
                          fill="#475569", font=("Segoe UI", 11))
            return

        # Sort by average ETA descending (worst zones first)
        items = []
        for z, s in self.zone_stats.items():
            avg = s["eta_sum"] / s["count"]
            items.append((z, avg, s["count"], s["at_risk"]))
        items.sort(key=lambda x: x[1], reverse=True)
        items = items[:10]

        row_h = (h - 60) / len(items)

        # Header
        c.create_text(30, 40, text="ZONE", anchor=tk.W,
                      fill="#64748b", font=("Consolas", 8, "bold"))
        c.create_text(200, 40, text="AVG ETA", anchor=tk.W,
                      fill="#64748b", font=("Consolas", 8, "bold"))
        c.create_text(320, 40, text="COUNT", anchor=tk.W,
                      fill="#64748b", font=("Consolas", 8, "bold"))
        c.create_text(420, 40, text="AT RISK", anchor=tk.W,
                      fill="#64748b", font=("Consolas", 8, "bold"))

        for i, (z, avg, cnt, risk) in enumerate(items):
            y = 60 + i * row_h + row_h / 2
            color = "#f43f5e" if risk > cnt * 0.4 else \
                    "#fbbf24" if risk > cnt * 0.2 else "#22c55e"

            c.create_text(30, y, text=z, anchor=tk.W, fill="#e2e8f0",
                          font=("Consolas", 9))
            c.create_text(200, y, text=f"{avg:.1f} min", anchor=tk.W,
                          fill=color, font=("Consolas", 9, "bold"))
            c.create_text(320, y, text=str(cnt), anchor=tk.W,
                          fill="#94a3b8", font=("Consolas", 9))
            c.create_text(420, y, text=str(risk), anchor=tk.W,
                          fill="#f43f5e" if risk > 0 else "#64748b",
                          font=("Consolas", 9, "bold"))

            # Mini bar
            bar_x = 480
            bar_w = w - bar_x - 30
            if bar_w > 50:
                max_avg = max(x[1] for x in items)
                ratio = avg / max_avg
                c.create_rectangle(bar_x, y - 6, bar_x + bar_w, y + 6,
                                   fill="#152036", outline="")
                c.create_rectangle(bar_x, y - 6, bar_x + bar_w * ratio, y + 6,
                                   fill=color, outline="")

    # ------------------------------------------------------------------
    # ANIMATION LOOP
    # ------------------------------------------------------------------
    def _animate(self):
        if not self.running:
            return

        self.clock.config(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

        # KPIs
        on_time_pct = (self.on_time_count / max(1, self.total_predicted)) * 100
        at_risk_pct = (self.at_risk_count / max(1, self.total_predicted)) * 100
        self.kpi_widgets["TOTAL PREDICTIONS"].config(text=str(self.total_predicted))
        self.kpi_widgets["ON-TIME SLA"].config(text=f"{on_time_pct:.1f}%")
        self.kpi_widgets["AT-RISK SLA"].config(text=f"{at_risk_pct:.1f}%")
        self.kpi_widgets["AVG ETA (min)"].config(text=f"{self.avg_eta:.1f}")
        self.kpi_widgets["AVG CONFIDENCE"].config(
            text=f"{self.avg_confidence:.1f}%")

        # Auto-predict
        if self.auto_running and random.random() < 0.05:
            self.dest_var.set(random.choice(list(DHAKA_ZONES.keys())))
            self.weather_var.set(random.choices(
                list(WEATHER.keys()),
                weights=[0.5, 0.2, 0.15, 0.1, 0.05])[0]
            )
            self.prio_var.set(random.choice(list(PRIORITIES.keys())))
            self.hour_var.set(random.randint(6, 22))
            self.hour = self.hour_var.get()
            self.hour_lbl.config(text=f"{self.hour:02d}:00")
            self.courier_var.set(random.choice(COURIERS))
            self.active_var.set(random.randint(0, 10))
            self.exp_var.set(round(random.uniform(0.3, 1.0), 1))
            self.weight_var.set(round(random.uniform(0.2, 8.0), 1))
            self.cod_var.set(random.choice([0, 0, 500, 1000, 2500]))
            self._predict_one()

        # Charts
        self._draw_trend()
        self._draw_zone_map()
        self._draw_factors()
        self._draw_stats()

        self.root.after(500, self._animate)

    # ------------------------------------------------------------------
    def on_close(self):
        self.running = False
        self.root.destroy()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = CourierTimePredictionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()