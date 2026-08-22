from __future__ import annotations

import sqlite3
import json
import time
import queue
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from config import DATA_DIR


@dataclass
class TrafficEvent:
    timestamp: float
    frame_idx: int
    event_type: str
    track_id: int
    vehicle_type: str
    lane: str
    direction: str
    plate_text: str = ""
    speed_kmh: float = 0.0
    confidence: float = 0.0
    bbox: str = ""
    crop_path: str = ""


class EventStorage:
    def __init__(self, db_path: str = str(DATA_DIR / "traffic_events.db")):
        self.db_path = db_path
        self.lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        # Enable WAL mode and normal synchronous for ultra-fast non-blocking I/O
        with self.lock:
            self.conn.execute("PRAGMA journal_mode = WAL;")
            self.conn.execute("PRAGMA synchronous = NORMAL;")
            self._create_tables()

        # Background async write worker
        self.write_queue = queue.Queue(maxsize=10000)
        self.running = True
        self.worker_thread = threading.Thread(target=self._async_writer_loop, daemon=True)
        self.worker_thread.start()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER UNIQUE,
                vehicle_type TEXT,
                plate_text TEXT,
                first_seen REAL,
                last_seen REAL,
                lane TEXT,
                direction TEXT,
                avg_speed REAL,
                total_frames INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                frame_idx INTEGER,
                event_type TEXT,
                track_id INTEGER,
                vehicle_type TEXT,
                lane TEXT,
                direction TEXT,
                plate_text TEXT,
                speed_kmh REAL,
                confidence REAL,
                bbox TEXT,
                crop_path TEXT
            );

            CREATE TABLE IF NOT EXISTS counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                frame_idx INTEGER,
                lane TEXT,
                direction TEXT,
                count INTEGER
            );

            CREATE TABLE IF NOT EXISTS plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                frame_idx INTEGER,
                track_id INTEGER,
                plate_text TEXT,
                confidence REAL,
                crop_path TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_vehicles_track ON vehicles(track_id);
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_vehicles_lane ON vehicles(lane);
            CREATE INDEX IF NOT EXISTS idx_vehicles_dir ON vehicles(direction);
        """)
        self.conn.commit()

    def _async_writer_loop(self):
        """Processes database write operations in batches asynchronously without blocking the CV pipeline."""
        while self.running or not self.write_queue.empty():
            batch = []
            try:
                # Grab first item
                item = self.write_queue.get(timeout=0.2)
                batch.append(item)
                # Grab any additional items available
                while len(batch) < 100:
                    try:
                        batch.append(self.write_queue.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                continue

            if batch:
                with self.lock:
                    try:
                        cur = self.conn.cursor()
                        for action, args in batch:
                            if action == "upsert_vehicle":
                                track_id, vehicle_type, plate_text, lane, direction, ts = args
                                cur.execute("SELECT id FROM vehicles WHERE track_id = ?", (track_id,))
                                row = cur.fetchone()
                                if row:
                                    cur.execute("""
                                        UPDATE vehicles SET
                                            last_seen = ?,
                                            lane = CASE WHEN ? != '' THEN ? ELSE lane END,
                                            direction = CASE WHEN ? != '' THEN ? ELSE direction END,
                                            total_frames = total_frames + 1
                                        WHERE track_id = ?
                                    """, (ts, lane, lane, direction, direction, track_id))
                                else:
                                    cur.execute("""
                                        INSERT INTO vehicles (track_id, vehicle_type, plate_text, first_seen, last_seen, lane, direction, avg_speed, total_frames)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, 0.0, 1)
                                    """, (track_id, vehicle_type, plate_text, ts, ts, lane, direction))
                            elif action == "log_event":
                                event = args
                                cur.execute("""
                                    INSERT INTO events (timestamp, frame_idx, event_type, track_id,
                                        vehicle_type, lane, direction, plate_text, speed_kmh,
                                        confidence, bbox, crop_path)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    event.timestamp, event.frame_idx, event.event_type,
                                    event.track_id, event.vehicle_type, event.lane,
                                    event.direction, event.plate_text, event.speed_kmh,
                                    event.confidence, event.bbox, event.crop_path,
                                ))
                            elif action == "log_plate":
                                frame_idx, track_id, plate_text, confidence, crop_path, ts = args
                                cur.execute("""
                                    INSERT INTO plates (timestamp, frame_idx, track_id, plate_text, confidence, crop_path)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                """, (ts, frame_idx, track_id, plate_text, confidence, crop_path))
                            elif action == "log_count":
                                frame_idx, lane, direction, count, ts = args
                                cur.execute("""
                                    INSERT INTO counts (timestamp, frame_idx, lane, direction, count)
                                    VALUES (?, ?, ?, ?, ?)
                                """, (ts, frame_idx, lane, direction, count))
                            self.write_queue.task_done()
                        self.conn.commit()
                    except Exception as e:
                        print(f"[STORAGE ERROR] Async write failed: {e}")

    def log_event(self, event: TrafficEvent):
        try:
            self.write_queue.put_nowait(("log_event", event))
        except queue.Full:
            pass

    def log_plate(self, frame_idx: int, track_id: int, plate_text: str, confidence: float, crop_path: str = ""):
        try:
            self.write_queue.put_nowait(("log_plate", (frame_idx, track_id, plate_text, confidence, crop_path, time.time())))
        except queue.Full:
            pass

    def log_count(self, frame_idx: int, lane: str, direction: str, count: int):
        try:
            self.write_queue.put_nowait(("log_count", (frame_idx, lane, direction, count, time.time())))
        except queue.Full:
            pass

    def upsert_vehicle(self, track_id: int, vehicle_type: str, plate_text: str = "",
                       lane: str = "", direction: str = "", speed: float = 0.0):
        try:
            self.write_queue.put_nowait(("upsert_vehicle", (track_id, vehicle_type, plate_text, lane, direction, time.time())))
        except queue.Full:
            pass

    # -------------------------------------------------------------------------
    # Analytics Queries (Fast read-only queries for Dashboard & Analytics)
    # -------------------------------------------------------------------------
    def get_summary(self) -> dict:
        with self.lock:
            cur = self.conn.cursor()
            summary = {}
            summary["total_vehicles"] = cur.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]
            summary["total_events"] = cur.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            summary["total_violations"] = cur.execute(
                "SELECT COUNT(*) FROM events WHERE event_type IN ('WRONG_WAY', 'CYCLE_ZONE_INTRUSION')"
            ).fetchone()[0]

            by_type = cur.execute("SELECT vehicle_type, COUNT(*) FROM vehicles GROUP BY vehicle_type").fetchall()
            summary["by_type"] = {row[0]: row[1] for row in by_type if row[0]}

            by_lane = cur.execute("SELECT lane, COUNT(*) FROM vehicles GROUP BY lane").fetchall()
            summary["by_lane"] = {row[0]: row[1] for row in by_lane if row[0]}

            by_dir = cur.execute("SELECT direction, COUNT(*) FROM vehicles GROUP BY direction").fetchall()
            summary["by_direction"] = {row[0]: row[1] for row in by_dir if row[0]}

            return summary

    def get_violations(self, limit: int = 50) -> list[dict]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT timestamp, frame_idx, track_id, vehicle_type, lane, event_type, direction "
                "FROM events WHERE event_type IN ('WRONG_WAY', 'CYCLE_ZONE_INTRUSION') "
                "ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [
                {
                    "timestamp": r[0],
                    "frame_idx": r[1],
                    "track_id": r[2],
                    "vehicle_type": r[3],
                    "lane": r[4],
                    "violation_type": r[5],
                    "description": r[6],
                }
                for r in cur
            ]

    def get_analytics_payload(self, time_range: str = "all") -> dict:
        """Returns structured data for the dedicated Traffic Intelligence Analytics dashboard."""
        with self.lock:
            cur = self.conn.cursor()
            payload = {}
            now = time.time()

            # Determine time cutoff based on filter
            cutoff = 0
            if time_range == "1m":
                cutoff = now - 60
            elif time_range == "5m":
                cutoff = now - 300
            elif time_range == "15m":
                cutoff = now - 900
            
            where_time = f"WHERE first_seen >= {cutoff}" if cutoff > 0 else ""
            where_event_time = f"WHERE timestamp >= {cutoff}" if cutoff > 0 else ""

            # 1. Active Tracks (Seen in last 3 seconds)
            active_tracks = cur.execute(
                f"SELECT COUNT(*) FROM vehicles WHERE last_seen >= {now - 3.0}"
            ).fetchone()[0]

            # Recent session increment (last 60s)
            recent_added = cur.execute(
                f"SELECT COUNT(*) FROM vehicles WHERE first_seen >= {now - 60.0}"
            ).fetchone()[0]

            # Latest activity timestamp
            latest_row = cur.execute("SELECT MAX(last_seen) FROM vehicles").fetchone()
            last_activity_sec = int(now - latest_row[0]) if (latest_row and latest_row[0]) else 0

            # 2. KPI Cards
            total_v = cur.execute(f"SELECT COUNT(*) FROM vehicles {where_time}").fetchone()[0]
            t2b_v = cur.execute(
                f"SELECT COUNT(*) FROM vehicles {'WHERE' if not where_time else where_time + ' AND'} direction = 'T2B'"
            ).fetchone()[0]
            b2t_v = cur.execute(
                f"SELECT COUNT(*) FROM vehicles {'WHERE' if not where_time else where_time + ' AND'} direction = 'B2T'"
            ).fetchone()[0]
            
            cars = cur.execute(
                f"SELECT COUNT(*) FROM vehicles {'WHERE' if not where_time else where_time + ' AND'} LOWER(vehicle_type) LIKE '%car%'"
            ).fetchone()[0]
            motorcycles = cur.execute(
                f"SELECT COUNT(*) FROM vehicles {'WHERE' if not where_time else where_time + ' AND'} (LOWER(vehicle_type) LIKE '%motor%' OR LOWER(vehicle_type) LIKE '%bike%')"
            ).fetchone()[0]
            buses = cur.execute(
                f"SELECT COUNT(*) FROM vehicles {'WHERE' if not where_time else where_time + ' AND'} LOWER(vehicle_type) LIKE '%bus%'"
            ).fetchone()[0]
            trucks = cur.execute(
                f"SELECT COUNT(*) FROM vehicles {'WHERE' if not where_time else where_time + ' AND'} LOWER(vehicle_type) LIKE '%truck%'"
            ).fetchone()[0]

            ev_clause = f"AND {where_event_time[6:]}" if where_event_time else ""
            total_viols = cur.execute(
                f"SELECT COUNT(*) FROM events WHERE event_type IN ('WRONG_WAY', 'CYCLE_ZONE_INTRUSION') {ev_clause}"
            ).fetchone()[0]
            wrong_way = cur.execute(
                f"SELECT COUNT(*) FROM events WHERE event_type = 'WRONG_WAY' {ev_clause}"
            ).fetchone()[0]
            cycle_zone = cur.execute(
                f"SELECT COUNT(*) FROM events WHERE event_type = 'CYCLE_ZONE_INTRUSION' {ev_clause}"
            ).fetchone()[0]

            payload["kpis"] = {
                "total_vehicles": total_v,
                "t2b_vehicles": t2b_v,
                "b2t_vehicles": b2t_v,
                "cars": cars,
                "motorcycles": motorcycles,
                "buses": buses,
                "trucks": trucks,
                "total_violations": total_viols,
                "wrong_way": wrong_way,
                "cycle_zone_violations": cycle_zone,
                "active_tracks": active_tracks,
                "recent_added": recent_added,
                "last_activity_sec": last_activity_sec,
            }

            # 3. Traffic Flow Timeline (Dynamic time binning)
            bin_size = 5 if (time_range in ("live", "1m", "5m") or total_v < 100) else 10
            flow_where = f"WHERE first_seen >= {cutoff}" if cutoff > 0 else "WHERE first_seen > 0"
            flow_rows = cur.execute(f"""
                SELECT 
                    CAST(first_seen / {bin_size} AS INTEGER) * {bin_size} as time_bin,
                    SUM(CASE WHEN direction = 'T2B' THEN 1 ELSE 0 END) as t2b_count,
                    SUM(CASE WHEN direction = 'B2T' THEN 1 ELSE 0 END) as b2t_count,
                    COUNT(*) as total
                FROM vehicles
                {flow_where}
                GROUP BY time_bin
                ORDER BY time_bin ASC
            """).fetchall()

            if flow_rows:
                base_t = flow_rows[0][0]
                payload["flow_timeline"] = [
                    {
                        "time_sec": int(r[0] - base_t),
                        "time_label": time.strftime("%H:%M:%S", time.localtime(r[0])),
                        "t2b": r[1],
                        "b2t": r[2],
                        "total": r[3],
                    }
                    for r in flow_rows
                ]
            else:
                payload["flow_timeline"] = []

            # 4. Lane Analytics (Lane x Direction)
            lane_where = f"{where_time} AND" if where_time else "WHERE"
            lane_stats = cur.execute(f"""
                SELECT 
                    lane,
                    COUNT(*) as total,
                    SUM(CASE WHEN direction = 'T2B' THEN 1 ELSE 0 END) as t2b,
                    SUM(CASE WHEN direction = 'B2T' THEN 1 ELSE 0 END) as b2t
                FROM vehicles
                {lane_where} lane != '' AND lane != 'unknown'
                GROUP BY lane
                ORDER BY total DESC
            """).fetchall()
            
            payload["lanes"] = [
                {"name": r[0], "total": r[1], "t2b": r[2], "b2t": r[3]}
                for r in lane_stats
            ]

            # 5. Vehicle Classes Distribution
            class_where = f"{where_time} AND" if where_time else "WHERE"
            class_stats = cur.execute(f"""
                SELECT vehicle_type, COUNT(*) as count
                FROM vehicles
                {class_where} vehicle_type != ''
                GROUP BY vehicle_type
                ORDER BY count DESC
            """).fetchall()
            payload["classes"] = [{"type": r[0], "count": r[1]} for r in class_stats]

            # 6. Violations Log
            viol_where = f"WHERE event_type IN ('WRONG_WAY', 'CYCLE_ZONE_INTRUSION') {ev_clause}"
            viol_rows = cur.execute(f"""
                SELECT timestamp, frame_idx, track_id, vehicle_type, lane, event_type, direction
                FROM events
                {viol_where}
                ORDER BY id DESC LIMIT 50
            """).fetchall()
            payload["recent_violations"] = [
                {
                    "timestamp": r[0],
                    "time_str": time.strftime("%H:%M:%S", time.localtime(r[0])),
                    "frame_idx": r[1],
                    "track_id": r[2],
                    "vehicle_type": r[3],
                    "lane": r[4],
                    "violation_type": r[5],
                    "description": r[6],
                }
                for r in viol_rows
            ]

            # 7. Detailed Vehicles Table
            veh_where = f"{where_time}" if where_time else ""
            veh_rows = cur.execute(f"""
                SELECT 
                    v.track_id, v.vehicle_type, v.direction, v.lane,
                    v.first_seen, v.last_seen,
                    (v.last_seen - v.first_seen) as duration_sec,
                    v.total_frames,
                    CASE WHEN e.id IS NOT NULL THEN 1 ELSE 0 END as has_violation,
                    e.event_type
                FROM vehicles v
                LEFT JOIN events e ON v.track_id = e.track_id AND e.event_type IN ('WRONG_WAY', 'CYCLE_ZONE_INTRUSION')
                {veh_where}
                GROUP BY v.track_id
                ORDER BY v.id DESC LIMIT 100
            """).fetchall()
            payload["vehicles_table"] = [
                {
                    "track_id": r[0],
                    "vehicle_type": r[1],
                    "direction": r[2],
                    "lane": r[3],
                    "first_seen": r[4],
                    "last_seen": r[5],
                    "first_seen_str": time.strftime("%H:%M:%S", time.localtime(r[4])),
                    "last_seen_str": time.strftime("%H:%M:%S", time.localtime(r[5])),
                    "duration": round(r[6], 1),
                    "total_frames": r[7],
                    "violation": r[9] if r[8] else "None",
                }
                for r in veh_rows
            ]

            return payload

    def close(self):
        self.running = False
        if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        with self.lock:
            try:
                self.conn.commit()
                self.conn.close()
            except Exception:
                pass
