from __future__ import annotations

import cv2
import json
import time
import threading
import numpy as np
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request, Response

from config import LaneConfig, OUTPUT_DIR
from src.storage.event_storage import EventStorage


# =============================================================================
# 1. LIVE STREAM & REGION SETUP HTML (Simple, focused live view)
# =============================================================================
LIVE_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Traffic Intelligence — Live Processing</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #080c14; color: #e2e8f0; min-height: 100vh; }
        
        .header { background: #0f172a; padding: 14px 28px; border-bottom: 1px solid #1e293b; display: flex; justify-content: space-between; align-items: center; }
        .header-title { display: flex; align-items: center; gap: 12px; }
        .header h1 { font-size: 16px; font-weight: 700; color: #38bdf8; letter-spacing: 0.5px; text-transform: uppercase; }
        .live-tag { display: inline-flex; align-items: center; gap: 6px; background: rgba(52, 211, 153, 0.12); color: #34d399; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(52, 211, 153, 0.25); }
        .status-dot { width: 7px; height: 7px; border-radius: 50%; background: #34d399; display: inline-block; box-shadow: 0 0 8px #34d399; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        
        .header-actions { display: flex; gap: 10px; align-items: center; }
        .btn { padding: 7px 14px; border-radius: 6px; font-weight: 600; font-size: 12px; cursor: pointer; border: none; transition: all 0.15s; display: inline-flex; align-items: center; gap: 6px; text-decoration: none; }
        .btn-primary { background: #38bdf8; color: #080c14; }
        .btn-primary:hover { background: #7dd3fc; }
        .btn-analytics { background: #0284c7; color: #fff; border: 1px solid #38bdf8; }
        .btn-analytics:hover { background: #0369a1; }
        .btn-secondary { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; }
        .btn-secondary:hover { background: #334155; }
        .btn-danger { background: #ef4444; color: #fff; }
        .btn-warning { background: #f59e0b; color: #080c14; }

        /* Main Container */
        .main-container { padding: 20px 28px; display: grid; grid-template-columns: 1fr 340px; gap: 20px; }
        @media(max-width: 1100px) { .main-container { grid-template-columns: 1fr; } }

        /* Stream & Setup Card */
        .stage-card { background: #0f172a; border-radius: 8px; border: 1px solid #1e293b; overflow: hidden; display: flex; flex-direction: column; }
        .stage-header { padding: 12px 18px; background: #131d31; border-bottom: 1px solid #1e293b; display: flex; justify-content: space-between; align-items: center; }
        .stage-title { font-size: 12px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
        
        .canvas-container { position: relative; width: 100%; background: #000; display: flex; justify-content: center; align-items: center; min-height: 480px; }
        #setupCanvas { max-width: 100%; height: auto; display: block; cursor: crosshair; }
        #videoFeed { max-width: 100%; height: auto; display: block; }

        /* Region Toolbar */
        .region-toolbar { padding: 12px 18px; background: #131d31; border-top: 1px solid #1e293b; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
        .mode-btn { padding: 6px 12px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; border: 1px solid #334155; background: #0f172a; color: #94a3b8; }
        .mode-btn.active { background: #38bdf8; color: #080c14; border-color: #38bdf8; }
        .mode-btn.zone-btn.active { background: #34d399; color: #080c14; border-color: #34d399; }

        /* Metrics Grid */
        .metrics-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 14px; }
        .metric-card { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 14px; }
        .metric-label { font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 4px; }
        .metric-value { font-size: 22px; font-weight: 800; color: #ffffff; }
        .metric-sub { font-size: 11px; color: #475569; margin-top: 2px; }

        /* Side Panel */
        .side-panel { display: flex; flex-direction: column; gap: 14px; }
        .panel-card { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; }
        .panel-title { font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }

        .region-item { background: #131d31; border: 1px solid #1e293b; border-radius: 6px; padding: 8px 12px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center; font-size: 12px; }
        .region-item-info { display: flex; align-items: center; gap: 8px; }
        .region-color-badge { width: 10px; height: 10px; border-radius: 2px; }
        .del-btn { background: transparent; border: none; color: #ef4444; cursor: pointer; font-size: 14px; font-weight: bold; }

        table { width: 100%; border-collapse: collapse; font-size: 12px; }
        th { text-align: left; padding: 6px 8px; color: #64748b; font-weight: 600; border-bottom: 1px solid #1e293b; }
        td { padding: 6px 8px; border-bottom: 1px solid #131d31; color: #cbd5e1; }
        tr:hover td { background: #131d31; }

        .hint { font-size: 12px; color: #38bdf8; line-height: 1.4; padding: 10px; background: rgba(56, 189, 248, 0.06); border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.2); }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">
            <h1>Traffic Intelligence</h1>
            <span class="live-tag"><span class="status-dot" id="statusDot"></span> LIVE</span>
        </div>
        <div class="header-actions">
            <a href="/analytics" class="btn btn-analytics">📊 Traffic Analytics Dashboard</a>
            <button class="btn btn-secondary" onclick="switchToSetupMode()">✏️ Redraw Regions</button>
            <button class="btn btn-warning" id="pauseBtn" onclick="togglePause()">⏸️ Pause</button>
            <button class="btn btn-danger" onclick="stopPipeline()">⏹️ Stop</button>
        </div>
    </div>

    <div class="main-container">
        <!-- Center Visual Stage -->
        <div class="stage-card">
            <div class="stage-header">
                <span class="stage-title" id="stageTitle">🎥 Video Stream & Region Setup</span>
                <span id="fpsBadge" style="font-size: 12px; color: #38bdf8; font-weight: 700;">NVIDIA GPU (CUDA)</span>
            </div>

            <div class="canvas-container" id="canvasContainer">
                <img id="videoFeed" alt="Video Stream" style="display: none; max-width: 100%; height: auto;">
                <canvas id="setupCanvas"></canvas>
            </div>

            <!-- Region Drawing Toolbar -->
            <div class="region-toolbar" id="setupToolbar">
                <button class="mode-btn active" id="btnLane1" onclick="selectTool('lane', 'lane_1', 'B2T')">🛣️ Lane 1 (B2T)</button>
                <button class="mode-btn" id="btnLane2" onclick="selectTool('lane', 'lane_2', 'T2B')">🛣️ Lane 2 (T2B)</button>
                <button class="mode-btn zone-btn" id="btnCycle" onclick="selectTool('zone', 'cycle_zone', '')">🚴 Cycle / Ped Zone</button>
                <button class="mode-btn" id="btnAddLane" onclick="addNewLanePrompt()">➕ Add Lane</button>
                
                <div style="flex: 1;"></div>
                <button class="btn btn-secondary" onclick="undoVertex()">↩️ Undo</button>
                <button class="btn btn-secondary" onclick="finishCurrentPolygon()">✅ Complete Shape</button>
                <button class="btn btn-secondary" onclick="clearAllPolygons()">🗑️ Clear</button>
                <button class="btn btn-primary" onclick="startInferenceWithRegions()">🚀 START INFERENCE</button>
            </div>
        </div>

        <!-- Right Side Live Metrics Panel -->
        <div class="side-panel">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Vehicles</div>
                    <div class="metric-value" id="valTotalVehicles">0</div>
                    <div class="metric-sub">Processed ROI</div>
                </div>
                <div class="metric-card" style="border-color: rgba(239, 68, 68, 0.3);">
                    <div class="metric-label" style="color: #f87171;">Violations</div>
                    <div class="metric-value" id="valViolations" style="color: #f87171;">0</div>
                    <div class="metric-sub">Wrong-Way / Zone</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Lane 1 (B2T)</div>
                    <div class="metric-value" id="valLane1" style="color: #34d399;">0</div>
                    <div class="metric-sub">Bottom → Top</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Lane 2 (T2B)</div>
                    <div class="metric-value" id="valLane2" style="color: #38bdf8;">0</div>
                    <div class="metric-sub">Top → Bottom</div>
                </div>
            </div>

            <!-- Configured Regions List -->
            <div class="panel-card" id="configuredRegionsCard">
                <div class="panel-title">
                    <span>Configured Regions</span>
                    <span id="regionCountBadge" style="color: #38bdf8;">0 active</span>
                </div>
                <div id="regionsListContainer">
                    <div class="hint">Click 4 corners around each lane on the first frame, then click <b>Start Inference</b>.</div>
                </div>
            </div>

            <!-- Live Rule Violations Table -->
            <div class="panel-card" style="border-color: rgba(239, 68, 68, 0.3);">
                <div class="panel-title" style="color: #f87171;">
                    <span>⚠️ Live Rule Violations</span>
                    <span id="violCountBadge" style="font-size: 11px; background: #ef4444; color: #fff; padding: 2px 6px; border-radius: 4px;">0 alerts</span>
                </div>
                <div style="max-height: 180px; overflow-y: auto;">
                    <table>
                        <thead>
                            <tr><th>Time</th><th>ID</th><th>Type</th><th>Violation</th></tr>
                        </thead>
                        <tbody id="violationsTableBody">
                            <tr><td colspan="4" style="color: #475569; text-align: center;">No violations detected</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Link to Analytics -->
            <div class="panel-card" style="background: linear-gradient(135deg, #0f172a, #0c4a6e); text-align: center; border-color: #0284c7;">
                <h3 style="font-size: 13px; margin-bottom: 6px; color: #38bdf8; font-weight: 700;">Live Analytics Center</h3>
                <p style="font-size: 11px; color: #94a3b8; margin-bottom: 12px;">Real-time traffic flow charts, lane distributions, and searchable vehicle logs.</p>
                <a href="/analytics" class="btn btn-analytics" style="width: 100%; justify-content: center;">📊 Open Analytics Dashboard</a>
            </div>
        </div>
    </div>

    <script>
        let isSetupMode = true;
        let activeTool = { type: 'lane', name: 'lane_1', direction: 'B2T' };
        let lanes = [];
        let zones = {};
        let currentPoints = [];
        let firstFrameImg = new Image();
        let canvas, ctx;
        let isPaused = false;

        const PALETTE = ['#38bdf8', '#34d399', '#f59e0b', '#ef4444', '#a855f7', '#06b6d4'];

        window.onload = function() {
            canvas = document.getElementById('setupCanvas');
            ctx = canvas.getContext('2d');
            
            canvas.addEventListener('click', handleCanvasClick);
            loadExistingRegions();
            loadFirstFrame();
            setInterval(updateStats, 1000);
        };

        function loadFirstFrame() {
            firstFrameImg = new Image();
            firstFrameImg.onload = function() {
                canvas.width = firstFrameImg.naturalWidth || firstFrameImg.width || 1280;
                canvas.height = firstFrameImg.naturalHeight || firstFrameImg.height || 720;
                redrawCanvas();
            };
            firstFrameImg.src = '/api/first_frame?t=' + Date.now();
        }

        function loadExistingRegions() {
            fetch('/api/get_regions').then(r => r.json()).then(data => {
                if (data.lanes && data.lanes.length > 0) {
                    lanes = data.lanes;
                    zones = data.zones || {};
                    updateRegionsListUI();
                    redrawCanvas();
                }
            });
        }

        function selectTool(type, name, direction) {
            if (currentPoints.length > 0) {
                if (confirm("Finish current polygon before switching?")) {
                    finishCurrentPolygon();
                } else {
                    currentPoints = [];
                }
            }
            activeTool = { type, name, direction };
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            if (name === 'lane_1') document.getElementById('btnLane1').classList.add('active');
            else if (name === 'lane_2') document.getElementById('btnLane2').classList.add('active');
            else if (type === 'zone') document.getElementById('btnCycle').classList.add('active');
        }

        function addNewLanePrompt() {
            const laneNum = lanes.length + 1;
            const name = prompt("Enter lane name:", "lane_" + laneNum);
            if (!name) return;
            const dir = prompt("Enter direction (T2B or B2T):", "T2B");
            selectTool('lane', name, dir || "T2B");
        }

        function handleCanvasClick(e) {
            if (!isSetupMode) return;
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const x = Math.round((e.clientX - rect.left) * scaleX);
            const y = Math.round((e.clientY - rect.top) * scaleY);

            currentPoints.push([x, y]);
            redrawCanvas();

            if (currentPoints.length === 4) {
                setTimeout(() => {
                    finishCurrentPolygon();
                }, 100);
            }
        }

        function undoVertex() {
            if (currentPoints.length > 0) {
                currentPoints.pop();
                redrawCanvas();
            }
        }

        function finishCurrentPolygon() {
            if (currentPoints.length < 3) {
                alert("Please click at least 3 corner points to form a region.");
                return;
            }

            if (activeTool.type === 'lane') {
                const existingIdx = lanes.findIndex(l => l.name === activeTool.name);
                const laneObj = {
                    name: activeTool.name,
                    direction: activeTool.direction,
                    polygon: [...currentPoints]
                };
                if (existingIdx >= 0) lanes[existingIdx] = laneObj;
                else lanes.push(laneObj);
            } else {
                zones[activeTool.name] = [...currentPoints];
            }

            currentPoints = [];
            updateRegionsListUI();
            redrawCanvas();
        }

        function clearAllPolygons() {
            if (confirm("Clear all defined lanes and zones?")) {
                lanes = [];
                zones = {};
                currentPoints = [];
                updateRegionsListUI();
                redrawCanvas();
            }
        }

        function removeRegion(type, name) {
            if (type === 'lane') {
                lanes = lanes.filter(l => l.name !== name);
            } else {
                delete zones[name];
            }
            updateRegionsListUI();
            redrawCanvas();
        }

        function updateRegionsListUI() {
            const container = document.getElementById('regionsListContainer');
            const totalCount = lanes.length + Object.keys(zones).length;
            document.getElementById('regionCountBadge').textContent = totalCount + ' active';

            if (totalCount === 0) {
                container.innerHTML = '<div class="hint">Click 4 corners around each lane on the first frame, then click <b>Start Inference</b>.</div>';
                return;
            }

            let html = '';
            lanes.forEach((l, idx) => {
                const col = PALETTE[idx % PALETTE.length];
                html += `
                    <div class="region-item">
                        <div class="region-item-info">
                            <div class="region-color-badge" style="background: ${col};"></div>
                            <b>${l.name}</b> (${l.direction}) - ${l.polygon.length} pts
                        </div>
                        <button class="del-btn" onclick="removeRegion('lane', '${l.name}')">&times;</button>
                    </div>`;
            });

            Object.entries(zones).forEach(([zname, pts]) => {
                html += `
                    <div class="region-item">
                        <div class="region-item-info">
                            <div class="region-color-badge" style="background: #34d399;"></div>
                            <b>${zname}</b> (Zone) - ${pts.length} pts
                        </div>
                        <button class="del-btn" onclick="removeRegion('zone', '${zname}')">&times;</button>
                    </div>`;
            });

            container.innerHTML = html;
        }

        function redrawCanvas() {
            if (!canvas || !ctx) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (firstFrameImg && (firstFrameImg.complete || firstFrameImg.naturalWidth > 0)) {
                ctx.drawImage(firstFrameImg, 0, 0, canvas.width, canvas.height);
            } else {
                ctx.fillStyle = "#080c14";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = "#38bdf8";
                ctx.font = "16px sans-serif";
                ctx.fillText("Loading video frame...", 30, 50);
            }

            lanes.forEach((l, idx) => {
                const col = PALETTE[idx % PALETTE.length];
                drawPoly(l.polygon, col, l.name + ' (' + l.direction + ')');
            });

            Object.entries(zones).forEach(([zname, pts]) => {
                drawPoly(pts, '#34d399', zname);
            });

            if (currentPoints.length > 0) {
                ctx.strokeStyle = '#f59e0b';
                ctx.fillStyle = '#f59e0b';
                ctx.lineWidth = 2;
                ctx.beginPath();
                currentPoints.forEach((pt, i) => {
                    ctx.fillRect(pt[0] - 4, pt[1] - 4, 8, 8);
                    if (i === 0) ctx.moveTo(pt[0], pt[1]);
                    else ctx.lineTo(pt[0], pt[1]);
                });
                ctx.stroke();
            }
        }

        function drawPoly(pts, color, label) {
            if (!pts || pts.length < 2) return;
            ctx.fillStyle = color + '33';
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            pts.forEach((p, i) => {
                if (i === 0) ctx.moveTo(p[0], p[1]);
                else ctx.lineTo(p[0], p[1]);
            });
            ctx.closePath();
            ctx.fill();
            ctx.stroke();

            const cx = pts.reduce((sum, p) => sum + p[0], 0) / pts.length;
            const cy = pts.reduce((sum, p) => sum + p[1], 0) / pts.length;
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 13px sans-serif';
            ctx.fillText(label, cx - 30, cy);
        }

        function startInferenceWithRegions() {
            fetch('/api/start_inference', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ lanes: lanes, zones: zones })
            }).then(r => r.json()).then(res => {
                if (res.status === 'ok') {
                    switchToLiveMode();
                }
            });
        }

        function switchToLiveMode() {
            isSetupMode = false;
            document.getElementById('setupCanvas').style.display = 'none';
            document.getElementById('setupToolbar').style.display = 'none';
            const vfeed = document.getElementById('videoFeed');
            vfeed.src = '/video_feed?t=' + Date.now();
            vfeed.style.display = 'block';
            document.getElementById('stageTitle').textContent = '🔴 Live Inference Stream';
        }

        function switchToSetupMode() {
            isSetupMode = true;
            const vfeed = document.getElementById('videoFeed');
            vfeed.src = '';
            vfeed.style.display = 'none';
            document.getElementById('setupCanvas').style.display = 'block';
            document.getElementById('setupToolbar').style.display = 'flex';
            document.getElementById('stageTitle').textContent = '✏️ Region Configuration (First Frame)';
            redrawCanvas();
        }

        function togglePause() {
            fetch('/api/pause', { method: 'POST' }).then(r => r.json()).then(res => {
                isPaused = res.is_paused;
                document.getElementById('pauseBtn').textContent = isPaused ? '▶️ Resume' : '⏸️ Pause';
            });
        }

        function stopPipeline() {
            if (confirm("Stop traffic intelligence inference?")) {
                fetch('/api/stop', { method: 'POST' }).then(() => {
                    document.getElementById('statusDot').style.background = '#ef4444';
                    document.getElementById('stageTitle').textContent = '⏹️ Inference Stopped';
                });
            }
        }

        function updateStats() {
            fetch('/api/summary').then(r => r.json()).then(data => {
                document.getElementById('valTotalVehicles').textContent = data.total_vehicles || 0;
                
                const laneCounts = data.by_lane || {};
                document.getElementById('valLane1').textContent = laneCounts['lane_1'] || 0;
                document.getElementById('valLane2').textContent = laneCounts['lane_2'] || 0;
            });

            fetch('/api/violations').then(r => r.json()).then(viols => {
                const tbody = document.getElementById('violationsTableBody');
                const badge = document.getElementById('violCountBadge');
                const valMetric = document.getElementById('valViolations');
                
                valMetric.textContent = viols.length;
                badge.textContent = viols.length + ' alert' + (viols.length === 1 ? '' : 's');

                if (viols && viols.length > 0) {
                    tbody.innerHTML = '';
                    viols.slice(0, 8).forEach(v => {
                        const typeTag = v.violation_type === 'WRONG_WAY' ? 
                            '<span style="color: #f87171; font-weight: bold;">WRONG WAY</span>' : 
                            '<span style="color: #f59e0b; font-weight: bold;">ZONE INTRUSION</span>';
                        tbody.innerHTML += `
                            <tr>
                                <td>${new Date(v.timestamp * 1000).toLocaleTimeString()}</td>
                                <td>#${v.track_id}</td>
                                <td>${v.vehicle_type}</td>
                                <td>${typeTag} <span style="color: #64748b; font-size: 11px;">(${v.lane})</span></td>
                            </tr>`;
                    });
                }
            });
        }
    </script>
</body>
</html>
"""


# =============================================================================
# 2. STATE-OF-THE-ART TRAFFIC INTELLIGENCE ANALYTICS DASHBOARD
# =============================================================================
ANALYTICS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Traffic Intelligence Platform — Operations Analytics</title>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg-base: #080c14;
            --bg-card: #0f172a;
            --bg-card-sub: #131d31;
            --border-color: #1e293b;
            --border-sub: #253349;
            
            --t2b-color: #38bdf8;
            --t2b-bg: rgba(56, 189, 248, 0.12);
            --b2t-color: #34d399;
            --b2t-bg: rgba(52, 211, 153, 0.12);
            --viol-color: #f87171;
            --viol-bg: rgba(248, 113, 113, 0.12);
            --amber-color: #fbbf24;
            
            --text-main: #f1f5f9;
            --text-sub: #94a3b8;
            --text-dim: #64748b;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg-base); color: var(--text-main); min-height: 100vh; padding-bottom: 30px; }

        /* Top Command Navigation */
        .navbar { background: var(--bg-card); padding: 12px 32px; border-bottom: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 100; }
        .nav-brand { display: flex; align-items: center; gap: 16px; }
        .brand-title { font-size: 15px; font-weight: 800; color: #fff; letter-spacing: 1px; text-transform: uppercase; }
        .brand-sub { font-size: 11px; color: var(--text-dim); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; border-left: 1px solid var(--border-color); padding-left: 12px; }

        .live-badge { display: inline-flex; align-items: center; gap: 6px; background: rgba(52, 211, 153, 0.12); color: var(--b2t-color); font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(52, 211, 153, 0.25); }
        .live-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--b2t-color); box-shadow: 0 0 6px var(--b2t-color); animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

        .nav-telemetry { display: flex; align-items: center; gap: 20px; font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
        .tele-item { display: flex; align-items: center; gap: 6px; color: var(--text-sub); }
        .tele-val { font-weight: 700; color: #fff; }

        .nav-actions { display: flex; gap: 10px; align-items: center; }
        .btn { padding: 6px 14px; border-radius: 6px; font-weight: 600; font-size: 12px; cursor: pointer; border: none; text-decoration: none; transition: all 0.15s; display: inline-flex; align-items: center; gap: 6px; }
        .btn-live { background: var(--t2b-color); color: #080c14; }
        .btn-live:hover { background: #7dd3fc; }
        .btn-sec { background: var(--bg-card-sub); color: var(--text-main); border: 1px solid var(--border-sub); }
        .btn-sec:hover { background: #1a2742; }

        /* Container */
        .container { max-width: 1480px; margin: 0 auto; padding: 20px 32px; }

        /* Time Filter Toolbar */
        .toolbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; }
        .time-pill-group { display: inline-flex; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 6px; padding: 2px; }
        .time-pill { padding: 5px 14px; border-radius: 4px; font-size: 11px; font-weight: 700; color: var(--text-dim); background: transparent; border: none; cursor: pointer; transition: all 0.15s; text-transform: uppercase; letter-spacing: 0.5px; }
        .time-pill.active { background: var(--border-sub); color: #fff; }

        /* Top 4 KPI Row */
        .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px; }
        @media(max-width: 1000px) { .kpi-row { grid-template-columns: repeat(2, 1fr); } }
        
        .kpi-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; padding: 16px 18px; display: flex; flex-direction: column; position: relative; overflow: hidden; }
        .kpi-card.kpi-t2b { border-top: 3px solid var(--t2b-color); }
        .kpi-card.kpi-b2t { border-top: 3px solid var(--b2t-color); }
        .kpi-card.kpi-viol { border-top: 3px solid var(--viol-color); }
        .kpi-card.kpi-total { border-top: 3px solid #6366f1; }

        .kpi-title { font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-dim); letter-spacing: 0.5px; margin-bottom: 8px; }
        .kpi-val { font-size: 28px; font-weight: 800; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; line-height: 1; margin-bottom: 6px; }
        .kpi-sub { font-size: 11px; color: var(--text-sub); display: flex; align-items: center; gap: 6px; }

        /* Flow & Direction Layout */
        .grid-flow { display: grid; grid-template-columns: 2.2fr 1fr; gap: 16px; margin-bottom: 20px; }
        @media(max-width: 1100px) { .grid-flow { grid-template-columns: 1fr; } }

        .card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; padding: 18px; display: flex; flex-direction: column; }
        .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
        .card-title { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-sub); display: flex; align-items: center; gap: 8px; }
        .card-sub { font-size: 11px; color: var(--text-dim); }

        .chart-box-large { position: relative; width: 100%; height: 260px; }
        .chart-box-donut { position: relative; width: 100%; height: 160px; }

        /* Direction Split Widget */
        .direction-balance { display: flex; flex-direction: column; gap: 14px; padding: 8px 0; }
        .dir-split-row { display: flex; justify-content: space-between; align-items: flex-end; }
        .dir-label { font-size: 12px; font-weight: 700; color: var(--text-sub); }
        .dir-num { font-size: 20px; font-weight: 800; font-family: monospace; }
        .split-bar-wrap { height: 10px; width: 100%; background: #131d31; border-radius: 5px; overflow: hidden; display: flex; }
        .split-bar-t2b { height: 100%; background: var(--t2b-color); transition: width 0.4s ease; }
        .split-bar-b2t { height: 100%; background: var(--b2t-color); transition: width 0.4s ease; }

        /* Dual Grid Section: Lane Traffic + Vehicle Mix */
        .grid-mid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
        @media(max-width: 900px) { .grid-mid { grid-template-columns: 1fr; } }

        /* Mix Table */
        .mix-container { display: grid; grid-template-columns: 140px 1fr; gap: 14px; align-items: center; }
        .mix-table { width: 100%; border-collapse: collapse; font-size: 12px; }
        .mix-table td { padding: 6px 4px; border-bottom: 1px solid var(--border-color); }
        .mix-color-dot { width: 8px; height: 8px; border-radius: 2px; display: inline-block; margin-right: 6px; }

        /* Violations & Recent Tables Grid */
        .grid-bottom { display: grid; grid-template-columns: 1.1fr 1.9fr; gap: 16px; margin-bottom: 20px; }
        @media(max-width: 1024px) { .grid-bottom { grid-template-columns: 1fr; } }

        /* Table Design */
        .table-wrap { width: 100%; max-height: 250px; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 12px; font-family: 'Inter', sans-serif; }
        th { text-align: left; padding: 8px 10px; color: var(--text-dim); font-weight: 700; text-transform: uppercase; font-size: 10px; letter-spacing: 0.5px; border-bottom: 1px solid var(--border-color); background: var(--bg-card-sub); position: sticky; top: 0; }
        td { padding: 7px 10px; border-bottom: 1px solid #131d31; color: var(--text-main); }
        tr:hover td { background: rgba(30, 41, 59, 0.4); }

        .badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: 700; text-transform: uppercase; }
        .badge-t2b { background: var(--t2b-bg); color: var(--t2b-color); border: 1px solid rgba(56, 189, 248, 0.3); }
        .badge-b2t { background: var(--b2t-bg); color: var(--b2t-color); border: 1px solid rgba(52, 211, 153, 0.3); }
        .badge-viol-red { background: var(--viol-bg); color: var(--viol-color); border: 1px solid rgba(248, 113, 113, 0.3); }
        .badge-viol-amber { background: rgba(251, 191, 36, 0.12); color: var(--amber-color); border: 1px solid rgba(251, 191, 36, 0.3); }
        .badge-clean { color: var(--text-dim); }

        /* Operations Footer Bar */
        .ops-footer { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: var(--text-dim); font-family: monospace; }
        .ops-item { display: flex; align-items: center; gap: 6px; }
        .ops-val { color: var(--text-main); font-weight: 700; }

        /* Empty State */
        .empty-state { display: none; padding: 60px 20px; text-align: center; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; margin-bottom: 20px; }
        .empty-state.show { display: block; }
        .empty-title { font-size: 16px; font-weight: 700; color: var(--t2b-color); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 1px; }
        .empty-sub { font-size: 12px; color: var(--text-dim); max-width: 420px; margin: 0 auto 16px; }
    </style>
</head>
<body>
    <!-- Top Navigation Header -->
    <div class="navbar">
        <div class="nav-brand">
            <div class="brand-title">TRAFFIC INTELLIGENCE</div>
            <div class="brand-sub">Operations Analytics</div>
            <span class="live-badge" id="liveBadge"><span class="live-dot" id="liveDot"></span> LIVE</span>
        </div>

        <div class="nav-telemetry">
            <div class="tele-item">FPS <span class="tele-val" id="topFps">31.2</span></div>
            <div class="tele-item">ACTIVE TRACKS <span class="tele-val" id="topActiveTracks">0</span></div>
            <div class="tele-item">UPDATED <span class="tele-val" id="topUpdated">just now</span></div>
        </div>

        <div class="nav-actions">
            <a href="/" class="btn btn-live">🎥 Live Camera Stream</a>
            <button class="btn btn-sec" onclick="exportCSV()">📥 Export CSV</button>
        </div>
    </div>

    <div class="container">
        <!-- Toolbar: Time Filter Range -->
        <div class="toolbar">
            <div class="time-pill-group">
                <button class="time-pill active" onclick="setTimeFilter('live', this)">LIVE</button>
                <button class="time-pill" onclick="setTimeFilter('1m', this)">Last 1 min</button>
                <button class="time-pill" onclick="setTimeFilter('5m', this)">Last 5 min</button>
                <button class="time-pill" onclick="setTimeFilter('15m', this)">Last 15 min</button>
                <button class="time-pill" onclick="setTimeFilter('all', this)">All Session</button>
            </div>
            <div style="font-size: 11px; color: var(--text-dim); font-family: monospace;" id="timeWindowLabel">
                Displaying real-time streaming window
            </div>
        </div>

        <!-- Waiting / Empty State (shown if 0 vehicles) -->
        <div class="empty-state" id="emptyStateBox">
            <div class="empty-title">Waiting For Traffic Data</div>
            <div class="empty-sub">Inference has not started or no vehicles have passed through the configured regions yet.</div>
            <a href="/" class="btn btn-live">Go to Live View to Start Inference</a>
        </div>

        <div id="analyticsDashboardContent">
            <!-- 1. Top KPI Row -->
            <div class="kpi-row">
                <div class="kpi-card kpi-total">
                    <div class="kpi-title">TOTAL VEHICLES</div>
                    <div class="kpi-val" id="kpiTotalVehicles">0</div>
                    <div class="kpi-sub" id="kpiRecentAdded">+0 in current window</div>
                </div>

                <div class="kpi-card kpi-t2b">
                    <div class="kpi-title" style="color: var(--t2b-color);">T2B (TOP → BOTTOM)</div>
                    <div class="kpi-val" style="color: var(--t2b-color);" id="kpiT2B">0</div>
                    <div class="kpi-sub" id="kpiT2BPct">0.0% of flow</div>
                </div>

                <div class="kpi-card kpi-b2t">
                    <div class="kpi-title" style="color: var(--b2t-color);">B2T (BOTTOM → TOP)</div>
                    <div class="kpi-val" style="color: var(--b2t-color);" id="kpiB2T">0</div>
                    <div class="kpi-sub" id="kpiB2TPct">0.0% of flow</div>
                </div>

                <div class="kpi-card kpi-viol">
                    <div class="kpi-title" style="color: var(--viol-color);">RULE VIOLATIONS</div>
                    <div class="kpi-val" style="color: var(--viol-color);" id="kpiViolations">0</div>
                    <div class="kpi-sub" id="kpiViolRate">0.0% violation rate</div>
                </div>
            </div>

            <!-- 2. Main Traffic Flow Area Chart + Traffic Direction Balance -->
            <div class="grid-flow">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">📈 TRAFFIC FLOW (VEHICLES OVER TIME)</div>
                        <div class="card-sub">Directional Volume Timeline</div>
                    </div>
                    <div class="chart-box-large">
                        <canvas id="trafficFlowCanvas"></canvas>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div class="card-title">⚖️ TRAFFIC DIRECTION BALANCE</div>
                        <div class="card-sub">Flow Proportion</div>
                    </div>
                    <div class="direction-balance">
                        <div class="dir-split-row">
                            <div>
                                <div class="dir-label" style="color: var(--t2b-color);">T2B</div>
                                <div class="dir-num" style="color: var(--t2b-color);" id="splitT2BCount">0</div>
                            </div>
                            <div style="text-align: right;">
                                <div class="dir-label" style="color: var(--b2t-color);">B2T</div>
                                <div class="dir-num" style="color: var(--b2t-color);" id="splitB2TCount">0</div>
                            </div>
                        </div>

                        <div class="split-bar-wrap">
                            <div class="split-bar-t2b" id="splitBarT2B" style="width: 50%;"></div>
                            <div class="split-bar-b2t" id="splitBarB2T" style="width: 50%;"></div>
                        </div>

                        <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-dim); font-family: monospace;">
                            <span id="splitT2BPct">50.0%</span>
                            <span id="splitB2TPct">50.0%</span>
                        </div>

                        <div style="margin-top: 10px; border-top: 1px solid var(--border-color); padding-top: 12px;">
                            <div style="font-size: 11px; color: var(--text-sub); margin-bottom: 6px; font-weight: 700;">ACTIVE FLOW DOMINANCE</div>
                            <div style="font-size: 12px; color: #fff;" id="dominanceText">Balanced bidirectional traffic</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 3. Dual Grid: Lane Traffic Analytics + Vehicle Mix -->
            <div class="grid-mid">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">🛣️ LANE TRAFFIC ANALYTICS</div>
                        <div class="card-sub">Directional Volume per Lane</div>
                    </div>
                    <div style="height: 180px; position: relative;">
                        <canvas id="laneChartCanvas"></canvas>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div class="card-title">🚗 VEHICLE MIX</div>
                        <div class="card-sub">Class Distribution</div>
                    </div>
                    <div class="mix-container">
                        <div class="chart-box-donut">
                            <canvas id="classDonutCanvas"></canvas>
                        </div>
                        <table class="mix-table">
                            <tbody id="classMixTableBody">
                                <tr><td><span class="mix-color-dot" style="background:#38bdf8;"></span>Cars</td><td style="text-align:right; font-weight:700;" id="cntCars">0</td></tr>
                                <tr><td><span class="mix-color-dot" style="background:#34d399;"></span>Motorcycles</td><td style="text-align:right; font-weight:700;" id="cntBikes">0</td></tr>
                                <tr><td><span class="mix-color-dot" style="background:#f59e0b;"></span>Buses</td><td style="text-align:right; font-weight:700;" id="cntBuses">0</td></tr>
                                <tr><td><span class="mix-color-dot" style="background:#ef4444;"></span>Trucks</td><td style="text-align:right; font-weight:700;" id="cntTrucks">0</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- 4. Violations Log & Live Vehicle Activity -->
            <div class="grid-bottom">
                <!-- Violations Panel -->
                <div class="card" style="border-color: rgba(248, 113, 113, 0.3);">
                    <div class="card-header">
                        <div class="card-title" style="color: var(--viol-color);">⚠️ RULE VIOLATIONS LOG</div>
                        <span id="violBadgeTotal" class="badge badge-viol-red">0 logged</span>
                    </div>
                    <div class="table-wrap">
                        <table>
                            <thead>
                                <tr><th>Time</th><th>ID</th><th>Lane</th><th>Type</th></tr>
                            </thead>
                            <tbody id="violLogTableBody">
                                <tr><td colspan="4" style="text-align:center; color:var(--text-dim);">No violations recorded</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Live Vehicle Activity -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">📋 LIVE VEHICLE ACTIVITY</div>
                        <input type="text" id="liveSearchInput" placeholder="🔍 Filter vehicles..." onkeyup="filterLiveVehicles()" style="background: #131d31; border: 1px solid var(--border-color); color: #fff; padding: 4px 8px; border-radius: 4px; font-size: 11px; width: 160px;">
                    </div>
                    <div class="table-wrap">
                        <table id="liveVehiclesTable">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>ID</th>
                                    <th>Type</th>
                                    <th>Lane</th>
                                    <th>Direction</th>
                                    <th>Duration</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody id="liveVehiclesTableBody">
                                <tr><td colspan="7" style="text-align:center; color:var(--text-dim);">Awaiting live tracking records...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- 5. Operations & System Status Bar -->
        <div class="ops-footer">
            <div class="ops-item">VIDEO SOURCE: <span class="ops-val" id="footVideo">traffic_2.mp4</span></div>
            <div class="ops-item">STATUS: <span class="ops-val" id="footStatus" style="color: var(--b2t-color);">● PROCESSING</span></div>
            <div class="ops-item">ENGINE: <span class="ops-val">YOLOv11 FP16 (CUDA)</span></div>
            <div class="ops-item">DATABASE: <span class="ops-val" style="color: var(--b2t-color);">● WAL CONNECTED</span></div>
            <div class="ops-item">LAST EVENT: <span class="ops-val" id="footLastEvent">0s ago</span></div>
        </div>
    </div>

    <!-- Live Polling & Chart Update Logic -->
    <script>
        let currentRange = 'live';
        let latestData = null;
        let flowChart = null;
        let laneChart = null;
        let classDonut = null;
        let lastUpdateTime = Date.now();

        window.onload = function() {
            initCharts();
            fetchData();
            // High-frequency, lightweight polling every 1.5 seconds
            setInterval(fetchData, 1500);
            setInterval(updateClockElapsed, 1000);
        };

        function setTimeFilter(range, btn) {
            currentRange = range;
            document.querySelectorAll('.time-pill').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const labels = {
                'live': 'Displaying real-time streaming window',
                '1m': 'Displaying records from the last 60 seconds',
                '5m': 'Displaying records from the last 5 minutes',
                '15m': 'Displaying records from the last 15 minutes',
                'all': 'Displaying all cumulative records from this session'
            };
            document.getElementById('timeWindowLabel').textContent = labels[range] || '';
            fetchData();
        }

        function updateClockElapsed() {
            const sec = Math.round((Date.now() - lastUpdateTime) / 1000);
            document.getElementById('topUpdated').textContent = sec <= 1 ? 'just now' : sec + 's ago';
        }

        function fetchData() {
            fetch('/api/analytics?time_range=' + currentRange)
                .then(r => r.json())
                .then(data => {
                    latestData = data;
                    lastUpdateTime = Date.now();
                    updateClockElapsed();
                    renderUI(data);
                })
                .catch(err => console.error("Analytics fetch error:", err));
        }

        function renderUI(data) {
            const kpis = data.kpis || {};
            const total = kpis.total_vehicles || 0;

            // Empty state toggle
            if (total === 0) {
                document.getElementById('emptyStateBox').classList.add('show');
                document.getElementById('analyticsDashboardContent').style.opacity = '0.4';
            } else {
                document.getElementById('emptyStateBox').classList.remove('show');
                document.getElementById('analyticsDashboardContent').style.opacity = '1';
            }

            // Top Telemetry
            if (data.system_status) {
                document.getElementById('topFps').textContent = data.system_status.fps.toFixed(1);
                document.getElementById('footVideo').textContent = data.system_status.video_name || 'traffic_2.mp4';
                if (!data.system_status.is_running) {
                    document.getElementById('footStatus').textContent = '⏹️ STOPPED';
                    document.getElementById('footStatus').style.color = '#ef4444';
                    document.getElementById('liveDot').style.background = '#ef4444';
                }
            }
            document.getElementById('topActiveTracks').textContent = kpis.active_tracks || 0;
            document.getElementById('footLastEvent').textContent = (kpis.last_activity_sec || 0) + 's ago';

            // Top KPIs
            document.getElementById('kpiTotalVehicles').textContent = total.toLocaleString();
            document.getElementById('kpiRecentAdded').textContent = `+${kpis.recent_added || 0} in last 60s`;

            const t2b = kpis.t2b_vehicles || 0;
            const b2t = kpis.b2t_vehicles || 0;
            const t2bPct = total > 0 ? ((t2b / total) * 100).toFixed(1) : '50.0';
            const b2tPct = total > 0 ? ((b2t / total) * 100).toFixed(1) : '50.0';

            document.getElementById('kpiT2B').textContent = t2b.toLocaleString();
            document.getElementById('kpiT2BPct').textContent = `${t2bPct}% of total flow`;

            document.getElementById('kpiB2T').textContent = b2t.toLocaleString();
            document.getElementById('kpiB2TPct').textContent = `${b2tPct}% of total flow`;

            const viols = kpis.total_violations || 0;
            const violRate = total > 0 ? ((viols / total) * 100).toFixed(1) : '0.0';
            document.getElementById('kpiViolations').textContent = viols.toLocaleString();
            document.getElementById('kpiViolRate').textContent = `${violRate}% of vehicles (${kpis.wrong_way || 0} Wrong-Way)`;

            // Direction Balance Widget
            document.getElementById('splitT2BCount').textContent = t2b;
            document.getElementById('splitB2TCount').textContent = b2t;
            document.getElementById('splitBarT2B').style.width = t2bPct + '%';
            document.getElementById('splitBarB2T').style.width = b2tPct + '%';
            document.getElementById('splitT2BPct').textContent = t2bPct + '%';
            document.getElementById('splitB2TPct').textContent = b2tPct + '%';

            if (t2b > b2t * 1.2) {
                document.getElementById('dominanceText').textContent = 'Heavy Top → Bottom (T2B) flow dominance';
            } else if (b2t > t2b * 1.2) {
                document.getElementById('dominanceText').textContent = 'Heavy Bottom → Top (B2T) flow dominance';
            } else {
                document.getElementById('dominanceText').textContent = 'Balanced bidirectional traffic volume';
            }

            // Update Charts in place without re-creating
            updateFlowChart(data.flow_timeline);
            updateLaneChart(data.lanes);
            updateClassMix(kpis, data.classes);
            updateViolationsLog(data.recent_violations, viols);
            updateVehiclesTable(data.vehicles_table);
        }

        // ================= CHART INITIALIZATION =================
        function initCharts() {
            // 1. Traffic Flow Area Line Chart
            const ctxFlow = document.getElementById('trafficFlowCanvas').getContext('2d');
            flowChart = new Chart(ctxFlow, {
                type: 'line',
                data: {
                    labels: ['0s'],
                    datasets: [
                        {
                            label: 'T2B (Top → Bottom)',
                            data: [0],
                            borderColor: '#38bdf8',
                            backgroundColor: 'rgba(56, 189, 248, 0.12)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.35,
                            pointRadius: 2,
                            pointHoverRadius: 5
                        },
                        {
                            label: 'B2T (Bottom → Top)',
                            data: [0],
                            borderColor: '#34d399',
                            backgroundColor: 'rgba(52, 211, 153, 0.12)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.35,
                            pointRadius: 2,
                            pointHoverRadius: 5
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false, // Prevents lag on rapid polling
                    scales: {
                        x: { grid: { color: '#131d31' }, ticks: { color: '#64748b', font: { size: 10 } } },
                        y: { grid: { color: '#131d31' }, ticks: { color: '#64748b', stepSize: 1, font: { size: 10 } }, beginAtZero: true }
                    },
                    plugins: {
                        legend: { position: 'top', align: 'end', labels: { color: '#cbd5e1', boxWidth: 10, font: { size: 11, weight: 'bold' } } },
                        tooltip: { backgroundColor: '#0f172a', borderColor: '#334155', borderWidth: 1 }
                    }
                }
            });

            // 2. Lane Traffic Horizontal Stacked Bar Chart
            const ctxLane = document.getElementById('laneChartCanvas').getContext('2d');
            laneChart = new Chart(ctxLane, {
                type: 'bar',
                data: {
                    labels: ['Lane 1', 'Lane 2'],
                    datasets: [
                        { label: 'T2B', data: [0, 0], backgroundColor: '#38bdf8', borderRadius: 3 },
                        { label: 'B2T', data: [0, 0], backgroundColor: '#34d399', borderRadius: 3 }
                    ]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    scales: {
                        x: { stacked: true, grid: { color: '#131d31' }, ticks: { color: '#64748b', stepSize: 1, font: { size: 10 } }, beginAtZero: true },
                        y: { stacked: true, grid: { display: false }, ticks: { color: '#cbd5e1', font: { size: 11, weight: 'bold' } } }
                    },
                    plugins: {
                        legend: { position: 'bottom', labels: { color: '#cbd5e1', boxWidth: 10, font: { size: 10 } } }
                    }
                }
            });

            // 3. Class Donut Chart
            const ctxClass = document.getElementById('classDonutCanvas').getContext('2d');
            classDonut = new Chart(ctxClass, {
                type: 'doughnut',
                data: {
                    labels: ['Cars', 'Motorcycles', 'Buses', 'Trucks'],
                    datasets: [{
                        data: [1, 0, 0, 0],
                        backgroundColor: ['#38bdf8', '#34d399', '#f59e0b', '#ef4444'],
                        borderColor: '#0f172a',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    cutout: '70%',
                    plugins: { legend: { display: false } }
                }
            });
        }

        // ================= IN-PLACE DATASET UPDATES =================
        function updateFlowChart(flow) {
            if (!flowChart) return;
            if (!flow || flow.length === 0) {
                flowChart.data.labels = ['0s'];
                flowChart.data.datasets[0].data = [0];
                flowChart.data.datasets[1].data = [0];
            } else {
                flowChart.data.labels = flow.map(f => f.time_label || (f.time_sec + 's'));
                flowChart.data.datasets[0].data = flow.map(f => f.t2b);
                flowChart.data.datasets[1].data = flow.map(f => f.b2t);
            }
            flowChart.update('none'); // Instant paint without layout jump
        }

        function updateLaneChart(lanes) {
            if (!laneChart) return;
            if (!lanes || lanes.length === 0) {
                laneChart.data.labels = ['No Lanes'];
                laneChart.data.datasets[0].data = [0];
                laneChart.data.datasets[1].data = [0];
            } else {
                laneChart.data.labels = lanes.map(l => l.name);
                laneChart.data.datasets[0].data = lanes.map(l => l.t2b);
                laneChart.data.datasets[1].data = lanes.map(l => l.b2t);
            }
            laneChart.update('none');
        }

        function updateClassMix(kpis, classes) {
            const cars = kpis.cars || 0;
            const bikes = kpis.motorcycles || 0;
            const buses = kpis.buses || 0;
            const trucks = kpis.trucks || 0;

            document.getElementById('cntCars').textContent = cars;
            document.getElementById('cntBikes').textContent = bikes;
            document.getElementById('cntBuses').textContent = buses;
            document.getElementById('cntTrucks').textContent = trucks;

            if (classDonut) {
                classDonut.data.datasets[0].data = [cars, bikes, buses, trucks];
                classDonut.update('none');
            }
        }

        function updateViolationsLog(viols, totalViols) {
            document.getElementById('violBadgeTotal').textContent = totalViols + ' logged';
            const tbody = document.getElementById('violLogTableBody');
            if (!viols || viols.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:var(--text-dim);">No violations recorded</td></tr>';
                return;
            }

            let html = '';
            viols.slice(0, 10).forEach(v => {
                const tagClass = v.violation_type === 'WRONG_WAY' ? 'badge-viol-red' : 'badge-viol-amber';
                const tagLabel = v.violation_type === 'WRONG_WAY' ? 'WRONG WAY' : 'ZONE INTRUSION';
                html += `
                    <tr>
                        <td style="font-family:monospace; color:var(--text-dim);">${v.time_str || new Date(v.timestamp*1000).toLocaleTimeString()}</td>
                        <td><b>#${v.track_id}</b></td>
                        <td>${v.lane}</td>
                        <td><span class="badge ${tagClass}">${tagLabel}</span></td>
                    </tr>`;
            });
            tbody.innerHTML = html;
        }

        function updateVehiclesTable(vehicles) {
            const tbody = document.getElementById('liveVehiclesTableBody');
            if (!vehicles || vehicles.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; color:var(--text-dim);">Awaiting live tracking records...</td></tr>';
                return;
            }

            let html = '';
            vehicles.slice(0, 20).forEach(v => {
                const dirBadge = v.direction === 'T2B' ? '<span class="badge badge-t2b">T2B</span>' : 
                                 (v.direction === 'B2T' ? '<span class="badge badge-b2t">B2T</span>' : v.direction);
                const violBadge = v.violation !== 'None' ? `<span class="badge badge-viol-red">${v.violation}</span>` : '<span class="badge badge-clean">Normal</span>';
                
                html += `
                    <tr>
                        <td style="font-family:monospace; color:var(--text-dim);">${v.last_seen_str || new Date(v.last_seen*1000).toLocaleTimeString()}</td>
                        <td><b>#${v.track_id}</b></td>
                        <td style="text-transform:capitalize;">${v.vehicle_type}</td>
                        <td>${v.lane}</td>
                        <td>${dirBadge}</td>
                        <td style="font-family:monospace;">${v.duration}s</td>
                        <td>${violBadge}</td>
                    </tr>`;
            });
            tbody.innerHTML = html;
        }

        function filterLiveVehicles() {
            const q = document.getElementById('liveSearchInput').value.toLowerCase();
            const rows = document.querySelectorAll('#liveVehiclesTableBody tr');
            rows.forEach(r => {
                const txt = r.textContent.toLowerCase();
                r.style.display = (q === '' || txt.includes(q)) ? '' : 'none';
            });
        }

        function exportCSV() {
            if (!latestData || !latestData.vehicles_table) return;
            let csv = "Track_ID,Vehicle_Type,Direction,Lane,First_Seen,Last_Seen,Duration_Sec,Total_Frames,Violation\\n";
            latestData.vehicles_table.forEach(v => {
                csv += `${v.track_id},${v.vehicle_type},${v.direction},${v.lane},${v.first_seen},${v.last_seen},${v.duration},${v.total_frames},${v.violation}\\n`;
            });
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.setAttribute('href', url);
            a.setAttribute('download', 'traffic_intelligence_export_' + Date.now() + '.csv');
            a.click();
        }
    </script>
</body>
</html>
"""


# =============================================================================
# 3. FLASK APP & ROUTE ORCHESTRATION
# =============================================================================
class Dashboard:
    def __init__(self, storage: EventStorage, pipeline=None, port: int = 5000):
        self.storage = storage
        self.pipeline = pipeline
        self.port = port
        self.app = Flask(__name__)
        self.inference_thread: threading.Thread | None = None
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route("/")
        def index():
            return render_template_string(LIVE_DASHBOARD_HTML)

        @self.app.route("/analytics")
        def analytics_page():
            return render_template_string(ANALYTICS_HTML)

        @self.app.route("/api/first_frame")
        def api_first_frame():
            frame = None
            if self.pipeline:
                frame = self.pipeline.get_first_frame()
            if frame is None:
                video_name = self.pipeline.config.video_path if self.pipeline else "traffic_2.mp4"
                candidates = [
                    video_name,
                    str(Path(video_name).resolve()),
                    str(Path.cwd() / video_name),
                    str(Path.cwd() / Path(video_name).name),
                    "traffic_2.mp4",
                    "traffic_1.mp4",
                ]
                for cand in candidates:
                    if Path(cand).is_file():
                        cap = cv2.VideoCapture(cand)
                        if cap.isOpened():
                            ret, f = cap.read()
                            cap.release()
                            if ret and f is not None:
                                frame = f
                                break
            if frame is None:
                frame = 20 * np.ones((720, 1280, 3), dtype=np.uint8)
                cv2.putText(frame, "No Frame Found (Check Video Path)", (320, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            resp = Response(buf.tobytes(), mimetype="image/jpeg")
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp

        @self.app.route("/api/get_regions")
        def api_get_regions():
            reg_path = Path(OUTPUT_DIR) / "lanes_config.json"
            if reg_path.is_file():
                try:
                    with open(reg_path, "r") as f:
                        return jsonify(json.load(f))
                except Exception:
                    pass
            if self.pipeline and self.pipeline.lanes:
                lanes_data = [
                    {"name": l.name, "direction": l.direction, "polygon": l.polygon}
                    for l in self.pipeline.lanes
                ]
                return jsonify({"lanes": lanes_data, "zones": self.pipeline.zones or {}})
            return jsonify({"lanes": [], "zones": {}})

        @self.app.route("/api/start_inference", methods=["POST"])
        def api_start_inference():
            data = request.get_json() or {}
            lanes_data = data.get("lanes", [])
            zones_data = data.get("zones", {})

            # Save configuration to lanes_config.json
            out_file = Path(OUTPUT_DIR) / "lanes_config.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w") as f:
                json.dump({"lanes": lanes_data, "zones": zones_data}, f, indent=2)

            lanes_objs = [
                LaneConfig(
                    name=l["name"],
                    direction=l.get("direction", "T2B"),
                    polygon=[(int(p[0]), int(p[1])) for p in l.get("polygon", [])],
                )
                for l in lanes_data
            ]

            if self.pipeline:
                self.pipeline.set_regions(lanes_objs, zones_data)
                if not self.inference_thread or not self.inference_thread.is_alive():
                    self.pipeline.running = True
                    self.pipeline.is_paused = False
                    self.inference_thread = threading.Thread(target=self.pipeline.run, daemon=True)
                    self.inference_thread.start()

            return jsonify({"status": "ok", "lanes_count": len(lanes_objs)})

        @self.app.route("/api/pause", methods=["POST"])
        def api_pause():
            if self.pipeline:
                self.pipeline.is_paused = not self.pipeline.is_paused
                return jsonify({"status": "ok", "is_paused": self.pipeline.is_paused})
            return jsonify({"status": "error"})

        @self.app.route("/api/stop", methods=["POST"])
        def api_stop():
            if self.pipeline:
                self.pipeline.running = False
                return jsonify({"status": "ok"})
            return jsonify({"status": "error"})

        @self.app.route("/api/summary")
        def api_summary():
            summary = self.storage.get_summary()
            if self.pipeline:
                summary["frame_idx"] = self.pipeline.frame_idx
                summary["is_running"] = self.pipeline.running
                summary["is_paused"] = self.pipeline.is_paused
                summary["fps"] = getattr(self.pipeline, "current_fps", 0.0)
            return jsonify(summary)

        @self.app.route("/api/violations")
        def api_violations():
            return jsonify(self.storage.get_violations(30))

        @self.app.route("/api/analytics")
        def api_analytics():
            """Returns rich analytics payload directly from SQLite without running CV."""
            time_range = request.args.get("time_range", "live")
            payload = self.storage.get_analytics_payload(time_range=time_range)
            
            # Attach live pipeline metadata
            payload["system_status"] = {
                "is_running": self.pipeline.running if self.pipeline else False,
                "is_paused": self.pipeline.is_paused if self.pipeline else False,
                "fps": getattr(self.pipeline, "current_fps", 0.0) if self.pipeline else 0.0,
                "frame_idx": getattr(self.pipeline, "frame_idx", 0) if self.pipeline else 0,
                "video_name": Path(self.pipeline.config.video_path).name if self.pipeline else "traffic_2.mp4",
            }
            return jsonify(payload)

        @self.app.route("/video_feed")
        def video_feed():
            def generate():
                init_frame = None
                if self.pipeline:
                    init_frame = self.pipeline.latest_annotated or self.pipeline.get_first_frame()
                if init_frame is not None:
                    ret, jpeg = cv2.imencode(".jpg", init_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        yield (b"--frame\r\n"
                               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

                while True:
                    if self.pipeline and self.pipeline.latest_annotated is not None:
                        frame = self.pipeline.latest_annotated
                        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ret:
                            yield (b"--frame\r\n"
                                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
                    time.sleep(0.033)  # ~30 FPS
            return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def run(self):
        import signal
        import sys

        def handle_shutdown(sig=None, frame=None):
            print("\n\n[SHUTDOWN] Ctrl+C received. Stopping Traffic Intelligence System...")
            if self.pipeline:
                self.pipeline.running = False
                try:
                    self.pipeline.storage.close()
                except Exception:
                    pass
            print("[SHUTDOWN] System stopped cleanly.")
            sys.exit(0)

        try:
            signal.signal(signal.SIGINT, handle_shutdown)
            signal.signal(signal.SIGTERM, handle_shutdown)
        except Exception:
            pass

        try:
            from werkzeug.serving import make_server
            server = make_server("0.0.0.0", self.port, self.app, threaded=True)
            server.serve_forever()
        except KeyboardInterrupt:
            handle_shutdown()
        except Exception:
            try:
                self.app.run(host="0.0.0.0", port=self.port, debug=False, threaded=True)
            except KeyboardInterrupt:
                handle_shutdown()
