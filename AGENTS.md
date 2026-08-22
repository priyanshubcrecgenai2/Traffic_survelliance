# AGENTS.md

# Traffic Intelligence System

Build a modular traffic-intelligence application that processes a fixed
traffic-camera video and converts the scene into structured, searchable
traffic data.

The current camera view contains a divided multi-lane road with vehicles
moving in two primary directions:

- T2B — Top to Bottom
- B2T — Bottom to Top

The system should look like a polished traffic analytics product rather than
a raw computer-vision demo.

---

# Primary Goals

The system should provide:

- vehicle detection
- vehicle instance segmentation
- multi-object tracking
- vehicle classification
- lane-wise counting
- direction-wise counting
- number-plate detection
- ANPR/OCR
- pedestrian region monitoring
- cyclist region monitoring
- traffic event storage
- analytics dashboard
- visual evidence for detected vehicles and events

The architecture must remain modular so each computer-vision component can be
developed, tested and improved independently.

---

# Input Video

The primary input is a locally downloaded traffic video.

The system should work with a normal local video file and should not depend on
a live YouTube stream.

The camera is fixed and provides a clear elevated view of a multi-lane road.

The important traffic directions are:

```text
T2B = Top → Bottom
B2T = Bottom → Top