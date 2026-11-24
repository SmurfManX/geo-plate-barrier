# Hardware integration

What is needed to run Plate Barrier on a real gate.

```
 ANPR IP camera ──PoE──► switch ──► Linux box (Docker: plate-barrier) ──► network relay ──► barrier controller "OPEN" input
                                          │
                                   operator UI (VPN)
```

## Bill of materials

| Item | Requirements | Notes |
|---|---|---|
| Camera | IP camera from an ANPR/LPR line; 2 MP; IR illuminator; shutter 1/1000 s or faster; varifocal lens; RTSP; PoE | Plate must be at least 150–200 px wide in the frame. Some models recognise plates on-board. |
| Computer | Mini-PC on Intel N100 (or better), 8 GB RAM, SSD, Linux, Docker | Raspberry Pi 5 also works, but more slowly. |
| Relay | Ethernet relay (HTTP or Modbus TCP), or USB relay, or relay HAT for Raspberry Pi | One normally-open contact is enough. |
| Network | PoE switch, separate VLAN for camera + computer | Operator UI via VPN only. |
| Power | Small UPS for computer, switch and camera | |

## Camera placement

- 3–6 m before the stop line, horizontal angle ≤ 30°, vertical angle ≤ 30°.
- Aim at where the car stops, not at the barrier arm.
- Adjust zoom so the plate is 150–200+ px wide, and check it at night with IR on.

## Wiring the barrier

Almost every barrier controller has an **OPEN** input (dry contact, often also used by a push-button or a remote receiver).

- The relay contact goes in parallel with that input; the software closes it for 0.5–1 s on `granted`.
- **Closing stays with the controller**: auto-close after the car passes the safety loop / photocells.
  The software must never close the arm, so remove the "Close" button in a real installation.
- Keep the existing remote/keypad as a fallback.

## Software setup

```bash
# on the Linux box (the image must be built there: a Mac build is arm64)
git clone <repo> && cd plate-barrier
# compose.yaml:  CAMERA_SOURCE: "rtsp://user:pass@<camera-ip>:554/stream1"
docker compose up -d --build
```

For a USB camera use `CAMERA_SOURCE: "/dev/video0"` and uncomment `devices` / `group_add` in `compose.yaml`.

## Still to implement before going live

1. **Relay driver**: send the OPEN pulse where `gate.py` decides `granted`.
2. **Authentication + HTTPS** for the web UI and API. Right now anyone on the network can open the gate.
3. **Database backup** (the `barrier-data` volume) and NTP time sync.
4. **Shadow mode**: run for a few days with the relay disconnected, check the log against real traffic,
   tune `MIN_AGREE` / `FRAME_CONF`, and fine-tune the OCR on recorded plates if needed.
