# 3D Viz Visualization Setup

## Problem

The 3D visualization (viz.html) cannot be loaded directly via `file://` protocol due to browser security restrictions. When you open index.html directly in a browser, the iframe will show "Error loading data: Failed to fetch".

## Solution

You need to serve the website through an HTTP server. Here are several options:

### Option 1: Python HTTP Server (Recommended)

From the project root directory, run:

```bash
# Python 3
python3 -m http.server 8000

# Or Python 2
python -m SimpleHTTPServer 8000
```

Then open: http://localhost:8000

### Option 2: Node.js HTTP Server

If you have Node.js installed:

```bash
# Install http-server globally
npm install -g http-server

# Run server
http-server -p 8000
```

Then open: http://localhost:8000

### Option 3: PHP Built-in Server

If you have PHP installed:

```bash
php -S localhost:8000
```

Then open: http://localhost:8000

### Option 4: VS Code Live Server Extension

If you're using VS Code:

1. Install the "Live Server" extension
2. Right-click on index.html
3. Select "Open with Live Server"

## File Structure

```
demos/
├── viz.html       # 3D visualization viewer (101KB)
├── data.bin       # Binary data file (68MB, 100 frames, 512x512)
├── visualize_viz_skip_occluded.py  # Python script to generate data.bin
└── README.md      # This file
```

## How viz.html Works

1. viz.html loads via iframe from index.html
2. It fetches `data.bin` from the same directory
3. data.bin contains:
   - RGB video frames (100 frames, 512x512x3)
   - Depth maps (encoded as RGB)
   - Camera intrinsics and extrinsics
   - 3D point trajectories
   - Metadata (FOV, depth range, etc.)
4. The viewer uses Three.js + WebGL to render the 3D scene

## Generating New data.bin

If you need to generate a new data.bin file, use the Python script:

```bash
python demos/visualize_viz_skip_occluded.py
```

The script requires:
- numpy
- PIL (Pillow)
- opencv-python (cv2)
- matplotlib
- open3d
- tqdm

## Troubleshooting

### "Error loading data: Failed to fetch"

**Cause**: You're opening index.html directly with `file://` protocol.

**Fix**: Use an HTTP server (see solutions above).

### Viz is blank or shows loading spinner forever

**Possible causes**:
1. data.bin is missing or corrupted
2. data.bin format is incorrect
3. Browser console shows errors

**Fix**:
1. Check that `demos/data.bin` exists (should be ~68MB)
2. Open browser DevTools (F12) and check Console for errors
3. Verify data.bin format with:
   ```bash
   hexdump -C demos/data.bin | head -5
   ```
   Should show JSON header starting with `{"rgb_video":`

### Performance issues

The viz viewer loads a large dataset (68MB). On slower machines:
- Initial loading may take 5-10 seconds
- Reduce quality in viz.html settings (lower resolution, fewer frames)
- Or regenerate data.bin with smaller dimensions using the Python script

## Notes

- The current data.bin contains 100 frames at 512x512 resolution
- Loading time depends on your connection speed (for remote servers) or disk speed (for local)
- The viewer supports playback controls, camera rotation, and point cloud visualization
