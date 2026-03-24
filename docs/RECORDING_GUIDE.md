# Demo Recording Guide

How to capture terminal recordings and screen recordings of GraphBot demos for the README, launch posts, and documentation.

## Quick Start

```bash
# First, verify demos run cleanly in dry-run mode
./scripts/record_demos.sh

# Then record with one of the tools below
```

## Terminal Recordings (asciinema)

[asciinema](https://asciinema.org/) captures terminal sessions as lightweight text-based recordings that can be embedded anywhere.

### Install

```bash
# macOS
brew install asciinema

# Linux (pip)
pip install asciinema

# Windows (WSL recommended)
wsl --install   # if not already set up
wsl pip install asciinema
```

### Record

```bash
# Record the flight search demo
asciinema rec demos/flight_search.cast \
    --title "GraphBot: Flight Search (AMS -> BCN)" \
    --command "python scripts/demo_flight_search.py --dry-run --verbose"

# Record the research report demo
asciinema rec demos/research_report.cast \
    --title "GraphBot: Research Report Pipeline" \
    --command "python scripts/demo_research_report.py --dry-run --verbose"
```

### Convert to GIF

```bash
# Install agg (asciinema gif generator)
cargo install --git https://github.com/asciinema/agg

# Convert .cast to .gif
agg demos/flight_search.cast demos/flight_search.gif --font-size 14 --cols 100 --rows 30
agg demos/research_report.cast demos/research_report.gif --font-size 14 --cols 100 --rows 30
```

### Upload for embedding

```bash
# Upload to asciinema.org (get embed link)
asciinema upload demos/flight_search.cast
asciinema upload demos/research_report.cast
```

Embed in README:
```markdown
[![Flight Search Demo](https://asciinema.org/a/YOUR_ID.svg)](https://asciinema.org/a/YOUR_ID)
```

## GIF Recordings (ScreenToGif)

[ScreenToGif](https://www.screentogif.com/) is the best option for Windows. It captures any region of the screen and exports directly to GIF.

### Install

Download from https://www.screentogif.com/ or:

```powershell
winget install NickeManarin.ScreenToGif
```

### Recording workflow

1. Open ScreenToGif, select "Recorder"
2. Position the capture region over the terminal window
3. Set frame rate to 15 FPS (good balance of quality/size)
4. Click Record, then run the demo script in the terminal
5. Click Stop when the demo finishes
6. In the editor: trim dead frames at start/end, add title frame if desired
7. Export as GIF to `demos/` directory

### Recommended settings

- Resolution: 1280x720 or 960x540 (for web embedding)
- Frame rate: 15 FPS
- Max file size target: 5 MB per GIF (GitHub renders up to 10 MB)
- Color depth: 256 colors

## Split-Screen Recording Concept

The most compelling demo visual is a **split-screen** showing the terminal pipeline on the left and the WhatsApp phone mockup on the right.

### Layout

```
+-----------------------------------+------------------+
|                                   |                  |
|  Terminal (GraphBot pipeline)     |  Phone mockup    |
|                                   |  (WhatsApp)      |
|  $ python demo_flight_search.py   |                  |
|  === Flight Search Demo ===       |  [chat bubbles]  |
|  DAG: 4 nodes                     |                  |
|  [1/4] Browser search...          |                  |
|  [2/4] Data extraction...         |  +----------+   |
|  [3/4] Formatting...              |  | Cheap     |   |
|  [4/4] Delivering message...      |  | flights:  |   |
|                                   |  | AMS->BCN  |   |
|  Results saved.                   |  | EUR 39... |   |
|  === Done ===                     |  +----------+   |
|                                   |                  |
+-----------------------------------+------------------+
```

### How to create the split-screen

**Option A: Two recordings composited (recommended)**

1. Record the terminal demo as a standalone GIF (see above)
2. Record the WhatsApp Web interface receiving the message (or mock it)
3. Composite side-by-side using FFmpeg:

```bash
ffmpeg -i demos/terminal.gif -i demos/whatsapp.gif \
    -filter_complex "[0]scale=960:720[left];[1]scale=480:720[right];[left][right]hconcat" \
    demos/split_screen.gif
```

**Option B: ScreenToGif with dual monitors**

1. Arrange terminal window on left half, WhatsApp Web on right half
2. Capture both with a single ScreenToGif region

**Option C: Static mockup (simplest)**

1. Take a screenshot of the terminal output
2. Create a WhatsApp phone mockup (use Figma or any mockup tool)
3. Composite in any image editor
4. Save as PNG to `demos/split_screen.png`

### Phone mockup resources

- Figma: Search "iPhone mockup" in community files
- Online: https://mockuphone.com/ or https://deviceframes.com/
- The WhatsApp message content is in `demos/flight_search_results.md` under "Formatted Message"

## Web Dashboard Recording

For recording the Next.js dashboard with live DAG visualization:

1. Start the dev server: `cd ui/frontend && npm run dev`
2. Open http://localhost:3000
3. Start a task via the UI or API
4. Record with ScreenToGif or OBS (for longer recordings)
5. Export to GIF or MP4

Key things to capture:
- DAG nodes appearing and animating through states (pending -> running -> complete)
- Knowledge graph visualization updating in real-time
- Cost footer showing near-zero cost

## File naming convention

```
demos/
  flight_search.gif          # Terminal recording of flight search demo
  research_report.gif        # Terminal recording of research report demo
  split_screen.gif           # Split-screen: terminal + WhatsApp
  dashboard_dag.gif          # Dashboard DAG visualization
  flight_search.cast         # asciinema raw recording
  research_report.cast       # asciinema raw recording
  flight_search_output.txt   # Text capture from record_demos.sh
  research_report_output.txt # Text capture from record_demos.sh
  flight_search_results.md   # Structured flight data output
  research_report.md         # Research report output
  README.md                  # Demo index
```

## Checklist before publishing

- [ ] GIFs are under 5 MB each (for fast loading)
- [ ] Terminal font is readable at the GIF resolution
- [ ] No API keys or personal info visible in recordings
- [ ] Demo runs cleanly without errors or warnings
- [ ] Split-screen mockup shows a realistic WhatsApp conversation
- [ ] README links point to the correct GIF files
