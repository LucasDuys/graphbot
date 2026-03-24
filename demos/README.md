# GraphBot Demos

End-to-end demonstrations of GraphBot's recursive DAG execution pipeline running on free LLMs.

## Demo 1: Flight Search with WhatsApp Notification

**Task:** "Find cheap flights from Amsterdam to Barcelona next week and message me the results on WhatsApp"

**What it demonstrates:**
- 4-node sequential DAG: browser search -> data extraction -> formatting -> WhatsApp send
- Browser automation (Playwright) for flight search
- Structured data extraction from raw HTML
- Cross-channel delivery (terminal + WhatsApp bridge)
- Mock mode for reproducible demos without real browser/API calls

**DAG structure:**
```
[browser] flight_search_browser
    |
[synthesis] flight_search_extract
    |
[synthesis] flight_search_format
    |
[comms] flight_search_send
```

**Run it:**
```bash
# Dry-run (mocked browser data, prints to console)
python scripts/demo_flight_search.py --dry-run

# Live (real Playwright browser + WhatsApp bridge)
python scripts/demo_flight_search.py --live
```

**Output:** `demos/flight_search_results.md`

---

## Demo 2: Research Report Pipeline

**Task:** "Research the top 5 AI agent frameworks in 2026 and write a comparison report"

**What it demonstrates:**
- Recursive decomposition: root task -> parallel web searches (5) -> synthesis
- Full Orchestrator pipeline (decompose -> schedule -> contextualize -> execute -> verify -> aggregate)
- Parallel execution of independent subtasks
- LLM synthesis combining multiple research results into a coherent report
- Pipeline stats: latency, cost, tokens, LLM calls

**DAG structure:**
```
[synthesis] root
    |
[web] research ----+----+----+----+----+
    |              |    |    |    |    |
[web] search1  search2  search3  search4  search5
                (LangGraph) (CrewAI) (AutoGen) (OpenAI) (Claude)
    |              |    |    |    |    |
    +--------------+----+----+----+----+
    |
[synthesis] synthesize
```

**Run it:**
```bash
# Dry-run (mocked LLM responses, no API calls)
python scripts/demo_research_report.py --dry-run

# Live (real API calls via OpenRouter)
python scripts/demo_research_report.py
```

**Output:** `demos/research_report.md`

---

## Split-Screen Demo Concept

The most visually compelling format for demos is a **split-screen view** showing the GraphBot terminal pipeline on the left and the WhatsApp phone interface on the right:

```
+-----------------------------------+------------------+
|                                   |                  |
|  Terminal                         |  WhatsApp        |
|                                   |                  |
|  $ python demo_flight_search.py   |  [phone mockup]  |
|  === Flight Search Demo ===       |                  |
|  DAG: 4 nodes                     |  User:           |
|    [browser] flight_search_brow.. |  "Find cheap     |
|    [synthesis] flight_search_ex.. |   flights AMS    |
|    [synthesis] flight_search_fo.. |   to BCN"        |
|    [comms] flight_search_send     |                  |
|                                   |  GraphBot:       |
|  [1/4] Browser search...          |  "Cheap flights: |
|        Extracted 412 chars        |   AMS -> BCN     |
|  [2/4] Data extraction...         |   Week of 30 Mar |
|        Found 6 flights            |   1. Ryanair     |
|  [3/4] Formatting results...      |      EUR 39      |
|  [4/4] Delivering message...      |   2. Transavia   |
|                                   |      EUR 49      |
|  Results saved.                   |   ..."           |
|  === Done ===                     |                  |
|                                   |  [cost: $0.00002]|
+-----------------------------------+------------------+
```

This demonstrates the full loop: user sends a message on WhatsApp, GraphBot decomposes the task, executes it autonomously, and delivers the result back to the same WhatsApp chat.

See `docs/RECORDING_GUIDE.md` for instructions on how to create this recording.

---

## Recording All Demos

```bash
# Capture terminal output from both demos
./scripts/record_demos.sh

# For GIF recordings, see docs/RECORDING_GUIDE.md
```

## Files in this directory

| File | Description |
|------|-------------|
| `README.md` | This file |
| `flight_search_results.md` | Structured output from flight search demo |
| `research_report.md` | Generated research report (after running demo 2) |
| `flight_search_output.txt` | Terminal capture from record_demos.sh |
| `research_report_output.txt` | Terminal capture from record_demos.sh |
| `*.gif` | GIF recordings (after following RECORDING_GUIDE.md) |
| `*.cast` | asciinema recordings (after following RECORDING_GUIDE.md) |
