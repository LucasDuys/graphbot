---
domain: phase8-generative-ui
status: approved
created: 2026-03-21
complexity: complex
linked_repos: []
---

# Phase 8: Generative UI -- Real-Time DAG Visualization Dashboard

## Overview

Build a web dashboard that visualizes GraphBot's pipeline execution in real-time:
DAG decomposition, parallel node execution, data flow, context assembly, knowledge
graph, and aggregation. Tech stack: Next.js + React Flow + ELK.js + Framer Motion + SSE.

Design: Vercel/Linear aesthetic. No purple. No emojis. Clean, technical, minimal.

## Requirements

### R001: Backend SSE Streaming Endpoint
FastAPI server that streams pipeline events to the frontend.
**Acceptance Criteria:**
- [ ] `ui/server.py` -- FastAPI app with SSE endpoint `/api/stream`
- [ ] Accepts task via POST `/api/task` with `{"message": "..."}`, returns task_id
- [ ] SSE stream emits events: node.created, node.status, edge.created, data.flow, context.injected, aggregation.complete
- [ ] Each event: `{"type": "event_type", "payload": {...}, "timestamp": ...}`
- [ ] Orchestrator patched to emit events via callback during execution
- [ ] CORS configured for localhost dev
- [ ] Test: SSE endpoint streams events for a simple task

### R002: DAG Visualization Component
Real-time task tree rendering with React Flow.
**Acceptance Criteria:**
- [ ] Next.js app in `ui/` directory
- [ ] React Flow canvas showing task decomposition as it happens
- [ ] Nodes appear with animation (Framer Motion fade-in + scale)
- [ ] Node states: pending (gray), running (amber), completed (green), failed (red)
- [ ] Status transitions animate smoothly (color + border glow)
- [ ] ELK.js auto-layout (top-down hierarchical)
- [ ] Edges show data flow labels (provides/consumes keys)
- [ ] Edge animation: data "pulse" travels along edge when data forwards
- [ ] Responsive: works on 1440px+ screens

### R003: Context Assembly Panel
Side panel showing what context was injected into each node.
**Acceptance Criteria:**
- [ ] Right panel shows selected node details
- [ ] Displays: description, domain, complexity, model used
- [ ] Shows injected context: entities, memories, token budget used
- [ ] Token budget visualized as a bar (used/total)
- [ ] Input data (from provides/consumes) shown with source node reference
- [ ] Output data shown after completion
- [ ] Timestamps and latency per node

### R004: Knowledge Graph Explorer
Bottom panel showing the knowledge graph state.
**Acceptance Criteria:**
- [ ] D3 force-directed graph showing entities + relationships
- [ ] Nodes colored by type (User, Project, Service, Memory, etc.)
- [ ] Edges labeled with relationship type
- [ ] Updates in real-time as GraphUpdater adds nodes during execution
- [ ] Click node to see properties
- [ ] Filter by node type
- [ ] Zoom and pan

### R005: Aggregation Visualization
Show how leaf outputs combine into the final response.
**Acceptance Criteria:**
- [ ] Bottom section of DAG view shows aggregation step
- [ ] Visual: leaf outputs flow into aggregator node
- [ ] Template slots highlighted with which leaf filled which slot
- [ ] Final output displayed with provenance markers (which leaf contributed each part)
- [ ] Confidence scores shown per leaf output (when available)

### R006: Design System
Consistent visual language across all components.
**Acceptance Criteria:**
- [ ] OKLCH color tokens in CSS variables (no purple anywhere)
- [ ] Status colors: pending=blue-gray, running=amber, success=green, failed=rust-red
- [ ] Neutral palette: off-white backgrounds, near-black text, subtle gray borders
- [ ] Data flow accent: cyan
- [ ] Typography: system-ui for labels, monospace for code/tokens/costs
- [ ] Border radius max 4px
- [ ] No gradients, no shadows > 4px blur
- [ ] No emojis anywhere in the UI
- [ ] Dark mode support (swap OKLCH lightness values)

## Future Considerations (NOT Phase 8)
- Pattern cache visualization (show when a cached pattern is used)
- Multi-session history timeline
- Embedding into Nanobot channels (Discord bot with DAG image)
- Mobile responsive layout
