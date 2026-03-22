---
domain: phase10b-ui-redesign
status: approved
created: 2026-03-22
complexity: complex
linked_repos: []
---

# Phase 10B: UI Redesign -- Linear-Style Premium Dashboard

## Overview

Rebuild the GraphBot UI from scratch with a Linear-inspired aesthetic: soft backgrounds,
subtle card-based layout, smooth animations, premium typography, and clear information
hierarchy. React + CSS (no Tailwind utility spam), Framer Motion for animations.

METHODOLOGY: Research-first. Study Linear's design system, collect references, define
the design language BEFORE writing any component code.

Design constraints: No purple. No emojis. Dark/light mode. Monospace for data.
System font for UI. 4px max border radius. Subtle shadows. Card-based layout.

## Requirements

### R001: Research -- Linear Design System Analysis
Deep research before any UI code.
**Acceptance Criteria:**
- [ ] docs/research/ui-design-system.md written
- [ ] Analysis of Linear's: color palette, typography scale, spacing system, shadow system, border treatment, card patterns
- [ ] Analysis of: animation timing, hover states, focus states, transition curves
- [ ] Analysis of: how Linear shows progress/status (issue states, project progress)
- [ ] Recommended color palette (OKLCH), type scale, spacing scale for GraphBot
- [ ] Component inventory: what components do we need and their design specs

### R002: Design Tokens + Global Styles
Complete CSS design system.
**Acceptance Criteria:**
- [ ] ui/frontend/src/styles/design-system.css with all tokens
- [ ] Color palette: neutral scale (12 steps), status colors (4), accent (2), all OKLCH
- [ ] Typography: 6-step type scale (xs, sm, base, lg, xl, 2xl) with line heights
- [ ] Spacing: 4px base unit, scale from 1-16 (4px to 64px)
- [ ] Shadows: 3 levels (sm, md, lg) -- subtle, not dramatic
- [ ] Borders: 1px, color from neutral scale, 4px max radius
- [ ] Transitions: default 150ms ease-out, slow 300ms for layout
- [ ] Dark mode via class toggle (not media query -- user control)
- [ ] No Tailwind classes in components -- pure CSS modules or inline styles

### R003: Layout Shell -- 3-Panel Dashboard
Main layout with resizable panels.
**Acceptance Criteria:**
- [ ] Left panel (60%): DAG canvas -- full React Flow visualization
- [ ] Right panel (40%): split into top (node details) and bottom (output/result)
- [ ] Top bar: logo, task input, status indicator, dark mode toggle
- [ ] Panels have subtle card-style borders with soft shadows
- [ ] Responsive: panels stack vertically below 1200px
- [ ] Smooth panel transitions when content changes

### R004: DAG Canvas Redesign
Rebuilt DAG visualization with premium feel.
**Acceptance Criteria:**
- [ ] React Flow canvas with custom node component
- [ ] Nodes: card-style with soft shadow, status color as left border accent (not full background)
- [ ] Node content: task label (truncated), domain badge, status text, timing
- [ ] Node states: idle (gray border), running (amber left border + pulse shadow), done (green left border + checkmark), failed (red left border)
- [ ] Edges: smooth bezier curves, subtle gray by default, cyan when data is flowing
- [ ] Edge animation: dashed line animation when data transfers (CSS dash-offset animation)
- [ ] ELK.js auto-layout with smooth transitions when nodes are added
- [ ] Empty state: centered text "Enter a task to see the execution graph"

### R005: Task Input Bar
Premium input experience.
**Acceptance Criteria:**
- [ ] Full-width input at top of page (like Linear's command bar)
- [ ] Subtle border, grows slightly on focus (shadow change)
- [ ] Placeholder: "What would you like GraphBot to do?"
- [ ] Submit with Enter or button
- [ ] Loading state: input border animates (subtle gradient sweep)
- [ ] Disabled during execution with visual feedback

### R006: Node Detail Panel
Right-side panel showing selected node info.
**Acceptance Criteria:**
- [ ] Click a node to populate detail panel
- [ ] Shows: description, domain (with badge), status, model used, tokens, latency, cost
- [ ] Input data section (if consumes): shows forwarded data with source node name
- [ ] Output section: shows node output (scrollable, monospace for code/JSON)
- [ ] Context section: shows what graph context was injected (if any)
- [ ] All sections are collapsible cards

### R007: Result Panel
Bottom-right panel showing final aggregated output.
**Acceptance Criteria:**
- [ ] Shows final output after task completes
- [ ] Markdown rendering for formatted output
- [ ] Metrics bar: total nodes, tokens, latency, cost -- with subtle badges
- [ ] Comparison line: "Cost: $0.0003 (vs ~$0.05 with GPT-4)" when applicable
- [ ] Copy button for output text
- [ ] Expand/collapse animation

### R008: Status + Progress Indicator
Clear communication of what's happening.
**Acceptance Criteria:**
- [ ] Horizontal stepper: Intake -> Decompose -> Execute -> Aggregate -> Done
- [ ] Current step highlighted with accent color, completed steps have checkmark
- [ ] Step transitions animate smoothly (200ms)
- [ ] Below stepper: plain-English narration ("Breaking task into 3 parts...")
- [ ] Timer: shows elapsed time during execution
