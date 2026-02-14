# Strategic Flow-Lenia — Design Review & Implementation Guidance

## Executive verdict

Yes: the proposed **game-ification direction is strongly aligned** with turning the project from an open-ended sandbox into a replayable strategy game.

Your plan introduces the three missing pillars of strategy systems:
1. **Agency** (deck-building + tactical deployment),
2. **Legibility** (roles + explicit modes),
3. **Constraints** (biomass economy + cooldowns).

This is exactly what converts emergent simulation into meaningful play.

---

## 1) Vision alignment check

### What currently works
- The current simulation excels at emergence, motion richness, and biological flavor.
- It already has computational depth (GPU flow, species differentiation) that can support a game layer.

### What is missing for “game feel”
- No explicit player verbs (choose, commit, counter, time).
- Weak short-term goals and payoff loops.
- Species behavior is hard to predict, which reduces tactical planning.

### Why your proposal fixes this
- **Deck** creates pre-run strategic commitment (macro decisions).
- **Cost/cooldown** creates tactical timing (micro decisions).
- **Role-based species** enables readable counters and composition logic.
- **Brain modes** create controlled, explainable adaptation in-run.
- **Economy loop** ties map control to deployment power.

Conclusion: this is a coherent and commercially viable direction.

---

## 2) State Machine vs continuous neural control

## Recommendation
Use a **hybrid architecture**:
- **Primary runtime controller:** finite state machine (FSM) with 4 modes.
- **Continuous parameters inside each state:** weighted sensor response.

In practice:
- FSM decides **intent** (`Forage/Aggro/Defend/Replicate`).
- Continuous genes tune **style** within intent (how aggressive, how sticky, how fast to switch).

### Why FSM first
- **Readable to players** (critical for deck games).
- **Designable and balanceable** by humans.
- **Debuggable** in tools (state overlays, transition logs).
- **Cheaper in shaders** than fully recurrent neural policies.

### Why not fully neural at this stage
- Hard to communicate “why this species acted that way”.
- Harder balancing and counter-play design.
- Higher risk of degenerate metas and opaque behavior.

### Future extension
Add optional “Advanced Brain” later:
- Keep FSM outer shell for legibility.
- Replace transition score calculation with small learned/weighted function.
- Preserve explainability by showing top trigger contributions.

---

## 3) Refined systems design

## A. Deck meta-loop
- **Library**: persistent species assets (genome + role + tags + rarity/performance stats).
- **Deck**: 4–8 cards with:
  - biomass cost,
  - cooldown,
  - role,
  - optional active spell.
- **Run start choice**: map + objective + deck lock.

Add one key rule:
- **Deployment cap by population pressure** (soft cap), to prevent spam snowball and promote composition.

## B. Brain model (FSM + triggers)
Each species has:
- `state_id` (2 bits enough for 4 states),
- `state_timer`,
- `cooldown_mask` (optional),
- transition thresholds/weights.

Suggested transition evaluation per step:
1. Compute normalized sensors (enemy density, ally density, resource gradient, own mass).
2. Compute score per state:
   - `score_s = bias_s + dot(weights_s, sensors)`
3. Apply hysteresis:
   - switch only if `score_new > score_current + delta` and `min_state_time` satisfied.

This prevents state thrashing.

## C. Economy loop
- Passive income + map-based harvesting.
- Spawn consumes biomass immediately.
- Destroyed species yields partial reclaim (optional 20–40%) for comeback dynamics.

Add two anti-snowball mechanics:
1. **Diminishing harvest returns** per local cluster.
2. **Upkeep** for high-pop armies.

## D. Win conditions (important)
Without clear objectives the loop can still feel like sandbox.
Pick one for first milestone:
- Territory control score over time,
- Core destruction,
- Biomass supremacy at timer end.

---

## 4) Technical implementation plan (practical)

## Phase 1 — Minimal playable core
1. Add deck data model (resource files for species cards).
2. Add biomass economy and spawn/cooldown rules.
3. Add 4-state FSM in GPU state texture.
4. Add simple UI deck bar + costs + cooldown indicators.
5. Add one objective mode (e.g., score to 1000 biomass).

Goal: playable prototype with tactical decisions in < 2 weeks.

## Phase 2 — Intelligence and legibility
1. Add transition hysteresis and minimum state duration.
2. Add visual state VFX tinting per mode.
3. Add combat log overlays (why state switched).
4. Balance role archetypes and counters.

## Phase 3 — Meta depth
1. Library progression and saved builds.
2. Draft mode / challenge runs.
3. Additional spells and map modifiers.

---

## 5) Shader/data architecture notes

Your proposed `tex_ai` is a good separation. Prefer explicit texture over overloading alpha if possible.

Suggested channels (example `rgba16f`):
- `R`: `state_id` (encoded as 0.0, 0.33, 0.66, 1.0)
- `G`: state timer (normalized)
- `B`: local cooldown / lockout
- `A`: reserved (confidence, fear, or debug)

Pipelines:
1. `compute_sense.glsl` (optional): precompute sensor fields.
2. `compute_decision.glsl`: evaluate transitions and write `tex_ai`.
3. `compute_flow_conservative.glsl`: read `tex_ai`, apply state multipliers.

State multipliers table (uniform/SSBO):
- speed,
- repulsion/attraction,
- growth rate,
- metabolism drain,
- damage/consumption coefficient.

---

## 6) Risks and mitigations

1. **Chaos remains too high**
   - Mitigation: stricter state hysteresis + role parameter bounds.
2. **Player cannot read combat**
   - Mitigation: mode colors/icons + transition popups + hover explainers.
3. **Economy snowball**
   - Mitigation: upkeep + reclaim tuning + map resource depletion.
4. **Balance complexity explosion**
   - Mitigation: start with 3 roles, then expand.

---

## 7) Direct answers to review questions

### Q1: Does this game-ification align with the vision?
**Yes.** It is the right strategic layer and complements the simulation instead of replacing it.

### Q2: FSM vs continuous neural approach?
**FSM-first is preferred** for clarity, balance, and UX. Use a hybrid later (FSM intent + continuous tuning) once the core game loop is proven fun.

---

## 8) Suggested MVP scope (strict)

- 4 cards per deck
- 3 roles (Harvester, Tank, DPS)
- 4 modes (Forage/Aggro/Defend/Replicate)
- 1 map
- 1 win condition
- 1 enemy AI script

If this MVP is fun, the rest scales naturally.
