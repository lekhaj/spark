"""Creator agent layer — the orchestrator + discipline minds that sit *on top* of
the shared deterministic substrate (``cyclezero.creator_agent``).

Structure (the "agents on top" refactor):

  orchestrator.run_turn   ── the only entry point: routes a turn to a discipline,
                             runs that agent's LLM tool-use, applies the proposed
                             tool calls via the shared deterministic gate, persists
                             memory + the active-game pointer.
  base.DisciplineAgent    ── a mind = system prompt + a tool subset + owned layers
                             + routing intents. Every agent writes through the SAME
                             shared tools (the blackboard), so adding a discipline is
                             dropping in a module.
  systems.SYSTEMS_AGENT   ── the first real discipline (mechanics/systems/factors/
                             economy/loops + relations). Today's mono-agent, extracted.
  registry.route          ── classifies a turn → discipline. Systems is implemented;
                             narrative/world/art are recognised and fall back to the
                             default until their modules land (grow-into-the-sweet-spot).
"""
