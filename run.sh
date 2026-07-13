#!/usr/bin/env bash
set -euo pipefail

cd ~/ralphton

echo "========================================"
echo " Step 4 - Implement Claude Agents"
echo "========================================"
echo

claude <<'EOF'
Implement the agents described in the previously approved Research Operating System.

Create these files under:

.claude/agents/

- director.md
- scholar.md
- strategist.md
- engineer.md
- operator.md
- verifier.md
- reviewer-2.md
- red-team.md
- archivist.md

Requirements:

- Follow the official Claude Code sub-agent specification exactly.
- Use only documented frontmatter fields.
- Do not invent undocumented configuration keys.
- Encode the authority model exactly as specified.
- Configure:
  - tools
  - disallowedTools
  - permissionMode
  - model
  - effort
  - memory
  - isolation
  - skills
  where appropriate.

Architecture requirements:

- Director manages portfolio only.
- Scholar handles literature only.
- Strategist plans experiments only.
- Engineer writes code only.
- Operator is the only agent allowed to launch experiments.
- Verifier independently checks numerical correctness.
- Reviewer-2 attacks scientific validity.
- Red-team attacks significance and novelty.
- Archivist maintains claims and long-term memory.

Constraints:

- Human approval is required before irreversible actions.
- No adjudication agent may modify production code.
- No production agent may certify its own output.
- Keep each agent concise (~60–80 lines).
- Reuse existing skills where appropriate.
- Do not modify existing project code.
- Only create/update files under .claude/agents/.

When finished, print a summary of all agents created and stop.
EOF
