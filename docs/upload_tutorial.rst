Empty Workspace Upload Tutorial (Claude Code / Codex)
======================================================

This tutorial shows how to run Colophon in a fresh, empty Claude Code or Codex workspace
by uploading files from ``/examples/``.

Use this when you want an upload-first workflow and do not want to pass every file path manually.

What To Upload From ``/examples/``
----------------------------------

Minimum required files:

- ``bibliography.json``
- ``outline.json``
- ``seed_graph.json``
- ``prompts.json``

Recommended upload set (for additional workflows):

- ``outline_preliminary.json``
- ``llm_openai.json`` / ``llm_codex.json`` / ``llm_claude.json`` / ``llm_pi.json``
- ``recommendation_openalex.json``
- ``recommendation_semantic_scholar.json``
- ``kg_update_local.json``
- ``kg_update_openai.json``
- ``notes/`` (if you also want note-import demos)

In an empty workspace, upload these into a folder named ``uploads/``.

Prepare Environment
-------------------

If Colophon is not already installed in the workspace:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -e .

Verify uploaded files:

.. code-block:: bash

   ls -la uploads

You should see at least ``bibliography.json``, ``outline.json``, ``seed_graph.json``, and ``prompts.json``.

Run In Codex
------------

.. code-block:: bash

   colophon \
     --runtime codex \
     --artifacts-dir uploads \
     --output build/upload_codex.md \
     --output-format markdown \
     --report build/upload_codex_diagnostics.json \
     --title "Upload Tutorial Draft (Codex)"

Run In Claude Code
------------------

.. code-block:: bash

   colophon \
     --runtime claude-code \
     --artifacts-dir uploads \
     --output build/upload_claude.md \
     --output-format markdown \
     --report build/upload_claude_diagnostics.json \
     --title "Upload Tutorial Draft (Claude Code)"

Optional: Interactive User Guidance
-----------------------------------

Add stage-aware guidance prompts (planning, recommendations, outline expansion, and coordination):

.. code-block:: bash

   colophon \
     --runtime codex \
     --artifacts-dir uploads \
     --request-user-guidance \
     --user-guidance-stages planning,recommendations,outline,coordination \
     --guidance-output build/upload_user_guidance.json \
     --output build/upload_guided.md \
     --output-format markdown \
     --report build/upload_guided_diagnostics.json \
     --title "Upload Tutorial Draft (Guided)"

Colophon caps user-input prompts to top 10 questions per stage. If more are available,
questions are ranked by importance and only the top 10 are asked.

Expected Outputs
----------------

After a successful run, you should have:

- manuscript output (``.md`` by default in this tutorial)
- diagnostics JSON with runtime + artifact resolution metadata
- optional user guidance JSON (if guidance mode is enabled)

Check diagnostics quickly:

.. code-block:: bash

   jq '.runtime, .input_artifacts, .user_guidance.requested' build/upload_guided_diagnostics.json

Troubleshooting
---------------

- If Colophon reports missing artifacts, confirm uploaded filenames match example names exactly.
- If you uploaded files to a different folder, set that folder in ``--artifacts-dir``.
- If both explicit input flags and ``--artifacts-dir`` are set, explicit flags take precedence.
- For Claude Code runtime, use ``--runtime claude-code`` (``claude_code`` also works).
