# Plan 004 final presentation corrections

Status: completed. Opened: 2026-11-11. Closed: 2026-11-11.
Branch: fix/004-final, from main (explicit delivery exception).
Related decision: D-072. Presentation-only bug fixes; no new scientific decision.

## Scope and preservation

1. Verify every report/presentation CAD marker from drawn objects, then fix the motion plot and publish its corrected version with exact original data fields.
2. Verify every viewer curve's line style against the committed theme, then apply supported Plotly dash values. Stop if the installed chart library does not support them.
3. Verify export from a temporary tracked-only worktree with no scratch directory, using cache-free figures. The producer creates its temporary directories.

Observe red before implementation and pass the full pytest suite before every commit. Source-code and generated-figure commits are separate so the figure records its actual committed renderer. Retain all historical figures/manifests under their original names; publish the one affected figure as v3. Registry, scientific settings/results, demo package, ENV v4 and collision calculations stay unchanged. Test figures read registry metrics only; the holdout remains closed. Never publish remote changes.

## Progress

- Correction 1 red: all 23 report/presentation render cases were drawn; 22 passed and the motion plot failed because its CAD marker was filled. Geometry-only panels use synthetic display fixtures, and coverage curves use synthetic stored traces; no geometry cache or environment is needed for this style gate.
- At the initial correction-1 checkpoint, the later corrections were still pending; outcomes follow.
- Correction 1 green: all 23 actual-drawing cases pass after the motion renderer uses the shared hollow-marker helper. Source commit 5272ba4 passed the full suite. The one affected motion figure is republished as v3 (PNG/PDF/SVG/JSON), with a separate manifest and unchanged v2 evidence. Its entire metadata equals v2 except the permitted render_commit; all original numeric/source fields are exact.
- Correction 2: installed Plotly 7.1.0 validates solid/dot/dash/dashdot, so the conditional stop rule did not trigger. Seven selected-method cases were observed red because line.dash was unset. The adapter now translates the committed base-style token for every selected and context trace; synthetic x/y data are asserted unchanged.
- Correction 3 red: a real detached worktree contained only tracked files, with no scratch directory, data cache or outputs. Running the report entry point failed at mkdtemp with FileNotFoundError. Green: the same worktree check exported two registry-only figures into scratch/generated and created a local Matplotlib config directory. It checked that caches/outputs stayed absent and Git remained clean, then removed the worktree. While implementation is uncommitted, only source under test is overlaid in memory; all resource files and the worktree stay unchanged.

## Closure

- All three fixes preserve theme v2, numeric/source figure fields, registry, frozen scientific results, demo package and ENV v4. No policy, training, setting, collision or test-holdout evaluation was performed.
- The 23 actual-canvas cases changed from 1 failed / 22 passed to 23 passed. Seven viewer-trace cases changed from 7 failed to 7 passed. The tracked-only clean-worktree check changed from 1 failed (missing scratch) to 1 passed. The v3 motion metadata equals v2 in every field except the permitted renderer commit.
- Separate source/evidence commits for correction 1 preserve renderer provenance: 5272ba4 (393 full tests passed), 143293c (394 passed). Correction 2 is cece865 (401 passed). Correction 3 includes this closure and the final full-suite validation.
- Regenerated artifacts: results/figures/p004_e_test_motion_time_v3.png, .pdf, .svg and .json, with results/figures/manifest_v3.json. All previous v1/v2 files, manifests and captures remain intact; no other published figure required regeneration.
- Exporters now support repeated --figure selections, --output-dir and --version, refuse existing targets before drawing, and clean their temporary staging directory. Both entry points prepare local Matplotlib configuration before numerical rendering imports; an explicit MPLCONFIGDIR is retained.
- The original source license-version debt and Viser's default panel font remain as already documented. Research Plan 003 remains an unapproved draft. After the final green test/commit, locally fast-forward main and remove fix/004-final; no remote publication.

## Open issues

- Existing unknown source license versions and the viewer's documented default font remain outside this correction's scope.
