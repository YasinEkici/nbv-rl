# NBV VAL demo bundle v1

Upload all seven files together: common.json, part_271875_v0.json, part_591247_v0.json,
part_37275_v0.json, schema.json, manifest.json and README.md. No test data is present.
Use schema.json and the semantic rules below; verify SHA-256 before displaying.

Arrays: base64 raw C-order bytes; <f8 = little-endian IEEE-754 binary64,
<u4 = little-endian uint32, |u1 = uint8 (byte order not applicable).
No lossy rounding/downcasting. Positions are mm, angles rad, time s.
Visibility is 48 pass-major rows, each 2,500 bytes. Big bit order: sample 0 is
the most significant bit of byte 0; exactly 20,000 bits per row, no padding.
Sample positions AND normals are variant-v0 values in their original correspondence
order. Display mesh is normalized nominal, not variant. Do not reorder samples.
Mesh nominal, points variant v0; deviations up to approximately 1.5 mm.

Robot uses standard DH: RotZ(theta+offset) TransZ(d) TransX(a) RotX(alpha),
with metres in DH, rad in joint states. Apply base yaw/translation, then flange TCP
translation along local z. Part/world frame: bbox bottom-center, z up. Capsule
coordinates are mm and DH frame indices; offsets are along joint-2 axis as in
configs/collision_v1.yaml. Sensor box is in sensor frame (x=sweep, z=view).
Never animate valid_q=0; those q entries contain declared zero placeholders.
The five nominal IK samples may differ by 2*pi lifts from an episode's executed
samples. Use episode joint_samples_rad, which preserve ENV v4 wrapping/limits.
Move starts at joint_start_rad, reaches the first sample, then scans through five
samples; no physical intermediate trajectory is certified. Scan-step cumulative
time excludes home. The final home row uses direct q_home return as ENV v4.
original_step retains the exact evaluation record, whose final cycle_time already
includes home. Avoid adding that home duration twice.

Episodes are stored-sequence replays, no policy calls. Full oracle A: 195 canonical
VAL baseline episodes, all 12 original step fields plus summaries exactly matched.
Partial oracle B: 900 learner VAL episodes, sequence and all 15 stored summary fields
exactly matched. Every exported episode carries its verification and manifest link.
Learner final coverage: derived from replay; consistent with partial oracle.
JSON reference decimals are parsed back to IEEE-754 binary64 and compared by bytes;
no tolerances or rounded display-cell comparisons. Historical learner joints/final
coverage were not independently stored. AUC is the step-function integral of primary
coverage on [0,180 s] / 180, with jumps at completed passes and constant home coverage;
matching AUC is one scalar consistency check, not a proof of final coverage.

Always show each part's name, author, Thingiverse/Thingi10K source link, recorded
CC BY / CC BY-SA family, version status and changes alongside its mesh/derivatives.
All three license versions are NOT recorded in the split metadata: do not infer 4.0.
official_specific_license_url is null; the CC overview link is only an overview.
BY-SA derivatives retain the same license; attribution, license link and changed-
material notice are required. Exact version-specific license links require resolving
the missing source metadata before public distribution. Transformations are listed
per component in attribution.changes; no display simplification was performed.
