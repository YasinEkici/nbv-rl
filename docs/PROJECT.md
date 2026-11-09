# Project

## What
Learn a next-best-view policy for a robot-mounted line-laser profilometer that reaches 95% observable surface coverage of cast parts with minimum estimated cycle time (arm motion + scan time), and generalizes to unseen part types (base-part split). Phase 1 uses classical RL without neural networks; Phase 2 replaces handcrafted features with a learned encoder.

## Assignment (verbatim, Turkish)
> **Ödev 7: 3B Lazer Tarama ve Kalite Kontrol Hücresinde En İyi Sonraki Bakış Açısı (Next-Best-View) Planlama**
> - Endüstriyel Problem: Karmaşık geometrili döküm veya metal parçaların robot kol üzerindeki 3B lazer profilometre ile taranması sırasında, kör noktaları (occlusion) kapatmak için gereksiz yere çok fazla tarama geçişi yapılarak çevrim süresinin uzatılması.
> - Durum: Şimdiye kadar taranmış 3B voksel/nokta bulutu doluluk haritası (taranmış, taranmamış, gölgede kalan bölgeler) ve sensörün mevcut 6-eksen pozu.
> - Eylem: Sensörün bir sonraki tarama adımı için gideceği yeni konum ve yönelim (küresel koordinatlarda Δφ, Δθ).
> - Ödül: Yeni kazanılan yüzey kapsama alanı (information gain) yüzdesi için pozitif; robot kolun kat ettiği mesafe ve harcanan tarama süresi için negatif ceza.
> - Önerilen Yöntem: SAC (Soft Actor-Critic).

## Why RL (and when not)
Model-based setting (D-029): the nominal CAD is known, as in industrial inspection, where scan plans are generated from the CAD and the scan is compared with it. Set cover + TSP is then a strong two-stage baseline, but it optimizes coverage and motion separately and is not exactly optimal for the joint objective. RL is used as a learned planner that produces a plan for a previously unseen part type directly from its CAD and optimizes the total cycle time (coverage + joint-space motion); this matters in high-mix / low-volume production. Cast deviations are a realism check: we report how much coverage a CAD-based plan loses on the real part. The industry-type CAD planner (ILP + TSP) and the joint optimum stay as references.

## Research question
On unseen part types (base-part split), can a learned NBV policy reach 95% coverage with lower estimated cycle time than greedy information-gain baselines, and how close does it get to the industry-type CAD planner (ILP + TSP) and the joint-optimum reference?

## Hypotheses
Thresholds and "if rejected" reports are predeclared in docs/plans/done/002-classical-rl.md, section 2.3.
- H1′: Phase 1 true online SARSA(λ) closes at least half of the normalized cycle-time gap between greedy-IG/cost and ILP + TSP.
- H-forward: true online SARSA(λ) has a lower normalized cycle time than the myopic agent.
- H-shaping: the choice of the shaping potential Φ changes learning speed but not the final performance markedly.
- H3: A camera-only occlusion model overestimates coverage compared with the dual-occlusion model (measured in Plan 001, reported descriptively).
- H2 (Phase 2 deep policy against Phase 1) is defined in Plan 003.
Null results are reportable.

## Primary data
- Base parts: 14 mechanical parts from Thingi10K (Zhou & Jacobson, 2016), CC BY or CC BY-SA only, clean meshes needing no repair (D-015), selected for occlusion features (internal cavities, ribs, deep holes, flanges). Scaled to a 200 mm longest side (D-017).
- Variants: 5 per part from a topology-preserving deviation generator: low-frequency surface noise ≤ 0.5 mm, Gaussian dents/bumps ≤ 1 mm, ±0.75% per-axis scale (D-031).
- Split: 8 train / 3 val / 3 test base parts, stratified by a hard-to-see ratio (D-016).
- Note: deviations are small (≤ ~1.5 mm) and barely change visibility. The generalization claim rests on the base-part split; yaw/position randomization has an interface only (`pose_id`); its computation is deferred to Plan 003 (D-030).
- Provenance and attribution: per-part Thingiverse link, author, license, and CC BY-SA flag in configs/split_v1.yaml; attribution table in README → Data. Variants of CC BY-SA parts must be shared under CC BY-SA if distributed. Meshes are not committed.

## Scope
- Must-have: environment with dual occlusion and joint-space cost; the canonical baselines, MCTS included (D-041); Phase 1 true online SARSA(λ) (D-045); Phase 2 masked DQN or MaskablePPO, and SAC as one of the main Phase 2 arms (D-008, PROTOCOL v2); evaluation on unseen part types.
- In PROTOCOL v1 the 128-candidate set is only the NSC reference (D-041).
- Stretch: GNN encoder.
- Out of scope: real robot, speckle/reflection noise, ROS/Gazebo, photorealistic simulators.

## Timeline
Plans 001 and 002 completed · Plan 004 visualization/VAL demos completed 8 November 2026 (D-070) for the 13 November midterm · Week 8–12 Plan 003 (deep RL, still a draft) · Week 13–14 Plan 005 (final + SİU draft, deadline to verify: 1 Feb 2027). Numbering follows D-068; no Plan 005 file is opened yet.

## Where things are
Protocol: docs/PROTOCOL.md · Decisions: docs/DECISIONS.md · Results: results/registry.csv · Project status: README → Abstract and status
