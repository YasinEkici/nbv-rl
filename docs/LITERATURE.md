# Literature

Status: unverified · secondary (YYYY-MM-DD) · verified-source (YYYY-MM-DD) · verified (<name>, YYYY-MM-DD)
- verified-source: the citation and the claim we use were seen in the source itself (abstract or full text).
- secondary: the claim was seen only as reported by another work.
- verified: the full paper was read and the citation checked against its DOI (D-044).
Role: classical NBV | RL-based NBV | supervised NBV | industrial inspection | evaluation practice

A note stays unverified until the citation and the claim we use have been seen in the source (verified-source) or as reported by another work (secondary); `verified` requires a full read and a DOI check. Do not fill note fields beyond what the recorded source supports.

## connolly1985 — Connolly, "The determination of next best views", ICRA 1985
Status: unverified
Role: classical NBV
Why it matters: Klasik NBV problemini tanımlayan erken çalışma.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## scott2003 — Scott, Roth & Rivest, "View planning for automated 3D object reconstruction and inspection", ACM Computing Surveys 35(1), 2003
Status: verified-source (2026-11-03)
Role: classical NBV
Why it matters: Bakış planlama derlemesi; set cover çerçevesi.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## isler2016 — Isler et al., "An information gain formulation for active volumetric 3D reconstruction", ICRA 2016
Status: unverified
Role: classical NBV
Why it matters: Hacimsel bilgi kazancı (IG) tanımları.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## delmerico2018 — Delmerico et al., "A comparison of volumetric information gain metrics…", Autonomous Robots 42, 2018
Status: unverified
Role: classical NBV
Why it matters: Hacimsel IG metriklerinin karşılaştırması.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## kaba2017 — Kaba, Uzunbas & Lim, "A reinforcement learning approach to the view planning problem", CVPR 2017 (arXiv 1610.06204)
Status: verified-source (2026-11-03)
Role: RL-based NBV
Why it matters: Klasik RL ile NBV; en yakın emsal.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## landgraf2021 — Landgraf et al., "A Reinforcement Learning Approach to View Planning for Automated Inspection Tasks", Sensors 21(6):2030, 2021
Status: verified-source (2026-11-03)
Role: industrial inspection
Why it matters: Endüstriyel muayenede RL ile bakış planlama (Q-learning/DQN/PPO).
Additional claim (2026-11-07): introduction section 1.2 reports that earlier approaches did not improve performance/efficiency over simple greedy methods (attributed there to Kaba et al.). This is a critique of those prior approaches, not a new quantitative comparison in our project. Source: https://www.mdpi.com/1424-8220/21/6/2030/xml.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## peralta2020 — Peralta et al., "Next-Best View Policy for 3D Reconstruction" (Scan-RL, Houses3K), ECCV Workshops 2020
Status: verified-source (2026-11-03)
Role: RL-based NBV
Why it matters: Derin RL ile NBV.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## zeng2020 — Zeng, Zhao & Liu, "PC-NBV", IROS 2020
Status: unverified
Role: supervised NBV
Why it matters: Nokta bulutu tabanlı NBV.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## mendoza2020 — Mendoza et al., NBV-Net, Pattern Recognition Letters 133, 2020
Status: unverified
Role: supervised NBV
Why it matters: 3B CNN kodlayıcı emsali.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## pan2022 — Pan, Hu & Wei, SCVP, IEEE RA-L 7(2), 2022
Status: unverified
Role: supervised NBV
Why it matters: Set cover tabanlı tek seferlik planlama.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## chen2024 — Chen et al., GenNBV, CVPR 2024
Status: verified-source (2026-11-03)
Role: RL-based NBV
Why it matters: PPO ile genelleşen NBV.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## objviewbench2026 — Pan et al., ObjView-Bench, arXiv 2605.10707, 2026
Status: verified-source (2026-11-03)
Role: evaluation practice
Why it matters: Benchmark protokolü; 128 aday.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## roos2025 — Roos-Hoefgeest, S., Roos-Hoefgeest, M., Álvarez & González, "Reinforcement Learning Approach to Optimizing Profilometric Sensor Trajectories for Surface Inspection", Sensors 25(7):2271, 2025
Status: verified-source (2026-11-03)
Role: industrial inspection
Why it matters: Profilometrik sensör yörüngeleri için RL, UR3e ile gerçek doğrulama. Aynı sensör türü, farklı problem: yüzey boyunca yol; bizde pas seçimi ve sıralaması.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## zussman1994 — Zussman, Schuler & Seliger, Int J Adv Manuf Technol 9:56–64, 1994
Status: verified-source (2026-11-03)
Role: industrial inspection
Why it matters: Laser-scanner planning constraints (field of view, depth of field, maximum view angle); cited for the standoff rule (D-020).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## spensieri2025 — Spensieri et al., Robot Comput-Integr Manuf 93:102932, 2025
Status: verified-source (2026-11-03)
Role: industrial inspection
Why it matters: Robot base placement chosen by coverage / cycle time rather than raw reachability; basis for the Plan 001 base-position decision point.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## ng1999 — Ng, Harada & Russell, "Policy invariance under reward transformations: theory and application to reward shaping", ICML 1999
Status: secondary (2026-11-03)
Role: evaluation practice
Why it matters: Potential-based shaping keeps the optimal policy; basis of the coverage term in the reward (D-030).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## grzes2017 — Grześ, "Reward shaping in episodic reinforcement learning", AAMAS 2017
Status: secondary (2026-11-03)
Role: evaluation practice
Why it matters: Potential must be zero at terminal states for policy invariance in episodic tasks (D-030, option (i)).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## huang2022 — Huang & Ontañón, "A closer look at invalid action masking in policy gradient algorithms", FLAIRS 35, 2022
Status: verified-source (2026-11-03)
Role: evaluation practice
Why it matters: Invalid actions are masked rather than penalized (D-030).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## nemhauser1978 — Nemhauser, Wolsey & Fisher, "An analysis of approximations for maximizing submodular set functions — I", Mathematical Programming 14, 1978
Status: secondary (2026-11-03)
Role: evaluation practice
Why it matters: Greedy max coverage guarantees 1 − 1/e of the optimum, not the optimum; basis of the ILP-bound consistency test (D-033).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## feige1998 — Feige, "A threshold of ln n for approximating set cover", Journal of the ACM 45(4), 1998
Status: secondary (2026-11-03)
Role: evaluation practice
Why it matters: The 1 − 1/e greedy bound for max coverage is tight (D-033).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## saravanan2020 — Saravanan, Berman & Sober, "Application of the hierarchical bootstrap to multi-level data in neuroscience", Neurons, Behavior, Data analysis, and Theory 3(5), 2020
Status: verified-source (2026-11-03)
Role: evaluation practice
Why it matters: Hierarchical bootstrap for nested data; summarizing at the top level avoids pseudo-replication but shrinks the sample (D-034).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## best2019 — Best, Fitch, Sukkar & Yoo, "Multi-robot region-of-interest reconstruction with Dec-MCTS", ICRA 2019, pp. 9101–9107
Status: verified-source (2026-11-03)
Role: classical NBV
Why it matters: Precedent for MCTS-based, non-myopic view planning with robot arms; basis of the MCTS baseline (D-033).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## kot — Kot, Bobovský, Heczko, Vysocký, Virgala & Prada, "Using Virtual Scanning to Find Optimal Configuration of a 3D Scanner Turntable for Scanning of Mechanical Parts", <journal, year>
Status: unverified
Role: industrial inspection
Why it matters: Turntable scanning scheme (number of steps and height) optimized with ray-traced virtual scans; the fixed-pattern baseline is its robot analogue (D-033).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## bircher — Bircher et al., aerial robotic worker inspection path planning, arXiv 1703.02640
Status: verified-source (2026-11-03)
Role: classical NBV
Why it matters: Splitting full-coverage inspection path planning into a minimal view set and then tour optimization does not guarantee optimality in general; motivates the joint-optimum reference (D-035).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## kool2019 — Kool, van Hoof & Welling, "Attention, Learn to Solve Routing Problems!", ICLR 2019
Status: verified-source (2026-11-03)
Role: evaluation practice
Why it matters: Learned heuristics for orienteering and prize-collecting TSP approach solver quality rather than beat it; optimality gaps reported against a solver reference (D-034, D-035).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## gocator2800 — LMI Technologies, Gocator 2800 series (dual-camera line profiler), product reference
Status: verified-source (2026-11-03)
Role: industrial inspection
Why it matters: A real dual-camera (dual-triangulation) profilometer; product reference for the dual-camera ablation.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## agarwal2021 — Agarwal, Schwarzer, Castro, Courville & Bellemare, "Deep Reinforcement Learning at the Edge of the Statistical Precipice", NeurIPS 2021, arXiv 2108.13264
Status: verified-source (2026-11-08; citation and abstract checked, not a full read)
Role: evaluation practice
Why it matters: Aralık tahminleri ve performans profilleri, yalnızca nokta tahminine göre değişkenliği daha açık gösterir. PROTOCOL v1'in istatistiği değişmez; kullanılmayan IQM yardımcı fonksiyonu D-067'de kaldırıldı, rliable kullanılmıyor.
Source: https://arxiv.org/abs/2108.13264 (abstract).
Software note (2026-11-08): google-research/rliable was archived by its owner on 15 October 2025 and is read-only, consistent with D-010. Source: https://github.com/google-research/rliable. No installation or adoption.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## kwon2019 — Kwon, Na & Song, "Rescan Strategy for Time Efficient View and Path Planning in Automated Inspection System", Int. J. Precis. Eng. Manuf. 20:1747–1756, 2019
Status: verified-source (2026-11-03)
Role: industrial inspection
Why it matters: Yapılandırılmış ışık tarayıcı + cobot + döner tabla; ilk taramadan sonra yeniden taranacak bölgeleri seçen strateji. D-032 onarımının endüstriyel karşılığı; D-027 döner tabla örneği.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## laserdesign_flipplate — Laser Design, Flip Plate Frame, technical data sheet (product reference)
Status: verified-source (2026-11-03)
Role: industrial inspection
Why it matters: Parça çıkarılmadan önce kapsama doğrulanır, eksik veri ek taramayla tamamlanır (D-032); çevirme fikstürü ikinci kurulum örneği (D-025, Plan 003).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## fraunhofer_igd_viewplanning — Fraunhofer IGD, dynamic view planning for automated 3D scanning (inVISION News article)
Status: verified-source (2026-11-03)
Role: industrial inspection
Why it matters: Hızlı ilk tarama, ardından nokta yoğunluğuna göre sürekli planlama (D-032).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## mueller2025 — Müller & Kudenko, "Improving the Effectiveness of Potential-Based Reward Shaping in Reinforcement Learning", AAMAS 2025 (extended abstract)
Status: verified-source (2026-11-03)
Role: evaluation practice
Why it matters: Politika değişmezliği öğrenme hızını garanti etmez; Φ'nin seçimi, ölçeği ve kaydırılması etkinliği belirler (D-038).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## canonaco2024 — Canonaco, Ardon, Pozanco & Borrajo, "On the Sample Efficiency of Abstractions and Potential-Based Reward Shaping in Reinforcement Learning", arXiv 2404.07826, 2024
Status: verified-source (2026-11-03)
Role: evaluation practice
Why it matters: Sonlu ufkun PBRS'de yarattığı yanlılık; Φ'nin optimal değer fonksiyonuna yakın seçilmesi önerisi (D-030, D-038).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## jeon2023 — Jeon et al., "Benchmarking Potential Based Rewards for Learning Humanoid Locomotion", 2023
Status: verified-source (2026-11-04)
Role: evaluation practice
Why it matters: PBRS'nin yakınsama hızına katkısı marjinal ama ölçeklemeye dayanıklı (D-038).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## wiewiora2003 — Wiewiora, "Potential-Based Shaping and Q-Value Initialization are Equivalent", JAIR 19:205–208, 2003, DOI 10.1613/jair.1190
Status: verified-source (2026-11-05; citation and claim supplied by the user from the source)
Role: evaluation practice
Why it matters: PBRS, Q değerlerini Φ ile başlatmaya eşdeğer; sonuç SARSA ve diğer TD yöntemlerine uzanır. Doğrusal özellik uzayında temsil edilen Φ için Slice 2 eşdeğerlik testi (D-038).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## dynamic_shaping_framework2026 — <authors>, "A Unified Framework for Dynamic Reward Shaping in Reinforcement Learning", arXiv 2608.08158, 2026
Status: verified-source (2026-11-03)
Role: evaluation practice
Why it matters: Ödül şekillendirme çerçevesi; Plan 002 Φ ablation'ı için okunacak.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## danbv2026 — <authors>, "DA-NBV: A Direction-Aware Next-Best-View Planner for Efficient 3D Reconstruction of Ships at Sea", arXiv 2608.08025, 2026
Status: verified-source (2026-11-03)
Role: RL-based NBV
Why it matters: Kapsama arttıkça artışa daha fazla ödül veren yaklaşım; dışbükey Φ seçeneğine benzer (D-038).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## larc2026 — Feng, Wu, Wang & Zhou, "LARC: Lazy Adaptive Reachability Certification of Robot Manipulator Trajectories", arXiv 2608.29767, 2026
Status: verified-source (2026-11-03)
Role: evaluation practice
Why it matters: Örnekleme tabanlı çarpışma denetimlerinin örnekler arasını kaçırma sınırlaması (D-021 maskeleme sınırlılığı).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## carmignato2012 — Carmignato, "Computed tomography as a promising solution for industrial quality control and inspection of castings", Metallurgical Science and Technology 30-1, 2012
Status: verified-source (2026-11-03)
Role: industrial inspection
Why it matters: İç yüzeyler için endüstriyel CT; erişilemeyen bölgeler için tek araç (D-025, %40 tavanı).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## sae2006_ct — Smith, Bischoff, Georgi, Hansen et al., "Production Uses of Computed Tomography - Samples in an Aluminum Foundry", SAE Technical Paper 2006-01-0510, 2006
Status: verified-source (2026-11-03)
Role: industrial inspection
Why it matters: Otomatik CT ölçümünün 4–15 saat sürmesi; CT'nin pahalı ve yavaş olduğu (D-025).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## dulacarnold2015 — Dulac-Arnold et al., <title> (Wolpertinger), <venue>, 2015
Status: unverified
Role: RL-based NBV
Why it matters: Sürekli eylemi en yakın ayrık eylemlere eşleme; SAC'ın sürekli eylemini erişilebilir adaya eşleme seçeneği (Plan 003 Open issues).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## thingi10k — Zhou & Jacobson, "Thingi10K: A Dataset of 10,000 3D-Printing Models", arXiv 1605.04797, 2016
Status: verified-source (2026-11-03)
Role: evaluation practice
Why it matters: Parça kaynağı (D-015); lisans ve mesh özellikleri.
What we use: data/raw/parts (14 parça, CC BY / CC BY-SA)
What we do differently: <...>
Key numbers (page/table): <...>

## vanseijen2016 — van Seijen, Mahmood, Pilarski, Machado & Sutton, "True Online Temporal-Difference Learning", JMLR 17(145), 2016
Status: verified-source (2026-11-03)
Role: evaluation practice
Why it matters: Aşama 1 ana yöntemi: true online SARSA(λ), doğrusal FA, dutch izleri.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## patterson2024 — Patterson, Neumann, White & White, "Empirical Design in Reinforcement Learning", JMLR 25(318):1–63, 2024
Status: verified-source (2026-11-03)
Role: evaluation practice
Why it matters: Hiperparametre ayarı ve seçici raporlama yanlılığı; ayar seed'lerinin resmi seed'lerden ayrılması.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## eimer2023 — Eimer, Lindauer & Raileanu, "Hyperparameters in Reinforcement Learning and How To Tune Them", ICML 2023
Status: verified-source (2026-11-03)
Role: evaluation practice
Why it matters: Ayar ve değerlendirme seed'lerinin ayrılması; eşit ayar bütçesi.
Additional claim (2026-11-06): performance on tuning seeds can differ substantially from separate evaluation seeds; tuning landscapes can overfit the tuning seed. Source: https://proceedings.mlr.press/v202/eimer23a.html.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## kaiser2025_thesis — Kaiser, J.-P., "Autonomous View Planning using Reinforcement Learning: Modeling and Application for Visual Inspection in Remanufacturing", doctoral thesis, KIT 2024; Shaker 2025 (Forschungsberichte aus dem wbk, Band 294)
Status: verified-source (2026-11-03)
Role: industrial inspection
Why it matters: RL ajanları kıyaslama sezgiseliyle karşılaştırılabilir, optimale yakın analitik çözümden zayıf (H1′ beklentisi).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## kaiser2025_jim — Kaiser, Gäbele, Koch, Schmid et al., "Adaptive acquisition planning for visual inspection in remanufacturing using reinforcement learning", J Intell Manuf 36(7):4867–4893, 2025
Status: verified-source (2026-11-03)
Role: industrial inspection
Why it matters: Endüstriyel görüş planlamasında RL.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## koch2025 — Koch, Kaiser, Stamer, Stark & Lanza, "Enhancing Visual Inspection in Remanufacturing: A Reinforcement Learning Approach with Integrated Robot Simulation", Procedia CIRP 134:939–944, 2025 (58th CIRP CMS)
Status: verified-source (2026-11-03)
Role: industrial inspection
Why it matters: RL görüş planlaması + robot simülasyonu, erişilebilirlik ve çarpışma, gerçek istasyonda doğrulama (D-042 ile ilgili).
Additional claim (2026-11-07): decision/computation time matters alongside acquisition quality in runtime inspection. Source: https://publikationen.bibliothek.kit.edu/1000182849/161897271 (section 2); not evidence of deployment maturity across industry.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## khalil2017 — Dai, Khalil, Zhang, Dilkina & Song, "Learning Combinatorial Optimization Algorithms over Graphs", NeurIPS 2017 (arXiv 1704.01665)
Status: verified-source (2026-11-03)
Role: evaluation practice
Why it matters: Görülmemiş örneklere genelleşen öğrenilmiş sezgiseller.
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## regq — Lim & Lee, "Regularized Q-Learning", Advances in NeurIPS 37 (NeurIPS 2024), arXiv 2202.05404
Status: verified-source (2026-11-05; citation and claim supplied by the user from the source)
Role: evaluation practice
Why it matters: Doğrusal FA ile Q-learning'in L2 düzenlileştirmeyle kararlı hali. Düzenlileştirme yanlılığı bazı durumlarda büyük olabilir (Manek & Kolter 2022, aktaran regq).
Corrected version caveat (2026-11-06, D-058): arXiv v1 Lemma 3.1's inverse-Gram expression concerns eta X^T D X regularization, while our NeurIPS 2024 update uses eta I. The old formula and approximately 1.73e16 diagnostic value are inapplicable to our implementation; this is not merely renumbering. The old formula came from the earlier user instruction. For our eta I form no theorem condition was evaluated; stability was observed empirically only (all learner runs 0/70 divergent; RegQ official seeds 0/10). Sources: https://arxiv.org/pdf/2202.05404v1 (equation 13, Lemma 3.1); https://papers.nips.cc/paper_files/paper/2024/file/ea6d17af54f827336fc8fed27ca0319d-Paper-Conference.pdf (section 3.2).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## meyn2024 — Meyn, <title>, 2024
Status: secondary (2026-11-03)
Role: evaluation practice
Why it matters: Ağırlık normuyla ölçeklenen sıcaklıklı softmax davranış politikası (aktaran arXiv 2509.26442).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## liu2025 — Liu et al., <title>, 2025
Status: secondary (2026-11-03)
Role: evaluation practice
Why it matters: Meyn 2024 çizgisinde kararlılık sonucu (aktaran arXiv 2509.26442).
What we use: <module/config; exact or adapted>
What we do differently: <...>
Key numbers (page/table): <...>

## samsonov2024 — Samsonov, Tiapkin, Naumov & Moulines, "Improved High-Probability Bounds for the Temporal Difference Learning Algorithm via Exponential Stability", COLT 2024, PMLR 247
Status: verified-source (2026-11-05; citation and claim supplied by the user from the source)
Role: evaluation practice
Why it matters: Sabit adımlı doğrusal TD'de kuyruk ortalamasının varyansı 1/k mertebesinde azaltması; Plan 002 son %10 ağırlık ortalaması (D-046). Değişen SARSA politikası için garanti olarak kullanılmaz.
What we use: adapted; common tail-mean policy rule for Phase 1 learners.

## huo2026 — Huo, Chen & Xie, "Bias and Extrapolation in Markovian Linear Stochastic Approximation with Constant Stepsizes", Mathematics of Operations Research, 2026, arXiv 2210.00953
Status: verified-source (2026-11-05; citation and claim supplied by the user from the source)
Role: evaluation practice
Why it matters: Markov verisinde sabit adımın asimptotik yanlılığını ortalama kaldırmaz; adım büyüklüğüyle orantılı yanlılık sınırlaması (D-046).

## zhang_xie2024 — Zhang & Xie, "Constant Stepsize Q-learning: Distributional Convergence, Bias and Extrapolation", Reinforcement Learning Journal 3:1168–1210, 2024
Status: verified-source (2026-11-05; citation and claim supplied by the user from the source)
Role: evaluation practice
Why it matters: Sabit davranış politikası altında Q-learning'de sabit adım yanlılığı ve ortalama sınırlamaları; değişen SARSA politikası için garanti değildir (D-046).

## kolter2009 — Kolter & Ng, "Regularization and Feature Selection in Least-Squares Temporal Difference Learning", ICML 2009
Status: verified-source (2026-11-05; citation and claim supplied by the user from the source)
Role: evaluation practice
Why it matters: L1 tabanlı öznitelik seçimi; 16 öznitelikli problemde önceden beyan edilen çıkar-ve-karşılaştır kontrolünün bağlamı (D-046).

## manek2022 — Manek & Kolter, 2022 (<title and venue not supplied>)
Status: secondary (2026-11-05; reported by regq, supplied by the user)
Role: evaluation practice
Why it matters: Düzenlileştirme yanlılığı bazı durumlarda büyük olabilir; RegQ η ızgarasının küçük tutulmasının gerekçesi (D-046).

## khuller1999 — Khuller, Moss & Naor, "The budgeted maximum coverage problem", Information Processing Letters 70(1):39–45, 1999
Status: verified-source (2026-11-06; citation supplied by user; claim checked in source)
Role: evaluation practice
Why it matters: Bütçeli maksimum kapsam için yeni kapsanan ağırlık / maliyet oranını kullanır. 1−1/e garantisi kısmi enumerasyon ile genişletilen algoritmaya aittir; oran greedy'sine tek başına veya durum-bağımlı robot hareket maliyetine aktarılmaz. Plan 002 miyop toplamsal puan ile oran sezgiseli yorumunu sınırlar.
Source: https://thibaut.horel.org/submodularity/papers/khuller1999.pdf (sections 2–3); citation also at https://www.cs.umd.edu/~samir/papers.html.
What we use: interpretation only; no environment or baseline change.

## dwork2015 — Dwork, Feldman, Hardt, Pitassi, Reingold & Roth, "The reusable holdout: Preserving validity in adaptive data analysis", Science 349(6248):636–638, 2015; and "Generalization in Adaptive Data Analysis and Holdout Reuse", NeurIPS 2015, arXiv 1506.02629
Status: verified-source (2026-11-06; citation/claim supplied by user; arXiv abstract checked)
Role: evaluation practice
Why it matters: Holdout'u uyarlamalı tekrar kullanmak holdout'a aşırı uyuma yol açabilir. Test değerlendirmesinin tek açılış koruması D-055; makalenin gizlilik temelli reusable-holdout algoritmasını uyguladığımız iddia edilmez.
Source: https://arxiv.org/abs/1506.02629; https://doi.org/10.1126/science.aaa9375.
What we use: conservative single-opening guard, not a reproduction.

## jordan2020 — Jordan, Chandak, Cohen, Zhang & Thomas, "Evaluating the Performance of Reinforcement Learning Algorithms", ICML 2020, PMLR 119:4962–4973
Status: verified-source (2026-11-06; citation and abstract checked)
Role: evaluation practice
Why it matters: RL değerlendirmesinde metrik seçimi ve raporlanan performansın tutarlılığı; resmi seed'ler, tüm kolların eğrileri ve ikincil son-ağırlık sonuçlarını birlikte sunma gerekçesi. Dondurulmuş PROTOCOL v1 istatistikleri değişmez.
Source: https://proceedings.mlr.press/v119/jordan20a.html.
What we use: evaluation interpretation only.

## chvatal1979 — Chvátal, "A greedy heuristic for the set-covering problem", Mathematics of Operations Research 4(3):233–235, 1979
Status: secondary (2026-11-06; reported by Neal E. Young, "Greedy Set-Cover Algorithms", Encyclopedia of Algorithms)
Role: evaluation practice
Why it matters: Ağırlıklı set cover greedy kuralı, yeni kapsanan eleman başına en düşük maliyettir; Plan 002 toplamsal miyop puan ile greedy oranı yorumunun kaynağı. Kendi durum-bağımlı rota problemimize doğrudan yaklaşım garantisi verilmez.
Source consulted (secondary): https://www.cs.ucr.edu/~neal/Young08SetCover.pdf; https://doi.org/10.1007/978-0-387-30162-4_175.
What we use: interpretation only; primary paper not claimed read.

## smith_winkler2006 — Smith & Winkler, "The optimizer's curse: Skepticism and postdecision surprise in decision analysis", Management Science 52(3):311–322, 2006, DOI 10.1287/mnsc.1050.0451
Status: verified-source (2026-11-06; citation and publisher abstract checked)
Role: evaluation practice
Why it matters: Gürültülü değer tahminleri içinden en iyisini seçmek, tahminler yansız olsa bile seçilmiş tahmini iyimser yapar. Ayar skorunun ayrı değerlendirmeye taşınmasındaki iyimserlik için yorum çerçevesi; tek başına bizim deneydeki nedenin kanıtı değil.
Source: https://pubsonline.informs.org/doi/abs/10.1287/mnsc.1050.0451.

## zhang2023_sarsa — Zhang, Tachet des Combes & Laroche, "On the Convergence of SARSA with Linear Function Approximation", ICML 2023, PMLR 202:41613–41646, arXiv 2202.06828
Status: verified-source (2026-11-06; citation, abstract and relevant introduction checked)
Role: evaluation practice
Why it matters: Doğrusal SARSA'da sınırlı bölgede salınım ve SARSA(lambda) dahil bölgeye yakınsama hızı. Gordon'un gözlemleri azalan adımla da salınım bildirir. Makalenin Lipschitz politika, ergodiklik ve adım varsayımları bizim epsilon-greedy, sabit-adımlı true-online SARSA uygulamamıza otomatik garanti değildir; eğrilerle tutarlı açıklama olarak kullanılır.
Source: https://proceedings.mlr.press/v202/zhang23al.html; https://proceedings.mlr.press/v202/zhang23al/zhang23al.pdf.

## gordon2001 — Gordon, "Reinforcement learning with function approximation converges to a region", 2001
Status: secondary (2026-11-06; reported by zhang2023_sarsa, introduction and references)
Role: evaluation practice
Why it matters: Trajectory-based doğrusal SARSA'nın epsilon-greedy politika iyileştirmeyle sınırlı bölgeye yakınsaması; asıl çalışma okunmuş iddia edilmez. Azalan adımda süren salınım gözlemi Zhang vd.'nin aktardığı Gordon 1996/Bertsekas–Tsitsiklis 1996 gözlemleriyle ayrıştırılır.
Source consulted: https://proceedings.mlr.press/v202/zhang23al/zhang23al.pdf.

## cheng2021_hurl — Cheng, Kolobov & Swaminathan, "Heuristic-Guided Reinforcement Learning", NeurIPS 2021, arXiv 2106.02757
Status: verified-source (2026-11-07; title, authors, introduction and abstract checked)
Role: evaluation practice
Why it matters: A heuristic estimates V*. Useful state ordering can support shorter-horizon decisions; PBRS alone does not guarantee faster learning. Our Phase 1 uses PBRS, not HuRL's modified discount.
Source: https://proceedings.neurips.cc/paper/2021/file/70d31b87bd021441e5e6bf23eb84a306-Paper.pdf (sections 1–2).

## wang2016_dueling — Wang, Schaul, Hessel, van Hasselt, Lanctot & de Freitas, "Dueling Network Architectures for Deep Reinforcement Learning", ICML 2016, PMLR 48:1995–2003
Status: verified-source (2026-11-07; citation and abstract checked)
Role: evaluation practice
Why it matters: Factor state value and action advantage; sharing value information helps when actions have similar values. Possible Phase 2 representation, not implemented in Phase 1.
Source: https://proceedings.mlr.press/v48/wangf16.html.

## silver2018_rpl — Silver, Allen, Tenenbaum & Kaelbling, "Residual Policy Learning", arXiv 1812.06298, 2018
Status: verified-source (2026-11-07; citation, abstract and related-work passage checked)
Role: evaluation practice
Why it matters: Learning a residual over an imperfect controller can improve data efficiency relative to learning from scratch. Greedy/MCTS-based residual planning remains a Phase 2 option; adapting continuous residual actions to masked pass choice needs design.
Source: https://arxiv.org/abs/1812.06298; https://arxiv.org/pdf/1812.06298.

## johannink2019 — Johannink et al., "Residual Reinforcement Learning for Robot Control"
Status: secondary (2026-11-07; reported by silver2018_rpl, related work and reference 28)
Role: evaluation practice
Why it matters: Concurrent residual RL for robot control, including real-robot insertion. The requested key retains 2019; Silver cites its 2018 arXiv preprint 1812.03201. We do not claim a full read or a checked conference bibliography.
Source consulted: https://arxiv.org/pdf/1812.06298 (section II, reference 28).

## tanaka2026 — Tanaka & Mahmood, "Performance Variation in Deep Reinforcement Learning", arXiv 2606.06746, 2026
Status: verified-source (2026-11-08; citation, abstract and section 6 checked)
Role: evaluation practice
Why it matters: Run-wise percentile highlighting retains individual trajectories and highlights 5th/50th/95th percentile runs. Plan 004 adapts this for ten recorded VAL seeds; no synthetic interpolated trajectory or new statistical claim.
Adaptation corrected by D-069 (2026-11-08): the paper uses 100 runs per task and whole-run average performance. We have ten seeds and rank by the registered official tail-policy VAL score, highlighting best/lower-median/worst actual runs; alternate whole-curve-mean ordering is metadata only. Time-wise bands can describe trends no individual run follows. Do not call our ten-run extremes 5th/95th percentiles.
Source: https://arxiv.org/abs/2606.06746; https://arxiv.org/html/2606.06746v1 (section 6).

## yi2025_viser — Yi et al., "Viser: Imperative, Web-based 3D Visualization in Python", arXiv 2507.22885, 2025
Status: verified-source (2026-11-08; citation and abstract checked)
Role: visualization software research
Why it matters: Python API and web-based 3D/GUI primitives for interactive visualization. Candidate for optional Plan 004 S3, not an adopted runtime or RL method.
Source: https://arxiv.org/abs/2507.22885. Software metadata: https://pypi.org/project/viser/; https://github.com/viser-project/viser. No package installed in S0.

## barr2015 — Barr, Harman, McMinn, Shahbaz & Yoo, "The Oracle Problem in Software Testing: A Survey", IEEE TSE 41(5):507–525, 2015, DOI 10.1109/TSE.2014.2372785
Status: verified-source (2026-11-08; citation and author-hosted survey checked)
Role: evaluation practice
Why it matters: Partial oracle information can support checks when full specifications are absent. D-069 distinguishes baseline full-step validation from learner aggregate consistency; neither makes missing historical traces appear.
Source: https://philmcminn.com/publications/barr2015.pdf; https://doi.org/10.1109/TSE.2014.2372785.

## weissgerber2015 — Weissgerber, Milic, Winham & Garovic, "Beyond Bar and Line Graphs: Time for a New Data Presentation Paradigm", PLOS Biology 13(4):e1002128, 2015, DOI 10.1371/journal.pbio.1002128
Status: verified-source (2026-11-08; citation and primary article checked)
Role: evaluation practice
Why it matters: Show individual observations in small samples and preserve pairing; aggregated bars can hide distributions and paired structure. Applied to descriptive Phase 1 visualization, not new inference.
Source: https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002128.

## wong2011 — Wong, "Points of view: Color blindness", Nature Methods 8(6):441, 2011, DOI 10.1038/nmeth.1618
Status: verified-source (2026-11-08; publisher citation/excerpt and user-supplied source claim checked; not a full read)
Role: visualization practice
Why it matters: Colorblind-friendly categorical design. D-069 explicitly supplies the eight-color Okabe–Ito palette; markers/line styles provide additional distinctions. Attribution belongs to Okabe and Ito, not an invented new palette.
Source: https://doi.org/10.1038/nmeth.1618; publisher correction https://www.nature.com/articles/s41592-023-01974-0. Full publisher text was not retrievable by this client; do not claim a full read.

## rougier2014 — Rougier, Droettboom & Bourne, "Ten Simple Rules for Better Figures"
Status: verified-source (2026-11-09; citation and used claims seen in the publisher source; not a full-read claim).

PLoS Computational Biology 10(9):e1003833, 2014. DOI: https://doi.org/10.1371/journal.pcbi.1003833
Source: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833
Used claims: choose a clear message and adapt complexity, line/point size and text to the display medium; oral-presentation figures need simpler, more visible content than article figures. Captions explain the encoding and its limits.
Plan 004b adapts these principles. The user-specified 16:9/1920×1080 format, minimum 18 pt text, grey context/highlight scheme and direct labels are our presentation choices, not reproduced experimental results or literal numeric prescriptions from the paper. Original report figures stay unchanged.

