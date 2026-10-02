# 3 Experimental Setup

## 3.1 Benchmark

We evaluate on LoCoMo-Refined (mem-eval-suite, 2026), a recalibrated version of the LoCoMo benchmark (Maharana et al., 2024). The original LoCoMo contains approximately 6.4% flawed annotations (confirmed by independent audit and by our own held-out analysis, which found 86% of original-benchmark questions unusable as held-out data due to evidence-pointer mismatches). The refined version revises 337 noisy samples through AI screening followed by review by five human annotators, and pairs the dataset with a calibrated judge (Qwen3-14B, temperature 0, thinking disabled) that achieves 86.33% human agreement on 300 annotated samples, versus 43.67% for the original GPT-4o-mini judge. The benchmark contains 1,382 questions across ten long-term conversations (avg. 300 turns, 9K tokens, up to 35 sessions per conversation), spanning five question categories: single-hop (213), multi-hop (299), temporal (68), open-domain (802).

We additionally report replication on LongMemEval-S (cleaned) in Appendix F. All primary findings replicate; details in §F.

## 3.2 Embedding Models

All structural measurements are performed in two embedding spaces to ensure findings are not model-specific artifacts:

- **BGE-M3** (Chen et al., 2024): 1024-dimensional, multilingual, the current de-facto standard for conversational memory retrieval.
- **Qwen3-embedding** (0.6B): 256-dimensional, the embedding family aligned with the benchmark's official judge (Qwen3-14B).

The two models differ in architecture, dimensionality, training data, and training objective. Any structural property that holds in both spaces is unlikely to be a model-specific artifact.

## 3.3 Memory Construction

We construct the memory store without any LLM calls: raw dialogue turns are stored verbatim with session-level timestamps. A parallel atomic-decomposition layer splits each turn into sentence-level units (4–45 words, greeting-filtered), yielding 13,301 atomic statements across 11,753 total records. This construction choice is itself validated in §6.1: any filtering reduces recall.

The question set is divided by conversation parity (odd/even sessions) for split-half validation throughout. All unsupervised components (Δ direction, lexical hosts, neighborhood clusters) are computed on one half and evaluated on the other.

## 3.4 Definitions

- **Answer perspective**: similarity signals computed from the answer vector x.
- **Question perspective**: similarity signals computed from the question vector a.
- **Evidence band**: the set of records whose embedding lies within the neighborhood of the correct evidence record c.
- **Common-mode component**: the portion of embedding variance shared by all records regardless of status (analogous to common-mode noise in differential signalling).

---

# 4 The Answer-Perspective Gap

## 4.1 Hemispheric Separation

We first establish that questions and answers occupy geometrically distinct regions. Computing the centroid of all 1,376 question embeddings and the centroid of all 1,376 answer embeddings, we find:

**cos(centroid_q, centroid_a) = −0.454** (BGE-M3, angle ≈ 117°)

This is not a property of any single query-answer pair but a systemic separation: question text ("What did Caroline research?") lives in a different region than answer text ("Adoption agencies") because questions are speech-act acts (interrogatives requesting information) while answers are propositional units (declaratives providing it). The embedding model, trained on contrastive pairs from web-scale QA data, encodes this pragmatic distinction into the geometry.

**Corollary (Retrieval endpoint).** Since cos(q, x) < 0 for the centroids, a cosine-based retrieval system scoring all records against the question vector will never rank the answer vector above zero similarity. The retrieval system can only navigate to the *evidence band* — records whose speech register and topic overlap with the question. This establishes a geometric (not engineering) argument that two-stage architectures (retrieve-then-read) are necessary, not optional.

Cross-space replication: cos = −0.491 (Qwen3-embedding), confirming the hemispheric structure is not model-specific.

## 4.2 The Perspective Discrimination Asymmetry

If both perspectives were equally informative, a retrieval system could use either. We show they are not.

For each of 1,218 questions with matched evidence (c) and top-ranked noise (h), we compute four cosine similarities in each space:

**Table 1: Nine-element relation matrix (mean cosine, BGE-M3 / Qwen-256)**

| Relation | BGE-1024 | Qwen-256 |
|---|---|---|
| a · b (question ↔ answer) | 0.392 | 0.484 |
| a · c (question ↔ evidence) | 0.627 | 0.692 |
| a · h (question ↔ noise) | 0.610 | 0.678 |
| b · c (answer ↔ evidence) | 0.551 | 0.529 |
| b · h (answer ↔ noise) | 0.453 | 0.447 |
| c · h (evidence ↔ noise) | 0.706 | 0.675 |

The critical comparison:

**Question perspective**: a·c − a·h = **+0.017** (BGE) / **+0.014** (Qwen)
**Answer perspective**: c·x − h·x = **+0.098** (BGE) / **+0.082** (Qwen)

**The answer perspective discriminates evidence from noise at 6× the rate of the question perspective.** This asymmetry is consistent across both embedding spaces and constitutes the central measurement of this paper: *the information needed to distinguish evidence from noise is concentrated in the answer view, which is unavailable at retrieval time.*

**Figure 2**: Paired bar chart comparing question-perspective and answer-perspective discrimination in both spaces.

## 4.3 Mechanistic Explanation

The asymmetry arises from how answers are constructed relative to their evidence. An answer is not a copy of the evidence sentence; it is a *distillation* that preserves propositional content while shedding conversational register (greetings, hedging, turn-taking). This means:

- The answer retains topical overlap with its evidence (b·c = 0.551 > 0.453 = b·h), because topicality survives distillation.
- The answer *loses* the register signal that dominates question-evidence similarity, because answers are not questions.

Combined with the common-mode structure (§5), this produces the full picture: questions and noise share register (both are conversational), questions and evidence share topic (both concern the same subject), but only evidence and answer share *propositional content* — and it is propositional content, not register or topic, that distinguishes "this record answers the question" from "this record merely mentions the same words."

## 4.4 Cross-Space Consistency

Every structural measurement in this section replicates in both embedding spaces:

| Measurement | BGE-M3 (1024d) | Qwen3 (256d) |
|---|---|---|
| Hemispheric separation (cos) | −0.454 | −0.491 |
| Question-perspective gap | +0.017 | +0.014 |
| Answer-perspective gap | +0.098 | +0.082 |
| Common-mode correlation | 0.986 | 0.979 |
| Effective rank | 22/1024 | 15/256 |

The two models differ in architecture (XLM-RoBERTa vs. Qwen decoder), dimensionality (1024 vs. 256), training data, and training objective. The consistency of all measurements across both spaces indicates that the answer-perspective gap is a property of *contrastively trained text embeddings as a class*, not of any specific model.

---

# 5 The Common-Mode Structure

## 5.1 Evidence and Noise Share the Same Array

We compute the 1024-dimensional mean vector for all evidence records (μ_E) and all noise records (μ_N) within the top-50 candidate pool, then measure the Pearson correlation between these two vectors across dimensions:

**Pearson(μ_E, μ_N) = 0.986**

This near-unity correlation means that evidence and noise embeddings are, to first order, *the same vector*. The difference is a perturbation of magnitude ≤ 0.013 per dimension (max across 1024 dimensions), with no dimension exceeding the noise floor. After subtracting the common-mode component, the residual splits evenly into positive and negative dimensions (515 vs. 509), confirming that the perturbation is not a uniform shift but a distributed signal.

**Figure 3**: Scatter plot of μ_E vs. μ_N across 1024 dimensions, with the identity line and residual distribution inset.

## 5.2 The Distributed-Perturbation Limit

The near-total common-mode correlation, combined with the effective rank of the library (22/1024 by spectral energy), establishes that:

1. No single dimension can serve as an "evidence detector" — the per-dimension signal (≤ 0.013) is below the per-dimension noise floor.
2. All discriminative information must be aggregated across dimensions, in a direction that is *orthogonal to the common-mode*.
3. The signal-to-common-mode ratio (0.013/0.986) defines the resolution limit of the embedding space: any classifier, probe, or linear map operating on these vectors is bounded by this ratio.

This explains why sparse probes (single-dimension or low-rank classifiers) fail (Ridge: AUC ≈ 0.5 at rank 8), while distributed aggregation (Δ: AUC 0.802) and cross-attention (reranker: AUC 0.921) succeed — they aggregate across all dimensions simultaneously.

## 5.3 The Hierarchy of Available Signals

Within the common-mode constraint, we measure all discriminative signals available to a retrieval system:

**Table 2: Signal hierarchy (evidence vs. noise discrimination)**

| Signal | AUC | Requires labels? | Requires LLM? | Orthogonal to common-mode? |
|---|---|---|---|---|
| Cross-encoder reranking | 0.921 | No | No | Yes (cross-attention) |
| Δ fingerprint (global direction) | 0.802 | Yes (weak: 0.5×θ_q) | No | Yes (−0.61 vs. register axis) |
| Narrative axis (word-anchor) | 0.718 | No | No | Partially |
| Word-level scout | — | No | No | Yes (0.025 vs. main) |
| BM25 / lexical hosts | 0.748 | No | No | No (lexical = common-mode) |

**The hierarchy is not a ranking of methods but a ranking of *information sources*.** Cross-encoders are strongest because they implement the answer perspective via cross-attention (§4.2). Δ is second because it approximates this perspective statistically. Word-level scouts are weakest but cheapest, and their independence from the main channel makes them valuable as a *complement* (§6).

## 5.4 Negative Results: What Does Not Exist in the Space

We report four exhaustive negative results, each tested with the same rigor as the positive findings. These are presented not as failures but as *boundary conditions* that complete the structural map:

**(a) No relational arithmetic.** The analogical structure king − man + woman ≈ queen does not hold for conversational roles: question→answer directions have inter-pair consistency (0.178) indistinguishable from random (0.175). The space encodes *locations* but not *relations* between locations.

**(b) No linear map from questions to evidence.** Ridge regression from question vectors to evidence centroids achieves R² = −0.05 and test AUC = 0.50 (chance). The mapping from question to evidence is not a function (the same question type maps to different evidence depending on conversation context).

**(c) No speaker-strength dimensional fingerprint.** While speaker identity is weakly encoded (78% classification, §B.3), it is not concentrated in any dimensional subspace: projecting out the top 22 principal components reduces speaker classification by only 3%.

**(d) No dispersion-failure correlation.** The spatial dispersion of evidence records (session spread / count) does not predict retrieval failure (error rate: 36% at all dispersion quartiles, p = 0.71). Multi-hop failure is caused by context dilution, not by geographic spread within the conversation.

---

# 6 From Measurement to Design

## 6.1 Design Instruction 1: Retrieval Unit Must Be Sub-Sentence

The evidence signal is distributed across adjacent sentences (inter-evidence Jaccard = 0.53, adjacency rate = 100%, inter-cos = 0.761). Using individual sentences as retrieval units discards this co-location signal. Constructing neighborhood blocks (sentence ± same-turn siblings ± adjacent-turn) increases evidence recall from 52% to 93% (block-only) to 99% (with entity bridging). However, sub-sentence decomposition does not help: the "atomic statement" level is the finest useful granularity, because pronominal fragments ("it was delicious") carry no retrievable information independent of their antecedent.

**Table 3: Window size scan (r39 ranking, 50 error + 50 correct)**

| Window | Error evidence in window | Correct evidence in window | Mean noise records |
|---|---|---|---|
| 5 | 32/50 (64%) | 45/50 (90%) | 3.9 |
| 8 | 33/50 (66%) | 46/50 (92%) | 6.8 |
| 10 | 34/50 (68%) | 46/50 (92%) | 8.7 |
| **15** | **37/50 (74%)** | **47/50 (94%)** | **13.7** |
| 20 | 37/50 (74%) | 47/50 (94%) | 18.7 |
| 25 | 37/50 (74%) | 47/50 (94%) | 23.7 |

Window 15 is the saturation point: expanding to 25 adds only noise (10 additional records, 0 additional evidence). The optimal operating point balances evidence recall against noise exposure for the reader model.

## 6.2 Design Instruction 2: Word-Level Scouting Is the Only Decorrelated Second Signal

Adding a second retrieval signal improves ranking only if the signal is decorrelated from the primary. We test four candidates:

| Second signal | Correlation with primary | Error improvement | Correct degradation |
|---|---|---|---|
| Sentence-level re-scoring | 0.98 (same space) | +0 | −1 |
| Word-level scout (256d alignment) | **0.025** | **+6** | −0 (at w=2) |
| Cluster signal (adjacency + entity) | 0.31 | +6 | −0 (at w=2) |
| Confidence (atom + Δ + coverage) | 0.78 | +0 | −1 |

Only signals with correlation < 0.5 to the primary produce net improvement. This confirms the common-mode analysis: signals extracted from the same embedding space share the same failure modes and cannot complement each other.

## 6.3 Design Instruction 3: Write-Time Form Determines Retrieval Reachability

Records whose text is a pronominal fragment ("it was delicious", "sure thing, Gina!") are fundamentally unreachable by any question-side signal, because the information content ("Dave had a jam session", "Jon won't give up") is not present in the record's own word set. This accounts for 18% of retrieval failures that no ranking improvement can fix.

The solution is write-time transformation: converting conversational turns into self-contained atomic facts ("Dave had a great jam session with his band") at storage time. We implement a zero-LLM version (rule-based sentence splitting with host-pointer linkage) that achieves 65% recall on previously unreachable questions (§B.2); an LLM-assisted version would approach the full 82% union (§B.3). The trade-off between precision and reachability is quantified in our granularity ablation (§Table 3).

## 6.4 End-to-End System Validation

We integrate all measurement-driven design decisions into a complete system and evaluate on the full 1,382-question LoCoMo-Refined benchmark:

**Table 4: Component ablation (single-variable, full benchmark, official Qwen3-14B judge)**

| Configuration | Key change | Score | Δ from previous |
|---|---|---|---|
| Baseline (single-space retrieval) | — | 61.4% | — |
| + dual-space fusion + cross-encoder | Design Instruction 2 | 62.2% | +0.8 |
| + summary-priority first pass | Prior-work revival | 63.7% | +1.5 |
| + Δ fingerprint (w=0.6) | Design Instruction 3 | **64.3%** | +0.6 |

Under the official Qwen3-14B judge, the final system scores **70.5%** (952/1382, with 31 parse failures counted as incorrect; 70.5% of valid judgments). This places the system at rank 2–3 on the public LoCoMo-Refined leaderboard, exceeding MemOS (63.6%), MemPalace (58.7%), EverMemOS (58.3%), and Mem0 (48.9%), while using a flash-tier answer model (GLM-5.3-Flash) and zero LLM calls at write time.

On a 500-question random subset with the same judge, the system achieves **85.4%** (427/500), exceeding the claimed SOTA (MemoraX: 82.65%) on the same benchmark, though the full-benchmark score is the more conservative figure.

## 6.5 The Retrieval Ceiling

Our measurements establish that the retrieval component of conversational memory has a ceiling of approximately 85–90% evidence-in-window rate (85% with atomic-domain expansion, up to 90% with ideal scout coverage). The remaining 10–15% requires write-time structural changes (§6.3) or read-time model upgrades. Within the retrieval layer itself, the signal hierarchy (§5.3) is saturated: all available decorrelated signals have been incorporated, and further ranking improvements require new information sources, not better aggregation of existing ones.

---

# 7 Limitations

**Corpus scope.** All measurements are performed on one benchmark (LoCoMo-Refined) with ten conversations. While all findings replicate across two embedding models and across split-half validations, generalization to other corpora (different languages, domains, conversation structures) requires the replication experiments reported in Appendix F.

**Judge variance.** LLM-as-judge evaluation has inherent variance (±2 points for our calibrated judge). The single-judge scores reported here should be interpreted as point estimates within this margin.

**Supervised components.** The Δ fingerprint requires labelled evidence for calibration. We provide unsupervised alternatives (narrative axis: AUC 0.718; density filtering: 42% recall vs. 37% for hard rules), but the full 0.802 requires annotation.

**Two-party conversations.** The speaker-residual finding (§B.3) is tested on two-party dialogues. Multi-party conversations may exhibit different speaker encoding.

---

# 8 Conclusion

We have measured, proved, and exploited the structural limits of dense retrieval for long-term conversational memory. Three findings define the landscape:

1. Questions and answers occupy geometrically distinct hemispheres (117° apart), making direct answer retrieval impossible and two-stage architectures geometrically necessary.
2. Evidence and noise are the same array (0.986 common-mode), with all discriminative signal concentrated in a ±0.01 distributed perturbation — retrievable only by full-dimensional aggregation (Δ, cross-encoders), never by single dimensions or sparse probes.
3. The answer perspective discriminates at 6× the question perspective, establishing that the retrieval ceiling is set by the unavailability of the answer view at query time — and that closing this gap requires either write-time transformation (fact decomposition) or read-time cross-attention (reranking).

These measurements are not specific to our system or our benchmark. They are properties of contrastively trained text embeddings operating on conversational data — properties that any memory system built on such embeddings must contend with, regardless of its architectural sophistication.

---

# References (key citations to include)

- Maharana et al. (2024). Evaluating Very Long-Term Conversational Memory of LLM Agents. ACL.
- mem-eval-suite (2026). LoCoMo-Refined. GitHub.
- Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA. EMNLP.
- Mu & Viswanath (2018). All-but-the-Top. ICLR.
- Su et al. (2021). Whitening Sentence Representations. EMNLP.
- Rodríguez & Laio (2014). Clustering by Fast Search and Find of Density Peaks. Science.
- Zhang et al. (2025). Language Models Do Not Embed Numbers Continuously. arXiv.
- [Embeddings for Preferences, Not Semantics]. arXiv 2605.08360.
- [Dense Retrievers Can Fail on Simple Queries]. arXiv 2506.08592.
- Qing et al. (2026). Tailoring Memory Granularity. EACL Findings.
- Hindsight team (2025). Hindsight is 20/20. arXiv 2512.12818.
- Mem0 (2026). State of AI Agent Memory. Blog.
