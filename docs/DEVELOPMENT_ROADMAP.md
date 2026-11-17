# Shu — Development Roadmap and Balanced GGUF Release Plan

**Status:** Proposal / planning document (not an implementation claim)  
**Created:** 2026-11-17  
**Repository:** https://github.com/Lord1Egypt/Shu  
**Scope:** Future Shu runtime improvements, ready-to-download GGUF releases, and measured low-resource inference.

> This document captures product decisions, hypotheses, and engineering ideas for future work. Nothing under "Proposed" or "Research" is implemented merely because it appears here. Existing measured evidence remains in [AUDIT.md](AUDIT.md) and the repository README.

## 1. Product decision and principles

**Decision:** Publish a **Shu Balanced Models** series consisting of ready-to-download GGUF model artifacts hosted on Hugging Face. The standard releases **must run directly in compatible LM Studio and llama.cpp versions without requiring Shu**. Shu can additionally offer a controlled runtime experience, but it must not be a mandatory dependency for the model series.

Shu's differentiation is **measured practicality**: preserve model quality as much as possible, lower memory demands where feasible, respect hardware limits when Shu is used, and disclose throughput/latency trade-offs.

Non-negotiable principles:

1. **Fit does not mean usable.** An oversized model that produces 0.6 tok/s can demonstrate feasibility but must not be presented as a good daily-use option.
2. **No "half the RAM" marketing.** A model normally observed at 32 GiB is not automatically a viable 16 GiB release. Quantized file size, actual resident memory, KV cache, runtime buffers, cache churn, and throughput are different measurements.
3. **Measure, then label.** Never state that a model meets an 8/16/32 GiB tier without actual measurements on identified hardware, engine, context length, and quantization.
4. **Preserve the original model identity.** A GGUF quantization is not a newly trained model. Credit and link the upstream base model.
5. **Quality is a release gate, not an assumption.** The model must still answer correctly; successful loading alone is insufficient.
6. **Protect everyday computers.** No overclocking requirement, unsafe thermal assumptions, forced GPU saturation, or excessive SSD write experiments.
7. **Keep failed results.** Publish/retain negative or slow internal findings; do not silently turn "could load" into "recommended".
8. **Preserve Shu's existing validated architecture.** Build incrementally and compare against the current baseline.

## 2. Current baseline (verified in the repository)

The README and [engineering audit](AUDIT.md) describe:

- Python standard-library CLI; GGUF inspection, planning, execution, measurement, and optional conversion.
- Ollama integration and caged llama.cpp server; LM Studio settings card rather than an LM Studio hard RAM cage.
- RAM/CPU hard limits via cgroup v2 **only in the caged serving path**; NVIDIA VRAM is planned/monitored, not kernel-caged by a precise per-process hard limit.
- Memory-mapped weights and partial GPU layer offload.
- Temperature guard, with available-sensor limitations.
- A real test host: Intel i7-9750H, WSL2 12.7 GiB RAM, RTX 2060 6 GiB.
- An observed pathological paging case: a ~5 GiB model within a 2 GiB cage re-read roughly 2.26 GiB per output token and generated about 0.34 tok/s. See README for exact run conditions.
- Measured and unmeasured values are distinguished; an unavailable measurement is not a PASS.

**New field observation, not yet independently verified in repo tests:** A user reports loading a ~367 GB quantized large MoE model with a 12 GB RAM setting and ~70% GPU-memory budget, around 4 minutes to load, long thinking output, and ~0.6 tok/s observed. Exact model GGUF build, engine, counters, cache behavior, timing breakdown, and resource enforcement are not yet captured in a reproducible Shu report. Do not publicize this as a validated benchmark. A ~520 GB alternate file was not used.

The experiment motivates **practical bounded inference**, not a claim of having accelerated a large model.

## 3. Release series: Shu Balanced Models

### 3.1 Packaging and naming (proposed)

Offer tested memory tiers **Shu-8G**, **Shu-16G**, and **Shu-32G**, plus explicitly experimental research builds. These are **verified deployment profiles**, not claimed universal memory requirements.

Example repository naming convention:

- Shu-Balanced-Qwen3-8B-GGUF (repository)
- Qwen3-8B-Shu-Q4_K_M.gguf (file)
- Model card: "Tested configuration: 16 GiB system RAM / specified available memory, context 4096, target engine/version, CPU/GPU details"

Naming must distinguish whether the artifact is:
- **Quantized-only:** original architecture and parameters; different storage precision.
- **Modified model:** structural pruning, fine-tuning, distillation, or merging; disclose the actual intervention and its impact.

**Direct-download compatibility is mandatory.** Do not depend on Shu-specific metadata or a special inference runtime to load the standard GGUF files. Optional Shu-specific plan manifests may accompany the files.

### 3.2 Deliverables per released model

- Compatible GGUF file(s), including split-shard instructions only when required.
- Full model card containing original author, base model revision, model license, modification method, quantization method/tool version, context and chat template, model capability limitations, and acknowledgements.
- SHA-256 hashes, file sizes, quantization types, supported engines/builds, and tested hardware.
- Measured RAM (process and whole-machine where possible), VRAM, context, load time, prompt processing speed, decode speed, startup and first-token latency, temperatures, and whether those are peaks/averages.
- A clear **minimum tested configuration** and **recommended usable configuration**. These may differ. State headroom for OS and background applications.
- Same-prompt quality regression results against a well-defined upstream baseline, including multilingual/coding/reasoning subsets as appropriate.
- Exact reproducible conversion command, artifact provenance, and a changelog.
- Optional **Shu runtime profile** for users wanting additional resource isolation. This is supplementary, never a substitute for the GGUF download.

### 3.3 Memory label policy

- "8G", "16G", and "32G" are release categories **only after measurement**; no fixed 50% reduction rule.
- Declare **total machine RAM** separately from **RAM budget available to the model**. A 16 GiB computer may have substantially less than 16 GiB free.
- Publish both **can load** and **usefully runs** outcomes. A file that loads under a cap by thrashing disk is NOT automatically eligible for the Balanced recommendation.
- Account for weight residency, KV cache, scratch/compute buffers, engine overhead, projector/multimodal extras, and OS memory.
- Distinguish **VRAM allocation/offload limit** from **GPU utilization percent**.
- For LM Studio, don't claim an enforced hard RAM/CPU limit from a GGUF alone; direct compatibility and optional settings are different from cgroup enforcement.

## 4. Legal and attribution gate (blocking)

No public derivative-weights upload without a model-specific license review.

1. Pin the exact original **weights** repository and revision. Code license alone is not enough.
2. Confirm permission to **modify, quantize, redistribute**, and use commercially if that use is planned.
3. Review any dependencies: tokenizer, adapters, distillation datasets, model-merging sources, or other borrowed weights.
4. Preserve upstream LICENSE / NOTICE requirements, copyright notices, required attribution, and restrictions on marks or names.
5. Mark the artifact explicitly as quantized/modified, describe modifications, and link to the upstream version.
6. Check distribution-specific terms for each model family/version (do not generalize from a family name).
7. Record reviewer, evidence links, disposition **ALLOW / CONDITIONAL / BLOCK**, and date.

**Preferred initial candidates:** models whose exact weights license permits redistribution under Apache-2.0; verify each target model revision before starting. Shu's MIT license does **not** relicense upstream weights.

## 5. LM Studio distribution and discovery

**User objective:** Ask the LM Studio team to consider Shu's validated GGUF releases for their **Suggested Models / discovery recommendations**.

What is currently practical:
- LM Studio can download compatible Hugging Face GGUF repositories by exact repository identifier or URL.
- A Hugging Face publication does **not** guarantee curated placement in Suggested Models.
- Suggested/featured inclusion is controlled by LM Studio; treat outreach as a future request, not a promised result.

Release readiness for outreach:
1. Stable public HF repository with a directly downloadable GGUF, README/model card, license and hashes.
2. LM Studio compatibility evidence (successful load, prompt processing, streamed output, no broken chat template).
3. Reproducible memory/speed measurements on entry-level hardware, including limitations.
4. Comparison against the unmodified quantization/baseline with the same test conditions.
5. Clear user benefit and maintenance/contact plan.
6. Contact the appropriate LM Studio public channel with exact links and evidence, without claiming endorsement.

**Status:** Not submitted to LM Studio. No model release is claimed by this roadmap.

Official references:
- https://lmstudio.ai/docs/app/basics/download-model
- https://lmstudio.ai/docs/cli/local-models/get
- https://huggingface.co/docs/hub/lmstudio
- https://huggingface.co/docs/transformers/main/quantization/gguf

## 6. Proposed runtime development

### P0 — Honest planning and practical viability (highest priority)

**Goal:** Make Shu distinguish a small safe oversubscription from pathological paging.

Proposed:
- Add a **balanced** planner objective alongside the existing **research/fit** path.
- Build a memory model for total weights, active execution working set, KV cache, buffers, CPU/GPU placement, and disk-backed pages.
- Report **estimated steady-state disk bytes read per generated token**, not just file-size-minus-RAM.
- Account for Dense versus MoE architectures; do not assume a MoE's entire file is touched for each token.
- Estimate prefill and decode separately; include load time and time-to-first-token.
- Offer candidate configurations ranked by predicted usable speed **subject to** memory, VRAM, and thermal constraints.
- Warn or decline a "Balanced" recommendation when a target speed is implausible. User can still explicitly choose research mode.
- Preserve and improve measurement coverage; no synthetic PASS for missing telemetry.

**Acceptance criteria:** Predictions are calibrated against measured results on multiple model families; calibration error is reported. The project must not promise a universal tok/s threshold.

### P1 — Safe adaptive optimization

Proposed:
- Optional user-selected goal such as 5 or 10 tok/s, treated as a **target** not a guarantee.
- Tune GPU layers, CPU threads, batch size, context size, quantization alternatives, and model variants within safe bounds.
- Avoid needless rewriting or re-quantizing a model on every run; cache reusable profiles keyed by file hash, engine version, hardware and settings.
- Use a bounded search with early stop, rollback on performance regression, and report all attempts.
- Separate performance constraints from thermals, power and battery.
- Offer a conservative **Safe Research** mode that stops on OOM, repeated faults, excessive temperature, or device instability. Never advertise that temperature alone proves safety.

**Acceptance criteria:** Improvement over the same unmodified engine/model/quantization baseline, with identical context, prompt and sampling; no unsubstantiated speedups.

### P2 — MoE-aware expert residency and caching (research)

**Hypothesis:** For MoE models, keeping shared compute and frequently selected experts resident while fetching only necessary routed experts could reduce storage traffic compared with whole-layer paging.

Design directions:
- Model **shared attention / routers / shared experts / per-layer routed experts / KV cache** as distinct residency classes.
- Measure real expert activation distributions, working-set size, reuse, cache hit rate, miss penalty, read amplification, and GPU upload cost.
- Keep shared/high-frequency tensors in RAM/VRAM where feasible; back cold experts with SSD.
- Research capacity-bounded expert cache eviction (LRU, frequency-aware, adaptive) and asynchronous prefetch.
- Investigate architecture-aware sparse tables or Engram-style memory access only for architectures that actually expose such tensors.
- Prefer existing supported engine mechanisms where applicable; adding a real expert-level scheduler may require llama.cpp changes or a new backend. **Current Shu cannot intercept the model's internal routing just by setting cgroup or layer counts.**
- Do not equate "experts" with external factual documents. Their weights participate in neural computation.
- Do not claim that active parameters for one token represent an unchanging set across all tokens.

**Acceptance criteria:** Fewer physical bytes read per token and lower decode latency at the same model, precision, context, output, and memory limit, without material output-quality regression.

### P3 — Streaming, cache and storage awareness (research)

- Test asynchronous expert/tensor prefetch against Linux's native readahead (which may already be strong on sequential scans).
- Separate cold load, warm cache and sustained decode measurements.
- Model SSD/HDD throughput and random I/O, page faults, reclaim and cache-thrashing.
- Consider io_uring or equivalent only after evidence that overhead is worth it.
- Minimize write amplification and unnecessary model conversion; never depend on a hazardous amount of temporary storage.
- Do not predict fast inference from raw sequential SSD bandwidth alone; account for access patterns, MoE expert selection and compute.
- Keep optional implementation portable where practical while noting initial Linux/WSL2 scope.

### P4 — Performance rating and transparent recommendations

A proposed **Shu Balanced Performance Rating** should be transparent and reproducible, not an unexplained marketing number.

Include separately:
- **Quality retention:** task-based baseline-relative results.
- **Decode throughput:** median and tail tok/s under fixed settings.
- **Latency:** cold-load, first-token, long-context and sustained generation.
- **Resource demand:** peak measured RAM/VRAM and CPU consumption.
- **Stability:** OOM, overshoots, thermal pauses, engine failures.
- **Storage overhead:** file size, disk reads per generated token, optional temp-storage/write cost.

Do **not** reduce dissimilar models to a single score without reporting its weighting, caveats and raw measurements. Rankings are only comparable within matching measurement protocols and model/task categories.

### P5 — Model reduction research (separate from runtime control)

- **Quantization:** primary near-term pathway to ship smaller GGUF artifacts. A lower bit width changes precision/file size but usually **does not reduce the number of parameters**.
- **Mixed-precision / activation-aware quantization:** preserve sensitive tensors at higher precision and assess quality loss carefully.
- **Structured pruning:** could actually remove architectural components, but risks destroying useful computation and frequently requires recovery training.
- **Distillation:** train a smaller student from a teacher's permitted outputs/data; teacher performance is **not automatically transferred**. Respect API, licensing and data-use terms.
- **RAG + domain adapters:** may widen accessible knowledge or improve specialties but are **not equivalent** to attaching a 500B model's neural weights to an 8B model.
- **Hybrid research track:** investigate small local base models with specialized adapters and retrieval. Keep separate from "same model at lower RAM" claims.

Do not promise a 1B model with all 8B capabilities, or a dense model with 500B-quality reasoning, without controlled evidence.

## 7. Delivery phases and acceptance gates

| Phase | Deliverable | Gate |
| --- | --- | --- |
| 0 — Baseline | Pin Shu revision; standard benchmark and quality harness; measured existing model | Reproducible numbers and no regressions |
| 1 — First artifact | One directly downloadable, legally cleared GGUF plus transparent model card | Opens and generates in vanilla LM Studio + llama.cpp |
| 2 — Tier validation | Measured 8/16/32 GiB class labels only where justified | RAM/VRAM/throughput/context/headroom documented |
| 3 — Balanced planner | Predicted viability, disk pressure warnings, candidate settings | Predictions compared to actual runs |
| 4 — Adaptive tuner | Optional safe settings search with rollback | Demonstrated gain on controlled baseline |
| 5 — MoE cache experiment | Prototype expert residency on one supported small MoE | Measurable I/O/latency reduction; quality maintained |
| 6 — Catalog outreach | Evidence-backed LM Studio suggestion request | Request sent; acceptance not guaranteed |

**Initial test strategy:** Start with a small license-cleared GGUF on the existing host. Avoid repeating the huge ~367 GB experiment merely to validate roadmap logic. Never require days-long training or large new downloads for first-stage validation.

## 8. Benchmark and quality protocol

For each comparison, record:
- System, RAM, CPU, GPU, VRAM, storage type, OS/WSL version, engine commit/version.
- Exact base model, weights revision, tokenizer, GGUF checksum, quantization method/version.
- Context length, number of active requests, prompt/input tokens, generated tokens, batch, threads, GPU layers, cache state, settings.
- Cold-load time, prefill time, first-token time, decode tokens/sec (exclude prompt processing from decode), and total wall clock.
- Peak system/process/cgroup RAM, VRAM measurements with attribution caveats, disk bytes/token, temperatures, throttling/sensor failures.
- A fixed quality set including instruction-following, factual retrieval, reasoning, coding and multilingual tasks where relevant. Evaluate per domain rather than claiming universal quality.
- Results from repeated short runs where feasible; provide variability and failure counts.
- Exit code/status distinct for **PASS / OVER-LIMIT / NOT MEASURED / TOO SLOW / QUALITY REGRESSION**. Do not treat "no reading" as zero usage.

Recommended artifacts: a JSON report, a Markdown summary table, scripts/commands for reproduction, and machine-readable release metadata.

## 9. Documentation and promotion requirements

For each HF upload:
- Base-model reference and accurate model license.
- GGUF download and standard LM Studio instructions that do **not** require Shu.
- Optional instructions to use Shu for enforced Linux/WSL resource budgets, clearly separated.
- A simple "works on this tested hardware" table and clear non-guarantees on other systems.
- Disclose that any 8/16/32G label depends on context, quantization, runtime, and free RAM.
- No "trained by Shu" claims for quantized upstream models.
- Optional inclusion in a Hugging Face Collection named Shu Balanced Models.
- Public suggestion outreach to LM Studio after actual release and compatibility evidence, **not before**.

## 10. Open questions (do not guess)

1. What exactly should "16G" mean in the public catalog: total physical RAM, RAM available to the process, or both?
2. What is an acceptable minimum usable decode speed by model class and use case?
3. Which model family and precise revision offer a practical first license-cleared benchmark?
4. Which quantization/compatibility target is best for direct LM Studio downloads?
5. Can MoE expert caching improve latency over the existing engine's mmap and caching at identical budgets?
6. Which telemetry can be trusted under WSL2 versus native Linux?
7. How should CPU-only, NVIDIA, and future non-NVIDIA configurations be labeled separately?
8. What maintenance/retest policy should apply when LM Studio or llama.cpp versions change?

## 11. Priority order

1. **Now:** Legal gate + one ready-to-download, bench-tested GGUF + transparent model card.
2. **Next:** Balanced planning, realistic memory tiers, performance evidence.
3. **Then:** Safe adaptive tuning and measured storage-aware optimizations.
4. **Research:** MoE expert residency, active working-set tracking, expert prefetch, hybrid intelligence.
5. **Distribution:** Once releases exist, request consideration by LM Studio for Suggested Models. The team decides whether to feature them.

**Project criterion:** A model is a **Balanced** release only if it is legal to redistribute, loads directly in ordinary supported runtimes, has verified resource claims, maintains acceptable quality, and performs well enough to be useful. Merely forcing an oversized model inside a memory cap is a **Research** result, not a Balanced release.
