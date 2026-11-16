# Engineering audit: status

An engineering audit of commit `e0c7eae` (when the project was still called Ptah)
listed the issues below. Each one was checked against the code and, where it held,
fixed and tested. "Verified" means there is a test or a measurement on real hardware
behind the claim; anything else is marked as open.

Test machine: Intel i7-9750H, 12.7 GiB RAM (WSL2), NVIDIA RTX 2060 6 GiB,
Ollama 0.40 with its bundled llama.cpp.

## Prioritized findings

| # | Finding | Priority | Verdict | What changed |
|---|---|---|---|---|
| 1 | Missing measurements could count as a pass | High | Confirmed, fixed | A value that could not be read is `None`, shown as `NOT MEASURED`, and the run exits with code 3. It never counts as inside the budget. |
| 2 | Thermal guard started after the model loaded | High | Confirmed, fixed | The guard starts with the engine process, before loading. It refuses to start when no sensor exists and pauses the engine if sensors stop answering for five polls. |
| 3 | The stop signal might hit `systemd-run`, not the engine | High | Not reproduced, hardened | `systemd-run --scope` replaces itself with the engine, so the process ID is the engine's (checked on the test machine). Pausing now freezes the whole cage (`cgroup.freeze`), stopping kills anything left in it (`cgroup.kill`), and the engine is terminated by the kernel if Shu dies. |
| 4 | Tuning only looked at VRAM | Medium | Confirmed, fixed | `--verify` now also corrects RAM (smaller batch, then smaller context) and CPU (one thread fewer), then spends spare VRAM on GPU layers. |
| 5 | The last tuned configuration could go unmeasured | Medium | Confirmed, fixed | Whatever is installed at the end is measured; the result carries `verified: true/false`. |
| 6 | Speed estimates used a fixed disk speed | Medium | Confirmed, fixed | The model's disk is timed with a parallel read probe. The paging model itself was also wrong and was replaced after measurement (see below). |
| 7 | Monitoring samples kept without limit | Medium | Confirmed, fixed | History is a bounded queue; peaks are tracked separately so they survive. |

## Other items from the audit

| Item | Status |
|---|---|
| Handle sensor failures and missing telemetry explicitly | Fixed (findings 1 and 2); sensor failures are counted and reported |
| Verify cgroup enforcement; separate hard limits from estimates | Fixed: `memory.max` and `cpu.max` are read back from the cage and every result says whether RAM and CPU were enforced by the kernel or only bounded by settings |
| Process lifecycle, no orphans | Fixed: engine dies with its parent (verified by killing the parent with SIGKILL), cage is emptied on stop, SIGTERM and Ctrl+C clean up |
| Recalculate estimates after every tuning step | Fixed: `plan.json` and the generated files are rewritten from the tuned settings |
| Replace fixed SSD assumptions with real measurements | Fixed (finding 6) |
| Do not unload unrelated Ollama models without permission | Fixed: Shu stops with an explanation unless `--unload-others` is given; the wizard asks |
| Handle out-of-memory kills and memory pressure | Fixed: an OOM kill inside the cage is detected and explained; memory pressure itself is the kernel reclaiming model pages, which is the intended behavior |
| Unavailable cgroup controllers | Handled: Shu runs without a cage and says the limits are not enforced |
| Why was Q3_K_M slower than Q4_K_M? | Resolved: it was not the quantization. The first cage design set `MemoryHigh`, which made the kernel stall a process reading a cold file. With the hard limit only, Q3_K_M measured 7.4 tokens/s against 6.7 for Q4_K_M |
| Extend tuning to inference speed | Partly: tuning moves layers to the GPU while VRAM allows, which is the main speed lever, and records tokens/s for every round. It does not search thread counts or batch sizes for speed |
| Validate architecture-specific memory estimates | Partly: checked by measurement on three architectures (qwen35 hybrid attention, gemma4, llama). Estimates err on the safe side; `--verify` corrects the rest. Other architectures are unmeasured |
| Improve GPU detection, support several GPUs | Open: Shu plans and measures the first NVIDIA GPU only and says so. It cannot be tested on the single-GPU test machine |
| Per-process VRAM instead of the change in total VRAM | Open: `nvidia-smi` reports no per-process memory under WSL2. Other GPU programs can still distort a measurement |

## What the measurements changed

Two results from re-measuring are worth more than the individual fixes.

**`MemoryHigh` was the wrong tool.** The first cage set a soft limit just under the
hard one. On a model larger than the RAM budget the kernel then stalled the engine on
purpose, on top of reclaiming its pages. The same run, a 5 GiB model in a 2 GiB cage,
went from about 35 seconds per token to 2.6 once only `MemoryMax` was set.

**The page cache does not help a model that does not fit.** The first estimate
assumed only the part that overflows RAM is re-read. Measurement says otherwise:
the layers are swept in order for every token, so once they no longer fit, the
cache always holds the wrong pages. With 2.2 GiB of CPU-side layers the engine
read 2.17 GiB from disk per token. The estimate is now simply

    seconds per token = bytes of CPU-side layers / measured disk speed

and for that run it predicts 2.4 seconds against 2.6 measured.

## Research directions (not implemented)

The audit also set out a longer plan. None of it exists in the code yet; it is
recorded here so the direction is not lost. Any of it has to beat the measured
baseline above to be worth building.

- **Stream**: read weights from disk ahead of the computation, predictively, so
  disk and compute overlap. The baseline to beat is the kernel's own readahead on a
  sequential sweep.
- **Cache**: classify tensors as hot or cold, compress the KV cache, evict under
  memory pressure.
- **Scheduler**: place layers across CPU, GPU and disk by measurement; run several
  models at once with isolation between them.
- **Optimizer**: profile the machine, choose quantization, threads and batch size,
  and keep the result as a reusable profile.
- **Runtime**: an independent lightweight GGUF runtime with out-of-core execution.
- **Platforms**: native Windows, macOS, Android ARM64, AMD and Intel GPUs, Vulkan.

Rules kept for all of it: compare against the unmodified engine with identical
settings, keep failed experiments, and claim no improvement without a
reproducible measurement.
