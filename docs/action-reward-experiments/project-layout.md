# Project layout and current experiment names

The cluster project now has a task-oriented entry point under `experiments/`. The cleanup separates completed evidence, the current selector-training task and later actor training. It does not change an experiment protocol, source hash, dataset or queued job path.

## Where to start

Paths in this document are relative to the cluster runtime project, `arm-generalization/`. The GitHub folder containing this document publishes the research record; it is not a checkout of the complete runtime.

```text
arm-generalization/
├── README.md
├── experiments/
│   ├── README.md
│   ├── selector-pilot/       completed pilot and length diagnostic
│   ├── selector-training/    current collection and four selector fits
│   │   ├── data/            existing collection artifacts
│   │   ├── models/          existing training artifacts and workflow state
│   │   ├── code/            frozen model-training implementation
│   │   ├── orchestration/   independent Slurm training branches
│   │   ├── collection.yaml  recorded data-collection configuration
│   │   ├── benchmark.yaml   observation/action/horizon configuration
│   │   └── README.md        links to methods, validation and evaluation scope
│   └── actor-sft/           later workstream; no active training
├── armbench/                shared benchmark and inference implementation
├── configs/                 versioned configurations and locks
├── scripts/                 existing runtime entry points
├── vendor/                  pinned upstream repositories
├── docs/experiment-log/     published methods and results
├── reports/                 diagnostics and operational history
└── artifacts/               authoritative versioned run records
```

The root also retains the physical `training_framework/` and `training_framework_v2/` directories. The former is superseded; the latter is the active implementation exposed through `experiments/selector-training/code/`.

## Friendly names and recorded paths

| Workspace entry | Existing authoritative target | Meaning |
|---|---|---|
| `experiments/selector-training/data/` | `artifacts/iterative-arm/v6i-webshop30-text50-1024-seed42/` | Current shared data collection |
| `experiments/selector-training/models/` | `artifacts/selector-training/v2-listwise-sol-preaction/` | Generative SelectionARM and listwise-head training |
| `experiments/selector-training/code/` | `training_framework_v2/` | Active training code |
| `experiments/selector-training/collection.yaml` | `configs/iterative_arm_v6i.yaml` | Collection protocol |
| `experiments/selector-training/benchmark.yaml` | `configs/experiment-webshop30.yaml` | WebShop 30-step / TextWorld 50-step configuration |
| `experiments/selector-pilot/metrics.json` | `docs/experiment-log/data/v3-sol-nativecmd-seed42.json` | Frozen portable pilot results |
| `experiments/selector-pilot/provenance.json` | `reports/pilot-seed42-artifacts.json` | Original pilot artifact hashes |

These are relative symbolic links, not duplicate copies. Each stage README links directly to the relevant method and result documents at their original location, preserving their internal Markdown links. Submitted Slurm scripts continue using their recorded physical paths. The navigation cleanup itself introduced no workflow. A subsequent user-requested [parallel scheduling change](parallel-training.md) added a CPU event-driven controller; no polling daemon is used.

The `v6i` and `v2` names came from different development histories: one versions collection fixes, the other versions training/head architecture. They are not comparable experiment-round numbers. In particular, the directory name `iterative-arm` does not mean iterative actor training is running.

## Read current state rather than old snapshots

- `experiments/selector-training/data/STATUS.md`: human-readable collection status.
- `experiments/selector-training/data/status.json`: audited counts and collection successor.
- `experiments/selector-training/models/parallel/workflow.json`: independent training branches and dependencies; the old `models/workflow.json` points here.
- `experiments/selector-training/models/validation/`: completed small-scale validation.
- `experiments/selector-training/models/research/`: formal-training artifacts, populated as stages finish.

The `models/` name identifies an output workspace; it is not a claim that formal models already exist. The final `data/datasets/round0-5000/` cohort and CPU preparation are now complete.

Current scope is [SelectionARM versus listwise head](selection-arm-head-plan.md), followed later by [actor SFT](qijia-training-workstream.md). The older pointwise `v1-sol-preaction` framework remains stopped. Earlier selector-DPO and scalar-ARM recipes remain historical material, not automatic next stages.

## Storage and retention

Project storage holds original observations, paid Sol labels, authoritative dataset freezes, final weights and provenance. After the October 20 recovery, 74 large raw JSONL files have verified gzip copies on project and uncompressed scratch working copies at their original paths. Five rebuildable `audited-current` views also use scratch; frozen `round0-*` cohorts stay on project. See the [completion record](collection-complete.md) and [restoration instructions](MAINTENANCE.md). Rebuildable caches, compact head features, tokenized inputs, redundant preparation exports and intermediate optimizer checkpoints use Scrubbed storage. Some paths under the workspace therefore resolve through a second link to scratch.

Scrubbed is subject to a 60-day inactivity purge. Navigation links do not make scratch persistent or replace a backup. The paper is a separate sibling checkout named `arm-paper/`; the shared base-model path is recorded in the training configuration.

## Naming future work

Use research-stage names for workspace entries. Record the environment, method, label policy, seed, protocol version and upstream revisions inside each run's configuration and provenance. Introduce a new immutable run for a scientific or implementation change; update the task entry only after validating it.

Keep one authoritative artifact location. Do not rename a frozen run in place or silently combine data/results from different horizons or label policies. Larger physical relocation can be considered after all jobs referencing the old paths have finished.
