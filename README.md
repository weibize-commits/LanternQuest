# LanternQuest

LanternQuest is a browser-based WebXR environment and research codebase for evidence-grounded interaction with Bianjing lantern heritage. The repository accompanies the manuscript on persistent, source-linked evidence obligations for auditable neuro-symbolic planning.

## Contents

- `src/lanternquest`: evidence obligations, symbolic authorization, evidence-debt scheduling and evaluation infrastructure.
- `game/app`: Three.js and WebXR Device API client used for the virtual task.
- `scripts`: frozen analyses and manuscript figure-generation code.
- `configs`: frozen machine-evaluation and extraction protocols.
- `kg`: public ontology, schemas and graph-processing code.
- `data`: anonymized analysis-ready participant outcomes and aggregate machine, expert, extraction and figure source data.

## Environment

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .[analysis,figures]
pytest
```

The WebXR client can be installed separately.

```bash
cd game/app
npm ci
npm run build
```

The formal participant sessions used a Meta Quest 3 head-mounted display with Meta Quest Touch Plus controllers. The application ran in standalone mode through Meta Quest Browser and the headset-native WebXR runtime.

## Data boundaries

The public human-study file uses release-specific participant and block identifiers. Direct identifiers, contact details, dates, timestamps, narratives, comments, raw event logs, photographs, consent forms and demographic microdata are excluded. The raw heritage media, the 160 restricted text segments, row-level extraction predictions and trained checkpoints are also excluded because their source agreements prohibit public redistribution. See `DATA_AVAILABILITY.md` and `data/DATA_DICTIONARY.md`.

## Citation

Please cite the associated LanternQuest manuscript and the tagged repository release. A journal citation and archival DOI will be added when available.
