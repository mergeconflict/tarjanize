# Analysis Scripts

Python scripts for validating the tarjanize cost model against actual cargo build times.

## Prerequisites

These scripts require:
- Python 3.8+
- A tarjanize symbol graph (e.g., `omicron.json`)
- Cargo timing HTML from `cargo build --timings`

### Getting cargo timing data

```bash
# Run a timed build
cargo build --timings

# The HTML file is at target/cargo-timings/cargo-timing-*.html
# Scripts accept the HTML file directly - no extraction needed
```

## Scripts

### Parallelism Analysis

| Script | Description |
|--------|-------------|
| `analyze_actual_parallelism.py` | Analyze actual build parallelism from cargo timing data (10s buckets) |
| `simulate_model_parallelism.py` | Simulate parallel build using cost model for comparison |
| `analyze_parallelism_1s.py` | High-resolution (1s) parallelism analysis |

**Comparison workflow:**
```bash
TIMING=~/omicron/target/cargo-timings/cargo-timing-*.html

# Analyze actual build
python3 analyze_actual_parallelism.py $TIMING omicron.json

# Simulate using cost model
python3 simulate_model_parallelism.py omicron.json

# Compare the outputs side-by-side to validate model accuracy
```

### Model Validation

| Script | Description |
|--------|-------------|
| `analyze_component_contribution.py` | Test impact of metadata/linking on R² |
| `analyze_frontend_codegen.py` | Breakdown of frontend vs codegen time |
| `analyze_rmeta_pipelining.py` | Evidence of cargo's rmeta pipelining |

### Key Findings

These scripts produced the findings documented in `docs/cost-model-validation.md`:

1. **Model accuracy**: LIB targets R² = 0.856, merged lib+test R² = 0.917
2. **Component contribution**: Metadata +10% R², linking +0.4% (negligible)
3. **Frontend dominance**: 70% of lib compilation is frontend (type checking)
4. **Rmeta pipelining**: Test targets start before lib finishes (uses rmeta, not rlib)

## Usage Examples

```bash
TIMING=~/omicron/target/cargo-timings/cargo-timing-*.html
GRAPH=~/omicron.json

# Full parallelism comparison
python3 analyze_actual_parallelism.py $TIMING $GRAPH > actual.txt
python3 simulate_model_parallelism.py $GRAPH > simulated.txt
diff -y actual.txt simulated.txt | less

# Investigate pipelining for a specific package
python3 analyze_rmeta_pipelining.py $TIMING omicron-nexus

# Check frontend/codegen split
python3 analyze_frontend_codegen.py $TIMING $GRAPH

# Validate component contribution
python3 analyze_component_contribution.py $TIMING $GRAPH
```

## File Formats

### cargo-timing-*.html

Cargo timing HTML file from `cargo build --timings`. Contains embedded JavaScript
with `const UNIT_DATA = [...]` array that the scripts extract automatically.

Each unit object has:
- `name`: package name
- `version`: package version (workspace crates are "0.1.0")
- `target`: target suffix (empty for lib, "(test)" for tests, etc.)
- `start`: start time in seconds
- `duration`: duration in seconds
- `sections`: optional frontend/codegen breakdown

### symbol_graph.json (omicron.json)

Tarjanize output from `cargo tarjanize`. Contains packages with targets:
```json
{
  "packages": {
    "package-name": {
      "targets": {
        "lib": {
          "root": { "symbols": {...}, "submodules": {...} },
          "metadata_ms": 1500,
          "linking_ms": 200,
          "dependencies": ["other-pkg/lib"]
        }
      }
    }
  }
}
```
