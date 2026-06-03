# Configuration Reference

ReFlex resolves configuration with an explicit precedence:

```
code defaults  <  YAML file  <  REFLEX_* environment variables  <  explicit overrides
```

Load it in code or via the CLI:

```python
from reflex import ReflexConfig
cfg = ReflexConfig.load("configs/example.yaml")   # YAML + env applied
```

```bash
reflex run --config configs/example.yaml
reflex config --config configs/example.yaml       # print the fully-resolved config
```

## Environment variables

Use the `REFLEX_` prefix and `__` to descend into nested sections. Values are parsed as YAML
scalars (so `true`, `42`, `0.5` get native types):

```bash
export REFLEX_LLM__PROVIDER=openai
export REFLEX_LLM__BASE_URL=http://localhost:8000/v1
export REFLEX_LLM__API_KEY=sk-...
export REFLEX_MEMORY__RETRIEVAL__TOTAL_K=12
```

Env vars override YAML, which keeps secrets out of checked-in files.

## Sections

### `agent`
| key | default | meaning |
|---|---|---|
| `name` | `reflex` | display name |
| `persona` | (grounded assistant) | system-prompt persona |
| `goals` | `[]` | standing goals injected into working memory |
| `session_id` | auto | fixed session id; auto-generated if unset |

### `llm`
| key | default | meaning |
|---|---|---|
| `provider` | `mock` | `mock` (offline) or `openai` (vLLM/SGLang/OpenAI) |
| `model` | `reflex-mock` | model id |
| `base_url` | none | OpenAI-compatible endpoint, e.g. `http://localhost:8000/v1` |
| `api_key` | none | bearer token (prefer the env var) |
| `temperature` / `max_tokens` | `0.4` / `1024` | sampling controls |
| `timeout_s` / `max_retries` | `60` / `2` | HTTP resilience (429/5xx retried with backoff) |

### `embeddings`
| key | default | meaning |
|---|---|---|
| `provider` | `hashing` | `hashing` (offline) or `sentence_transformers` |
| `model` | `hashing-256` | model id |
| `dim` | `256` | vector dimensionality (hashing only) |
| `normalize` | `true` | L2-normalise vectors |

### `memory`
| key | default | meaning |
|---|---|---|
| `db_path` | `reflex_memory.db` | SQLite file (`:memory:` for ephemeral) |
| `short_term_capacity` | `40` | short-term buffer size |
| `working_token_budget` | `2048` | working-memory token budget |
| `vector.backend` | `numpy` | `numpy` (exact) or `faiss` |
| `vector.metric` | `cosine` | `cosine` / `ip` / `l2` |
| `retrieval.*_k`, `total_k` | see default.yaml | per-tier and total retrieval depth |
| `retrieval.min_score` | `0.0` | drop hits below this similarity before fusion |
| `retrieval.recency_*` | half-life 1d, weight 0.15 | recency prior in score fusion |
| `compaction.*` | enabled, thr 200/500 | when/how to compress durable tiers |

### `reflection`
| key | default | meaning |
|---|---|---|
| `enabled` | `true` | run self-correction after each turn |
| `every_n_turns` | `1` | reflection cadence |
| `min_importance_to_store` | `0.55` | confidence gate for persisting facts |

### `integrity`
| key | default | meaning |
|---|---|---|
| `enabled` | `true` | run the consistency/hallucination guard |
| `support_threshold` | `0.18` | min grounding similarity before flagging |
| `block_threshold` | `0.8` | flag severity that blocks/triggers a revision |
| `on_violation` | `revise` | `flag` (keep), `revise` (re-prompt), or `raise` |
| `max_revisions` | `1` | revision attempts before giving up |

### `logging`
| key | default | meaning |
|---|---|---|
| `level` | `INFO` | log level |
| `rich` | `true` | use Rich console handler if available |

See [`configs/default.yaml`](../configs/default.yaml) (offline) and
[`configs/example.yaml`](../configs/example.yaml) (ROCm/vLLM) for complete, commented files.
