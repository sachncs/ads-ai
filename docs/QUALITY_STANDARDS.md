# Ads.ai Agent Quality Standards

This document defines the quantifiable expectations for each agent in the advertising pipeline,
plus the production-grade code standards that apply across all modules.

---

## 1. Code Standards (Applied Globally)

### 1.1 Logging & Instrumentation
- Every public method entry/exit uses ``logger.info()`` with ``perf_counter`` elapsed time.
- Failures use ``logger.exception()`` (which auto-attaches traceback) and re-raise a
  typed exception — never bare ``except Exception``.
- Log metadata uses ``key=value`` pairs (e.g. ``concept_label``, ``variant_count``).
- ``time.perf_counter()`` is used for wall-clock timing (not ``time.time()``).

### 1.2 Exception Handling
- Each agent module defines at least one named ``Error`` exception (e.g.
  ``VideoGenerationError``, ``FileOperationError``).
- ``try/except`` uses an explicit ``exc: Exception | None = None`` pattern so that
  ``finally`` blocks can log elapsed time regardless of whether an exception was raised.
- Docstrings list only the exceptions that are actually raised.

### 1.3 Type Safety
- The ``BaseAgent.generate()`` method uses ``@overload`` decorators to give callers a
  precise return type: ``BaseModel`` when ``response_schema`` is provided, ``str`` when
  it is ``None``.
- The ``save_json()`` function uses ``runtime_checkable`` Protocols (``ModelDumper``,
  ``LegacyDumper``) for structural subtyping of serializable objects.
- SDK imports suppress the ``[attr-defined]`` mypy error via ``# type: ignore`` since
  ``google-genai`` ships without type stubs.

### 1.4 Documentation
- Every ``__init__``, public method, and module-level constant has a docstring.
- Docstrings use Args / Returns / Raises / Attributes sections per the Google style.
- ``Raises:`` sections list only the exceptions that can actually be raised, not generic
  ``Exception``.

### 1.5 Linting
- All code passes ``ruff check ads_ai/`` (zero errors).
- Ruff configuration enforces: F (pyflakes), I (isort), N (naming), W (warnings), UP
  (modernization), I (imports). The project targets Python 3.10+ and uses ``X | None``
  union syntax throughout.

---

## 2. Strategy & Audience Intelligence

### URL Intelligence Agent
- **Expectation**: Must extract > 80% of core fields (``brand_name``, ``value_proposition``,
  ``target_audience``).
- **Quality Marker**: ``missing_information`` list should only include truly absent web data
  (e.g. internal pricing).

### Strategy Agent
- **Expectation**: ``PreReleaseTargets`` must define numeric thresholds for all 5 dimensions.
- **Quality Marker**: ``strategic_pillars`` must correlate with identified audience pain points.

---

## 3. Creative & Video Generation

### Creative Agent
- **Expectation**: Every ``AdScript`` must contain at least 3 ``Scene`` objects.
- **Quality Marker**: ``hook`` must be concise (< 15 words) and present in the first scene's
  visual cues.
- **Video Prompt**: Must include camera path, lighting, and cinematic quality descriptors.

### Video Generation Agent
- **Expectation**: Synthesized prompt must be a single paragraph > 100 words.
- **Quality Marker**: Must include at least 3 "Cinematic Markers"
  (e.g. "4k", "Atmospheric Lighting", "Macro-photography").
- **Technical**: Uses ``_synthesize_video_prompt()`` (LLM prompt architect step) before
  calling ``client.models.generate_videos()``. Polls operation status with a 300-second
  timeout. Raises ``VideoGenerationError`` on timeout or if no videos are returned.
- **Attributes**: The agent exposes ``self.video_model`` (Veo model identifier) in addition
  to the base ``self.model_name`` (text model used for prompt synthesis).

---

## 4. Evaluation & Feedback

### Evaluation Agents (Clarity, Brand, etc.)
- **Expectation**: Must provide specific ``IssuesIdentified`` and ``RecommendedFixes`` if
  score is below target.
- **Quality Marker**: Justifications for scores must cite specific lines or visual cues from
  the script.

### Scoring Agent
- **Expectation**: ``CompositeScore`` must be a weighted aggregate of all sub-evaluations.
- **Quality Marker**: ``is_ready`` should only be ``True`` if ``critical_defects`` is empty.

### Iteration Controller
- **Expectation**: ``IterationDirective`` must address the highest-impact failure points from
  the evaluation agents.
- **Quality Marker**: Directives must specify which stage (Hook, Body, CTA) to modify.

---

## 5. Run Artifact Management

### file_ops.py
- ``ensure_output_dir()`` creates a timestamped run directory (``run_YYYYMMDD_HHMMSS``).
- ``save_json()`` accepts any object with ``model_dump()`` (Pydantic v2), ``dict()``
  (Pydantic v1), or a plain ``dict`` / ``list`` / JSON scalar. Raises
  ``FileOperationError`` on serialization failure.
