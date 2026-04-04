# Ads.ai Agent Quality Standards

This document defines the quantifiable expectations for each agent in the advertising pipeline. These standards are used to validate agent outputs and provide feedback during the iteration loop.

## 1. Strategy & Audience Intelligence

### URL Intelligence Agent
- **Expectation**: Must extract > 80% of core fields (`brand_name`, `value_proposition`, `target_audience`).
- **Quality Marker**: `missing_information` list should only include truly absent web data (e.g. internal pricing).

### Strategy Agent
- **Expectation**: `PreReleaseTargets` must define numeric thresholds for all 5 dimensions.
- **Quality Marker**: `strategic_pillars` must correlate with identified audience pain points.

## 2. Creative & Video Generation

### Creative Agent
- **Expectation**: Every `AdScript` must contain at least 3 `Scene` objects.
- **Quality Marker**: `hook` must be concise (< 15 words) and present in the first scene's visual cues.
- **Video Prompt**: Must include camera path, lighting, and cinematic quality descriptors.

### Video Generation Agent
- **Expectation**: Synthesized prompt must be a single paragraph > 100 words.
- **Quality Marker**: Must include at least 3 "Cinematic Markers" (e.g., "4k", "Atmospheric Lighting", "Macro-photography").
- **Technical**: Must handle long-polling (retry/done) logic for generation operations.

## 3. Evaluation & Feedback

### Evaluation Agents (Clarity, Brand, etc.)
- **Expectation**: Must provide specific `IssuesIdentified` and `RecommendedFixes` if score is below target.
- **Quality Marker**: Justifications for scores must cite specific lines or visual cues from the script.

### Scoring Agent
- **Expectation**: `CompositeScore` must be a weighted aggregate of all sub-evaluations.
- **Quality Marker**: `is_ready` should only be true if `critical_defects` is empty.

### Iteration Controller
- **Expectation**: `IterationDirective` must address the highest-impact failure points from the evaluation agents.
- **Quality Marker**: Directives must specify which stage (Hook, Body, CTA) to modify.
