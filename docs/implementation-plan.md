# Colophon Implementation Plan

## 1) Product goal and constraints

Colophon is a cooperative multi-agent system for producing long-form narratives (articles, books, reports) from:

- a user bibliography,
- a preliminary knowledge graph,
- a hierarchical outline.

The system should preserve citation grounding, maintain cross-chapter consistency, and scale from paragraph-level drafting to full-book assembly.

### Core design principles

1. **Traceability first**: every claim should be linked to evidence and provenance metadata.
2. **Graph-aware generation**: writing decisions should be informed by entities, relations, and unresolved gaps in the KG.
3. **Hierarchical planning**: chapter/section/paragraph decomposition with explicit contracts between levels.
4. **Quality gates**: factuality, style, coherence, and citation checks before promotion to the next stage.
5. **Human-in-the-loop**: user can inspect/override plans, claims, and drafts at each level.

## 2) High-level architecture

```text
Inputs
  ├─ Bibliography (BibTeX, RIS, CSL JSON, PDFs)
  ├─ Preliminary KG (RDF/Property graph/JSON-LD)
  └─ Outline (book > chapter > section > subsection)

Ingestion + Indexing Layer
  ├─ Document parser + citation normalizer
  ├─ Chunker + embedding pipeline
  ├─ Entity/relation extraction
  └─ KG merge + conflict resolver

Knowledge Services Layer
  ├─ Vector retrieval (semantic)
  ├─ Graph retrieval (multi-hop neighborhood, path queries)
  ├─ Hybrid ranker (vector + graph + citation priors)
  └─ Claim memory store (accepted/rejected/uncertain claims)

Agent Orchestration Layer
  ├─ Planner agents (book/chapter/section)
  ├─ Claim author agents
  ├─ Paragraph synthesizer agents
  ├─ Reviewer/critic agents
  └─ Consistency + citation verifier agents

Composition Layer
  ├─ Section compiler
  ├─ Chapter compiler
  └─ Whole-book consistency pass

Interfaces
  ├─ CLI + API
  ├─ Review UI (diffs, evidence panes, graph views)
  └─ Exporters (Markdown, LaTeX, DOCX)
```

## 3) Data model and storage strategy

### Primary objects

- **SourceDocument**: bibliographic metadata, file refs, parsing quality.
- **EvidenceChunk**: text span, source pointer, embedding, section context.
- **Entity / Relation**: canonical IDs, aliases, confidence.
- **Claim**: normalized proposition + supporting/opposing evidence.
- **OutlineNode**: level, goals, constraints, dependencies.
- **DraftUnit**: paragraph/section/chapter text + provenance map.
- **ReviewArtifact**: quality scores, critiques, remediation actions.

### Suggested storage stack

- **Postgres** for metadata, jobs, agent state.
- **Vector DB** (pgvector, Qdrant, or Weaviate) for chunk/claim embeddings.
- **Graph DB** (Neo4j or Memgraph) for KG traversal and consistency checks.
- **Object store** for PDFs, parsed artifacts, and generated snapshots.

## 4) Multi-agent workflow (hierarchical)

### A. Planning stage

1. **Outline Planner** converts the user outline into explicit writing objectives per node:
   - target arguments,
   - required entities/claims,
   - expected evidence depth,
   - style/register constraints.
2. **Gap Analyzer** queries KG + corpus to detect missing evidence and weakly connected concepts.
3. **Plan Refiner** proposes additional subsections or claim targets where gaps are severe.

### B. Claim generation stage

1. **Claim Author** produces candidate claims for each section objective.
2. **Evidence Retriever** performs hybrid retrieval for each claim.
3. **Claim Verifier** checks support/contradiction/uncertainty status.
4. **Claim Curator** accepts, rewrites, or flags claims for human review.

### C. Drafting stage

1. **Paragraph Synthesizer** builds paragraphs from accepted claims + retrieved context.
2. **Local Coherence Critic** checks transitions, redundancy, and argument flow.
3. **Citation Binder** attaches in-text citations and bibliography keys.

### D. Compilation stage

1. **Section Composer** assembles paragraphs into sections with intro/body/conclusion logic.
2. **Chapter Composer** enforces chapter-level narrative arc and motif continuity.
3. **Book Consistency Agent** detects cross-chapter contradictions, term drift, and style variance.

### E. Final QA stage

- factuality and citation integrity checks,
- unresolved-claim report,
- completeness vs outline report,
- export pipeline validation.

## 5) Retrieval and KG integration design

### Hybrid retrieval scoring

Use a weighted score for candidate evidence:

`score = wv * vector_similarity + wg * graph_relevance + wc * citation_authority + wt * recency/topic fit`

Where:

- **graph_relevance** rewards chunks linked to required entities/relations for the active outline node,
- **citation_authority** can encode source reliability priors.

### KG-aware prompting

For every generation task, include:

1. top-k evidence chunks,
2. relevant 1-hop/2-hop KG neighborhood,
3. unresolved contradictions related to target entities,
4. explicit output schema (claims, rationale, citations).

### Claim provenance contract

Each claim must carry:

- `claim_id`,
- textual statement,
- support set (chunk IDs),
- contradiction set (if any),
- confidence,
- owning outline node.

## 6) Orchestration and runtime

### Orchestrator responsibilities

- maintain DAG of tasks from outline tree,
- enforce dependencies (claims before paragraphs, etc.),
- schedule retries with adjusted prompts,
- route flagged artifacts to human review queues.

### Implementation option

- Start with a Python orchestration service (FastAPI + Celery/Temporal).
- Use event-driven state transitions (e.g., `SECTION_PLAN_READY`, `CLAIMS_VALIDATED`, `CHAPTER_COMPILED`).

## 7) Evaluation framework

Define measurable quality metrics per layer:

1. **Claim-level**: support precision/recall, contradiction rate.
2. **Paragraph-level**: coherence score, citation density, unsupported sentence count.
3. **Section/chapter-level**: objective coverage, redundancy index, cross-reference quality.
4. **Book-level**: global consistency, style stability, unresolved issue count.

Add regression datasets with gold references:

- small “mini-book” benchmark,
- domain-specific chapter benchmark,
- adversarial contradiction benchmark.

## 8) Suggested phased roadmap

### Phase 0: Foundation (2-3 weeks)

- ingestion pipeline for bibliography + PDFs,
- chunking + embeddings,
- baseline vector retrieval,
- minimal CLI workflow.

### Phase 1: KG + claims (3-4 weeks)

- KG import/merge,
- claim schema + claim store,
- hybrid retrieval,
- claim verification loop.

### Phase 2: hierarchical drafting (4-6 weeks)

- planner, claim author, paragraph synthesizer, critics,
- section/chapter composition,
- basic quality gates.

### Phase 3: review UX + exports (3-4 weeks)

- reviewer UI with provenance panes,
- feedback loops to re-plan/rewrite,
- robust Markdown/LaTeX export.

### Phase 4: hardening (ongoing)

- benchmark-driven tuning,
- observability and cost controls,
- multi-book project support.

## 9) Minimal v1 API surface

- `POST /projects` – create project with outline + bibliography.
- `POST /projects/{id}/ingest` – parse/index sources.
- `POST /projects/{id}/plan` – generate hierarchical writing plan.
- `POST /projects/{id}/draft` – run claim→paragraph→section pipeline.
- `POST /projects/{id}/review` – run QA and produce issues.
- `GET /projects/{id}/artifacts` – retrieve drafts, claim ledgers, QA reports.

## 10) Risks and mitigations

- **Hallucinated synthesis across weak evidence**  
  Mitigation: strict claim provenance and unsupported-sentence detector.
- **KG drift or entity duplication**  
  Mitigation: canonicalization pipeline + human-validated merge rules.
- **Cost/latency explosion on book-scale runs**  
  Mitigation: caching at claim/evidence level, incremental recompute, batch retrieval.
- **Inconsistent voice across chapters**  
  Mitigation: global style guide + chapter-level style critic + final harmonization pass.

## 11) Immediate next actions

1. Freeze the canonical schemas (`Claim`, `OutlineNode`, `DraftUnit`, `ReviewArtifact`).
2. Stand up a thin ingestion + retrieval spike using 20-50 sample sources.
3. Implement one end-to-end path for a single chapter:
   - outline node → claims → paragraphs → section draft → QA report.
4. Instrument quality metrics from day one to avoid opaque “it sounds good” evaluation.
