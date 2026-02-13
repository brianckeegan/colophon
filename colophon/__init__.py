"""Colophon package."""

from .coordination import (
    BookCoordinationAgent,
    ChapterCoordinationAgent,
    MessageBus,
    ParagraphCoordinationAgent,
    SectionCoordinationAgent,
)
from .agents import OutlineExpanderAgent
from .kg_update import KGUpdateConfig, KnowledgeGraphGeneratorUpdater
from .llm import LLMConfig, create_llm_client
from .functional_forms import SoftValidationFinding, SoftValidationResult, run_soft_validation, select_functional_form
from .genre_ontology import build_genre_ontology_context
from .writing_ontology import (
    WritingOntologyValidationFinding,
    WritingOntologyValidationResult,
    build_writing_ontology_context,
    run_writing_ontology_validation,
)
from .note_import import NotesImportConfig, NotesImportResult, NotesKnowledgeGraphImporter
from .models import (
    Chapter,
    Claim,
    CoordinationMessage,
    Figure,
    GapRequest,
    Manuscript,
    Paragraph,
    RecommendationProposal,
    Section,
    Source,
)
from .ontology import (
    normalize_functional_forms_catalog,
    normalize_genre_ontology,
    normalize_writing_companion_ontology,
)
from .pipeline import ColophonPipeline, PipelineConfig
from .recommendations import (
    OpenAlexSearchClient,
    PaperRecommendationWorkflow,
    RecommendationConfig,
    SemanticScholarSearchClient,
)
from .vectors import EmbeddingConfig, InMemoryVectorDB, create_embedding_client

__all__ = [
    "BookCoordinationAgent",
    "Chapter",
    "ChapterCoordinationAgent",
    "Claim",
    "ColophonPipeline",
    "CoordinationMessage",
    "EmbeddingConfig",
    "Figure",
    "GapRequest",
    "InMemoryVectorDB",
    "KGUpdateConfig",
    "KnowledgeGraphGeneratorUpdater",
    "LLMConfig",
    "MessageBus",
    "Manuscript",
    "NotesImportConfig",
    "NotesImportResult",
    "NotesKnowledgeGraphImporter",
    "OpenAlexSearchClient",
    "OutlineExpanderAgent",
    "PaperRecommendationWorkflow",
    "ParagraphCoordinationAgent",
    "Paragraph",
    "PipelineConfig",
    "RecommendationConfig",
    "RecommendationProposal",
    "SemanticScholarSearchClient",
    "Section",
    "SectionCoordinationAgent",
    "SoftValidationFinding",
    "SoftValidationResult",
    "Source",
    "create_embedding_client",
    "build_writing_ontology_context",
    "normalize_functional_forms_catalog",
    "normalize_genre_ontology",
    "normalize_writing_companion_ontology",
    "build_genre_ontology_context",
    "create_llm_client",
    "run_writing_ontology_validation",
    "run_soft_validation",
    "select_functional_form",
    "WritingOntologyValidationFinding",
    "WritingOntologyValidationResult",
]
