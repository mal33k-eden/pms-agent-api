import logging
from abc import ABC, abstractmethod
from typing import TypedDict, Dict, Optional

from app.services.pub_med_client import PubMedClient

log = logging.getLogger(__name__)


class DrugAnalysisResult(TypedDict, total=False):
    """
    Base drug analysis result structure returned by all analyzers.

    This TypedDict defines the common fields that all analyzer implementations
    must return. Enhanced analyzers may add additional fields.
    """

    # Basic identification
    drug_name: str

    # Core safety assessments
    pregnancy_safety: str  # 'safe', 'caution', 'avoid', 'unknown'
    breastfeeding_safety: str  # 'safe', 'caution', 'avoid', 'unknown'

    # Warnings and recommendations
    warnings: list[str]
    summary: str

    # Analysis metadata
    confidence: float  # 0.0 to 1.0
    sources_used: list[str]  # e.g., ['fda', 'dailymed', 'pubmed']


class EnhancedAnalysisResult(DrugAnalysisResult):
    """
    Extended result structure for enhanced analysis.

    Contains all base fields plus additional detailed information
    from multiple sources.
    """

    # Medical context
    medical_context: Dict[str, any]  # {'is_pregnant': bool, 'is_breastfeeding': bool, 'trimester': str}

    # Enhanced data
    sources_available: Dict[str, any]  # FDA, DailyMed, PubMed availability and study counts
    extracted_data: Dict[str, any]  # BioBERT extracted entities
    research_quality: Dict[str, any]  # Study counts, recent studies, quality indicators
    safety_assessment: Dict[str, any]  # Detailed contextualized safety information

    # Context-specific details
    trimester_specific: Optional[Dict[str, str]]  # Pregnancy trimester information
    breastfeeding_specific: Optional[Dict[str, str]]  # Lactation information
    relevant_warnings: list[Dict]  # Warnings prioritized by context


class BaseDrugAnalyzer(ABC):
    """
    Abstract base class for drug safety analyzers.

    All analyzer implementations must inherit from this class and implement
    the required abstract methods. This ensures consistent interfaces across
    basic and enhanced analyzers.

    Subclasses should:
    - Implement all abstract methods
    - Return DrugAnalysisResult from basic analysis
    - Return EnhancedAnalysisResult from enhanced analysis
    - Maintain consistent field naming and types
    """

    def __init__(self):
        self.pubmed = PubMedClient()
        self.fda_data = None

    @abstractmethod
    async def fetch_and_analyze(
            self,
            drug_name: str,
            is_pregnant: bool = None,
            is_breastfeeding: bool = None,
            trimester: str = None
    ) -> DrugAnalysisResult:
        """
        Fetch drug data and perform analysis in a single operation.

        This is the main entry point for drug analysis. It handles fetching
        the necessary data and performing the analysis.

        Args:
            drug_name: Name of the drug to analyze

        Returns:
            DrugAnalysisResult with safety assessment

        Raises:
            Exception: If analysis fails, should return fallback response
        """
        pass

    @staticmethod
    def _create_fallback_response(drug_name: str) -> DrugAnalysisResult:
        """
        Create a safe fallback response when analysis fails.

        All analyzers should use this method for consistent error handling.

        Args:
            drug_name: Name of the drug being analyzed

        Returns:
            DrugAnalysisResult with safe default values
        """
        return {
            "drug_name": drug_name,
            "pregnancy_safety": "unknown",
            "breastfeeding_safety": "unknown",
            "warnings": ["Consult healthcare provider"],
            "summary": f"Unable to analyze {drug_name}. Please consult your healthcare provider.",
            "confidence": 0.0,
            "sources_used": [],
        }

    def validate_analysis_result(self, result: DrugAnalysisResult) -> bool:
        """
        Validate that an analysis result has all required fields.

        Args:
            result: The analysis result to validate

        Returns:
            True if all required fields are present and valid
        """
        required_fields = {
            "drug_name": str,
            "pregnancy_safety": str,
            "breastfeeding_safety": str,
            "warnings": list,
            "summary": str,
            "confidence": (int, float),
            "sources_used": list,
        }

        for field, expected_type in required_fields.items():
            if field not in result:
                return False
            if isinstance(expected_type, tuple):
                if not isinstance(result[field], expected_type):
                    return False
            else:
                if not isinstance(result[field], expected_type):
                    return False

        # Validate confidence is between 0.0 and 1.0
        if not 0.0 <= result["confidence"] <= 1.0:
            return False

        # Validate safety values
        valid_safety_values = {"safe", "caution", "avoid", "unknown"}
        if result["pregnancy_safety"] not in valid_safety_values:
            return False
        if result["breastfeeding_safety"] not in valid_safety_values:
            return False

        return True

    def normalize_safety_value(self, value: str) -> str:
        """
        Normalize safety values to standard format.

        Args:
            value: The safety value to normalize

        Returns:
            Normalized safety value: 'safe', 'caution', 'avoid', or 'unknown'
        """
        if not value:
            return "unknown"

        normalized = value.strip().lower()

        # Handle variations
        if normalized in ("safe", "ok", "yes", "recommended"):
            return "safe"
        elif normalized in ("caution", "warn", "maybe", "probably safe"):
            return "caution"
        elif normalized in ("avoid", "no", "contraindicated"):
            return "avoid"
        else:
            return "unknown"

    async def get_pubmed_count(self, drug_name: str) -> int:
        """Get the number of relevant PubMed studies for a drug."""
        try:
            result = await self.pubmed.search_pregnancy_breastfeeding_studies(drug_name)
            return result.get('total_studies', 0) if result else 0
        except Exception as e:
            log.error(f"PubMed count fetch failed for {drug_name}: {e}", exc_info=True)
            return 0
