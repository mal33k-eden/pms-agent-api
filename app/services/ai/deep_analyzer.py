import os

from langchain_anthropic import ChatAnthropic

from app.services.ai.base_analyzer_class import BaseDrugAnalyzer, EnhancedAnalysisResult
from app.services.ai.utils import AnalysisUtility, SynthesisOrchestrator
from app.services.fda_client import FDAClient
from app.services.daily_med_client import DailyMedClient
from app.services.pub_med_client import PubMedClient
from app.services.ai.bio_bert_analyzer import BioBERTAnalyzer
from app.services.ai.basic_analyzer import DrugSafetyAI
import asyncio
from typing import Dict, TypedDict, Optional
import logging

logger = logging.getLogger(__name__)


class DrugAnalysisState(TypedDict, total=False):
    """State for FDA drug analysis workflow"""
    drug_name: str
    fda_data: Dict
    pregnancy_safety: str
    breastfeeding_safety: str
    warnings: list
    summary: str
    error: Optional[str]


class EnhancedDrugAnalyzer(BaseDrugAnalyzer):
    """Orchestrates all data sources and analysis"""

    def __init__(self):
        try:
            super().__init__()
            # Initialize data source clients
            self.fda = FDAClient()
            self.dailymed = DailyMedClient()
            self.biobert = BioBERTAnalyzer()

            # Keep reference to basic analyzer for synthesis (still used)
            self.ai = DrugSafetyAI()

            # Initialize LLM client for direct FDA analysis (decoupled from basic analyzer)
            self.client = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=0.1,
                max_tokens=1024
            )

        except Exception as e:
            logger.error(f"Error initializing EnhancedDrugAnalyzer: {e}", exc_info=True)
            raise

    async def _safe_fda_fetch(self, drug_name: str):
        """Safely fetch FDA data with error handling"""
        try:
            self.fda_data = await self.fda.search_drug_label(drug_name)
            return self.fda_data
        except Exception as e:
            logger.error(f"FDA fetch error for {drug_name}: {e}", exc_info=True)
            return None

    async def _safe_dailymed_fetch(self, drug_name: str):
        """Safely fetch DailyMed data with error handling"""
        try:
            return await self.dailymed.search_spl(drug_name)
        except Exception as e:
            logger.error(f"DailyMed fetch error for {drug_name}: {e}", exc_info=True)
            return None

    async def _safe_pubmed_fetch(self, drug_name: str, medical_context: Dict = None):
        """
        Safely fetch PubMed data with error handling, filtering by medical context.

        Args:
            drug_name: Name of the drug
            medical_context: Dictionary containing is_pregnant, is_breastfeeding, trimester

        Returns:
            Dictionary with PubMed study data
        """
        try:
            result = await self.pubmed.search_pregnancy_breastfeeding_studies(drug_name)

            # Filter results based on medical context if provided
            if medical_context and result:
                if medical_context.get('is_pregnant'):
                    # Emphasize pregnancy-related studies
                    result['context'] = 'pregnancy'
                elif medical_context.get('is_breastfeeding'):
                    # Emphasize breastfeeding-related studies
                    result['context'] = 'breastfeeding'
                else:
                    result['context'] = 'general'

            return result
        except Exception as e:
            logger.error(f"PubMed fetch error for {drug_name}: {e}", exc_info=True)
            return {
                'total_studies': 0,
                'recent_studies': 0,
                'has_rct': False,
                'has_meta_analysis': False,
                'context': 'error'
            }

    async def fetch_and_analyze(
            self,
            drug_name: str,
            is_pregnant=None,
            is_breastfeeding=None,
            trimester=None
    ) -> EnhancedAnalysisResult:
        """
        Fetch from all sources and analyze drug safety with contextual parameters.

        Args:
            drug_name: Name of the drug to analyze
            is_pregnant: Whether the user is pregnant
            is_breastfeeding: Whether the user is breastfeeding
            trimester: Pregnancy trimester ("first", "second", "third") if applicable

        Returns:
            Comprehensive analysis dictionary with safety assessment
        """
        try:
            logger.info(
                f"Starting comprehensive analysis for {drug_name} - "
                f"Pregnant: {is_pregnant}, Breastfeeding: {is_breastfeeding}, "
                f"Trimester: {trimester}"
            )

            # Determine medical context for analysis
            medical_context = {
                'is_pregnant': is_pregnant,
                'is_breastfeeding': is_breastfeeding,
                'trimester': trimester,
                'needs_pregnancy_info': is_pregnant or trimester is not None,
                'needs_breastfeeding_info': is_breastfeeding
            }

            # Parallel fetch from all sources with error handling
            tasks = [
                self._safe_fda_fetch(drug_name),
                self._safe_dailymed_fetch(drug_name),
                self._safe_pubmed_fetch(drug_name, medical_context)
            ]
            fda_data, dailymed_data, pubmed_data = await asyncio.gather(*tasks)

            # Use BioBERT to extract structured data
            biobert_extracted = {}
            try:
                if fda_data and (fda_data.get('pregnancy_text') or fda_data.get('breastfeeding_text')):
                    pregnancy_text = fda_data.get('pregnancy_text', '')
                    breastfeeding_text = fda_data.get('breastfeeding_text', '')
                    combined_text = f"{pregnancy_text}\n{breastfeeding_text}"
                    biobert_extracted = self.biobert.extract_structured_data(
                        combined_text,
                        dailymed_data or {}
                    )
                    logger.debug(f"BioBERT extraction successful for {drug_name}")
            except Exception as e:
                logger.error(f"BioBERT extraction error for {drug_name}: {e}", exc_info=True)
                biobert_extracted = {}

            # Have orchestrator synthesize everything with medical context
            orchestrator = SynthesisOrchestrator(self.ai.client)
            synthesis = await orchestrator.synthesize_all_sources({
                'fda': fda_data,
                'dailymed': dailymed_data,
                'pubmed': pubmed_data,
                'biobert_extracted': biobert_extracted,
                'medical_context': medical_context
            })

            # Calculate final confidence
            confidence = AnalysisUtility.calculate_confidence(
                has_fda=bool(fda_data),
                has_dailymed=bool(dailymed_data),
                study_count=pubmed_data.get('total_studies', 0) if pubmed_data else 0,
                has_meta_analysis=pubmed_data.get('has_meta_analysis', False) if pubmed_data else False,
                medical_context=medical_context
            )

            # Filter and contextualize safety information based on medical situation
            contextualized_assessment = AnalysisUtility.contextualize_assessment(
                synthesis,
                is_pregnant,
                is_breastfeeding,
                trimester
            )

            # Build result with both base and enhanced fields
            result: EnhancedAnalysisResult = {
                # Base DrugAnalysisResult fields
                'drug_name': drug_name,
                'pregnancy_safety': synthesis.get('pregnancy_safety', 'unknown'),
                'breastfeeding_safety': synthesis.get('breastfeeding_safety', 'unknown'),
                'warnings': synthesis.get('warnings', []),
                'summary': synthesis.get('summary', 'Consult your healthcare provider.'),
                'confidence': confidence,
                'sources_used': [s for s in ['fda', 'dailymed', 'pubmed', 'biobert']
                                 if
                                 s == 'fda' and fda_data or s == 'dailymed' and dailymed_data or s == 'pubmed' and pubmed_data or s == 'biobert' and biobert_extracted],
                # Enhanced fields
                'medical_context': {
                    'is_pregnant': is_pregnant,
                    'is_breastfeeding': is_breastfeeding,
                    'pregnancy_trimester': trimester
                },
                'sources_available': {
                    'fda': bool(fda_data),
                    'dailymed': bool(dailymed_data),
                    'pubmed_studies': pubmed_data.get('total_studies', 0) if pubmed_data else 0
                },
                'extracted_data': biobert_extracted,
                'research_quality': {
                    'total_studies': pubmed_data.get('total_studies', 0) if pubmed_data else 0,
                    'recent_studies': pubmed_data.get('recent_studies', 0) if pubmed_data else 0,
                    'has_high_quality': (pubmed_data.get('has_meta_analysis', False) or
                                         pubmed_data.get('has_rct', False)) if pubmed_data else False
                },
                'safety_assessment': contextualized_assessment,
            }

            logger.info(
                f"Comprehensive analysis complete for {drug_name} - "
                f"Confidence: {confidence:.2%}, Sources: FDA={bool(fda_data)}, "
                f"DailyMed={bool(dailymed_data)}, Studies={pubmed_data.get('total_studies', 0) if pubmed_data else 0}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {drug_name}: {e}", exc_info=True)
            raise
