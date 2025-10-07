from app.services.fda_client import FDAClient
from app.services.daily_med_client import DailyMedClient
from app.services.pub_med_client import PubMedClient
from app.services.ai.bio_bert_analyzer import BioBERTAnalyzer
from app.services.ai.basic_analyzer import DrugSafetyAI
import asyncio
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class EnhancedDrugAnalyzer:
    """Orchestrates all data sources and analysis"""

    def __init__(self):
        try:
            self.fda = FDAClient()
            self.dailymed = DailyMedClient()
            self.pubmed = PubMedClient()
            self.biobert = BioBERTAnalyzer()
            self.ai = DrugSafetyAI()
        except Exception as e:
            logger.error(f"Error initializing EnhancedDrugAnalyzer: {e}", exc_info=True)
            raise

    async def analyze_drug_comprehensive(self, drug_name: str) -> Dict:
        """Fetch from all sources and analyze"""
        try:
            # Parallel fetch from all sources with error handling
            tasks = [
                self._safe_fda_fetch(drug_name),
                self._safe_dailymed_fetch(drug_name),
                self._safe_pubmed_fetch(drug_name)
            ]
            fda_data, dailymed_data, pubmed_data = await asyncio.gather(*tasks)

            # Use BioBERT to extract structured data
            biobert_extracted = {}
            try:
                if fda_data and fda_data.get('pregnancy_text'):
                    pregnancy_text = fda_data.get('pregnancy_text', '')
                    breastfeeding_text = fda_data.get('breastfeeding_text', '')
                    combined_text = f"{pregnancy_text}\n{breastfeeding_text}"
                    biobert_extracted = self.biobert.extract_structured_data(
                        combined_text,
                        dailymed_data or {}
                    )
            except Exception as e:
                logger.error(f"BioBERT extraction error: {e}", exc_info=True)
                biobert_extracted = {}

            # Have AI synthesize everything
            synthesis = await self.ai.synthesize_all_sources({
                'fda': fda_data,
                'dailymed': dailymed_data,
                'pubmed': pubmed_data,
                'biobert_extracted': biobert_extracted
            })

            # Calculate final confidence
            confidence = self._calculate_confidence(
                has_fda=bool(fda_data),
                has_dailymed=bool(dailymed_data),
                study_count=pubmed_data.get('total_studies', 0) if pubmed_data else 0,
                has_meta_analysis=pubmed_data.get('has_meta_analysis', False) if pubmed_data else False
            )

            return {
                'drug_name': drug_name,
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
                'safety_assessment': synthesis,
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {drug_name}: {e}", exc_info=True)
            raise

    async def _safe_fda_fetch(self, drug_name: str):
        """Safely fetch FDA data with error handling"""
        try:
            return await self.fda.search_drug_label(drug_name)
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

    async def _safe_pubmed_fetch(self, drug_name: str):
        """Safely fetch PubMed data with error handling"""
        try:
            return await self.pubmed.search_pregnancy_breastfeeding_studies(drug_name)
        except Exception as e:
            logger.error(f"PubMed fetch error for {drug_name}: {e}", exc_info=True)
            return {
                'total_studies': 0,
                'recent_studies': 0,
                'has_rct': False,
                'has_meta_analysis': False
            }

    def _calculate_confidence(self, has_fda, has_dailymed, study_count, has_meta_analysis):
        """Calculate confidence score based on data availability"""
        score = 0.0

        # Base scores for data sources
        if has_fda:
            score += 0.3
        if has_dailymed:
            score += 0.2

        # Research quality
        if study_count > 100:
            score += 0.3
        elif study_count > 50:
            score += 0.2
        elif study_count > 10:
            score += 0.1

        # High-quality research bonus
        if has_meta_analysis:
            score += 0.2

        return min(score, 1.0)
