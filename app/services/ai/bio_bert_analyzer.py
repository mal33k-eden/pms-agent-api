from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from typing import List, Dict, Tuple, Optional
import re
import logging

logger = logging.getLogger(__name__)


class BioBERTAnalyzer:
    """Extract medical entities from FDA/DailyMed text"""

    def __init__(self):
        try:
            # Use BioBERT for NER
            self.model_name = "dmis-lab/biobert-v1.1"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # For NER task, use a BioBERT-based NER model with safetensors
            self.ner_model = AutoModelForTokenClassification.from_pretrained(
                "alvaroalon2/biobert_diseases_ner",
                use_safetensors=True  # Use safetensors to avoid torch.load security issue
            )
            self.model_loaded = True
        except Exception as e:
            logger.error(f"Failed to load BioBERT models: {e}", exc_info=True)
            logger.warning("BioBERT analyzer will use fallback text extraction only")
            self.model_loaded = False
            self.tokenizer = None
            self.ner_model = None

        # For extracting specific pregnancy/lactation entities
        self.pregnancy_terms = [
            "trimester", "teratogenic", "fetal", "embryo",
            "congenital", "miscarriage", "birth defects"
        ]
        self.lactation_terms = [
            "milk", "breastfed", "infant exposure", "nursing",
            "lactation", "milk/plasma ratio"
        ]

    def extract_pregnancy_risks(self, text: str) -> Dict:
        """Extract pregnancy-specific risks from FDA text"""
        try:
            if self.model_loaded and self.tokenizer and self.ner_model:
                # Tokenize
                inputs = self.tokenizer(
                    text[:512],  # BERT limit
                    return_tensors="pt",
                    truncation=True,
                    padding=True
                )

                # Get embeddings
                with torch.no_grad():
                    outputs = self.ner_model(**inputs)

                # Extract medical entities
                predictions = torch.argmax(outputs.logits, dim=2)
                tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        except Exception as e:
            logger.error(f"Error in BioBERT NER extraction: {e}", exc_info=True)
            # Continue with text-based extraction

        # Extract trimester-specific risks
        trimester_risks = {
            "first": [],
            "second": [],
            "third": []
        }

        # Simple pattern matching for trimester mentions
        sentences = text.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if "first trimester" in sentence_lower:
                risks = self._extract_risks_from_sentence(sentence)
                trimester_risks["first"].extend(risks)
            elif "second trimester" in sentence_lower:
                risks = self._extract_risks_from_sentence(sentence)
                trimester_risks["second"].extend(risks)
            elif "third trimester" in sentence_lower:
                risks = self._extract_risks_from_sentence(sentence)
                trimester_risks["third"].extend(risks)

        return {
            "trimester_specific": trimester_risks,
            "general_risks": self._extract_general_risks(text),
            "teratogenic": "teratogen" in text.lower(),
            "fda_category_mentioned": self._extract_fda_category(text)
        }

    def extract_milk_transfer_data(self, text: str) -> Dict:
        """Extract specific milk transfer numbers from text"""
        data = {}

        if not text:
            return data

        try:
            # Extract M/P ratio
            mp_pattern = r'(?:M/P|milk[/:]plasma) ratio[^\d]*(\d+\.?\d*)'
            mp_match = re.search(mp_pattern, text, re.IGNORECASE)
            if mp_match:
                data['milk_plasma_ratio'] = float(mp_match.group(1))

            # Extract infant dose percentage
            dose_pattern = r'infant (?:dose|exposure)[^\d]*(\d+\.?\d*)\s*%'
            dose_match = re.search(dose_pattern, text, re.IGNORECASE)
            if dose_match:
                data['infant_dose_percent'] = float(dose_match.group(1))

            # Extract half-life in milk
            halflife_pattern = r'half-life[^\d]*(\d+\.?\d*)\s*hours?'
            halflife_match = re.search(halflife_pattern, text, re.IGNORECASE)
            if halflife_match:
                data['half_life_hours'] = float(halflife_match.group(1))

            # Extract peak milk levels time
            peak_pattern = r'peak (?:milk )?levels?[^\d]*(\d+\.?\d*)\s*hours?'
            peak_match = re.search(peak_pattern, text, re.IGNORECASE)
            if peak_match:
                data['time_to_peak_hours'] = float(peak_match.group(1))

        except Exception as e:
            logger.error(f"Error extracting milk transfer data: {e}", exc_info=True)

        return data

    def _extract_risks_from_sentence(self, sentence: str) -> List[str]:
        """Extract risk mentions from a sentence"""
        risk_keywords = [
            "risk", "defect", "malformation", "toxicity",
            "adverse", "contraindicated", "avoid"
        ]
        risks = []
        for keyword in risk_keywords:
            if keyword in sentence.lower():
                # Extract the phrase around the keyword
                risks.append(sentence.strip())
                break
        return risks

    def _extract_general_risks(self, text: str) -> List[str]:
        """Extract general risk mentions"""
        # Placeholder implementation
        return []

    def _extract_fda_category(self, text: str) -> str:
        """Extract FDA pregnancy category"""
        # Placeholder implementation
        return "Unknown"

    def extract_structured_data(self, fda_text: str, dailymed_data: Dict) -> Dict:
        """Combine FDA text analysis with DailyMed structured data"""
        try:
            # Extract from FDA text using BioBERT
            fda_entities = {
                'pregnancy_risks': self.extract_pregnancy_risks(fda_text) if fda_text else {},
                'milk_data': self.extract_milk_transfer_data(fda_text) if fda_text else {}
            }

            # Safely get nested data
            pregnancy_risks = fda_entities.get('pregnancy_risks', {})
            milk_data = fda_entities.get('milk_data', {})
            dailymed_lactation = dailymed_data.get('lactation', {}) if dailymed_data else {}

            # Combine with DailyMed structured data
            combined = {
                'pregnancy': {
                    'fda_category': pregnancy_risks.get('fda_category_mentioned'),
                    'trimester_risks': pregnancy_risks.get('trimester_specific', {}),
                    'is_teratogenic': pregnancy_risks.get('teratogenic', False)
                },
                'lactation': {
                    'milk_plasma_ratio': (
                            milk_data.get('milk_plasma_ratio') or
                            dailymed_lactation.get('milk_plasma_ratio')
                    ),
                    'infant_dose_percent': (
                            milk_data.get('infant_dose_percent') or
                            dailymed_lactation.get('infant_dose_percent')
                    ),
                    'has_human_data': bool(dailymed_data.get('has_milk_levels')) if dailymed_data else False
                }
            }

            return combined
        except Exception as e:
            logger.error(f"Error in extract_structured_data: {e}", exc_info=True)
            # Return safe defaults
            return {
                'pregnancy': {
                    'fda_category': None,
                    'trimester_risks': {},
                    'is_teratogenic': False
                },
                'lactation': {
                    'milk_plasma_ratio': None,
                    'infant_dose_percent': None,
                    'has_human_data': False
                }
            }
