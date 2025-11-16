from typing import Dict, Optional

from app.services.ai.base_analyzer_class import DrugAnalysisResult


class SynthesisOrchestrator:
    """
    Handles synthesis of drug safety data from multiple sources.

    Orchestrates the Claude LLM-based synthesis of FDA, DailyMed, PubMed,
    and BioBERT data into comprehensive safety assessments.
    """

    def __init__(self, llm_client):
        """
        Initialize the synthesis orchestrator.

        Args:
            llm_client:  LLM client for synthesis (e.g claude or openai gpt-3.5-turbo)))
        """
        self.client = llm_client

    async def synthesize_all_sources(self, all_data: Dict) -> DrugAnalysisResult:
        """
        Synthesize drug safety data from multiple sources using Claude LLM.

        Args:
            all_data: Dictionary containing:
                - 'fda': FDA label data
                - 'dailymed': DailyMed SPL data
                - 'pubmed': PubMed research data
                - 'biobert_extracted': BioBERT entities
                - 'medical_context': User's medical situation (optional)

        Returns:
            DrugAnalysisResult with synthesized assessment
        """
        import json
        import logging
        logger = logging.getLogger(__name__)

        fda_data = all_data.get('fda') or {}
        dailymed_data = all_data.get('dailymed') or {}
        pubmed_data = all_data.get('pubmed') or {}
        biobert_extracted = all_data.get('biobert_extracted') or {}

        # Get drug name from available data
        drug_name = fda_data.get('generic_names', ['Unknown'])[0] if fda_data.get('generic_names') else 'Unknown'

        # Build comprehensive context for Claude
        context = self._build_synthesis_context(fda_data, dailymed_data, pubmed_data, biobert_extracted)

        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content="""You are a pharmacist synthesizing drug safety data from multiple authoritative sources.
            Analyze all available data and provide a comprehensive safety assessment.
            RESPOND WITH ONLY VALID JSON (no markdown, no extra text). JSON keys required: pregnancy_safety, breastfeeding_safety, warnings, summary, evidence_quality.
            pregnancy_safety and breastfeeding_safety must be 'safe', 'caution', or 'avoid'.
            warnings must be an array of strings (max 5).
            evidence_quality must be 'high', 'moderate', or 'low'."""),
            HumanMessage(content=context)
        ]

        try:
            response = self.client.invoke(messages)

            # Debug logging
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response: {response}")

            # Extract content from response
            content = response.content if hasattr(response, 'content') else str(response)

            if not content or not content.strip():
                logger.warning(f"Empty response content from LLM")
                # Return safe default
                return {
                    "drug_name": drug_name,
                    "pregnancy_safety": "unknown",
                    "breastfeeding_safety": "unknown",
                    "warnings": ["Consult healthcare provider"],
                    "summary": "Unable to analyze drug. Please consult your healthcare provider.",
                    "confidence": 0.3,
                    "sources_used": []
                }

            # Extract JSON from response, handling markdown code blocks
            json_str = content.strip()

            # Remove markdown code blocks if present
            if json_str.startswith("```"):
                # Remove opening markdown block (```json or ```)
                json_str = json_str.split("```", 1)[1]
                # Remove closing markdown block
                if "```" in json_str:
                    json_str = json_str.rsplit("```", 1)[0]
                json_str = json_str.strip()

                # Remove language identifier if present (e.g., "json")
                if json_str.startswith("json"):
                    json_str = json_str[4:].lstrip()

            synthesis = json.loads(json_str)

            # Count available sources
            sources_used = []
            if fda_data:
                sources_used.append('fda')
            if dailymed_data:
                sources_used.append('dailymed')
            if pubmed_data:
                sources_used.append('pubmed')
            if biobert_extracted:
                sources_used.append('biobert')

            # Calculate confidence based on available sources
            confidence = 0.5
            if len(sources_used) >= 3:
                confidence = 0.8
            elif len(sources_used) >= 2:
                confidence = 0.7

            result: DrugAnalysisResult = {
                "drug_name": drug_name,
                "pregnancy_safety": synthesis.get("pregnancy_safety", "unknown"),
                "breastfeeding_safety": synthesis.get("breastfeeding_safety", "unknown"),
                "warnings": synthesis.get("warnings", ["Consult healthcare provider"]),
                "summary": synthesis.get("summary", "Consult your healthcare provider."),
                "confidence": confidence,
                "sources_used": sources_used
            }
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.error(f"Raw content received: {json_str if 'json_str' in locals() else 'N/A'}")
            logger.error(f"Content length: {len(json_str) if 'json_str' in locals() else 'N/A'}")
            # Return safe default when JSON parsing fails
            return {
                "drug_name": drug_name,
                "pregnancy_safety": "unknown",
                "breastfeeding_safety": "unknown",
                "warnings": ["Consult healthcare provider"],
                "summary": "Unable to analyze drug. Please consult your healthcare provider.",
                "confidence": 0.3,
                "sources_used": []
            }
        except Exception as e:
            logger.error(f"Synthesis error: {e}", exc_info=True)
            raise

    @staticmethod
    def _build_synthesis_context(fda_data: Dict, dailymed_data: Dict, pubmed_data: Dict,
                                 biobert_data: Dict) -> str:
        """
        Build comprehensive context from all data sources for Claude LLM.

        Args:
            fda_data: FDA label data
            dailymed_data: DailyMed SPL data
            pubmed_data: PubMed research data
            biobert_data: BioBERT extracted entities

        Returns:
            Formatted context string for Claude LLM
        """
        context_parts = []

        # FDA data
        if fda_data:
            drug_name = fda_data.get('generic_names', ['Unknown'])[0] if fda_data.get('generic_names') else 'Unknown'
            context_parts.append(f"Drug: {drug_name}")

            if fda_data.get('pregnancy_text'):
                context_parts.append(f"\nFDA Pregnancy Data:\n{fda_data['pregnancy_text'][:800]}")

            if fda_data.get('breastfeeding_text'):
                context_parts.append(f"\nFDA Breastfeeding Data:\n{fda_data['breastfeeding_text'][:800]}")

        # DailyMed data
        if dailymed_data and dailymed_data.get('spl_data'):
            context_parts.append(f"\nDailyMed SPL Data:\n{str(dailymed_data['spl_data'])[:500]}")

        # PubMed research data
        if pubmed_data:
            context_parts.append(f"\nResearch Evidence:")
            context_parts.append(f"- Total studies: {pubmed_data.get('total_studies', 0)}")
            context_parts.append(f"- Recent studies (last 5 years): {pubmed_data.get('recent_studies', 0)}")
            context_parts.append(f"- Has randomized controlled trials: {pubmed_data.get('has_rct', False)}")
            context_parts.append(f"- Has meta-analysis: {pubmed_data.get('has_meta_analysis', False)}")

            if pubmed_data.get('key_findings'):
                context_parts.append(f"\nKey Research Findings:\n{pubmed_data['key_findings'][:500]}")

        # BioBERT extracted entities
        if biobert_data:
            context_parts.append(f"\nExtracted Medical Entities:\n{str(biobert_data)[:300]}")

        context_parts.append(
            """
                Based on all available data, provide:
                1. pregnancy_safety: 'safe', 'caution', or 'avoid'
                2. breastfeeding_safety: 'safe', 'caution', or 'avoid'
                3. warnings: Array of key warnings (max 5)
                4. summary: Patient-friendly 2-3 sentence summary
                5. evidence_quality: 'high', 'moderate', or 'low'
            """
        )

        return "\n".join(context_parts)


class AnalysisUtility:
    """
    Shared utility methods for drug analysis.

    Contains common analysis methods that both basic and enhanced analyzers
    can use to avoid code duplication. These include:
    - Synthesis of multiple data sources
    - Contextualization of safety assessments
    - Confidence calculation
    - FDA data analysis
    """

    @staticmethod
    def analyze_fda_data_utility(
            drug_name: str,
            fda_data: Dict,
            pregnancy_safety: str,
            breastfeeding_safety: str,
            warnings: list,
            summary: str
    ) -> DrugAnalysisResult:
        """
        Build a standardized FDA analysis result.

        Args:
            drug_name: Name of the drug
            fda_data: FDA label data
            pregnancy_safety: Assessed pregnancy safety
            breastfeeding_safety: Assessed breastfeeding safety
            warnings: Extracted warnings
            summary: Analysis summary

        Returns:
            DrugAnalysisResult with FDA analysis
        """
        return {
            "drug_name": drug_name,
            "pregnancy_safety": pregnancy_safety,
            "breastfeeding_safety": breastfeeding_safety,
            "warnings": warnings,
            "summary": summary,
            "confidence": 0.6,
            "sources_used": ["fda"]
        }

    @staticmethod
    def synthesize_all_sources_utility(
            drug_name: str,
            fda_data: Dict,
            dailymed_data: Dict,
            pubmed_data: Dict,
            biobert_data: Dict,
            synthesis: Dict
    ) -> DrugAnalysisResult:
        """
        Build a standardized synthesis result from multiple sources.

        Args:
            drug_name: Name of the drug
            fda_data: FDA label data
            dailymed_data: DailyMed data
            pubmed_data: PubMed research data
            biobert_data: BioBERT extracted data
            synthesis: AI synthesis result

        Returns:
            DrugAnalysisResult with synthesized analysis
        """
        # Count available sources
        sources_used = []
        if fda_data:
            sources_used.append('fda')
        if dailymed_data:
            sources_used.append('dailymed')
        if pubmed_data:
            sources_used.append('pubmed')
        if biobert_data:
            sources_used.append('biobert')

        # Calculate confidence based on available sources
        confidence = 0.5
        if len(sources_used) >= 3:
            confidence = 0.8
        elif len(sources_used) >= 2:
            confidence = 0.7

        return {
            "drug_name": drug_name,
            "pregnancy_safety": synthesis.get("pregnancy_safety", "unknown"),
            "breastfeeding_safety": synthesis.get("breastfeeding_safety", "unknown"),
            "warnings": synthesis.get("warnings", ["Consult healthcare provider"]),
            "summary": synthesis.get("summary", "Consult your healthcare provider."),
            "confidence": confidence,
            "sources_used": sources_used
        }

    @staticmethod
    def contextualize_assessment(
            safety_assessment: Dict,
            is_pregnant: bool = None,
            is_breastfeeding: bool = None,
            trimester: str = None
    ) -> Dict:
        """
        Contextualize the safety assessment based on medical situation.

        Filters and emphasizes relevant safety information for the user's specific
        medical context (pregnancy vs breastfeeding, trimester, etc.).

        Args:
            safety_assessment: Raw safety assessment from AI synthesis
            is_pregnant: Whether user is pregnant
            is_breastfeeding: Whether user is breastfeeding
            trimester: Pregnancy trimester if applicable

        Returns:
            Contextualized assessment dictionary
        """
        contextualized = dict(safety_assessment)

        # Add context-specific warnings and recommendations
        contextualized['context'] = {
            'is_pregnant': is_pregnant,
            'is_breastfeeding': is_breastfeeding,
            'pregnancy_trimester': trimester
        }

        # Emphasize trimester-specific information if available
        if is_pregnant and trimester:
            contextualized['trimester_specific'] = {
                'trimester': trimester.lower() if trimester else None,
                'note': f"Information specific to {trimester.lower() if trimester else 'unknown'} trimester extracted from available data"
            }

        # Add breastfeeding-specific recommendations
        if is_breastfeeding:
            contextualized['breastfeeding_specific'] = {
                'note': "Information specific to breastfeeding mothers extracted from available data",
                'infant_risk': safety_assessment.get('infant_risk', 'Information not available'),
                'milk_transfer': safety_assessment.get('milk_transfer', 'Information not available')
            }

        # Prioritize warnings based on context
        contextualized['relevant_warnings'] = AnalysisUtility._prioritize_warnings(
            safety_assessment,
            is_pregnant,
            is_breastfeeding
        )

        return contextualized

    @staticmethod
    def _prioritize_warnings(
            assessment: Dict,
            is_pregnant: bool = None,
            is_breastfeeding: bool = None
    ) -> list:
        """
        Prioritize warnings based on medical context.

        Returns warnings most relevant to the user's situation first.
        """
        warnings = assessment.get('warnings', [])
        if not warnings:
            return []

        prioritized = []

        for warning in warnings:
            if isinstance(warning, dict):
                context = warning.get('context', [])

                # Prioritize if relevant to user's context
                if (is_pregnant and 'pregnancy' in context) or \
                        (is_breastfeeding and 'breastfeeding' in context):
                    prioritized.insert(0, warning)  # Add to front
                else:
                    prioritized.append(warning)  # Add to end
            else:
                prioritized.append(warning)

        return prioritized

    @staticmethod
    def calculate_confidence(
            has_fda: bool,
            has_dailymed: bool,
            study_count: int,
            has_meta_analysis: bool,
            medical_context: Dict = None
    ) -> float:
        """
        Calculate confidence score based on data availability and medical context.

        Args:
            has_fda: Whether FDA data was found
            has_dailymed: Whether DailyMed data was found
            study_count: Number of relevant studies found
            has_meta_analysis: Whether meta-analysis studies exist
            medical_context: Medical context for context-specific adjustments

        Returns:
            Confidence score between 0.0 and 1.0
        """
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

        # Context-specific adjustments
        if medical_context:
            # Reduce confidence if insufficient pregnancy-specific data
            if medical_context.get('needs_pregnancy_info') and not has_fda:
                score -= 0.1
            # Reduce confidence if insufficient breastfeeding-specific data
            if medical_context.get('needs_breastfeeding_info') and not has_dailymed:
                score -= 0.05

        return max(0.0, min(score, 1.0))

    @staticmethod
    def build_fda_analysis_result(
            drug_name: str,
            workflow_result: Dict
    ) -> DrugAnalysisResult:
        """
        Build FDA analysis result from LangGraph workflow output.

        Converts the raw workflow result into a standardized DrugAnalysisResult.
        Handles missing or incomplete workflow results with sensible defaults.

        Args:
            drug_name: Name of the drug being analyzed
            workflow_result: Output dictionary from LangGraph workflow

        Returns:
            DrugAnalysisResult with standardized FDA analysis
        """
        return {
            "drug_name": drug_name,
            "pregnancy_safety": workflow_result.get("pregnancy_safety", "unknown"),
            "breastfeeding_safety": workflow_result.get("breastfeeding_safety", "unknown"),
            "warnings": workflow_result.get("warnings", ["Consult healthcare provider"]),
            "summary": workflow_result.get("summary",
                                           f"Unable to analyze {drug_name}. Please consult your healthcare provider."),
            "confidence": 0.6,  # Moderate confidence for FDA-only analysis
            "sources_used": ["fda"]
        }

    @staticmethod
    def run_fda_workflow(
            drug_name: str,
            fda_data: Dict,
            client,  # ChatAnthropic client
            workflow  # Compiled StateGraph workflow
    ) -> DrugAnalysisResult:
        """
        Execute FDA analysis using LangGraph workflow.

        Runs the compiled workflow with FDA data and returns standardized results.
        This is the shared implementation used by both basic and enhanced analyzers.

        Args:
            drug_name: Name of the drug to analyze
            fda_data: FDA label data
            client: ChatAnthropic LLM client
            workflow: Compiled LangGraph StateGraph workflow

        Returns:
            DrugAnalysisResult with FDA analysis from workflow

        Raises:
            Exception: Propagates exceptions from workflow execution
        """
        # Import here to avoid circular imports
        from typing_extensions import TypedDict

        # Define inline the state structure
        class DrugAnalysisState(TypedDict, total=False):
            drug_name: str
            fda_data: Dict
            pregnancy_safety: str
            breastfeeding_safety: str
            warnings: list
            summary: str
            error: Optional[str]

        # Initialize workflow state
        initial_state: DrugAnalysisState = {
            "drug_name": drug_name,
            "fda_data": fda_data,
            "pregnancy_safety": "",
            "breastfeeding_safety": "",
            "warnings": [],
            "summary": "",
            "error": None
        }

        # Run workflow
        result = workflow.invoke(initial_state)

        # Check if result is None or empty
        if not result:
            raise ValueError("Workflow returned None or empty result")

        # Build and return standardized result
        return AnalysisUtility.build_fda_analysis_result(drug_name, result)
