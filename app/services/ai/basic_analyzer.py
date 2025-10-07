import json
import os
from typing import Dict, Optional, Annotated
import logging
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class DrugAnalysisState(TypedDict):
    """State for drug analysis workflow"""
    drug_name: str
    fda_data: Dict
    pregnancy_safety: str
    breastfeeding_safety: str
    warnings: list
    summary: str
    error: Optional[str]


class DrugSafetyAI:
    def __init__(self):
        self.client = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.1,
            max_tokens=1024
        )
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """Build the LangGraph workflow for drug analysis"""
        workflow = StateGraph(DrugAnalysisState)

        # Add nodes
        workflow.add_node("validate_data", self._validate_data)
        workflow.add_node("analyze_pregnancy", self._analyze_pregnancy)
        workflow.add_node("analyze_breastfeeding", self._analyze_breastfeeding)
        workflow.add_node("extract_warnings", self._extract_warnings)
        workflow.add_node("generate_summary", self._generate_summary)
        workflow.add_node("handle_error", self._handle_error)

        # Define edges
        workflow.set_entry_point("validate_data")
        workflow.add_conditional_edges(
            "validate_data",
            lambda state: "error" if state.get("error") else "continue",
            {
                "error": "handle_error",
                "continue": "analyze_pregnancy"
            }
        )
        workflow.add_edge("analyze_pregnancy", "analyze_breastfeeding")
        workflow.add_edge("analyze_breastfeeding", "extract_warnings")
        workflow.add_edge("extract_warnings", "generate_summary")
        workflow.add_edge("generate_summary", END)
        workflow.add_edge("handle_error", END)

        return workflow.compile()

    def _validate_data(self, state: Dict) -> Dict:
        """Validate input data"""
        updates = {}
        if not state.get("drug_name"):
            updates["error"] = "Drug name is required"
        elif not state.get("fda_data"):
            updates["error"] = "FDA data is required"
        return updates

    def _analyze_pregnancy(self, state: Dict) -> Dict:
        """Analyze pregnancy safety"""
        drug_name = state.get("drug_name", "")
        fda_data = state.get("fda_data") or {}

        pregnancy_text = fda_data.get('pregnancy_text') or 'No data'
        pregnancy_category = fda_data.get('pregnancy_category') or 'Unknown'

        messages = [
            SystemMessage(content="""You are a pharmacist analyzing drug safety for pregnancy.
            Respond with only one word: 'safe', 'caution', or 'avoid'."""),
            HumanMessage(content=f"""
            Analyze {drug_name} pregnancy safety:
            Category: {pregnancy_category}
            Information: {pregnancy_text[:500] if pregnancy_text else 'No data'}
            """)
        ]

        try:
            response = self.client.invoke(messages)
            return {"pregnancy_safety": response.content.strip().lower()}
        except Exception as e:
            logger.error(f"Pregnancy analysis error: {e}")
            return {"pregnancy_safety": "unknown"}

    def _analyze_breastfeeding(self, state: Dict) -> Dict:
        """Analyze breastfeeding safety using Claude"""
        drug_name = state.get("drug_name", "")
        fda_data = state.get("fda_data") or {}

        breastfeeding_text = fda_data.get('breastfeeding_text') or 'No data'

        messages = [
            SystemMessage(content="""You are a pharmacist analyzing drug safety for breastfeeding.
            Respond with only one word: 'safe', 'caution', or 'avoid'."""),
            HumanMessage(content=f"""
            Analyze {drug_name} breastfeeding safety:
            Information: {breastfeeding_text[:500] if breastfeeding_text else 'No data'}
            """)
        ]

        try:
            response = self.client.invoke(messages)
            return {"breastfeeding_safety": response.content.strip().lower()}
        except Exception as e:
            logger.error(f"Breastfeeding analysis error: {e}")
            return {"breastfeeding_safety": "unknown"}

    def _extract_warnings(self, state: Dict) -> Dict:
        """Extract key warnings using Claude"""
        drug_name = state.get("drug_name", "")
        fda_data = state.get("fda_data") or {}

        pregnancy_text = fda_data.get('pregnancy_text') or 'No data'
        breastfeeding_text = fda_data.get('breastfeeding_text') or 'No data'

        messages = [
            SystemMessage(content="""You are a pharmacist extracting key warnings from drug labels.
            Respond with a JSON array of warning strings. Maximum 5 warnings."""),
            HumanMessage(content=f"""
            Extract key warnings for {drug_name}:
            Pregnancy: {pregnancy_text[:300] if pregnancy_text else 'No data'}
            Breastfeeding: {breastfeeding_text[:300] if breastfeeding_text else 'No data'}
            """)
        ]

        try:
            response = self.client.invoke(messages)
            warnings = json.loads(response.content)
            return {"warnings": warnings if isinstance(warnings, list) else ["Consult healthcare provider"]}
        except Exception as e:
            logger.error(f"Warning extraction error: {e}")
            return {"warnings": ["Consult healthcare provider"]}

    def _generate_summary(self, state: Dict) -> Dict:
        """Generate patient-friendly summary using Claude"""
        drug_name = state.get("drug_name", "this medication")
        pregnancy = state.get("pregnancy_safety", "unknown")
        breastfeeding = state.get("breastfeeding_safety", "unknown")

        messages = [
            SystemMessage(content="""You are a pharmacist providing clear, patient-friendly drug safety summaries.
            Keep the summary brief (2-3 sentences) and actionable."""),
            HumanMessage(content=f"""
            Create a summary for {drug_name}:
            Pregnancy safety: {pregnancy}
            Breastfeeding safety: {breastfeeding}
            """)
        ]

        try:
            response = self.client.invoke(messages)
            return {"summary": response.content.strip()}
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return {"summary": f"Consult your healthcare provider about {drug_name} safety."}

    def _handle_error(self, state: Dict) -> Dict:
        """Handle errors in the workflow"""
        drug_name = state.get("drug_name", "this medication")
        return {
            "pregnancy_safety": "unknown",
            "breastfeeding_safety": "unknown",
            "warnings": ["Consult healthcare provider"],
            "summary": f"Unable to analyze {drug_name}. Please consult your healthcare provider."
        }

    async def analyze_fda_data(self, drug_name: str, fda_data: Dict) -> Dict:
        """Analyze FDA data using LangGraph workflow with Claude"""
        initial_state: DrugAnalysisState = {
            "drug_name": drug_name,
            "fda_data": fda_data,
            "pregnancy_safety": "",
            "breastfeeding_safety": "",
            "warnings": [],
            "summary": "",
            "error": None
        }

        try:
            result = self.workflow.invoke(initial_state)

            # Check if result is None or empty
            if not result:
                logger.error("Workflow returned None or empty result")
                return self._fallback_response(drug_name)

            return {
                "pregnancy_safety": result.get("pregnancy_safety", "unknown"),
                "breastfeeding_safety": result.get("breastfeeding_safety", "unknown"),
                "warnings": result.get("warnings", ["Consult healthcare provider"]),
                "summary": result.get("summary",
                                      f"Unable to analyze {drug_name}. Please consult your healthcare provider.")
            }
        except Exception as e:
            logger.error(f"Workflow execution error: {e}", exc_info=True)
            return self._fallback_response(drug_name)

    def _fallback_response(self, drug_name: str) -> Dict:
        return {
            "pregnancy_safety": "unknown",
            "breastfeeding_safety": "unknown",
            "warnings": ["Consult healthcare provider"],
            "summary": f"Unable to analyze {drug_name}. Please consult your healthcare provider."
        }

    async def synthesize_all_sources(self, all_data: Dict) -> Dict:
        """Synthesize data from multiple sources (FDA, DailyMed, PubMed, BioBERT)"""
        fda_data = all_data.get('fda') or {}
        dailymed_data = all_data.get('dailymed') or {}
        pubmed_data = all_data.get('pubmed') or {}
        biobert_extracted = all_data.get('biobert_extracted') or {}

        # Build comprehensive context for Claude
        context = self._build_synthesis_context(fda_data, dailymed_data, pubmed_data, biobert_extracted)

        messages = [
            SystemMessage(content="""You are a pharmacist synthesizing drug safety data from multiple authoritative sources.
            Analyze all available data and provide a comprehensive safety assessment.
            Respond in JSON format with keys: pregnancy_safety, breastfeeding_safety, warnings, summary, evidence_quality"""),
            HumanMessage(content=context)
        ]

        try:
            response = self.client.invoke(messages)
            synthesis = json.loads(response.content)
            return synthesis
        except Exception as e:
            logger.error(f"Synthesis error: {e}", exc_info=True)
            # Fallback to basic FDA analysis if synthesis fails
            if fda_data:
                return await self.analyze_fda_data(
                    fda_data.get('generic_names', ['Unknown'])[0] if fda_data.get('generic_names') else 'Unknown',
                    fda_data
                )
            return self._fallback_response("this medication")

    def _build_synthesis_context(self, fda_data: Dict, dailymed_data: Dict, pubmed_data: Dict,
                                 biobert_data: Dict) -> str:
        """Build comprehensive context from all data sources"""
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

    async def get_pubmed_count(self, drug_name: str) -> int:
        """Get research count (simplified for demo)"""
        # In real implementation, would call PubMed API
        # For now, return mock data
        mock_counts = {
            "tylenol": 1250,
            "zoloft": 845,
            "amoxicillin": 2100
        }
        return mock_counts.get(drug_name.lower(), 10)
