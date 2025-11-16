import json
import os
from typing import Dict, Optional, Annotated
import logging
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from app.services.ai.base_analyzer_class import BaseDrugAnalyzer, DrugAnalysisResult
from app.services.fda_client import FDAClient

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


class DrugSafetyAI(BaseDrugAnalyzer):
    def __init__(self):
        super().__init__()
        self.client = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.1,
            max_tokens=1024
        )
        self.fda_client = FDAClient()
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
        return updates

    def _analyze_pregnancy(self, state: Dict) -> Dict:
        """Analyze pregnancy safety"""
        drug_name = state.get("drug_name", "")
        fda_data = state.get("fda_data") or {}

        pregnancy_text = fda_data.get('pregnancy_text') or 'No data'

        messages = [
            SystemMessage(content="""You are a pharmacist analyzing drug safety for pregnancy.
            Respond with only one word: 'safe', 'caution', or 'avoid'."""),
            HumanMessage(content=f"""
            Analyze {drug_name} pregnancy safety:
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

    async def fetch_and_analyze(
            self, drug_name: str,
            is_pregnant=None,
            is_breastfeeding=None,
            trimester=None
    ) -> DrugAnalysisResult:
        """
        Fetch FDA data and perform basic analysis in a single operation.

        This is the main entry point for basic drug analysis. It handles
        fetching FDA data and performing the analysis through the LangGraph workflow.

        Args:
            drug_name: Name of the drug to analyze

        Returns:
            DrugAnalysisResult with safety assessment
        """
        try:
            # Fetch FDA data
            fda_data = await self.fda_client.search_drug_label(drug_name)

            if not fda_data:
                logger.warning(f"No FDA data found for {drug_name}")
                return self._create_fallback_response(drug_name)
            self.fda_data = fda_data
            # Initialize workflow state with fetched FDA data
            initial_state: DrugAnalysisState = {
                "drug_name": drug_name,
                "fda_data": fda_data,
                "pregnancy_safety": "",
                "breastfeeding_safety": "",
                "warnings": [],
                "summary": "",
                "error": None
            }

            # Run the workflow
            try:
                result = self.workflow.invoke(initial_state)

                if not result:
                    logger.error(f"Workflow returned empty result for {drug_name}")
                    return self._create_fallback_response(drug_name)

                # Build and return result with metadata
                analysis_result: DrugAnalysisResult = {
                    "drug_name": drug_name,
                    "pregnancy_safety": result.get("pregnancy_safety", "unknown"),
                    "breastfeeding_safety": result.get("breastfeeding_safety", "unknown"),
                    "warnings": result.get("warnings", ["Consult healthcare provider"]),
                    "summary": result.get("summary",
                                          f"Unable to analyze {drug_name}. Please consult your healthcare provider."),
                    "confidence": 0.6,  # Moderate confidence for FDA-only analysis
                    "sources_used": ["fda"]
                }
                return analysis_result
            except Exception as e:
                logger.error(f"Workflow execution error for {drug_name}: {e}", exc_info=True)
                return self._create_fallback_response(drug_name)

        except Exception as e:
            logger.error(f"Error in fetch_and_analyze for {drug_name}: {e}", exc_info=True)
            return self._create_fallback_response(drug_name)
