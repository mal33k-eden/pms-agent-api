from typing import Union

from fastapi import APIRouter, Query, HTTPException
import logging

from app.data.models import DrugSafetyResponse
from app.services.ai.basic_analyzer import DrugSafetyAI
from app.services.ai.deep_analyzer import EnhancedDrugAnalyzer
from app.services.fda_client import FDAClient
from setup.db.config import db

logger = logging.getLogger(__name__)

router = APIRouter()
fda_client = FDAClient()

# Lazy load enhanced analyzer to avoid startup errors
enhanced_analyzer = None


def get_analyzer(enhanced: bool = False) -> Union[DrugSafetyAI, EnhancedDrugAnalyzer]:
    """
    Get appropriate analyzer instance based on enhanced parameter.

    Args:
        enhanced: If True, returns EnhancedDrugAnalyzer, otherwise returns basic DrugSafetyAI

    Returns:
        Union[DrugSafetyAI, EnhancedDrugAnalyzer]: The appropriate analyzer instance

    Raises:
        HTTPException: If enhanced analyzer fails to initialize
    """
    if enhanced:
        try:
            return EnhancedDrugAnalyzer()
        except Exception as e:
            logger.error(f"Failed to initialize EnhancedDrugAnalyzer: {e}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="Enhanced analysis is currently unavailable. Please try basic analysis or contact support."
            )
    else:
        try:
            return DrugSafetyAI()
        except Exception as e:
            logger.error(f"Failed to initialize DrugSafetyAI: {e}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="Basic analysis is currently unavailable. Please try again later."
            )


@router.get("/api/drug/{drug_name}", response_model=DrugSafetyResponse)
async def get_drug_safety(
        drug_name: str,
        enhanced: bool = Query(False,
                               description="Use enhanced analysis with multiple data sources (FDA, DailyMed, PubMed, BioBERT)")
):
    """
    Fetch drug safety data for a given drug name.

    Args:
        drug_name (str): Name of the drug to lookup
        enhanced (bool): If True, performs comprehensive analysis using multiple data sources

    Returns:
        DrugSafetyResponse: Safety data including pregnancy and breastfeeding information

    Example response:
    {
      "drug_name": "Atorvastatin",
      "pregnancy_category": null,
      "pregnancy_safety": "avoid",
      "breastfeeding_safety": "avoid",
      "recommendations": "Atorvastatin should not be taken during pregnancy as it may harm fetal development, and women who become pregnant while taking it should stop immediately and contact their doctor. The drug should also be avoided during breastfeeding since it may pass into breast milk and affect the nursing infant. Women who need cholesterol management during pregnancy or breastfeeding should discuss alternative options with their healthcare provider.",
      "confidence": "moderate",
      "warnings": [
        "Discontinue atorvastatin when pregnancy is recognized",
        "May affect synthesis of cholesterol and other biologically active substances",
        "Safety during breastfeeding has not been established"
      ]
    }
    """
    try:
        # Validate drug name
        if not drug_name or not drug_name.strip():
            raise HTTPException(status_code=400, detail="Drug name cannot be empty")
        # Check database first
        drug_data = await get_from_database(drug_name, enhanced)
        if drug_data:
            return drug_data

        # Not in DB - fetch and analyze
        drug_data = await analyze(enhanced=enhanced, drug_name=drug_name)

        return drug_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing drug safety request for {drug_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request. Please try again later."
        )


async def analyze(enhanced, drug_name):
    """
    Analyze drug safety data using the provided analyzer.

    Args:
        enhanced (bool): Whether to use enhanced analysis mode
        drug_name (str): Name of the drug to analyze

    Returns:
        DrugSafetyResponse: Analyzed drug safety data
    """
    analyzer = get_analyzer(enhanced)
    if enhanced:
        drug_data = await fetch_and_analyze_enhanced(analyzer, drug_name)
    else:
        drug_data = await fetch_and_analyze(analyzer, drug_name)
    return drug_data


async def get_from_database(drug_name: str, enhanced: bool = False):
    try:
        async with db.pool.acquire() as conn:
            # If enhanced, prefer enhanced data source
            data_source_filter = "'enhanced_multi_source'" if enhanced else "'fda_ai'"

            result = await conn.fetchrow(f"""
                                         SELECT d.*, ds.*
                                         FROM drugs d
                                                  JOIN drug_safety_data ds ON d.id = ds.drug_id
                                         WHERE (LOWER(d.name) = LOWER($1)
                                            OR LOWER(d.generic_name) = LOWER($1))
                                             AND ds.expires_at > NOW()
                                             AND ds.data_source = {data_source_filter}
                                             ORDER BY ds.fetched_at DESC
                                             LIMIT 1
                                         """, drug_name)

            if result:
                return DrugSafetyResponse(
                    drug_name=result['name'],
                    pregnancy_category=result['pregnancy_category'],
                    pregnancy_safety=result['pregnancy_safety'],
                    breastfeeding_safety=result['breastfeeding_safety'],
                    recommendations=result['ai_summary'],
                    confidence="high" if result['confidence_score'] > 0.7 else "moderate",
                    warnings=result['key_warnings'] if result['key_warnings'] else []
                )

        return None
    except Exception as e:
        logger.error(f"Database error while fetching {drug_name}: {e}", exc_info=True)
        # Return None to fallback to fresh analysis
        return None


async def fetch_and_analyze(analyzer: DrugSafetyAI, drug_name: str):
    """Fetch from FDA and analyze with AI"""
    try:
        # Get FDA data
        fda_data = await fda_client.search_drug_label(drug_name)
        if not fda_data:
            # Pure AI fallback
            return DrugSafetyResponse(
                drug_name=drug_name,
                pregnancy_category="Unknown",
                pregnancy_safety="unknown",
                breastfeeding_safety="unknown",
                recommendations="No FDA data available. Consult healthcare provider.",
                confidence="low"
            )

        # Analyze with AI
        ai_analysis = await analyzer.analyze_fda_data(drug_name, fda_data)

        # Get research count
        study_count = await analyzer.get_pubmed_count(drug_name)

        # Store in database (don't fail if storage fails)
        try:
            await store_drug_data(drug_name, fda_data, ai_analysis, study_count, data_source='fda_ai')
        except Exception as e:
            logger.error(f"Failed to store drug data for {drug_name}: {e}", exc_info=True)

        return DrugSafetyResponse(
            drug_name=drug_name,
            pregnancy_category=fda_data.get('pregnancy_category'),
            pregnancy_safety=ai_analysis.get('pregnancy_safety', 'unknown'),
            breastfeeding_safety=ai_analysis.get('breastfeeding_safety', 'unknown'),
            recommendations=ai_analysis.get('summary', 'Consult healthcare provider.'),
            confidence="high" if study_count > 100 else "moderate",
            warnings=ai_analysis.get('warnings', [])
        )
    except Exception as e:
        logger.error(f"Error in fetch_and_analyze for {drug_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Unable to analyze drug data. Please try again later."
        )


async def fetch_and_analyze_enhanced(analyzer: EnhancedDrugAnalyzer, drug_name: str):
    """Fetch from multiple sources and perform comprehensive analysis"""
    try:

        # Use EnhancedDrugAnalyzer to get data from all sources
        comprehensive_analysis = await analyzer.analyze_drug_comprehensive(drug_name)

        # Extract synthesis results
        synthesis = comprehensive_analysis.get('safety_assessment', {})
        sources = comprehensive_analysis.get('sources_available', {})

        pregnancy_category = None
        if sources.get('fda'):
            # TODO:// Figure a way to extract such information from FDA
            pregnancy_category = None

        # Store enhanced data in database (don't fail if storage fails)
        try:
            await store_enhanced_drug_data(drug_name, comprehensive_analysis, pregnancy_category)
        except Exception as e:
            logger.error(f"Failed to store enhanced drug data for {drug_name}: {e}", exc_info=True)

        confidence_value = comprehensive_analysis.get('confidence', 'low')
        if isinstance(confidence_value, (int, float)):
            if confidence_value >= 0.8:
                confidence_str = "high"
            elif confidence_value >= 0.5:
                confidence_str = "moderate"
            else:
                confidence_str = "low"
        else:
            confidence_str = confidence_value

        return DrugSafetyResponse(
            drug_name=drug_name,
            pregnancy_category=pregnancy_category,
            pregnancy_safety=synthesis.get('pregnancy_safety', 'unknown'),
            breastfeeding_safety=synthesis.get('breastfeeding_safety', 'unknown'),
            recommendations=synthesis.get('summary', 'Consult healthcare provider.'),
            confidence=confidence_str,
            warnings=synthesis.get('warnings', [])
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in enhanced analysis for {drug_name}: {e}", exc_info=True)
        # Fallback to basic analysis if enhanced fails
        logger.info(f"Falling back to basic analysis for {drug_name}")
        analyzer = get_analyzer(enhanced=False)
        return await fetch_and_analyze(analyzer, drug_name)


async def store_drug_data(drug_name, fda_data, ai_analysis, study_count, data_source='fda_ai'):
    """Store analyzed data in database"""
    try:
        async with db.pool.acquire() as conn:
            # Insert or get drug
            drug_id = await conn.fetchval(
                """
                INSERT INTO drugs (name, generic_name)
                VALUES ($1, $2) ON CONFLICT (name) DO
                UPDATE SET name = $1
                    RETURNING id
                """,
                drug_name,
                fda_data.get('generic_names', [None])[0]
            )

            # Store safety data
            await conn.execute(
                """
                INSERT INTO drug_safety_data
                (drug_id, pregnancy_category, pregnancy_text, breastfeeding_text,
                 pregnancy_safety, breastfeeding_safety, ai_summary, key_warnings,
                 data_source, confidence_score, study_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                drug_id,
                fda_data.get('pregnancy_category'),
                fda_data.get('pregnancy_text'),
                fda_data.get('breastfeeding_text'),
                ai_analysis['pregnancy_safety'],
                ai_analysis['breastfeeding_safety'],
                ai_analysis['summary'],
                ai_analysis.get('warnings', []),
                data_source,
                0.9 if study_count > 100 else 0.6,
                study_count
            )
    except Exception as e:
        logger.error(f"Error storing drug data: {e}", exc_info=True)
        raise


async def store_enhanced_drug_data(drug_name: str, comprehensive_analysis: dict, pregnancy_category: str = None):
    """Store enhanced multi-source analysis data in database"""
    try:
        async with db.pool.acquire() as conn:
            synthesis = comprehensive_analysis['safety_assessment']
            sources = comprehensive_analysis['sources_available']
            research = comprehensive_analysis.get('research_quality', {})

            # Determine generic name from available sources
            generic_name = drug_name  # Default to drug_name if not found

            # Insert or get drug
            drug_id = await conn.fetchval(
                """
                INSERT INTO drugs (name, generic_name)
                VALUES ($1, $2) ON CONFLICT (name) DO
                UPDATE SET name = $1
                    RETURNING id
                """,
                drug_name,
                generic_name
            )

            # Calculate confidence score based on comprehensive analysis
            confidence_score = comprehensive_analysis.get('confidence', 0.5)
            if isinstance(confidence_score, str):
                # Convert string confidence to numeric
                confidence_map = {'high': 0.9, 'moderate': 0.6, 'low': 0.3}
                confidence_score = confidence_map.get(confidence_score.lower(), 0.5)

            # Store enhanced safety data
            await conn.execute(
                """
                INSERT INTO drug_safety_data
                (drug_id, pregnancy_category, pregnancy_text, breastfeeding_text,
                 pregnancy_safety, breastfeeding_safety, ai_summary, key_warnings,
                 data_source, confidence_score, study_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                drug_id,
                pregnancy_category,  # TODO:// i should figure out a way to extract this information
                f"Enhanced analysis from {sources}",
                f"Enhanced analysis from {sources}",
                synthesis.get('pregnancy_safety', 'unknown'),
                synthesis.get('breastfeeding_safety', 'unknown'),
                synthesis.get('summary', 'Consult healthcare provider.'),
                synthesis.get('warnings', []),
                'enhanced_multi_source',
                confidence_score,
                research.get('total_studies', 0)
            )
    except Exception as e:
        logger.error(f"Error storing enhanced drug data: {e}", exc_info=True)
        raise
