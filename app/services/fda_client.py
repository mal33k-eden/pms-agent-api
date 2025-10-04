"""
 * Author: Emmanuel Kwami Tartey
 * Date: 03 Oct, 2025
 * Time: 11:14 PM
 * Project: pms-agent
 * gitHub: https://github.com/mal33k-eden
"""
import os
import httpx
import logging
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class FDAClient:
    BASE_URL = os.getenv('FDA_API_URL', '')

    async def search_drug_label(self, drug_name: str) -> Optional[Dict]:
        """Fetch drug label from FDA"""
        params = {
            'search': f'(openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}")',
            'limit': 1
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(self.BASE_URL, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('results'):
                        return self._extract_relevant_sections(data['results'][0])
            except Exception as e:
                logger.error(f"FDA API error: {e}")

        return None

    def _extract_relevant_sections(self, label_data: Dict) -> Dict:
        """Extract pregnancy and nursing sections"""
        return {
            'brand_names': label_data.get('openfda', {}).get('brand_name', []),
            'generic_names': label_data.get('openfda', {}).get('generic_name', []),
            'pregnancy_category': self._extract_first_or_none(
                label_data.get('pregnancy_category', [])
            ),
            'pregnancy_text': self._extract_first_or_none(
                label_data.get('pregnancy', [])
            ),
            'breastfeeding_text': self._extract_first_or_none(
                label_data.get('nursing_mothers', [])
            ),
            'warnings': self._extract_first_or_none(
                label_data.get('warnings', [])
            )
        }

    def _extract_first_or_none(self, data_list):
        return data_list[0] if data_list else None
