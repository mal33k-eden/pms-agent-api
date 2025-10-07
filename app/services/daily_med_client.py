import httpx
import xml.etree.ElementTree as ET
from typing import Optional, Dict


class DailyMedClient:
    """DailyMed has more structured data than FDA labels"""
    BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/webservices/v2"

    async def search_spl(self, drug_name: str) -> Optional[Dict]:
        """Search for Structured Product Label"""
        # First, search for the drug
        search_url = f"{self.BASE_URL}/spls.json"
        params = {"drug_name": drug_name}

        async with httpx.AsyncClient() as client:
            response = await client.get(search_url, params=params)
            if response.status_code != 200:
                return None

            data = response.json()
            if not data.get('data'):
                return None

            # Get the first result's setid
            setid = data['data'][0]['setid']

            # Fetch detailed SPL data
            return await self.fetch_spl_details(setid)

    async def fetch_spl_details(self, setid: str) -> Optional[Dict]:
        """Fetch detailed lactation/pregnancy data"""
        # DailyMed provides XML with structured sections
        url = f"{self.BASE_URL}/spls/{setid}.xml"

        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code != 200:
                return None

            # Parse XML for specific sections
            root = ET.fromstring(response.text)

            # Extract lactation-specific data
            lactation_data = self._extract_lactation_section(root)
            pregnancy_data = self._extract_pregnancy_section(root)

            return {
                'setid': setid,
                'lactation': lactation_data,
                'pregnancy': pregnancy_data,
                'has_milk_levels': bool(lactation_data.get('milk_levels')),
                'has_effects_in_infants': bool(lactation_data.get('infant_effects'))
            }

    def _extract_lactation_section(self, root) -> Dict:
        """Extract structured lactation data from XML"""
        # DailyMed uses LOINC codes for sections
        # 77306-9 = Lactation section
        lactation_section = root.find(".//section[code[@code='77306-9']]")

        if not lactation_section:
            return {}

        return {
            'summary': self._get_text(lactation_section.find('.//text')),
            'milk_levels': self._extract_milk_levels(lactation_section),
            'infant_effects': self._extract_infant_effects(lactation_section),
            'clinical_considerations': self._get_text(
                lactation_section.find(".//subsection[code[@code='34077-8']]")
            )
        }

    def _extract_pregnancy_section(self, root) -> Dict:
        """Extract pregnancy section data"""
        # Placeholder implementation
        return {}

    def _get_text(self, element) -> str:
        """Extract text from XML element"""
        if element is None:
            return ""
        return ''.join(element.itertext()).strip()

    def _extract_milk_levels(self, section) -> Optional[str]:
        """Extract milk level information"""
        # Placeholder implementation
        return None

    def _extract_infant_effects(self, section) -> Optional[str]:
        """Extract infant effects information"""
        # Placeholder implementation
        return None
