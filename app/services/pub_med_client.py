"""
 * Author: Emmanuel Kwami Tartey
 * Date: 04 Oct, 2025
 * Time: 12:49 AM
 * Project: pms-agent
 * gitHub: https://github.com/mal33k-eden
"""
import httpx
from typing import List, Dict
import xml.etree.ElementTree as ET


class PubMedClient:
    """Fetch actual research counts and recent studies"""
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    async def search_pregnancy_breastfeeding_studies(self, drug_name: str) -> Dict:
        """Get research data from PubMed"""
        # Search for pregnancy studies
        pregnancy_query = f"{drug_name} AND (pregnancy OR pregnant)"
        pregnancy_count = await self._get_count(pregnancy_query)

        # Search for breastfeeding studies
        breastfeeding_query = f"{drug_name} AND (breastfeeding OR lactation)"
        breastfeeding_count = await self._get_count(breastfeeding_query)

        # Get recent relevant studies
        combined_query = f"{drug_name} AND (pregnancy OR breastfeeding OR lactation)"
        recent_studies = await self._get_recent_studies(combined_query, limit=5)

        # Check for high-quality studies
        meta_analysis = await self._check_study_type(
            drug_name, "meta-analysis"
        )
        rct_count = await self._check_study_type(
            drug_name, "randomized controlled trial"
        )

        return {
            'total_studies': pregnancy_count + breastfeeding_count,
            'pregnancy_studies': pregnancy_count,
            'breastfeeding_studies': breastfeeding_count,
            'recent_studies': recent_studies,
            'has_meta_analysis': meta_analysis > 0,
            'has_rct': rct_count > 0,
            'confidence_score': self._calculate_confidence(
                pregnancy_count + breastfeeding_count,
                meta_analysis,
                rct_count
            )
        }

    async def _get_count(self, query: str) -> int:
        """Get count of studies matching query"""
        url = f"{self.BASE_URL}/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmode': 'json',
            'retmax': 0  # Just want count
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                return int(data['esearchresult']['count'])
        return 0

    async def _get_recent_studies(self, query: str, limit: int = 5) -> List[Dict]:
        """Get recent study titles and PMIDs"""
        # First search
        search_url = f"{self.BASE_URL}/esearch.fcgi"
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmode': 'json',
            'retmax': limit,
            'sort': 'date'
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(search_url, params=search_params)
            if response.status_code != 200:
                return []

            data = response.json()
            pmids = data['esearchresult']['idlist']

            if not pmids:
                return []

            # Fetch summaries
            summary_url = f"{self.BASE_URL}/esummary.fcgi"
            summary_params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'json'
            }

            response = await client.get(summary_url, params=summary_params)
            if response.status_code != 200:
                return []

            summaries = response.json()
            studies = []
            for pmid in pmids:
                if pmid in summaries['result']:
                    study = summaries['result'][pmid]
                    studies.append({
                        'pmid': pmid,
                        'title': study.get('title', ''),
                        'authors': study.get('authors', []),
                        'year': study.get('pubdate', '').split()[0],
                        'journal': study.get('source', '')
                    })

            return studies

    async def _check_study_type(self, drug_name: str, study_type: str) -> int:
        """Check for specific study types"""
        query = f"{drug_name} AND {study_type}"
        return await self._get_count(query)

    def _calculate_confidence(self, total_studies: int, meta_analysis: int, rct_count: int) -> float:
        """Calculate confidence score based on research quality"""
        score = 0.0

        # Study count score
        if total_studies > 100:
            score += 0.5
        elif total_studies > 50:
            score += 0.3
        elif total_studies > 10:
            score += 0.2

        # High-quality research bonus
        if meta_analysis > 0:
            score += 0.3
        if rct_count > 0:
            score += 0.2

        return min(score, 1.0)