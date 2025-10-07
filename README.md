# Pregnancy & Medications System (PMS) Agent - Drug Safety

**A feature idea for [WITHME-HEALTH](linkedin.com/company/withme-health/) web-guide application**

PMS Agent helps healthcare providers and patients make informed decisions about medication use during pregnancy and
breastfeeding by:

- **Fetching official drug labels** from FDA and DailyMed databases
- **Analyzing safety data** using AI (Anthropic Claude) and BioBERT medical language models
- **Cross-referencing research** from PubMed to validate findings
- **Providing clear recommendations** with confidence scores and warnings
- **Storing results** in PostgreSQL to improve response times

## Features

- **Basic Analysis**: Fast lookups using FDA data and AI analysis
- **Enhanced Analysis**: Comprehensive multi-source analysis combining FDA, DailyMed, PubMed, and BioBERT
- **Structured Responses**: Standardized safety assessments with pregnancy category, recommendations, and warnings

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your database and API credentials

# Run the API
uvicorn main:app --reload
```

## API Usage

### Basic Drug Lookup

```bash
GET /api/drug/{drug_name}
```

### Enhanced Multi-Source Analysis

```bash
GET /api/drug/{drug_name}?enhanced=true
```

### Example Response

```json
{
  "drug_name": "Atorvastatin",
  "pregnancy_category": null,
  "pregnancy_safety": "avoid",
  "breastfeeding_safety": "avoid",
  "recommendations": "Atorvastatin should not be taken during pregnancy...",
  "confidence": "moderate",
  "warnings": [
    "Discontinue atorvastatin when pregnancy is recognized",
    "May affect synthesis of cholesterol and other biologically active substances"
  ]
}
```

## Technology Stack

- **FastAPI**: Modern async web framework
- **PostgreSQL**: Persistent storage with asyncpg
- **Anthropic Claude**: AI-powered medical text analysis
- **BioBERT**: Specialized biomedical language model
- **Transformers/PyTorch**: NLP processing pipeline

## Technical Implementation

### Asynchronous Architecture

- **Async/Await Pattern**: All I/O operations (database, HTTP requests, AI calls) use Python's `asyncio` for
  non-blocking execution
- **Connection Pooling**: PostgreSQL connections managed via `asyncpg` pool for optimal resource utilization
- **Async Context Managers**: FastAPI lifespan events handle graceful startup/shutdown of database connections

### Parallel Processing

- **Concurrent Data Fetching**: Enhanced analysis mode fetches from FDA, DailyMed, and PubMed simultaneously using
  `asyncio.gather()`
- **Independent API Calls**: Multiple external API requests execute in parallel to minimize total response time
- **Fault Tolerance**: Each data source wrapped in safe fetch methods with individual error handling to prevent cascade
  failures

### Performance Optimizations

- **Lazy Loading**: Enhanced analyzer (BioBERT/transformers) initialized on-demand to avoid startup overhead
- **Database Indexing**: Optimized queries with indexes on drug names and safety data lookups
- **Query Optimization**: Single-query joins to minimize database round trips

### Error Handling & Resilience

- **Graceful Degradation**: Enhanced analysis falls back to basic analysis if specialized models fail to load
- **Per-Source Error Isolation**: Failures in one data source don't block results from others
- **Comprehensive Logging**: Structured logging at error boundaries for debugging and monitoring
- **HTTP Exception Handling**: Proper status codes (400, 503, 500) with user-friendly error messages

 
