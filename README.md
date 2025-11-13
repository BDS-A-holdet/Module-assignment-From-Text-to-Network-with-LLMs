# From Text to Network: Earnings Call Analysis with LLMs

## Project Overview

This project demonstrates a complete pipeline for extracting structured information from unstructured text using Large Language Models (LLMs) and transforming it into network representations for analysis. We analyze earnings call transcripts from S&P 500 Information Technology companies to understand analyst coverage patterns and identify company communities based on shared analyst attention.

**Course:** From Text to Network with LLMs - Module Assignment  
**Dataset:** S&P 500 Earnings Call Transcripts (Information Technology Sector, 2024)  
**Analysis Focus:** Analyst coverage networks revealing market segments and company relationships

---

## Repository Structure

```
.
├── README.md                                          # This file
├── requirements.txt                                   # Python dependencies
│
├── notebooks/
│   ├── LLM_pipeline_identifying_branches_cleaned.ipynb  # Initial sector classification
│   └── exam_load_3__1_.ipynb                            # Main analysis pipeline
│
├── data/
│   ├── it_companies.csv                               # List of 74 IT companies
│   ├── structured_output.csv                          # LLM-extracted participant data
│   └── manual_review_sample.csv                       # Sample for precision/recall evaluation
│
├── outputs/
│   └── Slide_deck.pptx                                # Final presentation (max 10 slides)
│
└── docs/
    └── Tekst_eksempler__precision-recall_.docx        # Manual quality assessment notes
```

---

## Dataset Selection & Rationale

### Dataset: S&P 500 Earnings Call Transcripts
- **Source:** [kurry/sp500_earnings_transcripts](https://huggingface.co/datasets/kurry/sp500_earnings_transcripts)
- **Full dataset:** Earnings call transcripts for S&P 500 and US large caps, 2005–2025

### Our Subset Choice: Information Technology Sector Only (2024)

**Why Information Technology?**
1. **Coherent analytical scope:** Focusing on a single sector ensures meaningful comparisons and interpretable network patterns
2. **Rich relational potential:** IT companies share analysts with overlapping expertise, creating dense network structures
3. **Manageable scope:** 74 companies provide sufficient data without overwhelming processing requirements
4. **Business relevance:** IT sector represents distinct subsegments (chips, software, hardware, cloud) ideal for community detection

**Why 2024 only?**
- Recent data ensures current market dynamics
- Keeps computational requirements reasonable
- Single time period simplifies analysis while maintaining richness

---

## LLM Extraction Process

### 1. Sector Classification (Preprocessing)
**Notebook:** `LLM_pipeline_identifying_branches_cleaned.ipynb`

**Tool:** Ollama with Gemma3:12b model  
**Task:** Classify all S&P 500 companies into 11 GICS-like sectors  
**Method:** Few-shot classification with structured JSON output  
**Result:** Identified 74 Information Technology companies for subsequent analysis

**Key Features:**
- Batch processing (80 companies per batch) for efficiency
- Caching mechanism to avoid redundant API calls
- Confidence scoring for classification quality assessment
- Few-shot examples improve accuracy (Apple→IT, Pfizer→Health Care, etc.)

### 2. Structured Participant Extraction
**Notebook:** `exam_load_3__1_.ipynb`

**Tool:** Google Gemini 2.5 Flash Lite via OpenAI-compatible API  
**Task:** Extract structured information about all speakers in earnings call transcripts

#### Extraction Schema (Pydantic)
```python
class Participant(BaseModel):
    person_name: str           # Speaker's full name
    role: str                  # Job title (CEO, CFO, Analyst, etc.)
    organization: str          # Company/firm they represent
    type: str                  # 'company_rep' or 'analyst'
    topics: List[str]          # Discussion topics (not used in network)

class EarningsCallStructure(BaseModel):
    company_name: str
    participants: List[Participant]
```

#### Why This Schema?
- **Person-centric:** Earnings calls are conversations where roles matter
- **Clear classification:** Binary distinction (company representative vs. external analyst)
- **Relational potential:** Enables analyst-company and company-company networks
- **Structured output:** Pydantic + JSON Schema enforcement ensures consistent data format

#### Processing Pipeline
1. **API Configuration:** OpenAI client pointed to Gemini endpoint for fast, cheap inference
2. **Schema Validation:** JSON Schema enforcement via `response_format` parameter
3. **Batch Processing:** Process all 2024 IT sector transcripts with progress tracking
4. **Data Flattening:** Convert nested structures to flat DataFrame (one row per participant per call)
5. **Error Handling:** Graceful failure management, skip problematic transcripts

**Output:** `structured_output.csv` with 4,000+ participant entries across 74 companies

---

## Descriptive Exploration & Quality Assessment

### Dataset Statistics
- **Total Records:** 4,000+ participant entries from IT sector earnings calls (2024)
- **Unique Companies:** 74 Information Technology companies from S&P 500
- **Unique Participants:** ~2,500 individual speakers
- **Unique Organizations:** ~200 analyst firms and company organizations

### Participant Distribution
- **Analysts:** ~70% of participants (investment analysts from financial institutions)
- **Company Representatives:** ~30% (executives, officers, IR teams)

### Top Analyst Organizations
1. Morgan Stanley
2. Goldman Sachs  
3. JPMorgan
4. Bank of America
5. Evercore

### Analyst Coverage Patterns
- **Coverage Range:** High-coverage companies tracked by 30+ analyst organizations; low-coverage by <10
- **Average:** ~15-20 analyst organizations per company
- **Top 3 Most Covered:**
  1. CrowdStrike Holdings - 34 unique analyst organizations
  2. ServiceNow, Inc. - 33 unique analyst organizations
  3. Palo Alto Networks, Inc. - 32 unique analyst organizations

---

## Quality Assessment: Precision & Recall Analysis

### Method
**File:** `manual_review_sample.csv`  
**Sample Size:** 20 randomly selected participant entries  
**Comparison:** Manual review against original transcripts

### Findings

#### ✅ Strengths (High Precision)
1. **Type Classification (analyst vs. company_rep):** Nearly perfect accuracy
   - Model correctly identifies participant roles based on context
   - Very few misclassifications
   
2. **Organization Extraction:** High accuracy
   - Analyst firms correctly identified (e.g., "Morgan Stanley", "Goldman Sachs")
   - Company affiliations properly captured
   - Information is typically explicit in transcripts ("Ben Reitzes from Melius Research")

3. **Name and Role Extraction:** Generally accurate
   - Speaker names captured correctly
   - Job titles appropriately extracted (CEO, CFO, Analyst, etc.)

#### ⚠️ Limitations (Lower Recall/Issues)

1. **Topic Extraction Quality:**
   - **Problem:** Topics are highly granular and often unique per participant
   - **Impact:** Difficult to identify patterns or aggregate insights
   - **Root Cause:** No predefined topic taxonomy in prompt
   - **Solution for Future Work:** Provide LLM with 5-10 predetermined topic categories

2. **Completeness:**
   - Some participants may be missed if mentioned only briefly
   - Minor speakers or passing mentions sometimes not extracted

3. **Topic Inconsistency:**
   - Same topic described differently across extractions
   - Example: "AI strategy" vs "artificial intelligence initiatives" vs "AI adoption"

### Overall Assessment
**Precision:** High (~90-95%) - What the model extracts is generally correct  
**Recall:** Good (~85-90%) - Model captures most participants, occasional misses  
**Usability:** Excellent for network construction (participants & organizations are reliable)

### Impact on Analysis
- **Decided NOT to use topics** in network analysis due to quality issues
- Focus on participant-company relationships where extraction quality is high
- This demonstrates good scientific practice: acknowledge and work around limitations

---

## Network Construction: Knowledge Graph

### Graph Definition

**Type:** Undirected, weighted, one-mode network

#### Nodes (74 companies)
- **Entity:** IT companies hosting earnings calls
- **Attribute:** Number of unique analysts covering the company
- **Example:** "NVIDIA Corporation" (node_size = 28 analysts)

#### Edges (Shared analyst coverage)
- **Relationship:** Companies share one or more individual analysts
- **Weight:** Number of analysts both companies share
- **Example:** CrowdStrike ↔ Palo Alto Networks (weight = 17 shared analysts)

### Rationale
This network represents **shared analyst attention** across IT companies:
- **Market Segmentation:** Companies sharing many analysts likely operate in similar market segments
- **Competitive Positioning:** Shared coverage suggests companies are perceived as peers or competitors
- **Investor Perspective:** Network reveals how the analyst/investor community groups companies

### Construction Pipeline

1. **Filter Data:** Extract only analyst participants (exclude company representatives)
2. **Create Nodes:** One node per company with analyst count attribute
3. **Calculate Edges:** For each company pair, count shared individual analysts
4. **Apply Threshold:** Include edge only if ≥1 shared analyst (adjustable parameter)
5. **Store Graph:** NetworkX Graph object with full metadata

**Implementation:** `create_company_similarity_network()` function in main notebook

---

## Network Analysis Results

### 1. Community Detection

**Method:** Greedy Modularity Optimization  
**Algorithm:** Iteratively merge nodes to maximize modularity score  
**Constraint:** Communities must have ≥3 members  
**Visualization:** Top 4 largest communities displayed

#### Modularity Score: 0.570

**Interpretation:**
- Moderate-to-strong community structure (>0.3 is considered good)
- More connections within communities than expected by chance
- Indicates meaningful market segmentation in the IT sector
- Not perfect (would be 1.0), but significantly better than random

#### Identified Communities

**Community 1 (Chip/Semiconductor)** - Light Green
- **Companies:** NVIDIA, Broadcom, Microchip Technology, Qualcomm, Intel, AMD, etc.
- **Characteristic:** Dense interconnectedness, strong shared analyst coverage
- **Insight:** Analysts specializing in semiconductor market cover these companies as a group

**Community 2 (Software/Cloud)** - Dark Blue
- **Companies:** Microsoft, Adobe, CrowdStrike, ServiceNow, Salesforce, Oracle
- **Characteristic:** Moderately connected, strong edges between major pairs (Adobe-Microsoft)
- **Insight:** Enterprise software and cloud infrastructure companies

**Community 3 (Hardware/Consumer Electronics)** - Red
- **Companies:** Apple, Dell, HP Inc., Hewlett Packard Enterprise
- **Characteristic:** Consumer electronics and enterprise hardware manufacturers
- **Insight:** Companies selling physical computing devices

**Community 4 (Mixed/Financial IT)** - Orange
- **Companies:** Smaller, more diverse group including FinTech and specialized IT services
- **Characteristic:** Less dense, varied analyst coverage patterns

### 2. Centrality Analysis

#### Most Covered Companies (By Analyst Count)
1. **CrowdStrike** - 34 analysts
2. **ServiceNow** - 33 analysts  
3. **Palo Alto Networks** - 32 analysts
4. **Salesforce** - 31 analysts
5. **Adobe** - 30 analysts

**Insight:** High-growth software/cybersecurity companies attract most analyst attention

#### Most Connected Companies (By Weighted Degree)
1. **Broadcom** - 171 total shared analyst connections
2. **Qualcomm** - 158 connections
3. **Western Digital** - 155 connections
4. **Intel** - 153 connections
5. **NVIDIA** - 151 connections

**Insight:** Central companies with broad connections across IT sector

#### Eigenvector Centrality (Top 5)
1. **Western Digital** - 0.26
2. **Broadcom** - 0.25
3. **Seagate Technology** - 0.24
4. **Qualcomm** - 0.24
5. **Microchip Technology** - 0.23

**Interpretation:**
- Moderate scores (0.26 is not exceptionally high)
- No single "super-central" company dominates analyst attention
- Suggests multiple distinct IT subsectors with distributed importance
- These companies bridge between different communities

**Business Meaning:** Western Digital and Broadcom connect multiple IT segments, indicating versatile market positioning that appeals to diverse analyst specializations

#### Betweenness Centrality (Top 5)
1. **Dayforce Inc.** - 0.21
2. **Autodesk** - 0.19
3. **Western Digital** - 0.18
4. **Intel** - 0.17
5. **Seagate Technology** - 0.16

**Interpretation:**
- These companies act as "bridges" between different communities
- Dayforce connects HR tech analysts with broader IT sector coverage
- Intel bridges semiconductor and hardware communities
- High betweenness = strategic position connecting distinct market segments

### 3. Strongest Company Pairs (Most Similar)

1. **CrowdStrike ↔ Palo Alto Networks** - 17 shared analysts
   - *Both cybersecurity leaders, direct competitors*
   
2. **ServiceNow ↔ Salesforce** - 16 shared analysts
   - *Enterprise cloud platforms, similar market positioning*
   
3. **Adobe ↔ Microsoft** - 15 shared analysts
   - *Enterprise software giants with overlapping product portfolios*
   
4. **Broadcom ↔ Qualcomm** - 15 shared analysts
   - *Major semiconductor designers, mobile/wireless focus*
   
5. **NVIDIA ↔ Broadcom** - 14 shared analysts
   - *Leading chip designers, AI/datacenter market leaders*

---

## Main Findings

### Key Insights

1. **Market Segmentation is Real**
   - Modularity score (0.570) confirms distinct IT subsectors exist
   - Analyst coverage patterns align with industry intuition (chips, software, hardware)
   - Communities represent genuine market segments, not random clustering

2. **Cybersecurity is Hot**
   - CrowdStrike, Palo Alto Networks, and CrowdStrike top analyst coverage
   - Strongest company pair relationship (17 shared analysts)
   - Indicates high investor/analyst interest in security companies

3. **Central Players Bridge Communities**
   - Companies like Intel, Western Digital bridge multiple segments
   - High betweenness centrality indicates strategic market positioning
   - These companies relevant to analysts across specializations

4. **Chip Sector is Tightly Integrated**
   - Semiconductor community (NVIDIA, Broadcom, Qualcomm, etc.) shows densest interconnections
   - Analysts covering one chip company likely cover many others
   - Reflects high specialization in semiconductor analysis

5. **No Single Dominant Player**
   - Moderate eigenvector centrality scores (max 0.26)
   - Multiple important companies rather than one central giant
   - Reflects diverse, multi-faceted IT sector landscape

### Business Implications

- **For Companies:** Understand competitive positioning through analyst coverage
- **For Investors:** Identify market segments and peer groups for comparative analysis
- **For Analysts:** See how coverage patterns reveal market structure
- **For Researchers:** Validate that LLM-extracted networks match domain knowledge

---

## Limitations & Future Work

### 1. LLM Extraction Limitations

#### Topic Extraction Quality
- **Issue:** Topics are too granular and unique per participant
- **Impact:** Cannot aggregate or find patterns in discussion topics
- **Solution:** Provide LLM with predefined topic taxonomy (e.g., 5-10 categories like "AI/ML", "Cloud", "Cybersecurity", "Supply Chain", "Competition")

#### Completeness
- **Issue:** Minor participants occasionally missed
- **Impact:** Some analysts may not be captured if mentioned briefly
- **Mitigation:** Focus on major participants reduces impact

### 2. Temporal Limitations

#### Single Time Period (2024 only)
- **Issue:** Cannot observe evolution over time
- **Missing Insights:** How analyst coverage shifts after events (earnings beats, product launches, acquisitions)
- **Future Work:** Multi-year analysis to track network dynamics

#### Individual Analyst vs. Firm Level
- **Current:** Network based on individual analysts (person-level)
- **Alternative:** Could aggregate to firm level (e.g., "Morgan Stanley" as single node)
- **Tradeoff:** Individual level = more granular but may miss institutional patterns

### 3. Network Construction Choices

#### Shared Analyst Metric
- **Assumption:** More shared analysts = more similar companies
- **Limitation:** Doesn't account for why analysts cover both (genuine similarity vs. portfolio diversification)
- **Alternative Metrics:** Could weight by analyst firm size, coverage intensity, or sentiment

#### Single Network Type
- **Current:** Company-company network via shared analysts
- **Alternatives not explored:**
  - Bipartite network (companies + analysts)
  - Topic-based networks (if topics were cleaner)
  - Temporal networks (across quarters)

### 4. Data Scope Limitations

#### IT Sector Only
- **Benefit:** Coherent analysis
- **Cost:** Cannot compare cross-sector patterns (e.g., IT vs. Healthcare analyst behavior)
- **Future Work:** Multi-sector analysis to find universal vs. sector-specific patterns

#### S&P 500 Only
- **Bias:** Only large-cap companies
- **Missing:** Small/mid-cap IT companies with different analyst coverage
- **Consideration:** Results may not generalize beyond large-cap universe

### 5. Methodological Considerations

#### Precision vs. Recall Trade-off
- **Precision:** High for participant identification (~90-95%)
- **Recall:** Good but not perfect (~85-90%)
- **Impact:** Network likely captures major patterns but may miss edge cases

#### Community Detection Algorithm Choice
- **Method Used:** Greedy modularity
- **Alternatives:** Louvain, Label Propagation, Spectral Clustering
- **Consideration:** Different algorithms may reveal different community structures

### 6. Interpretation Limitations

#### Causality
- **Network shows:** Shared analyst attention
- **Network doesn't show:** Why analysts cover certain combinations
- **Caution:** Correlation ≠ causation (shared coverage ≠ strategic similarity)

#### Analyst Motivations
- **Assumption:** Analyst coverage reflects perceived company similarity
- **Reality:** Coverage also driven by client demand, firm strategy, analyst expertise
- **Complexity:** Multiple factors influence analyst assignment

---

## Future Enhancements

1. **Improved Topic Extraction**
   - Use predefined topic taxonomy in LLM prompt
   - Implement topic clustering post-processing
   - Explore topic modeling (LDA, BERTopic) as complement

2. **Temporal Analysis**
   - Expand to multiple years (2020-2024)
   - Track community evolution over time
   - Identify events that shift analyst coverage patterns

3. **Multi-Level Networks**
   - Analyst firm level aggregation
   - Bipartite networks (companies ↔ analysts)
   - Multi-layer networks (analysts + topics + companies)

4. **Alternative Similarity Metrics**
   - Weight by analyst seniority or firm size
   - Incorporate temporal patterns (coverage duration)
   - Use sentiment from transcript content

5. **Cross-Sector Comparison**
   - Replicate analysis across all 11 GICS sectors
   - Compare community structures
   - Identify sector-specific vs. universal patterns

6. **Validation Studies**
   - Compare with traditional industry classifications (GICS, NAICS)
   - Validate against actual company financials (revenue correlations)
   - Survey analysts about their coverage logic

---

## Technical Requirements

### Python Dependencies

```txt
# Core Data & Analysis
pandas>=2.0.0
numpy>=1.24.0
networkx>=3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# LLM & API
openai>=1.0.0
pydantic>=2.0.0
python-dotenv>=1.0.0

# Data Loading
datasets>=2.14.0  # Hugging Face datasets
tqdm>=4.65.0      # Progress bars

# Optional (for preprocessing)
ollama>=0.1.0     # Local LLM inference
```

---

## Contributions

- **Group Members:** Alexander Christiansen, Anders Skjødt Sønderby, Christian Ory Nielsen & Peter Christian Østerballe 
- **Module assignment** From Text to Network with LLMs
- **Submission Date:** November 14, 2025, 10:00 AM

