# From Text to Network with LLMs

**Authors:** Alexander Christiansen, Anders Skjødt Sønderby, Christian Ory Nielsen & Peter Christian Østerballe  
**Course:** From Text to Network with LLMs - Module Assignment  
**Date:** November 14, 2025

## Dataset

**Source:** [S&P 500 Earnings Call Transcripts](https://huggingface.co/datasets/kurry/sp500_earnings_transcripts) (2005-2025)  
**Subset:** Information Technology sector, 2024 (74 companies, 4,000+ participant entries)

**Rationale:**
- **Sector focus:** IT companies share analysts, enabling analysis of market segmentation (chips, software, hardware, cloud)
- **Temporal scope:** Single year provides current dynamics with manageable computational requirements
- **Relational richness:** Sufficient overlap for meaningful network construction

## Methodology

### LLM Extraction
1. **Sector classification:** Ollama Gemma3:12b with few-shot learning identified 76 IT companies from S&P 500
2. **Participant extraction:** Google Gemini 2.5 Flash Lite with Pydantic schema extracted structured data (name, role, organization, type) from transcripts
3. **Quality:** Manual review showed strong precision, and somewhat strong recall for participant identification

### Network Construction
- **Type:** Undirected, weighted company-company network
- **Nodes:** 74 IT companies (size = unique analyst count)
- **Edges:** Shared analyst relationships (weight = number of shared analysts)
- **Logic:** Companies sharing analysts likely operate in similar segments or are competitive peers

### Analysis
- **Community detection:** Greedy modularity optimization (modularity = 0.506)
- **Centrality metrics:** Degree, eigenvector, and betweenness centrality
- **Result:** 4 communities identified (Semiconductors, Software/Cloud, Hardware, Mixed/FinTech)

## Main Findings

1. **Distinct market segmentation:** Modularity (0.506) confirms clear IT subsectors aligned with industry structure
2. **Cybersecurity dominance:** CrowdStrike, Palo Alto, ServiceNow show highest coverage (30+ analyst organizations)
3. **Semiconductor integration:** Chip companies form densest community, indicating high analyst specialization
4. **Bridge companies:** Western Digital, Broadcom, Intel connect multiple segments (high betweenness centrality)
5. **Distributed landscape:** No single dominant player (max eigenvector = 0.26), reflecting diverse IT sector

## Limitations

1. **Topic extraction:** Too granular for pattern analysis; future work needs predefined taxonomy
2. **Temporal scope:** Single period limits analysis of coverage evolution
3. **Incomplete recall:** ~10-15% of minor participants missed
4. **Large-cap bias:** S&P 500 only; may not generalize to smaller firms
5. **Causality:** Shared coverage correlates with but doesn't prove strategic similarity
6. **Network scope:** Only explored company-company networks; bipartite and temporal alternatives not tested
7. **Algorithm choice:** Greedy modularity used; other methods (Louvain, spectral) may reveal different structures

## Repository Structure

```
├── Notebooks/
│   ├── Final notebook.ipynb                             # Main pipeline
│   └── LLM pipeline identifying branches cleaned.ipynb  # Sector classification
├── Structured outputs/
│   ├── structured output.csv                            # Extracted data
│   └── manual review sample.csv                         # Quality evaluation
├── Word files/
│   └── Tekst eksempler.docx                             # Informal precision/recall
├── Analyst company network viz.png                      # Knowledge graph
└──requirements.txt                                      # Dependencies
```

## Requirements

See `requirements.txt`. Key packages: pandas, numpy, networkx, matplotlib, seaborn, openai, pydantic, datasets
