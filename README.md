# Catalogue of Life Agent

The Catalogue of Life (COL) Agent is a Python-based AI agent for the iChatBio platform. It lets biodiversity researchers, educators, and enthusiasts query global species data using natural language. The agent extracts scientific names from user queries and fetches authoritative taxonomic information from the Catalogue of Life (COL) database.

## Tech Stack

- **Language:** Python 3.12+
- **Agent Framework:** LangChain
- **LLM:** OpenAI GPT-4o-mini
- **Testing:** pytest, Allure, DeepEval
- **Agent Protocol:** iChatBio A2A
- **Server:** Uvicorn (FastAPI/Starlette)

## Key Components
- `COL_Agent` – Main agent class
- `COLClient` – REST client to query COL API
- `models.py` – Defines species, synonyms, classification, etc.

## Setup & Run

1. **Clone the repo**

   ```bash
   git clone https://github.com/Krishnamodi1200702/catalogue_of_life_agent.git
   cd col_agent

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt

3. **Setup environment variables**
   
   create a .env file and 
   ```bash
   OPENAI_API_KEY=your-openai-api-key

4. **Run the agent**
    
   ```bash
   python main.py

## COL API Endpoints used

- `/dataset/3LR/nameusage/search` - Search for species
- `/dataset/3LR/taxon/{id}` - Get taxon details
- `/dataset/3LR/taxon/{id}/synonyms` - Get synonyms
- `/dataset/3LR/taxon/{id}/vernacular` - Get common names
- `/dataset/3LR/taxon/{id}/classification` - Get hierarchy
- `/dataset/3LR/taxon/{id}/children` - Get child taxa
- `/dataset/3LR/taxon/{id}/distribution` - Get distribution
- `/dataset/3LR/taxon/{id}/references` - Get references

## Available Tools

**Data Tools:**
- search_species
- get_taxon_details
- get_synonyms
- get_vernacular_names
- get_classification
- get_taxon_children
- get_distribution
- get_references

## Documentation

- ChecklistBank API: https://api.checklistbank.org
- LangChain: https://python.langchain.com/
- Allure Framework: https://docs.qameta.io/allure/
