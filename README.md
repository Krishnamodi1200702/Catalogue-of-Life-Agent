# Catalogue of Life Agent

The Catalogue of Life (COL) Agent is a Python-based AI agent for the iChatBio platform. It lets biodiversity researchers, educators, and enthusiasts query global species data using natural language. The agent extracts scientific names from user queries and fetches authoritative taxonomic information from the Catalogue of Life (COL) database.

## Tech Stack

- Language: Python 3.10+
- LLM Integration: OpenAI SDK
- Structured Output: Instructor
- Agent Framework: iChatBio Agent SDK
- Data Modeling: Pydantic
- Async API Calls: httpx
- Configuration: python-dotenv

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

- GET /nameusage/search?name={query}
