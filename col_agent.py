from dotenv import load_dotenv
import os
import requests
import json
from instructor import patch
import openai
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List

# loading the .env file to access API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# patching OpenAI client with Instructor to use response models
client = patch(openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
))

# data model to hold search parameters extracted from the query
class CoLQueryParams(BaseModel):
    q: str = Field(..., description="Scientific or common name keyword")
    rank: Optional[str] = Field(None, description="Taxonomic rank")
    limit: Optional[int] = Field(10, description="Number of results to return")

# model for each species entry
class NameInfo(BaseModel):
    scientificName: str
    rank: str
    link: Optional[HttpUrl]
    acceptedName: Optional[str] = None
    classification: Optional[List[str]] = None

# response structure after processing all results
class ColResponse(BaseModel):
    results: List[NameInfo]
    query_url: str
    total: int

# asking the user for a query input
user_message = input("Enter your species/taxonomy-related query: ")

# telling GPT what to extract from the user input
query_instructions = """
You are a helpful assistant extracting Catalogue of Life API parameters from user queries.
Return a JSON object with:
- q: a keyword like a species or genus name
- rank: taxonomic rank (like 'species', 'genus', etc.)
- limit: max number of results
"""

# get structured params from the user message
query_params = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": query_instructions},
        {"role": "user", "content": user_message}
    ],
    response_model=CoLQueryParams,
    temperature=0
)

# preparing to hit the CoL API with the extracted query
col_base_url = "https://api.checklistbank.org/dataset/3LR/nameusage/search"
params = {
    "q": query_params.q,
    "rank": query_params.rank or "species",
    "limit": query_params.limit or 10
}

# making the GET request
response = requests.get(col_base_url, params=params)
query_url = response.url

# parsing response from CoL
results = []
total = 0

if response.status_code == 200:
    data = response.json()
    total = data.get("total", 0)
    for item in data.get("result", []):
        usage = item.get("usage", {})
        name_info = usage.get("name", {})
        classification = [entry.get("name") for entry in item.get("classification", [])]

        try:
            result = NameInfo(
                scientificName=name_info["scientificName"],
                rank=name_info["rank"],
                link=f"https://www.catalogueoflife.org/data/taxon/{usage.get('id')}",
                acceptedName=usage.get("accepted", {}).get("name", {}).get("scientificName"),
                classification=classification or None
            )
            results.append(result)
        except Exception as e:
            print("Skipping invalid result:", e)
else:
    print("API call failed:", response.status_code)

# creating final structured object
structured_response = ColResponse(
    results=results,
    query_url=query_url,
    total=total
)

# printing results without GPT summary
print("\nStructured Catalogue of Life Results")
print("Total matching records:", structured_response.total)
print("Query URL:", structured_response.query_url)
print("Results:")

for i, result in enumerate(structured_response.results, 1):
    classification = f" | Classification: {' > '.join(result.classification)}" if result.classification else ""
    accepted = f" (Accepted name: {result.acceptedName})" if result.acceptedName else ""
    print(f"{i}. {result.scientificName} ({result.rank}) - {result.link or 'No link'}{accepted}{classification}")

# saving results as JSON
if structured_response.total > 0:
    json_filename = "col_results.json"
    with open(json_filename, mode="w") as file:
        # Convert Pydantic model to dict with string conversion for URLs
        data_dict = structured_response.model_dump()
        # Convert HttpUrl objects to strings
        for result in data_dict.get('results', []):
            if result.get('link'):
                result['link'] = str(result['link'])
        json.dump(data_dict, file, indent=2)
    print(f"Results saved to {json_filename}\n")