import os
import urllib.parse
from typing import Optional, Literal, override, AsyncGenerator
from urllib.parse import urlencode

import dotenv
import instructor
import pydantic
import requests
from instructor.exceptions import InstructorRetryException
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic import Field

from ichatbio.agent import IChatBioAgent
from ichatbio.types import AgentCard, AgentEntrypoint, ProcessMessage
from ichatbio.types import Message, TextMessage, ArtifactMessage

dotenv.load_dotenv()

TaxonomicResponseFormat = Literal["json", "summary"]


class GetTaxonomicDataParameters(BaseModel):
    format: TaxonomicResponseFormat = "json"


class CatalogueOfLifeAgent(IChatBioAgent):
    def __init__(self):
        self.agent_card = AgentCard(
            name="Catalogue of Life",
            description="Retrieves taxonomic and species data from the Catalogue of Life database.",
            icon=None,
            entrypoints=[
                AgentEntrypoint(
                    id="get_taxonomic_data",
                    description="Returns taxonomic information for species or higher taxa",
                    parameters=GetTaxonomicDataParameters
                )
            ]
        )

    @override
    def get_agent_card(self) -> AgentCard:
        return self.agent_card

    @override
    async def run(self, request: str, entrypoint: str, params: Optional[BaseModel]) -> AsyncGenerator[Message, None]:
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        instructor_client = instructor.patch(openai_client)

        try:
            yield ProcessMessage(summary="Searching for taxonomic data", description="Generating search parameters")

            taxon: TaxonModel = await instructor_client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_model=TaxonModel,
                messages=[
                    {"role": "system",
                     "content": "You translate user requests into Catalogue of Life API parameters for taxonomic searches."},
                    {"role": "user", "content": request}
                ],
                max_retries=3
            )

            url = taxon.to_url(params.format)

            yield ProcessMessage(
                summary="Retrieving taxonomic data",
                description=f"Search parameters",
                data={
                    "search_parameters": taxon.model_dump(exclude_none=True)
                })

            yield ProcessMessage(description=f"Sending GET request to {url}")

            response = requests.get(url)

            yield ProcessMessage(summary="Taxonomic data retrieved", description=f"Found {response.json().get('total', 0)} results")

            yield ArtifactMessage(
                mimetype="application/json",
                description=f"Taxonomic data for \"{taxon.search_term}\"" if taxon.search_term else "Taxonomic search results",
                content=response.content,
                metadata={
                    "api_query_url": url
                }
            )

            yield TextMessage(text="The generated artifact contains the requested taxonomic information. Note that the artifact's "
                                   "api_query_url may return different results over time as the database is updated.")

        except InstructorRetryException as e:
            yield TextMessage(text="Sorry, I couldn't find any taxonomic data.")


class TaxonModel(BaseModel):
    """API parameters for Catalogue of Life."""

    search_term: Optional[str] = Field(None,
                                      description="Scientific name, common name, or taxonomic keyword to search for.",
                                      examples=["Ursus maritimus", "polar bear", "Felidae"])
    rank: Optional[str] = Field(None, description="Taxonomic rank filter (species, genus, family, etc.)")
    limit: Optional[int] = Field(20, description="Maximum number of results to return. Default is 20.")

    @pydantic.field_validator("limit")
    @classmethod
    def validate_limit(cls, v):
        if v <= 0:
            raise ValueError("limit must be positive")
        return v

    def to_url(self, format: TaxonomicResponseFormat):
        url = "https://api.checklistbank.org/dataset/3LR/nameusage/search"
        params = {}

        if format == "summary":
            params |= {"format": "summary"}

        if self.search_term:
            params |= {"q": self.search_term}

        if self.rank:
            params |= {"rank": self.rank}

        if self.limit:
            params |= {"limit": self.limit}

        if params:
            url += "?" + urlencode(params)

        return url


if __name__ == "__main__":
    from ichatbio.server import run_agent_server
    
    agent = CatalogueOfLifeAgent()
    print("Starting Catalogue of Life Agent...")
    print("Agent card will be available at: http://localhost:9000/.well-known/agent.json")
    
    run_agent_server(agent, host="0.0.0.0", port=9000)