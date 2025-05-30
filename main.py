from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import openai
import properties_data_api
import about_us_api
# Import your main functionality from the provided code
# Assuming you encapsulate the logic above into callable functions
from amlak_core import (
    extract_filters_with_gpt,
    run_qdrant_search,
    generate_reply
)

app = FastAPI()

class QueryRequest(BaseModel):
    user_query: str

class QueryResponse(BaseModel):
    intent: Optional[str]
    reply: str
    display_results: Optional[List[dict]] = []
@app.post("/query", response_model=QueryResponse)
async def query_property(req: QueryRequest):
    user_query = req.user_query

    # Step 1: Classify the intent using a simple GPT call
    resp = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "You are a real estate assistant trained for Dubai."},
            {"role": "user", "content": f"What is the user's intent for this query? Just return the intent.\n\nQuery: {user_query}"}
        ],
        functions=[
            {
                "name": "classify_intent",
                "description": "Determine the user's intent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "enum": ["search_properties", "about_info", "contact_info", "company_info", "greeting", "help", "fallback"]
                        }
                    },
                    "required": ["intent"]
                }
            }
        ],
        function_call={"name": "classify_intent"}
    )

    intent = resp.choices[0].message.function_call.arguments
    intent = eval(intent)["intent"]  # convert string dict to actual dict

    # Handle each intent accordingly
    if intent == "search_properties":
        filters = extract_filters_with_gpt(user_query)
        emb = openai.Embedding.create(model="text-embedding-ada-002", input=user_query)
        vector = emb["data"][0]["embedding"]
        results = run_qdrant_search(filters, vector=vector, limit=20)
        display_results = results.get("display_results", [])
        reply = generate_reply(display_results, user_query)
        return QueryResponse(intent=intent, reply=reply, display_results=display_results)

    elif intent in ("about_info", "contact_info"):
        reply = generate_reply(None, user_query)
        return QueryResponse(intent=intent, reply=reply, display_results=[])

    else:
        reply = generate_reply(None, user_query)
        return QueryResponse(intent=intent, reply=reply, display_results=[])



if __name__ == "__main__":
    properties_data_api.properties_data_upload()
    about_us_api.about_us_api()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
