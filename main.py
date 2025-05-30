import subprocess
import time
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import openai
import properties_data_api
import about_us_api
from amlak_core import (
    extract_filters_with_gpt,
    run_qdrant_search,
    generate_reply
)

# ─── FastAPI Setup ─────────────────────────────────────────────────────
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

    intent = eval(resp.choices[0].message.function_call.arguments)["intent"]

    if intent == "search_properties":
        filters = extract_filters_with_gpt(user_query)
        emb = openai.Embedding.create(model="text-embedding-ada-002", input=user_query)
        vector = emb["data"][0]["embedding"]
        results = run_qdrant_search(filters, vector=vector, limit=20)
        display_results = results.get("display_results", [])
        reply = generate_reply(display_results, user_query)
        return QueryResponse(intent=intent, reply=reply, display_results=display_results)

    reply = generate_reply(None, user_query)
    return QueryResponse(intent=intent, reply=reply, display_results=[])

# ─── Docker Start Function ─────────────────────────────────────────────
def start_qdrant_docker():
    print("📦 Checking Qdrant Docker container...")
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "ancestor=qdrant/qdrant", "--format", "{{.ID}}"],
            capture_output=True, text=True
        )
        if result.stdout.strip() == "":
            print("🚀 Starting Qdrant Docker container...")
            subprocess.Popen(["docker", "run", "-p", "6333:6333", "qdrant/qdrant"])
            time.sleep(5)
        else:
            print("✅ Qdrant container is already running.")
    except Exception as e:
        print(f"❌ Docker startup error: {e}")

# ─── Entry Point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    start_qdrant_docker()
    properties_data_api.properties_data_upload()
    about_us_api.about_us_api()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
