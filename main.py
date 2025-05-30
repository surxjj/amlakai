import subprocess
import time
import requests
import uvicorn
import threading
import schedule
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# ─── Docker & Qdrant Startup ─────────────────────────────────────────
def start_qdrant_docker():
    print("📦 Checking Qdrant Docker container...")
    result = subprocess.run(
        ["docker", "ps", "--filter", "ancestor=qdrant/qdrant", "--format", "{{.ID}}"],
        capture_output=True, text=True
    )
    if result.stdout.strip() == "":
        print("🚀 Starting Qdrant Docker container...")
        subprocess.Popen(["docker", "run", "-p", "6333:6333", "qdrant/qdrant"])
    else:
        print("✅ Qdrant is already running.")

def wait_for_qdrant_ready(timeout=60):
    print("⏳ Waiting for Qdrant to be ready...")
    for _ in range(timeout):
        try:
            if requests.get("http://localhost:6333/collections").status_code == 200:
                print("✅ Qdrant is ready.")
                return
        except:
            time.sleep(1)
    raise Exception("❌ Qdrant did not start in time.")

# ─── FastAPI Setup ────────────────────────────────────────────────────
app = FastAPI()

class QueryRequest(BaseModel):
    user_query: str

class QueryResponse(BaseModel):
    intent: Optional[str]
    reply: str
    display_results: Optional[List[dict]] = []

@app.post("/query", response_model=QueryResponse)
async def query_property(req: QueryRequest):
    import openai
    from amlak_core import extract_filters_with_gpt, run_qdrant_search, generate_reply

    user_query = req.user_query
    resp = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "You are a real estate assistant trained for Dubai."},
            {"role": "user", "content": f"What is the user's intent for this query? Just return the intent.\n\nQuery: {user_query}"}
        ],
        functions=[{
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
        }],
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

# ─── Scheduler Loop ───────────────────────────────────────────────────
def schedule_jobs():
    import properties_data_api
    import about_us_api

    schedule.every(12).hours.do(properties_data_api.job)
    schedule.every(12).hours.do(about_us_api.job)

    while True:
        schedule.run_pending()
        time.sleep(60)

# ─── Entry Point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    start_qdrant_docker()
    wait_for_qdrant_ready()

    import properties_data_api
    import about_us_api

    print("📦 Uploading property data...")
    properties_data_api.job()
    print("📦 Uploading about us data...")
    about_us_api.job()

    # Start background scheduler
    threading.Thread(target=schedule_jobs, daemon=True).start()

    # Start API
    print("🚀 Starting FastAPI server...")
    uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=8000,
    ssl_keyfile="/home/ubuntu/.ssl/key.pem",
    ssl_certfile="/home/ubuntu/.ssl/cert.pem"
)

