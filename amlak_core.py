import os
import re
import json
import asyncio
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http.models import (
    Filter as QFilter,
    FieldCondition as QFieldCondition,
    Range as QRange
)
from qdrant_client.http.models import MatchValue

from qdrant_client.http.models import Filter as QFilter, FieldCondition, MatchValue, Range as QRange

# ─── CONFIG ─────────────────────────────────────────────────────────
openai.api_key  = os.getenv("OPENAI_API_KEY")  
QDRANT_HOST     = "localhost"
QDRANT_PORT     = 6333
COLLECTION      = "properties_collection"
VECTOR_DIM      = 384
INFO_COLLECTION = "info_collection"
from sentence_transformers import SentenceTransformer

# right after your other imports
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# ─── INIT QDRANT ─────────────────────────────────────────────────────
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
if not qdrant.collection_exists(collection_name=COLLECTION):
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
if not qdrant.collection_exists(collection_name=INFO_COLLECTION):
    raise RuntimeError(f"{INFO_COLLECTION} not found! Did you ingest your About/Contact docs?")
# ─── CHAT HISTORY ────────────────────────────────────────────────────
chat_history = [
    {"role": "system", "content": "You are Amlak, a friendly Dubai real estate assistant."}
]
# ─── RIGHT AFTER you import / define intent_functions ─────────────────
get_property_detail_spec = {
    "name": "get_property_detail",
    "description": "Fetch a single field (price, size_sq_ft, bedroom, description, etc.) for a previously returned property",
    "parameters": {
        "type": "object",
        "properties": {
            "project_name": {
                "type": "string",
                "description": "Exact project_name of the property"
            },
            "field": {
                "type": "string",
                "description": "Which field to retrieve (e.g. price, size_sq_ft, bedroom, description)"
            }
        },
        "required": ["project_name", "field"]
    }
}

calculate_roi_spec = {
    "name": "calculate_roi",
    "description": "Calculate ROI (Return on Investment) for a given property",
    "parameters": {
        "type": "object",
        "properties": {
            "project_name": {
                "type": "string",
                "description": "Exact project_name of the property"
            }
        },
        "required": ["project_name"]
    }
}



# ─── INTENT FUNCTION ─────────────────────────────────────────────────
intent_functions = [
    {
        "name": "classify_intent",
        "description": "Determine the user's intent based on their query. For example, if the user says 'how can I contact APIL Properties?', return 'contact_info'.",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [
                        "search_properties",
                        "about_info",
                        "contact_info",
                        "company_info",
                        "greeting",
                        "help",
                        "fallback"
                    ]
                }
            },
            "required": ["intent"]
        }
    }
]

# Then, *after* your intent_functions list:
all_function_specs = intent_functions + [
    get_property_detail_spec,
    calculate_roi_spec
]
def map_info_intent(user_query: str) -> str:
    low = user_query.lower()
    if any(w in low for w in ("who is apil", "about apil", "tell me about apil")):
        return "about_info"
    if any(w in low for w in ("contact", "reach", "email", "phone", "call")):
        return "contact_info"
    return None

def add_to_chat_history(role: str, content: str, max_tokens=3000):
    chat_history.append({"role": role, "content": content})
    # Optional: keep only the last N messages based on estimated token count
    while len(chat_history) > 10:  # keep it lean, or tune the number
        chat_history.pop(1)  # keep system prompt, trim oldest user/assistant messages


# ─── FILTER EXTRACTION FUNCTION ──────────────────────────────────────
filter_functions = [
    {
        "name": "extract_filters",
        "description": "Extract search filters from real-estate query",
        "parameters": {
            "type": "object",
            "properties": {
                "project_name":         {"type": "string", "description": "Exact property name, if mentioned"},
                "category":     {"type": "string", "description": "Apartment, villa, penthouse, etc."},
                "city":         {"type": "string"},
                "community_area":{"type": "string"},
                "developer":    {"type": "string"},
                "size_sq_ft_lt": { "type": "number", "description": "Max area in sqft" },
                "size_sq_ft_gt": { "type": "number", "description": "Min area in sqft" },

                "bedroom":      {"type": "integer"},
                "price_lt":     {"type": "number", "description": "Maximum price"},
                "price_gt":     {"type": "number", "description": "Minimum price"}
            }
        }
    }
]

# ─── UTIL: CALL GPT TO EXTRACT FILTERS ───────────────────────────────
def extract_filters_with_gpt(user_query: str) -> dict:
    prompt = (
        "Given this real-estate search query, return a JSON object with any of these keys "
        "(project_name, category, city, community_area, developer, size_sq_ft_lt, size_sq_ft_gt, bedroom, price_lt, price_gt) "
        "that you can extract. Return only the JSON."
        f"\n\nQuery: {user_query}\n\n"
    )

    # Force GPT to call *only* your extract_filters function:
    resp = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "You are Amlak, a friendly real estate assistant."},
            {"role": "user",   "content": prompt}
        ],
        functions=filter_functions,
        function_call={"name": "extract_filters"},
        temperature=0
    )

    # Now we can be sure function_call is present:
    args = resp.choices[0].message.function_call.arguments
    try:
        return json.loads(args)
    except json.JSONDecodeError:
        return {}

def get_known_values():
    keys = ["project_name","category","city","community_area","property_type","bedroom", "size_sq_ft"]
    cache = {k:set() for k in keys}
    pts,_ = qdrant.scroll(
        collection_name=COLLECTION,
        limit=10000,
        with_payload=True,
        with_vectors=False
    )
    for p in pts:
        for k in keys:
            v = p.payload.get(k)
            if v is not None:
                cache[k].add(str(v))
    return {k:list(cache[k]) for k in keys}
import re
import difflib

# ─── your existing synonyms dict ─────────────────────────────────────
synonyms = {
    "flat": "apartment", "flats": "apartment", "apartments": "apartment",
    "condo": "apartment", "home": "villa", "house": "villa", "bungalow": "villa",
    "penthouses": "penthouse", "townhouses": "townhouse"
}

# ─── build known_values once at startup ──────────────────────────────
# e.g. known_values["category"] = ["Apartment", "Villa", ...]
known_values = get_known_values()

# ─── improved normalize_to_payload (with fuzzy for NAME only) ─────────
import difflib

def normalize_to_payload(val: str, key: str) -> str:
    """
    Map a user‐supplied value (lowercased) to the exact payload string.
    Steps:
      1) apply any manual synonyms (e.g. 'apartments' → 'apartment')
      2) look for an exact lowercase match in known_values[key]
      3) if none, do a difflib fuzzy match on known_values[key]
      4) if still none, Title-case the input and return it
    """ 
    low = val.strip().lower()

    # 1) synonyms
    if low in synonyms:
        low = synonyms[low]

    # pull your list once
    candidates = known_values.get(key, [])        # e.g. ['Apartment','Villa',...]
    lower_cands = [c.lower() for c in candidates]

    # 2) exact lowercase
    if low in lower_cands:
        return candidates[lower_cands.index(low)]

    # 3) fuzzy match against all knowns
    match = difflib.get_close_matches(low, lower_cands, n=1, cutoff=0.6)
    if match:
        return candidates[lower_cands.index(match[0])]

    # 4) fallback
    return val.title()

# ─── enhanced extract_filters ────────────────────────────────────────
def extract_filters(text: str) -> dict:
    f = {}
    low = text.lower()

    # bedrooms: catch "3 br", "3 bhk"
    m = re.search(r"(\d+)\s*(?:bhk|br)\b", low)
    # if m:
    #     f["bedroom"] = int(m.group(1))
    if re.search(r"\bstudio\b", low):
        f["bedroom"] = "Studio"
    else:
        # 1) bedrooms: catch "3 br", "3 bhk"
        m = re.search(r"(\d+)\s*(?:bhk|br)\b", low)
        if m:
            f["bedroom"] = int(m.group(1))
    # price ranges
    m = re.search(r"under\s*([\d,]+)", low)
    if m:
        f["price_lt"] = float(m.group(1).replace(",", ""))
    m = re.search(r"above\s*([\d,]+)", low)
    if m:
        f["price_gt"] = float(m.group(1).replace(",", ""))
    
    m = re.search(r"under\s*([\d,]+)\s*(?:sq\.?\s*ft|sqft)\b", low)
    if m:
        f["size_sq_ft_lt"] = float(m.group(1).replace(",", ""))

    m = re.search(r"above\s*([\d,]+)\s*(?:sq\.?\s*ft|sqft)\b", low)
    if m:
        f["size_sq_ft_gt"] = float(m.group(1).replace(",", ""))

    # free-form fields
    for key in ("category", "city", "community_area", "developer", "property_type", "proeject_name", "project_name"):
        for cand in known_values.get(key, []):
            if re.search(rf"\b{re.escape(cand.lower())}\b", low):
                f[key] = cand
                break
        # also catch synonyms/plurals for category
        if key == "category" and "category" not in f:
            for syn, real in synonyms.items():
                if re.search(rf"\b{re.escape(syn)}\b", low):
                    f["category"] = real
                    break

    return f
def run_info_search(user_query: str, limit: int = 1) -> list[dict]:
    # 1) embed with the SAME model you used at ingestion
    vec = st_model.encode(user_query).tolist()

    # 2) search
    hits = qdrant.search(
        collection_name=INFO_COLLECTION,
        query_vector=vec,
        limit=limit,
        with_payload=True
    )
    return [h.payload for h in hits]

def run_qdrant_search(filters: dict, vector: list = None, limit: int = 20) -> list:
    conds = []

    # 1) category
    if "category" in filters:
        exact_cat = normalize_to_payload(filters["category"], "category")
        conds.append(FieldCondition(
            key="category",
            match=MatchValue(value=exact_cat)
        ))

    # 2) price ranges
    if "price_lt" in filters:
        conds.append(FieldCondition(
            key="price",
            range=QRange(lt=filters["price_lt"])
        ))
    if "price_gt" in filters:
        conds.append(FieldCondition(
            key="price",
            range=QRange(gt=filters["price_gt"])
        ))
    if "size_sq_ft_lt" in filters:
        conds.append(FieldCondition(
            key="size_sq_ft",
            range=QRange(lt=filters["size_sq_ft_lt"])
        ))
    if "size_sq_ft_gt" in filters:
        conds.append(FieldCondition(
            key="size_sq_ft",
            range=QRange(gt=filters["size_sq_ft_gt"])
        ))

    # 3) project_name / name (only if a real match)
    for name_key in ("project_name", "name"):
        if name_key in filters:
            exact = normalize_to_payload(filters[name_key], name_key)
            # if fuzzy didn't land on a known payload value, bail out
            if exact.lower() not in (v.lower() for v in known_values.get(name_key, [])):
                return []
            conds.append(FieldCondition(
                key=name_key,
                match=MatchValue(value=exact)
            ))

    # 4) other string fields (excluding category which we already did)
    for key in ("city", "community_area", "developer", "property_type"):
        if key in filters:
            exact = normalize_to_payload(filters[key], key)
            conds.append(FieldCondition(
                key=key,
                match=MatchValue(value=exact)
            ))

    # 5) bedroom
    if "bedroom" in filters:
        conds.append(FieldCondition(
            key="bedroom",
            match=MatchValue(value=str(filters["bedroom"]))
        ))

    if conds:
        all_pts, offset = [], None
        qf = QFilter(must=conds)
        while True:
            batch, offset = qdrant.scroll(
                collection_name=COLLECTION,
                scroll_filter=qf,
                with_payload=True,
                with_vectors=False,
                limit=500,
                offset=offset
            )
            all_pts.extend(batch)
            if offset is None:
                break

        # now that we've collected every point, pull out the payloads
        all_payloads = [pt.payload for pt in all_pts]

        return {
            "display_results": all_payloads[:limit],   # first 20
            "all_results":       all_payloads          # everything
        }
    else:
        return {
            "display_results": [],   # first 20
            "all_results":       []          # everything
        }
        

    # 7) semantic fallback
    # if vector:
    #     hits = qdrant.search(
    #         collection_name=COLLECTION,
    #         query_vector=vector,
    #         limit=limit,
    #         with_payload=True
    #     )
    #     display = [h.payload for h in hits]
    #     # in vector mode we don't scroll the entire set, so just treat these as both
    #     return {"display_results": display, "all_results": display}

    # return {"display_results": [], "all_results": []}


# ─── SUMMARIZATION ───────────────────────────────────────────────────
from typing import List


def generate_reply(
    results: List[dict] = None,
    user_query: str = "",
    extra_system: str = ""
) -> str:
    """
    Build a prompt from the user_query + any search results + extra_system instructions,
    then ask GPT to reply in a warm, conversational tone.
    """
    # 1) Summarize the facts
    fact_line = ""
    if results is None:
        fact_line = ""
    elif not results:
        fact_line = "No matches found."
    else:
        fact_line = f"Found {len(results)} properties."

    # 2) Build the assistant prompt
    system_msg = (


        f"{extra_system}\n"
        f"User asked: '{user_query}'. {fact_line}\n"

        "Always reply like a friendly, conversational human. "
        "You work for APIL Properties, a real estate brokerage company based in Dubai established in 2014. "
        "Use natural tone, be emotionally engaging, and guide the user."
        "if the price is ""None"" then it's empty and don't give those options"
        "this all properties are in dubai, so response according to dubai"
        "if anyone ask for personal number of anyone , you have to always refer to APIL contact details and it "
        "Use fuzzy LIKE matching on address or community_area_name when location is mentioned."
        "You must infer '2 bhk', '3bhk', '4 bedroom' → `bedroom`, and 'apartment', 'villa', 'penthouse' → `category`. "
        "Correct spelling automatically using fuzzy logic (like 'emmar' → 'emaar'). "
        "Always reply like a friendly, conversational human. Use a natural tone, be emotionally engaging, and guide the user. "
        "please give answer as short and sweet as you can give"
        "don't search any of property by yourself, showing properties to user by only our database."
        "please give shortest meaning full answer"
        "don't give any property detail which is not in our database."
        "here is contact info for apil properties, if user asked for contact info, then provide him this : 📞 +971585112282 💬 +971585112282 📧 inquiry@apilproperties.com  🏢 Supreme Court Complex — 5 — 1st Floor, Office No: 112 Riyadh St — Umm Hurair 2, Umm Hurair I , Dubai , UAE"    

"I want to train     you that if a user asks about a 1-year legal notice in Dubai, Abu Dhabi, or Sharjah, you need to explain in which circumstances a landlord can give a 1-year legal notice to a tenant to vacate the property. The answer must be short, human-like, and to the point, including only the most important legal information. Furthermore, always recommend working with legal professionals or property consultants when dealing with such situations, and mention that their service fees may apply. Always refer to APIL Properties as the source of guidance—not anyone else. Advise users to confirm current costs with responsible entities or professionals for accuracy."
"please answer of freehold and leasehold is : Freehold areas allow foreigners to fully own the property and land. Leasehold areas offer long-term leases (usually 30–99 years) but ownership of the land stays with the local landlord."

    )
    # 3) Add into history and ask GPT
    
    add_to_chat_history("system",  system_msg)
    resp = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=chat_history
    )
    answer = resp.choices[0].message.content.strip()
    add_to_chat_history("assistant",answer)
    return answer



def summarize(results: list, user_query: str) -> str:
    msg = (
        f"User asked: '{user_query}'. "
        
         +   ("No matches found." if not results else f"Found {len(results)} properties." )
        + " Reply in 1–2 friendly sentences."
        +"You are APIL Properties, a warm, helpful real estate assistant for Dubai properties. apil properties established in 2014."
        +"Always reply like a friendly, conversational human. Use a natural tone, be emotionally engaging, and guide the user. "
         +   "You work for APIL Properties , a real estate brokerage company based in Dubai. "
          +  "If a user asks what APIL Properties is, respond with: 'APIL Properties is a real estate brokerage company, based in Dubai.'"
           +             "Match synonymous terms like: 'house' → 'villa', 'condo' → 'apartment', 'flat' → 'apartment', 'home' → 'villa', 'bungalow' → 'villa'. "
           + "you're trained for only dubai's real estate , so please answer of questions related real estate, properties in dubai and everything about the properties"

    )
    add_to_chat_history("user",msg)
    resp = openai.ChatCompletion.create(model="gpt-4-0613", messages=chat_history)
    ans = resp.choices[0].message.content.strip()
    add_to_chat_history("assistant",ans)
    return ans

def ask_show_more() -> bool:
    """Prompt the user in a natural tone and interpret a range of yes/no replies."""
    affirmatives = {"yes","y","sure","ok","okay","please do","go on","show me more",
                    "absolutely","of course","yeah","yep","sure thing","why not"}
    negatives    = {"no","n","no thanks","not now","maybe later","nah","nope","don't","do not"}
    while True:
        resp = input("Would you like to see more listings? ").strip().lower()
        # exact match
        if resp in affirmatives:
            return True
        if resp in negatives:
            return False
        # contains
        if any(word in resp for word in affirmatives):
            return True
        if any(word in resp for word in negatives):
            return False
        # fallback
        print("Sorry, I didn’t catch that—just let me know if you’d like to see more or not.")




# ─── ADD AT TOP, ALONGSIDE classify_intent ────────────────────────────



COLLECTION1 = "info_collection"
model1 = SentenceTransformer("all-MiniLM-L6-v2")
def search_info_collection(query: str, limit: int = 3):
    vector = model1.encode(query).tolist()
    results = qdrant.search(
        collection_name=COLLECTION1,
        query_vector=vector,
        limit=limit,
        with_payload=True
    )
    return results
# —————————————————————————————————————————————————————————————
# 1) At the top of your file, **alongside** intent_functions, add:
# — get_property_detail spec with optional filters ————————
get_property_detail_spec = {
    "name": "get_property_detail",
    "description": "Fetch a single field (price, size_sq_ft, bedroom, description, etc.) for a previously returned property",
    "parameters": {
        "type": "object",
        "properties": {
            "project_name": {"type": "string", "description": "Exact project_name of the property"},
            "field":        {"type": "string", "description": "Which field to retrieve (e.g. price, size_sq_ft, bedroom, description)"}
        },
        "required": ["project_name", "field"]
    }
}

all_function_specs = intent_functions + [get_property_detail_spec]

async def main():
    last_results = []  # will hold payloads from the most recent search_properties
    print("🤖 Amlak: Hi! I'm Amlak, Ask me about properties or type 'exit' to quit.")

    while True:
        user = input("You: ").strip()
        if user.lower() in ("exit", "quit"):
            print("Amlak: Goodbye! 👋")
            break

        # 1) Append user message
        add_to_chat_history("user", user)

        # 2) Single GPT call for either intent or detail
        resp = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=chat_history,
            functions=all_function_specs,
            function_call="auto"
        )
        msg = resp.choices[0].message


        if getattr(msg, "function_call", None) and msg.function_call.name == "get_property_detail":
            args    = json.loads(msg.function_call.arguments)
            project = args["project_name"].lower()
            field   = args["field"]

            # fuzzy‐match project name
            from difflib import get_close_matches
            names = [p.get("project_name","") for p in last_results if p.get("project_name")]
            match = get_close_matches(project, names, n=1, cutoff=0.7)
            prop = next((p for p in last_results if p.get("project_name","") == match[0]), None) if match else None

            if not prop:
                print("Amlak: Sorry, I couldn’t find that property—please check the name.")
                continue

            value = prop.get(field)
            if value is None:
                print(f"Amlak: There’s no '{field}' field for {match[0]}.")
                continue

            # If they're asking for the 'description' field, humanize it:
            if field == "description":
                # 1) strip HTML tags
                clean = re.sub(r"<[^>]+>", "", value).strip()

                # 2) ask GPT to rewrite it warmly
                humanized = openai.ChatCompletion.create(
                    model="gpt-4-0613",
                    messages=[
                        {"role":"system","content":"You are a warm, conversational real-estate assistant."},
                        {"role":"user","content":
                            "Here’s a property description. Please rewrite it in friendly, humanized language, 2–3 sentences:\n\n"
                            + clean
                        }
                    ],
                    temperature=0.7
                ).choices[0].message.content.strip()

                print(f"Amlak: {humanized}")
            else:
                # numeric or text fields—just echo
                print(f"Amlak: The {field.replace('_',' ')} of '{prop['project_name']}' is {value}.")
            continue

        

        # 3) If GPT chose to fetch a property detail:
        if msg.get("function_call") and msg.function_call.name == "get_property_detail":
            args  = json.loads(msg.function_call.arguments)
            proj  = args["project_name"].lower()
            field = args["field"]
            br    = args.get("bedroom")
            cat   = args.get("category", "").lower() if args.get("category") else None

            # filter out only valid-named entries
            matches = [
                p for p in last_results
                if isinstance(p.get("project_name"), str) and p["project_name"].lower() == proj
            ]
            if br is not None:
                matches = [p for p in matches if p.get("bedroom") == br]
            if cat:
                matches = [p for p in matches
                           if isinstance(p.get("category"), str)
                           and p["category"].lower() == cat]

            if not matches:
                print("Amlak: I couldn’t find a match with those details—please check the name or filters.")
            else:
                prop = matches[0]
                if field in prop and prop[field] is not None:
                    print(f"Amlak: The {field.replace('_',' ')} of '{prop['project_name']}' is {prop[field]}.")
                else:
                    print("Amlak: I found that property, but that field isn’t available—sorry!")
            continue
        

        # 4) If GPT returned plain text (“small‐talk” / loans / etc.)
        if msg.content:
            add_to_chat_history("assistant", msg.content)
            print("Amlak:", msg.content.strip())
            continue

        # 5) Otherwise, it’s classify_intent:
        intent = json.loads(msg.function_call.arguments)["intent"]
        print("intent:", intent)

        # ─── Route your existing intents ─────────────────────────────────

        if intent == "search_properties":
            # extract filters & run Qdrant
            filters = extract_filters_with_gpt(user)
            if filters.get("category","").lower() == "studio":
                filters.pop("category"); filters["bedroom"] = "Studio"
            if filters.get("bedroom") == 0:
                filters["bedroom"] = "Studio"

            print(filters)

            emb    = openai.Embedding.create(model="text-embedding-ada-002", input=user)
            vector = emb["data"][0]["embedding"]
            out    = run_qdrant_search(filters, vector=vector, limit=20)
            all_matches = out["all_results"]

            last_results = all_matches  # store for detail lookups

            # paginate & display
            idx = 0
            while idx < len(all_matches):
                chunk = all_matches[idx:idx+20]
                print("\n--- Listings ---")
                for i,p in enumerate(chunk, start=idx+1):
                    print(f"{i}) {p['project_name']} | {p['bedroom']} BHK | "
                          f"{p['category']} | AED {p['price']} | SIZE {p.get('size_sq_ft','N/A')}")
                idx += 20
                if idx < len(all_matches) and ask_show_more():
                    continue
                break

            # hand off to GPT for summary
            # 4) hand off to GPT only for a humanized summary
            props = all_matches[:idx]
            bullet_list = "\n".join(
                    f"- {p['project_name']}: AED {p['price'] or 'N/A'}, "
                    f"{p['bedroom']} BHK, {p['category']}, {p.get('size_sq_ft','N/A')} sqft"
                    for p in props
                )

# Call GPT with a minimal system prompt:
            summary = openai.ChatCompletion.create(
                model="gpt-4-0613",
                messages=[
                    {"role": "system", "content":
                        "You are a friendly Dubai real-estate assistant. "
                        "Below are the only villas the user asked for. "
                        "Write a very short, human-sounding reply that summarizes them and asks which one they'd like more info on. "
                        "Do NOT add any other properties, do NOT include contact info, and do NOT apologize or fallback."
                        "please answer in 50 words"
                    },
                    {"role": "system", "content": bullet_list},
                    {"role": "user", "content": f"I want details on 2 BHK villas. Here are the results: {bullet_list}"}
                ],
                temperature=0.7
            )

            print("Amlak:", summary.choices[0].message.content.strip())


      
        elif intent == "about_info":
            docs = run_info_search(user, limit=1)
            if docs and "apil_properties_detail" in docs[0]:
                detail = docs[0]["apil_properties_detail"]
                first  = detail.get("property_first_description", "")
                second = detail.get("property_second_description", "")
                raw = f"{first}\n\n{second}".strip()
                # strip any HTML
                clean = re.sub(r"<[^>]+>", "", raw).strip()

                # ask GPT for a warm rewrite
                humanized = openai.ChatCompletion.create(
                    model="gpt-4-0613",
                    messages=[
                        {"role":"system","content":"You are a warm, conversational real-estate assistant."},
                        {"role":"user","content":
                            "Here’s some about-us content. Please rewrite it in friendly, humanized language, 2–3 sentences:\n\n"
                            + clean
                        }
                    ],
                    temperature=0.7
                ).choices[0].message.content.strip()

                print("Amlak:", humanized, "\n")
            else:
                # ultimate fallback
                print("Amlak: APIL Properties is a Dubai-based real estate brokerage established in 2014.\n")
        elif intent == "contact_info":
            docs = run_info_search(user, limit=1)
            if docs:
                c = docs[0]
                contact = (
                    "📞 +971585112282\n"
                    "💬 +971585112282\n"
                    "📧 inquiry@apilproperties.com\n"
                    "🏢 Supreme Court Complex — 5 — 1st Floor, Office No: 112 Riyadh St — Umm Hurair 2, Umm Hurair I , Dubai , UAE"
                )
            else:
                contact = "Sorry, I don’t have that info right now."
            print("Amlak:", generate_reply(None, user, contact), "\n")

        else:
            # greeting, help, company_info, fallback
            print("Amlak:", generate_reply(None, user, ""), "\n")

if __name__ == "__main__":
    asyncio.run(main())
