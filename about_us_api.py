import os
import uuid
import requests
import time
import schedule
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

def job():
    # ─── CONFIG ─────────────────────────────────────────────────────────
    OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "sk-proj-...")
    ABOUT_API         = "https://admin.apilproperties.com/api/about-apil"
    CONTACT_API       = "https://admin.apilproperties.com/api/contact-us"
    COLLECTION_NAME   = "info_collection"
    EMBED_MODEL_ID    = "all-MiniLM-L6-v2"
    QDRANT_HOST       = "localhost"
    QDRANT_PORT       = 6333

    # ─── INIT QDRANT ─────────────────────────────────────────────────────
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # ─── LOAD EMBEDDING MODEL ────────────────────────────────────────────
    model = SentenceTransformer(EMBED_MODEL_ID)

    points = []

    # ─── INGEST ABOUT US ─────────────────────────────────────────────────
    about_resp = requests.get(ABOUT_API)
    about_resp.raise_for_status()
    for item in about_resp.json().get("data", []):
        text = "\n\n".join([
            item["apil_properties_detail"]["property_title"],
            item["apil_properties_detail"]["property_first_description"],
            item["apil_properties_detail"]["property_second_description"],
            item["why_choose_us"]["choose_title"],
            item["why_choose_us"]["choose_description"],
            item["our_commitment"]["commitment_title"],
            item["our_commitment"]["commitment_description"]
        ])
        vector = model.encode(text).tolist()

        payload = {
            "id":               item["id"],
            "type":             "about",
            "property_title":   item["apil_properties_detail"]["property_title"],
            "first_desc":       item["apil_properties_detail"]["property_first_description"],
            "second_desc":      item["apil_properties_detail"]["property_second_description"],
            "why_choose_title": item["why_choose_us"]["choose_title"],
            "why_choose_desc":  item["why_choose_us"]["choose_description"],
            "commit_title":     item["our_commitment"]["commitment_title"],
            "commit_desc":      item["our_commitment"]["commitment_description"],
            "seo":              item.get("seo_data", {}),
            "raw":              item
        }

        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

    # ─── INGEST CONTACT US ────────────────────────────────────────────────
    contact_resp = requests.get(CONTACT_API)
    contact_resp.raise_for_status()
    for item in contact_resp.json().get("data", []):
        text = "\n\n".join([
            item["title"],
            item["description"],
            f"Email: {item['email']}",
            f"Phone: {item['phone_number']}",
            f"WhatsApp: {item['whatsapp_number']}",
            f"Address: {item['address']} ({item['city']}, {item['country']} {item['postal_code']})"
        ])
        vector = model.encode(text).tolist()

        payload = {
            "id":               item["id"],
            "type":             "contact",
            "title":            item["title"],
            "description":      item["description"],
            "email":            item["email"],
            "phone_number":     item["phone_number"],
            "whatsapp_number":  item["whatsapp_number"],
            "address":          item["address"],
            "city":             item["city"],
            "postal_code":      item["postal_code"],
            "country":          item["country"],
            "social_media":     item.get("social_media", []),
            "footer_description": item.get("footer_description"),
            "seo":              item.get("seo_data", {}),
            "raw":              item
        }

        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

    # ─── UPSERT TO QDRANT ────────────────────────────────────────────────
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Upserted {len(points)} documents into “{COLLECTION_NAME}”")
# def about_us_api():
#     # Run the job once at script start
#     job()

#     # Schedule to run every 12 hours
#     schedule.every(12).hours.do(job)

#     # Keep running
#     while True:
#         schedule.run_pending()
#         time.sleep(60)
