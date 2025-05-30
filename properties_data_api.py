import requests
import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer
import uuid
import schedule
import time

def job():
    qdrant = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')

    collection_name = "properties_collection"

    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    page = 1
    while True:
        url = "https://admin.apilproperties.com/api/properties"
        response = requests.get(url, params={"page": page})

        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            break

        items = response.json().get("data", [])
        if not items:
            print("✅ All data fetched and pushed to Qdrant.")
            break

        points = []
        for item in items:
            try:
                text = f"{item['name']} - {item['description']} - {item['address']}"
                vector = model.encode(text).tolist()

                payload = {
                    "id": item["id"],
                    "name": item["name"],
                    "slug": item["slug"],
                    "project_name": item["project_name"],
                    "description": item["description"],
                    "address": item["address"],
                    "latitude": item["latitude"],
                    "longitude": item["longitude"],
                    "permit_number": item["permit_number"],
                    "unit_reference": item["unit_reference"],
                    "added_on_date": item["added_on_date"],
                    "price": item["price"],
                    "size_sq_ft": item["size_sq_ft"],
                    "no_of_parking": item["no_of_parking"],
                    "no_of_bathroom": item["no_of_bathroom"],
                    "status": item["status"],
                    "luxury": item["luxury"],
                    "ask_for_price": item["ask_for_price"],
                    "ask_for_size": item["ask_for_size"],
                    "virtual_tour_url": item.get("virtual_tour_url"),
                    "brochure": item.get("brochure"),
                    "bedroom": item["bedroom"],
                    "category": item["category"],
                    "city": item["city"],
                    "district": item["district"],
                    "community_area": item["community_area"],
                    "sub_community": item.get("sub_community"),
                    "developer": item["developer"],
                    "property_type": item["property_type"],
                    "meta_title": item["meta_title"],
                    "meta_description": item["meta_description"],
                    "meta_keyword": item["meta_keyword"],
                    "canonical_url": item["canonical_url"],
                    "floor_plan_image": item.get("floor_plan_image"),
                    "floor_plan_description": item.get("floor_plan_description"),
                    "video_id": item.get("video_id"),
                    "video_title": item.get("video_title"),
                    "video_description": item.get("video_description"),
                    "video_image": item.get("video_image"),
                    "video_img_alt": item.get("video_img_alt"),
                    "video_uploade_date": item.get("video_uploade_date"),
                    "video_duration": item.get("video_duration"),
                    "highlights": json.dumps(item.get("highlights", [])),
                    "features_and_amenities": json.dumps(item.get("feature_and_amenities", {})),
                    "payment_plans": json.dumps(item.get("payment_plans", []))
                }

                points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

            except Exception as e:
                print(f"⚠️ Skipped one item due to error: {e}")
                continue

        qdrant.upsert(collection_name=collection_name, points=points)
        print(f"✅ Page {page} inserted ({len(points)} items)")
        page += 1
# def properties_data_upload():
#     schedule.every(12).hours.do(job)

#     job()

#     while True:
#         schedule.run_pending()
#         time.sleep(60)

