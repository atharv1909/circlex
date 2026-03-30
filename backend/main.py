"""
main.py — CircleX Backend API
Deploy to Render: https://render.com
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
import os
import math

# ─── Load model lazily (Render cold start optimization) ───
_embed_model = None
_embeddings  = None
_df          = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_resources():
    global _embed_model, _embeddings, _df
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model...")
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')

        _embeddings = np.load(os.path.join(BASE_DIR, "data", "embeddings.npy"))
        _df = pd.read_csv(os.path.join(BASE_DIR, "data", "companies.csv"))

        print(f"Loaded {len(_df)} listings, embeddings shape: {_embeddings.shape}")
    return _embed_model, _embeddings, _df

# ─── Emission factors ───
EMISSION_FACTORS = {
    "steel_scrap":    1.85,
    "aluminum_scrap": 9.20,
    "copper_wire":    3.80,
    "iron_powder":    1.60,
    "hdpe_pellets":   1.73,
    "pet_flakes":     2.15,
    "pp_granules":    1.96,
    "mixed_plastic":  1.50,
    "cardboard":      0.94,
    "kraft_paper":    0.86,
    "wood_chips":     0.72,
    "sawdust":        0.65,
    "pcb_scrap":      5.40,
    "used_lubricant": 2.85,
    "fly_ash":        0.48,
}

MATERIAL_LABELS = {
    "steel_scrap":    "Steel Scrap",
    "aluminum_scrap": "Aluminium Scrap",
    "copper_wire":    "Copper Wire Scrap",
    "iron_powder":    "Iron Powder",
    "hdpe_pellets":   "HDPE Plastic Pellets",
    "pet_flakes":     "PET Flakes",
    "pp_granules":    "PP Granules",
    "mixed_plastic":  "Mixed Plastic Scrap",
    "cardboard":      "Cardboard / OCC",
    "kraft_paper":    "Kraft Paper Waste",
    "wood_chips":     "Wood Chips",
    "sawdust":        "Sawdust",
    "pcb_scrap":      "PCB / E-Waste Scrap",
    "used_lubricant": "Used Lubricant Oil",
    "fly_ash":        "Fly Ash",
}

# ─── App setup ───
app = FastAPI(
    title="CircleX API",
    description="AI-powered industrial waste matchmaking",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request models ───
class MatchRequest(BaseModel):
    material_description: str
    quantity_kg: float
    city: str
    lat: Optional[float] = 20.5937
    lng: Optional[float] = 78.9629
    top_k: Optional[int] = 3

class ImpactRequest(BaseModel):
    material_type: str
    quantity_kg: float
    price_per_kg: Optional[float] = 50.0

# ─── Utility functions ───
def haversine(lat1, lng1, lat2, lng2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng/2)**2)
    return round(R * 2 * math.asin(math.sqrt(a)), 1)

# 🔥 NEW: multi-factor scoring
def compute_score(sim, dist, price, qty_available, qty_requested):
    dist_score = 1 / (1 + dist / 500)
    price_score = 1 / (1 + price / 100)
    qty_score = min(qty_available / (qty_requested + 1), 1)

    return (
        0.5 * sim +
        0.2 * dist_score +
        0.15 * price_score +
        0.15 * qty_score
    )

def do_match(query: str, top_k: int, query_lat: float, query_lng: float, qty_requested: float):
    from sklearn.metrics.pairwise import cosine_similarity
    model, embeddings, df = get_resources()

    query_emb = model.encode([query])
    scores    = cosine_similarity(query_emb, embeddings)[0]

    enriched_results = []
    seen = set()

    for idx in np.argsort(scores)[::-1]:
        row = df.iloc[idx]
        cid = row["company_id"]

        if cid in seen:
            continue
        seen.add(cid)

        dist = haversine(query_lat, query_lng, row["lat"], row["lng"])

        final_score = compute_score(
            sim=scores[idx],
            dist=dist,
            price=row["price_per_kg"],
            qty_available=row["quantity_kg"],
            qty_requested=qty_requested
        )

        enriched_results.append((final_score, idx))

    # sort by final score
    enriched_results.sort(reverse=True, key=lambda x: x[0])

    results = []
    for score, idx in enriched_results[:top_k]:
        row = df.iloc[idx]
        dist = haversine(query_lat, query_lng, row["lat"], row["lng"])

        results.append({
            "company_id":      row["company_id"],
            "company_name":    row["company_name"],
            "sector":          row["sector"],
            "city":            row["city"],
            "state":           row["state"],
            "lat":             float(row["lat"]),
            "lng":             float(row["lng"]),
            "material_type":   row["material_type"],
            "material_label":  row["material_label"],
            "quantity_kg":     int(row["quantity_kg"]),
            "price_per_kg":    float(row["price_per_kg"]),
            "emission_factor": float(row["emission_factor"]),
            "confidence_score": round(score * 100, 1),  # now real combined score
            "verified":        bool(row["verified"]),
            "rating":          float(row["rating"]),
            "distance_km":     dist,
        })

    return results

def do_impact(material_type: str, quantity_kg: float, price_per_kg: float):
    factor    = EMISSION_FACTORS.get(material_type, 1.2)
    co2_saved = quantity_kg * factor
    return {
        "material_label":          MATERIAL_LABELS.get(material_type, material_type),
        "quantity_kg":             quantity_kg,
        "co2_saved_kg":            round(co2_saved, 2),
        "co2_saved_tonnes":        round(co2_saved / 1000, 3),
        "trees_equivalent":        round(co2_saved / 21.77, 1),
        "car_km_avoided":          round(co2_saved / 0.192, 0),
        "landfill_cost_saved_inr": round(quantity_kg * 8.5, 0),
        "revenue_generated_inr":   round(quantity_kg * price_per_kg, 0),
    }

# ─── Routes ───
@app.get("/")
def root():
    return {"message": "CircleX API is live", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/match")
def match_materials(req: MatchRequest):
    if not req.material_description.strip():
        raise HTTPException(400, "material_description cannot be empty")
    if req.quantity_kg <= 0:
        raise HTTPException(400, "quantity_kg must be positive")

    try:
        matches = do_match(
            req.material_description,
            req.top_k,
            req.lat,
            req.lng,
            req.quantity_kg  # ✅ now used
        )

        return {
            "status":   "success",
            "query":    req.material_description,
            "city":     req.city,
            "quantity": req.quantity_kg,
            "matches":  matches,
            "count":    len(matches),
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/impact")
def get_impact(req: ImpactRequest):
    try:
        return do_impact(req.material_type, req.quantity_kg, req.price_per_kg)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/materials")
def list_materials():
    return [
        {"type": k, "label": v, "emission_factor": EMISSION_FACTORS[k]}
        for k, v in MATERIAL_LABELS.items()
    ]

@app.get("/stats")
def platform_stats():
    try:
        _, _, df = get_resources()
        total_co2 = float((df["quantity_kg"] * df["emission_factor"]).sum())
        return {
            "total_listings":      int(len(df)),
            "total_companies":     int(df["company_id"].nunique()),
            "total_surplus_kg":    float(df["quantity_kg"].sum()),
            "total_co2_potential": round(total_co2, 1),
            "total_value_inr":     float((df["quantity_kg"] * df["price_per_kg"]).sum()),
            "cities_covered":      int(df["city"].nunique()),
        }
    except Exception as e:
        raise HTTPException(500, str(e))
