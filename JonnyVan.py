import requests
import pandas as pd
import folium
import random
import math
import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull


AMAP_TILE_URL = "https://webrd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scale=1&style=8"
KM_PER_DEG = 111.0


def fallback_fetch_pois(city, api_key, keywords, max_pages=8):
    all_pois = []
    for kw in keywords:
        pois = []
        for page in range(1, max_pages + 1):
            url = "https://restapi.amap.com/v3/place/text"
            params = {
                "keywords": kw,
                "city": city,
                "output": "json",
                "offset": 25,
                "page": page,
                "key": api_key,
            }
            try:
                r = requests.get(url, params=params, timeout=10)
                j = r.json()
            except:
                break
            if j.get("status") != "1" or not j.get("pois"):
                break
            for poi in j["pois"]:
                loc = poi.get("location")
                if loc:
                    lng, lat = map(float, loc.split(","))
                    pois.append((lng, lat))
        all_pois.extend(pois)
    df = pd.DataFrame(all_pois, columns=["lng", "lat"])
    return df
def fallback_generate_grid(min_lng, max_lng, min_lat, max_lat, size_km):
    step = size_km / KM_PER_DEG
    lngs = np.arange(min_lng, max_lng, step)
    lats = np.arange(min_lat, max_lat, step)

    grid = []
    for lng in lngs:
        for lat in lats:
            grid.append({
                "lng": float(lng + step / 2),
                "lat": float(lat + step / 2),
            })
    return pd.DataFrame(grid)
def fallback_density(grid_df, poi_df, radius_km=3):
    r = radius_km / KM_PER_DEG
    densities = []
    if poi_df.empty:
        return grid_df.assign(density=0)
    for _, g in grid_df.iterrows():
        count = (
            (abs(poi_df["lng"] - g["lng"]) < r)
            & (abs(poi_df["lat"] - g["lat"]) < r)
        ).sum()
        densities.append(int(count))

    grid_df["density"] = densities
    return grid_df
def fallback_merge_secondary(df, threshold_km):
    if df.empty:
        return pd.DataFrame(columns=["lng", "lat", "parents"])
    threshold_deg = threshold_km / KM_PER_DEG
    used = set()
    merged = []
    for i in range(len(df)):
        if i in used:
            continue
        cluster = [i]
        for j in range(i + 1, len(df)):
            if j in used:
                continue
            d = math.dist((df.iloc[i]["lng"], df.iloc[i]["lat"]),
                          (df.iloc[j]["lng"], df.iloc[j]["lat"]))
            if d < threshold_deg:
                cluster.append(j)
                used.add(j)
        used.add(i)
        pts = df.iloc[cluster]
        lng = float(pts["lng"].mean())
        lat = float(pts["lat"].mean())
        parents = pts[["parent_lng", "parent_lat"]].values.tolist()
        merged.append({"lng": lng, "lat": lat, "parents": parents})
    return pd.DataFrame(merged)
def fallback_filter_secondary(sec_df, primaries, min_dist_km):
    if sec_df.empty:
        return sec_df
    min_deg = min_dist_km / KM_PER_DEG
    filtered = []
    for _, s in sec_df.iterrows():
        ok = True
        for _, p in primaries.iterrows():
            if [p["lng"], p["lat"]] not in s["parents"]:
                d = math.dist((s["lng"], s["lat"]), (p["lng"], p["lat"]))
                if d < min_deg:
                    ok = False
                    break
        if ok:
            filtered.append(s)
    return pd.DataFrame(filtered) if filtered else pd.DataFrame(columns=sec_df.columns)


def run_ga(city, api_key):
    KEYWORDS = ["商场", "餐饮服务", "酒店", "超市", "写字楼", "地铁站"]
    pois = fallback_fetch_pois(city, api_key, KEYWORDS)
    if pois.empty:
        raise RuntimeError(f"{city}：无POI数据")
    min_lng, max_lng = pois["lng"].min(), pois["lng"].max()
    min_lat, max_lat = pois["lat"].min(), pois["lat"].max()
    grid = fallback_generate_grid(min_lng, max_lng, min_lat, max_lat, 5)
    grid = fallback_density(grid, pois)
    grid_sorted = grid.sort_values("density", ascending=False).reset_index(drop=True)
    L1_count = random.randint(10, 15)
    level1 = grid_sorted.head(L1_count).copy().reset_index(drop=True)
    filtered_L1 = []
    min_L1_deg = 5 / KM_PER_DEG
    for _, row in level1.iterrows():
        ok = True
        for p in filtered_L1:
            if math.dist((row["lng"], row["lat"]), (p["lng"], p["lat"])) < min_L1_deg:
                ok = False
                break
        if ok:
            filtered_L1.append(row)
    level1 = pd.DataFrame(filtered_L1)
    if len(level1) >= 3:
        hull = ConvexHull(level1[["lng", "lat"]])
        polygon = Polygon(level1.iloc[hull.vertices][["lng", "lat"]].values).buffer(-0.02)
    else:
        polygon = Polygon([
            (min_lng - 0.02, min_lat - 0.02),
            (min_lng - 0.02, max_lat + 0.02),
            (max_lng + 0.02, max_lat + 0.02),
            (max_lng + 0.02, min_lat - 0.02)
        ])
    L2_raw = []
    for _, l1 in level1.iterrows():
        for _ in range(5):
            angle = random.uniform(0, 2 * math.pi)
            dist_deg = random.uniform(5, 10) / KM_PER_DEG
            lng = l1["lng"] + dist_deg * math.cos(angle)
            lat = l1["lat"] + dist_deg * math.sin(angle)
            if not polygon.contains(Point(lng, lat)):
                L2_raw.append({
                    "lng": lng,
                    "lat": lat,
                    "parent_lng": l1["lng"],
                    "parent_lat": l1["lat"],
                })
    L2 = pd.DataFrame(L2_raw)
    L2 = fallback_merge_secondary(L2, 2)
    L2 = fallback_filter_secondary(L2, level1, 5)
    center = [pois["lat"].mean(), pois["lng"].mean()]
    m = folium.Map(location=center, zoom_start=11, tiles=None)
    folium.TileLayer(tiles=AMAP_TILE_URL, attr="高德地图").add_to(m)
    for _, r in pois.sample(min(400, len(pois))).iterrows():
        folium.CircleMarker([r["lat"], r["lng"]], radius=2, color="#999", fill=True, fill_opacity=0.3).add_to(m)
    for _, r in level1.iterrows():
        folium.CircleMarker([r["lat"], r["lng"]], radius=7, color="red", fill=True).add_to(m)
    for _, r in L2.iterrows():
        folium.CircleMarker([r["lat"], r["lng"]], radius=5, color="blue", fill=True).add_to(m)
        for parent in r["parents"]:
            folium.PolyLine([[r["lat"], r["lng"]], [parent[1], parent[0]]], color="green", weight=1).add_to(m)
    info = {
        "algo": "fallback_GA",
        "city": city,
        "poi_count": len(pois),
        "level1_count": len(level1),
        "level2_count": len(L2),
    }
    return m, info


