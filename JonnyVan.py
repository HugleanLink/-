import requests
import pandas as pd
import folium
import random
import math
import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull

KM_PER_DEG = 111.0
AMAP_TILE_URL = "https://webrd02.is.autonavi.com/appmaptile?style=7&x={x}&y={y}&z={z}"

# ------------------ POI 采集 ------------------
def fetch_pois(city, api_key, keywords):
    all_pois = []
    for kw in keywords:
        for page in range(1, 8):
            url = "https://restapi.amap.com/v3/place/text"
            params = {
                "keywords": kw,
                "city": city,
                "output": "json",
                "offset": 25,
                "page": page,
                "key": api_key
            }
            r = requests.get(url, params=params)
            data = r.json()

            if data.get("status") != "1":
                break

            pois = data.get("pois", [])
            if not pois:
                break

            for p in pois:
                try:
                    lng, lat = map(float, p["location"].split(","))
                    all_pois.append((lng, lat))
                except:
                    continue

    df = pd.DataFrame(all_pois, columns=["lng", "lat"])
    return df


# ------------------ 网格 + 密度 ------------------
def generate_grid(min_lng, max_lng, min_lat, max_lat, size_km):
    step = size_km / KM_PER_DEG
    lngs = np.arange(min_lng, max_lng, step)
    lats = np.arange(min_lat, max_lat, step)
    grid = []

    for lng in lngs:
        for lat in lats:
            grid.append({"lng": lng + step / 2, "lat": lat + step / 2})

    return pd.DataFrame(grid)


def compute_density(grid, pois, radius_km=3):
    r_deg = radius_km / KM_PER_DEG
    densities = []

    for _, g in grid.iterrows():
        count = ((abs(pois["lng"] - g["lng"]) < r_deg) &
                 (abs(pois["lat"] - g["lat"]) < r_deg)).sum()
        densities.append(count)

    grid["density"] = densities
    return grid


# ------------------ 二级站过滤 ------------------
def merge_secondary(df, threshold_km):
    thr = threshold_km / KM_PER_DEG
    if df.empty:
        return df

    merged = []
    used = set()

    for i in range(len(df)):
        if i in used:
            continue
        cluster = [i]
        for j in range(i + 1, len(df)):
            if j in used:
                continue
            if math.dist((df.iloc[i]["lng"], df.iloc[i]["lat"]),
                         (df.iloc[j]["lng"], df.iloc[j]["lat"])) < thr:
                cluster.append(j)
                used.add(j)
        used.add(i)

        pts = df.iloc[cluster]
        merged.append({
            "lng": pts["lng"].mean(),
            "lat": pts["lat"].mean(),
            "parents": pts[["parent_lng", "parent_lat"]].values.tolist()
        })

    return pd.DataFrame(merged)


def filter_secondary(df, level1, min_dist_km):
    if df.empty:
        return df

    minn = min_dist_km
    kept = []

    for _, r in df.iterrows():
        ok = True
        for _, p in level1.iterrows():
            d = math.dist((r["lng"], r["lat"]), (p["lng"], p["lat"])) * KM_PER_DEG
            if d < minn:
                ok = False
                break
        if ok:
            kept.append(r)

    return pd.DataFrame(kept)


# ===========================================================
#                     主函数（供 Streamlit 调用）
# ===========================================================
def run_ga(city, api_key):
    keywords = ["商场", "酒店", "超市", "写字楼", "餐饮服务", "地铁站"]

    pois = fetch_pois(city, api_key, keywords)
    if pois.empty:
        raise RuntimeError(f"{city} 无 POI 数据")

    # 边界
    min_lng, max_lng = pois["lng"].min(), pois["lng"].max()
    min_lat, max_lat = pois["lat"].min(), pois["lat"].max()

    # 网格
    grid = generate_grid(min_lng, max_lng, min_lat, max_lat, 5)
    grid = compute_density(grid, pois)

    # 一级站
    grid_sorted = grid.sort_values("density", ascending=False).reset_index(drop=True)
    N1 = random.randint(10, 15)
    level1 = grid_sorted.head(N1).copy().reset_index(drop=True)

    # 一级去重
    filtered = []
    min_deg = 5 / KM_PER_DEG
    for _, r in level1.iterrows():
        ok = True
        for p in filtered:
            if math.dist((r["lng"], r["lat"]), (p["lng"], p["lat"])) < min_deg:
                ok = False
                break
        if ok:
            filtered.append(r)
    level1 = pd.DataFrame(filtered)

    # 凸包
    if len(level1) >= 3:
        hull = ConvexHull(level1[["lng", "lat"]])
        polygon = Polygon(level1.iloc[hull.vertices][["lng", "lat"]].values).buffer(-0.02)
    else:
        polygon = Polygon([(min_lng, min_lat), (min_lng, max_lat), (max_lng, max_lat), (max_lng, min_lat)])

    # 二级站
    sec_raw = []
    for _, p in level1.iterrows():
        for _ in range(5):
            ang = random.uniform(0, 2 * math.pi)
            dist = random.uniform(5, 10) / KM_PER_DEG
            lng = p["lng"] + dist * math.cos(ang)
            lat = p["lat"] + dist * math.sin(ang)
            if not polygon.contains(Point(lng, lat)):
                sec_raw.append({
                    "lng": lng,
                    "lat": lat,
                    "parent_lng": p["lng"],
                    "parent_lat": p["lat"]
                })

    L2 = pd.DataFrame(sec_raw)
    L2 = merge_secondary(L2, 2)
    L2 = filter_secondary(L2, level1, 5)

    # 地图
    center = [pois["lat"].mean(), pois["lng"].mean()]
    m = folium.Map(location=center, zoom_start=11, tiles=None)
    folium.TileLayer(tiles=AMAP_TILE_URL, attr="高德地图").add_to(m)

    for _, r in pois.sample(min(400, len(pois))).iterrows():
        folium.CircleMarker([r["lat"], r["lng"]], radius=2, color="#999", fill=True).add_to(m)

    for _, r in level1.iterrows():
        folium.CircleMarker([r["lat"], r["lng"]], radius=7, color="red", fill=True).add_to(m)

    for _, r in L2.iterrows():
        folium.CircleMarker([r["lat"], r["lng"]], radius=5, color="blue", fill=True).add_to(m)
        for p in r["parents"]:
            folium.PolyLine([[r["lat"], r["lng"]], [p[1], p[0]]], color="green", weight=1).add_to(m)

    info = {
        "algorithm": "GENETIC_APPROX",
        "poi_count": len(pois),
        "level1": len(level1),
        "level2": len(L2),
    }

    return m, info, level1, L2, pois
