import requests
import pandas as pd
import folium
import random
import math
import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull
import time

# ========================= 默认配置 =========================
DEFAULT_KEYWORDS = ["商场", "餐饮服务", "写字楼", "酒店", "超市"]
GRID_SIZE_KM = 5
LEVEL1_COUNT = (8, 12)
LEVEL2_PER_L1 = 4
MERGE_DISTANCE_KM = 2
MIN_DIST_TO_OTHER_L1_KM = 5

CITY = ""
API_KEY = ""
KEYWORDS_LIST = []


# ========================= 工具函数：城市中心回退 =========================
def geocode_city(city_name, api_key, timeout=10):
    """使用高德地理编码获取城市中心"""
    try:
        url = "https://restapi.amap.com/v3/geocode/geo"
        params = {"address": city_name, "key": api_key}
        r = requests.get(url, params=params, timeout=timeout)
        j = r.json()
        if j.get("status") == "1" and j.get("geocodes"):
            loc = j["geocodes"][0].get("location")
            if loc:
                lng, lat = map(float, loc.split(","))
                return lat, lng
    except:
        pass
    return None, None


# ========================= 经过重构的 POI 请求 =========================
def fetch_pois_for_keyword(keyword, max_pages=3, page_size=20, retries=3):
    """稳定版：一个关键词一个请求"""
    results = []

    for page in range(1, max_pages + 1):
        params = {
            "keywords": keyword,
            "city": CITY,
            "output": "json",
            "offset": page_size,
            "page": page,
            "key": API_KEY
        }

        success = False
        for attempt in range(retries):
            try:
                r = requests.get(
                    "https://restapi.amap.com/v3/place/text",
                    params=params,
                    timeout=25
                )
                data = r.json()
                success = True
                break
            except:
                time.sleep(1.5 * (attempt + 1))

        if not success:
            print(f"❌ {keyword} 第 {page} 页重试失败，跳过该页")
            continue

        if data.get("status") != "1" or not data.get("pois"):
            break

        for poi in data["pois"]:
            loc = poi.get("location")
            if loc:
                lng, lat = map(float, loc.split(","))
                results.append((lng, lat))

    return results


def fetch_all_pois():
    """新版流程：分关键词采集 + 合并"""
    all_results = []

    for kw in KEYWORDS_LIST:
        print(f"正在采集 {kw} ...")
        sub = fetch_pois_for_keyword(kw)
        all_results.extend(sub)

    if not all_results:
        return pd.DataFrame(columns=["lng", "lat"])

    df = pd.DataFrame(all_results, columns=["lng", "lat"])
    df.drop_duplicates(inplace=True)
    return df


# ========================= 计算辅助 =========================
def generate_grid(min_lng, max_lng, min_lat, max_lat, size_km):
    km_per_deg = 111
    step = size_km / km_per_deg
    lngs = np.arange(min_lng, max_lng, step)
    lats = np.arange(min_lat, max_lat, step)
    grid = []
    for lng in lngs:
        for lat in lats:
            grid.append({"lng": lng + step / 2, "lat": lat + step / 2})
    return pd.DataFrame(grid)


def calc_density(grid_df, poi_df, radius_km=3):
    km_per_deg = 111
    r = radius_km / km_per_deg
    density = []
    for _, g in grid_df.iterrows():
        count = ((abs(poi_df["lng"] - g["lng"]) < r) &
                 (abs(poi_df["lat"] - g["lat"]) < r)).sum()
        density.append(count)
    grid_df["density"] = density
    return grid_df


def merge_close_points(df, threshold_km):
    km_per_deg = 111
    th = threshold_km / km_per_deg

    merged, used = [], set()
    for i in range(len(df)):
        if i in used:
            continue

        cluster = [i]
        for j in range(i+1, len(df)):
            if j in used:
                continue
            d = math.dist(
                (df.iloc[i]["lng"], df.iloc[i]["lat"]),
                (df.iloc[j]["lng"], df.iloc[j]["lat"])
            )
            if d < th:
                cluster.append(j)
                used.add(j)

        used.add(i)
        subset = df.iloc[cluster]
        merged.append({
            "lng": subset["lng"].mean(),
            "lat": subset["lat"].mean(),
            "parents": subset[["parent_lng", "parent_lat"]].values.tolist()
        })

    return pd.DataFrame(merged)


def filter_near_other_primaries(level2_df, level1_sites, min_dist_km):
    km_per_deg = 111
    res = []
    for _, row in level2_df.iterrows():
        keep = True
        for _, p in level1_sites.iterrows():
            if [p["lng"], p["lat"]] not in row["parents"]:
                d_km = math.dist((row["lng"], row["lat"]), (p["lng"], p["lat"])) * km_per_deg
                if d_km < min_dist_km:
                    keep = False
                    break
        if keep:
            res.append(row)
    return pd.DataFrame(res)


# ========================= 主函数：run_ga =========================
def run_ga(city_name, api_key, keywords=DEFAULT_KEYWORDS):
    global CITY, API_KEY, KEYWORDS_LIST
    CITY = city_name
    API_KEY = api_key

    # keywords 可传字符串或列表
    if isinstance(keywords, str):
        KEYWORDS_LIST = [k.strip() for k in keywords.split(",") if k.strip()]
    else:
        KEYWORDS_LIST = list(keywords)

    # -------- 1. 获取 POI --------
    poi_df = fetch_all_pois()

    # 若 POI 为空，使用城市中心回退构造一个点
    if poi_df.empty:
        lat_c, lng_c = geocode_city(CITY, API_KEY)
        if lat_c is None:
            raise RuntimeError("GA 无法获取 POI 且地理编码也失败，请检查城市名或 API Key。")
        poi_df = pd.DataFrame([{"lng": lng_c, "lat": lat_c}])

    # -------- 2. 获取边界 --------
    min_lng, max_lng = poi_df["lng"].min(), poi_df["lng"].max()
    min_lat, max_lat = poi_df["lat"].min(), poi_df["lat"].max()

    # 防止 bbox 过窄导致凸包等出错
    if abs(max_lng - min_lng) < 1e-5:
        min_lng -= 0.01
        max_lng += 0.01
    if abs(max_lat - min_lat) < 1e-5:
        min_lat -= 0.01
        max_lat += 0.01

    grid_df = generate_grid(min_lng, max_lng, min_lat, max_lat, GRID_SIZE_KM)
    grid_df = calc_density(grid_df, poi_df)

    # -------- 3. 一 级 站 --------
    grid_sorted = grid_df.sort_values("density", ascending=False)
    num_l1 = random.randint(*LEVEL1_COUNT)
    level1_sites = grid_sorted.head(num_l1).copy()

    # -------- 4. 凸 包（稳健版）--------
    try:
        if len(level1_sites) >= 3:
            hull = ConvexHull(level1_sites[["lng", "lat"]])
            polygon = Polygon(level1_sites.iloc[hull.vertices][["lng", "lat"]].values).buffer(-0.02)
        else:
            # 回退 polygon = bbox
            polygon = Polygon([
                (min_lng - 0.02, min_lat - 0.02),
                (min_lng - 0.02, max_lat + 0.02),
                (max_lng + 0.02, max_lat + 0.02),
                (max_lng + 0.02, min_lat - 0.02)
            ])
    except:
        polygon = Polygon([
            (min_lng - 0.02, min_lat - 0.02),
            (min_lng - 0.02, max_lat + 0.02),
            (max_lng + 0.02, max_lat + 0.02),
            (max_lng + 0.02, min_lat - 0.02)
        ])

    # -------- 5. 二 级 站 --------
    l2 = []
    km_per_deg = 111

    for _, l1 in level1_sites.iterrows():
        for _ in range(LEVEL2_PER_L1):
            angle = random.uniform(0, 2 * math.pi)
            dist_deg = random.uniform(4, 10) / km_per_deg
            lng = l1["lng"] + dist_deg * math.cos(angle)
            lat = l1["lat"] + dist_deg * math.sin(angle)

            if not polygon.contains(Point(lng, lat)):
                l2.append({
                    "lng": lng, "lat": lat,
                    "parent_lng": l1["lng"], "parent_lat": l1["lat"]
                })

    l2_df = pd.DataFrame(l2)
    merged_l2 = merge_close_points(l2_df, MERGE_DISTANCE_KM) if not l2_df.empty else pd.DataFrame()
    final_l2 = filter_near_other_primaries(merged_l2, level1_sites, MIN_DIST_TO_OTHER_L1_KM) \
        if not merged_l2.empty else pd.DataFrame()

    # -------- 6. 地 图 中 心 --------
    lat_mean = float(poi_df["lat"].mean())
    lng_mean = float(poi_df["lng"].mean())

    if not (np.isfinite(lat_mean) and np.isfinite(lng_mean)):
        # 若失效，回退地理编码
        lat_geo, lng_geo = geocode_city(CITY, API_KEY)
        if lat_geo is None:
            raise RuntimeError("无法确定地图中心。")
        lat_mean, lng_mean = lat_geo, lng_geo

    map_center = [lat_mean, lng_mean]

    # -------- 7. 绘 地 图 --------
    m = folium.Map(
    location=map_center,
    zoom_start=12,
    tiles=None
)

# 作为底图，禁止 control 和 overlay
folium.TileLayer(
    tiles="https://webrd02.is.autonavi.com/appmaptile?style=7&x={x}&y={y}&z={z}",
    attr="高德地图(AMap)",
    name="AMap Base",
    overlay=False,
    control=False
).add_to(m)


    # POI
    for _, r in poi_df.sample(min(len(poi_df), 300)).iterrows():
        folium.CircleMarker([r["lat"], r["lng"]], radius=2,
                            color="#aaaaaa", fill=True, fill_opacity=0.4).add_to(m)

    # 一级站
    for _, r in level1_sites.iterrows():
        folium.CircleMarker([r["lat"], r["lng"]], radius=7,
                            color="red", fill=True).add_to(m)
        if len(level1_sites) >= 3:
            pts = level1_sites[["lng", "lat"]].values
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
    
    # 转换为 (lat, lng) 顺序
    path = [(p[1], p[0]) for p in hull_pts]
    path.append(path[0])  # 闭合
    
    folium.PolyLine(path, color="red", weight=3).add_to(m)

    # 二级站
    for _, r in final_l2.iterrows():
        folium.CircleMarker([r["lat"], r["lng"]], radius=5,
                            color="blue", fill=True).add_to(m)
        for parent in r["parents"]:
            folium.PolyLine([[r["lat"], r["lng"]], [parent[1], parent[0]]],
                            color="green", weight=1).add_to(m)

    info = {
        "城市": CITY,
        "一级站数量": len(level1_sites),
        "二级站数量": len(final_l2),
        "关键词": KEYWORDS_LIST
    }

    return m, info


