import requests
import pandas as pd
import folium
import random
import math
import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull
import time

# ========================= 配置 =========================
DEFAULT_KEYWORDS = ["商场", "餐饮服务", "写字楼", "酒店", "超市"]
GRID_SIZE_KM = 5
LEVEL1_COUNT = (8, 12)
LEVEL2_PER_L1 = 4
MERGE_DISTANCE_KM = 2
MIN_DIST_TO_OTHER_L1_KM = 5

# 动态参数
CITY = ""
API_KEY = ""
KEYWORDS_LIST = []   # 新结构：关键词列表

# ========================= 重构版：一个 keyword 一次采集 =========================

def fetch_pois_for_keyword(keyword, max_pages=3, page_size=20, retries=3):
    """单个关键词的 POI（稳定版）"""
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

                # 网络成功
                success = True
                break

            except Exception:
                # 重试前等待：指数退避
                time.sleep(1.5 * (attempt + 1))

        if not success:
            print(f"❌ {keyword} 第 {page} 页连续失败，跳过该页")
            continue

        # 没有 POI 时停止
        if data.get("status") != "1" or not data.get("pois"):
            break

        # 存入
        for poi in data["pois"]:
            loc = poi.get("location")
            if loc:
                lng, lat = map(float, loc.split(","))
                results.append((lng, lat))

    return results


def fetch_all_pois():
    """新版：分 keyword 采集，合并稳定高效"""
    all_results = []

    for kw in KEYWORDS_LIST:
        print(f"正在采集 {kw} ...")
        sublist = fetch_pois_for_keyword(kw)
        all_results.extend(sublist)

    if not all_results:
        return pd.DataFrame(columns=["lng", "lat"])

    df = pd.DataFrame(all_results, columns=["lng", "lat"])
    df.drop_duplicates(inplace=True)
    return df


# ========================= 工具函数（不变） =========================

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

    merged = []
    used = set()
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
            parent_list = row["parents"]

            # p 不是这个二级站的父节点
            if [p["lng"], p["lat"]] not in parent_list:
                d_km = math.dist((row["lng"], row["lat"]), (p["lng"], p["lat"])) * km_per_deg
                if d_km < min_dist_km:
                    keep = False
                    break
        if keep:
            res.append(row)
    return pd.DataFrame(res)


# ========================= 主入口（兼容你的 Streamlit） =========================

def run_ga(city_name, api_key, keywords=DEFAULT_KEYWORDS):
    global CITY, API_KEY, KEYWORDS_LIST
    CITY = city_name
    API_KEY = api_key
    KEYWORDS_LIST = keywords  # 新版：关键词为列表

    # -------- 1. 获取 POI --------
    poi_df = fetch_all_pois()
    if poi_df.empty:
        raise RuntimeError("未抓到任何 POI，请检查 API Key 或关键词")

    # -------- 2. 生成网格 --------
    min_lng, max_lng = poi_df["lng"].min(), poi_df["lng"].max()
    min_lat, max_lat = poi_df["lat"].min(), poi_df["lat"].max()

    grid_df = generate_grid(min_lng, max_lng, min_lat, max_lat, GRID_SIZE_KM)
    grid_df = calc_density(grid_df, poi_df)

    # -------- 3. 一级站 = 密度最高网格 --------
    grid_sorted = grid_df.sort_values("density", ascending=False)
    num_l1 = random.randint(*LEVEL1_COUNT)
    level1_sites = grid_sorted.head(num_l1).copy()

    # -------- 4. 凸包 --------
    hull = ConvexHull(level1_sites[["lng", "lat"]])
    poly = Polygon(level1_sites.iloc[hull.vertices][["lng", "lat"]].values).buffer(-0.02)

    # -------- 5. 二级站生成 --------
    l2 = []
    km_per_deg = 111

    for _, l1 in level1_sites.iterrows():
        for _ in range(LEVEL2_PER_L1):
            angle = random.uniform(0, 2*math.pi)
            dist_deg = random.uniform(4, 10) / km_per_deg

            lng = l1["lng"] + dist_deg * math.cos(angle)
            lat = l1["lat"] + dist_deg * math.sin(angle)

            if not poly.contains(Point(lng, lat)):
                l2.append({
                    "lng": lng, "lat": lat,
                    "parent_lng": l1["lng"], "parent_lat": l1["lat"]
                })

    l2_df = pd.DataFrame(l2)

    merged_l2 = merge_close_points(l2_df, MERGE_DISTANCE_KM)
    final_l2 = filter_near_other_primaries(merged_l2, level1_sites, MIN_DIST_TO_OTHER_L1_KM)

    # -------- 6. 绘图 --------
    m = folium.Map(location=[poi_df["lat"].mean(), poi_df["lng"].mean()], zoom_start=11)

    # 画POI
    for _, r in poi_df.sample(min(len(poi_df), 300)).iterrows():
        folium.CircleMarker([r["lat"], r["lng"]], radius=2,
                            color="#aaaaaa", fill=True, fill_opacity=0.4).add_to(m)

    # 一级
    for _, r in level1_sites.iterrows():
        folium.CircleMarker([r["lat"], r["lng"]],
                            radius=7, color="red", fill=True).add_to(m)

    # 二级
    for _, r in final_l2.iterrows():
        folium.CircleMarker([r["lat"], r["lng"]],
                            radius=5, color="blue", fill=True).add_to(m)
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
