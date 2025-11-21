# JonnyVan.py

import random
import math
import time
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import requests
import folium
from shapely.geometry import Point, Polygon, MultiPoint, mapping
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans

# ------------------------
# 常量
# ------------------------
DEFAULT_KEYWORDS = ["商场", "餐饮服务", "写字楼", "酒店", "超市"]
AMAP_TILE_URL = "https://webrd02.is.autonavi.com/appmaptile?style=7&x={x}&y={y}&z={z}"
KM_PER_DEG = 111.0


# ------------------------
# 工具函数
# ------------------------
def _meters_to_deg_lat(m):
    return m / 111000.0

def _meters_to_deg_lng(m, lat):
    return m / (111000.0 * math.cos(math.radians(lat)) + 1e-12)

def _haversine_km_array(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371.0
    lat1r = math.radians(lat1)
    lon1r = math.radians(lon1)
    lat2r = np.radians(lat2_arr)
    lon2r = np.radians(lon2_arr)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def fetch_pois_amap(city: str, api_key: str, keywords=DEFAULT_KEYWORDS,
                    page_size=25, max_pages=3, timeout=15, retries=3):

    all_pois = []
    kws = keywords if isinstance(keywords, (list, tuple)) else [keywords]

    for kw in kws:
        for page in range(1, max_pages + 1):
            url = "https://restapi.amap.com/v3/place/text"
            params = {
                "keywords": kw,
                "city": city,
                "output": "json",
                "offset": page_size,
                "page": page,
                "key": api_key,
            }

            success = False
            for attempt in range(retries):
                try:
                    r = requests.get(url, params=params, timeout=timeout)
                    j = r.json()
                    success = True
                    break
                except Exception:
                    time.sleep(1.2 * (attempt + 1))

            if not success or not j or j.get("status") != "1" or not j.get("pois"):
                break

            for poi in j.get("pois", []):
                loc = poi.get("location")
                if not loc:
                    continue
                try:
                    lng, lat = map(float, loc.split(","))
                except Exception:
                    continue

                all_pois.append({
                    "lng": lng,
                    "lat": lat,
                    "name": poi.get("name", ""),
                    "category": kw,
                })

    df = pd.DataFrame(all_pois)
    if not df.empty:
        df = df.drop_duplicates(subset=["lng", "lat", "name"])
    return df


def generate_candidate_grid(min_lng, max_lng, min_lat, max_lat, size_km=5.0):
    step_lat = size_km / KM_PER_DEG
    center_lat = (min_lat + max_lat) / 2.0
    step_lng = size_km / (KM_PER_DEG * math.cos(math.radians(center_lat)) + 1e-12)

    lngs = np.arange(min_lng, max_lng + 1e-9, step_lng)
    lats = np.arange(min_lat, max_lat + 1e-9, step_lat)

    pts = []
    for lng in lngs:
        for lat in lats:
            pts.append({
                "lng": float(lng + step_lng / 2.0),
                "lat": float(lat + step_lat / 2.0),
            })

    return pd.DataFrame(pts)


# ================================
#           GA Planner
# ================================
class GAPlanner:
    def __init__(self,
                 city: str,
                 api_key: str,
                 keywords=DEFAULT_KEYWORDS,
                 grid_size_km: float = 5.0,
                 num_primaries: int = 10,
                 city_center: Tuple[float, float] = None,
                 verbose: bool = False,
                 seed: Optional[int] = None):

        self.city = city
        self.api_key = api_key
        self.keywords = keywords
        self.grid_size_km = grid_size_km
        self.num_primaries = num_primaries
        self.city_center = city_center
        self.verbose = verbose
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.pois = pd.DataFrame()
        self.candidates = pd.DataFrame()
        self.cand_coords = None
        self.bounds = None

    # ------------------------
    # 预处理（获取 POI + 网格）
    # ------------------------
    def prepare(self, page_size=25, max_pages=3):
        if self.verbose:
            print(f"[GAPlanner] fetching POIs for {self.city} ...")

        self.pois = fetch_pois_amap(
            self.city,
            self.api_key,
            keywords=self.keywords,
            page_size=page_size,
            max_pages=max_pages,
        )

        # 如果没有 POI，退回地理编码
        if self.pois.empty:
            lat, lng = self._geocode_city(self.city)
            if lat is None:
                raise RuntimeError("无法获取 POI，也无法通过地理编码回退。")
            self.pois = pd.DataFrame([{
                "lng": lng,
                "lat": lat,
                "name": self.city,
                "category": "geocode_fallback",
            }])

        # 扩展边界框
        min_lng, max_lng = self.pois["lng"].min(), self.pois["lng"].max()
        min_lat, max_lat = self.pois["lat"].min(), self.pois["lat"].max()

        pad_lng = (max_lng - min_lng) * 0.08 + 0.01
        pad_lat = (max_lat - min_lat) * 0.08 + 0.01

        self.bounds = (
            min_lng - pad_lng,
            max_lng + pad_lng,
            min_lat - pad_lat,
            max_lat + pad_lat,
        )

        # 网格生成
        self.candidates = generate_candidate_grid(
            self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3],
            size_km=self.grid_size_km
        )

        # 限制最大候选点数量
        max_candidates = 800
        if len(self.candidates) > max_candidates:
            self.candidates = self.candidates.sample(max_candidates, random_state=self.seed).reset_index(drop=True)

        self.cand_coords = self.candidates[["lat", "lng"]].to_numpy()

    def _geocode_city(self, name):
        try:
            url = "https://restapi.amap.com/v3/geocode/geo"
            r = requests.get(url, params={"address": name, "key": self.api_key}, timeout=10)
            j = r.json()
            if j.get("status") == "1" and j.get("geocodes"):
                lng, lat = map(float, j["geocodes"][0]["location"].split(","))
                return lat, lng
        except:
            pass
        return None, None

    # ------------------------
    # 染色体相关
    # ------------------------
    def _random_chromosome(self) -> np.ndarray:
        n = len(self.candidates)
        if self.num_primaries >= n:
            return np.arange(n)
        return np.array(random.sample(range(n), self.num_primaries))

    def _chromosome_to_coords(self, chrom):
        return self.cand_coords[chrom]

    # ------------------------
    # 适应度函数
    # ------------------------
    def fitness(self, chrom, weight_repulsion=50.0):
        centers = self._chromosome_to_coords(chrom)
        poi_coords = self.pois[["lat", "lng"]].to_numpy()

        # POI → 最近一级站距离平方和
        lat_p = poi_coords[:, 0][:, None]
        lng_p = poi_coords[:, 1][:, None]
        lat_c = centers[:, 0][None, :]
        lng_c = centers[:, 1][None, :]

        R = 6371.0
        dlat = np.radians(lat_c - lat_p)
        dlon = np.radians(lng_c - lng_p)
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat_p)) * np.cos(np.radians(lat_c)) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        dmat = R * c

        best = np.min(dmat, axis=1)
        part1 = np.sum(best ** 2)

        # 一级站之间的惩罚项
        if centers.shape[0] >= 2:
            rep_pen = 0
            latc = centers[:, 0]
            lngc = centers[:, 1]
            for i in range(len(centers)):
                di = _haversine_km_array(latc[i], lngc[i], latc, lngc)
                di[i] = 1e9
                threshold = max(self.grid_size_km * 0.5, 1.0)
                close = di < threshold
                if np.any(close):
                    rep_pen += np.sum((threshold - di[close]) ** 2)
            part2 = weight_repulsion * rep_pen
        else:
            part2 = 0

        return float(part1 + part2)

    # ------------------------
    # 选择/交叉/变异
    # ------------------------
    def _select_parents(self, pop, fits, tsize=3):
        idxs = list(range(len(pop)))
        cont = random.sample(idxs, min(tsize, len(pop)))
        best = min(cont, key=lambda i: fits[i])
        return pop[best]

    def _crossover(self, p1, p2):
        set1 = set(p1.tolist())
        set2 = set(p2.tolist())
        child_set = set()

        # 一半来自 p1
        n_take = max(1, len(p1) // 2)
        child_set.update(random.sample(list(set1), n_take))

        # 再从 p2 填满
        for x in p2.tolist():
            if len(child_set) >= len(p1):
                break
            child_set.add(x)

        # 若还不够，从全集随机补
        all_idx = set(range(len(self.candidates)))
        remaining = list(all_idx - child_set)

        while len(child_set) < len(p1):
            child_set.add(random.choice(remaining))
            remaining = list(all_idx - child_set)

        return np.array(sorted(child_set))

    def _mutate(self, chrom, rate=0.15):
        n = len(chrom)
        for i in range(n):
            if random.random() < rate:
                all_idx = set(range(len(self.candidates)))
                used = set(chrom.tolist())
                choices = list(all_idx - used)
                if choices:
                    chrom[i] = random.choice(choices)

        uniq = list(dict.fromkeys(chrom.tolist()))
        if len(uniq) < n:
            remaining = list(set(range(len(self.candidates))) - set(uniq))
            while len(uniq) < n:
                uniq.append(random.choice(remaining))

        return np.array(uniq)

    # ------------------------
    # 主流程
    # ------------------------
    def run(self,
            pop_size=60,
            generations=40,
            elite_frac=0.12,
            mutation_rate=0.15,
            weight_repulsion=50.0,
            tournament_size=3,
            verbose=True,
            page_size=25,
            max_pages=3):

        start = time.time()

        # 数据准备
        self.prepare(page_size=page_size, max_pages=max_pages)

        if pop_size < 8:
            pop_size = 8

        # 初始种群
        population = [self._random_chromosome() for _ in range(pop_size)]
        fitnesses = [self.fitness(ch, weight_repulsion) for ch in population]

        elite_n = max(1, int(pop_size * elite_frac))
        best_idx = int(np.argmin(fitnesses))
        best_chrom = population[best_idx]
        best_fit = fitnesses[best_idx]

        history = {"best": [], "mean": []}

        # ------------------------
        # GA 主循环
        # ------------------------
        for gen in range(generations):

            # 选精英
            sorted_idx = np.argsort(fitnesses)
            new_pop = [population[i] for i in sorted_idx[:elite_n]]

            # 生成子代
            while len(new_pop) < pop_size:
                p1 = self._select_parents(population, fitnesses, tsize=tournament_size)
                p2 = self._select_parents(population, fitnesses, tsize=tournament_size)
                child = self._crossover(p1, p2)
                child = self._mutate(child, mutation_rate)
                new_pop.append(child)

            population = new_pop
            fitnesses = [self.fitness(ch, weight_repulsion) for ch in population]

            gen_best_idx = int(np.argmin(fitnesses))
            gen_best_fit = fitnesses[gen_best_idx]
            gen_mean = float(np.mean(fitnesses))

            history["best"].append(gen_best_fit)
            history["mean"].append(gen_mean)

            # 全局最优更新
            if gen_best_fit < best_fit:
                best_fit = gen_best_fit
                best_chrom = population[gen_best_idx]

        centers_coords = self._chromosome_to_coords(best_chrom)

        # ------------------------
        # 生成二级站
        # ------------------------
        secondaries = []
        sec_per_primary = max(3, int(len(self.pois) / 200))

        for c in centers_coords:
            latc, lngc = float(c[0]), float(c[1])
            for _ in range(sec_per_primary):
                angle = random.uniform(0, 2 * math.pi)
                dist_km = random.uniform(self.grid_size_km * 0.6, self.grid_size_km * 1.4)

                dlat = (dist_km / KM_PER_DEG) * math.cos(angle)
                dlng = (dist_km / (KM_PER_DEG * math.cos(math.radians(latc)))) * math.sin(angle)

                secondaries.append({
                    "lat": latc + dlat,
                    "lng": lngc + dlng,
                })

        sec_df = pd.DataFrame(secondaries)

        # 合并二级站（KMeans）
        if not sec_df.empty:
            try:
                ksec = min(len(sec_df), max(1, int(len(centers_coords) * sec_per_primary / 1.5)))
                km = KMeans(n_clusters=ksec, random_state=self.seed, n_init=10)
                labs = km.fit_predict(sec_df[["lat", "lng"]])
                sec_df["cluster"] = labs

                merged = []
                for lab in np.unique(labs):
                    grp = sec_df[sec_df["cluster"] == lab]
                    merged.append({
                        "lat": float(grp["lat"].mean()),
                        "lng": float(grp["lng"].mean()),
                        "count": len(grp),
                    })

                secondaries = merged
            except:
                secondaries = sec_df.to_dict(orient="records")

        # ------------------------
        # 生成地图
        # ------------------------
        map_center = [self.pois["lat"].mean(), self.pois["lng"].mean()]
        m = folium.Map(location=map_center, zoom_start=11, tiles=None)
        folium.TileLayer(tiles=AMAP_TILE_URL, attr="高德地图").add_to(m)

        # POI 简略展示
        for _, r in self.pois.sample(min(len(self.pois), 400)).iterrows():
            folium.CircleMarker(
                [r["lat"], r["lng"]],
                radius=2,
                color="#999",
                fill=True,
                fill_opacity=0.5
            ).add_to(m)

        primaries = []
        for i, c in enumerate(centers_coords, 1):
            folium.CircleMarker(
                [c[0], c[1]],
                radius=7,
                color="red",
                fill=True,
                fill_opacity=0.9,
                popup=f"一级站 P{i}"
            ).add_to(m)
            primaries.append({"lat": float(c[0]), "lng": float(c[1])})

        # 一级站凸包
        if len(centers_coords) >= 3:
            pts = np.array([[p[1], p[0]] for p in centers_coords])
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                path = [(float(p[1]), float(p[0])) for p in hull_pts]
                path.append(path[0])
                folium.PolyLine(path, color="red", weight=3).add_to(m)
            except:
                pass

        # 二级站
        for j, s in enumerate(secondaries, 1):
            folium.CircleMarker(
                [s["lat"], s["lng"]],
                radius=4,
                color="blue",
                fill=True
            ).add_to(m)

        # GA 结果
        info = {
            "city": self.city,
            "poi_count": int(len(self.pois)),
            "candidate_count": int(len(self.candidates)),
            "num_primaries": int(len(centers_coords)),
            "num_secondaries": int(len(secondaries)),
            "best_fitness": float(best_fit),
            "history": history,
        }

        return m, info


# ======================================================
#             Streamlit 的统一调用接口
# ======================================================
def run_ga(city: str, api_key: str):
    """
    给 Streamlit 调用的统一接口
    """
    planner = GAPlanner(
        city=city,
        api_key=api_key,
        grid_size_km=5.0,
        num_primaries=10,
        verbose=False,
        seed=42
    )

    m, info = planner.run(
        pop_size=60,
        generations=40,
        elite_frac=0.12,
        mutation_rate=0.15,
        weight_repulsion=50.0,
        tournament_size=3,
        verbose=False
    )
    return m, info
