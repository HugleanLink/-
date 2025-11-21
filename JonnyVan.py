from typing import List, Tuple, Optional, Dict, Any
import random
import time
import math
import requests
import numpy as np
import pandas as pd
import folium
from shapely.geometry import Point, Polygon, MultiPoint, mapping
from scipy.spatial import ConvexHull
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans


DEFAULT_KEYWORDS = ["商场", "餐饮服务", "写字楼", "酒店", "超市"]
AMAP_TILE_URL = "https://webrd02.is.autonavi.com/appmaptile?style=7&x={x}&y={y}&z={z}"
KM_PER_DEG = 111.0
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
def _coords_to_np(df: pd.DataFrame) -> np.ndarray:
    return df[["lat", "lng"]].to_numpy()
def fetch_pois_amap(city: str, api_key: str, keywords=DEFAULT_KEYWORDS,
                    page_size=25, max_pages=3, timeout=15, retries=3) -> pd.DataFrame:
    all_pois = []
    kws = keywords if isinstance(keywords, (list,tuple)) else [keywords]
    for kw in kws:
        for page in range(1, max_pages+1):
            url = "https://restapi.amap.com/v3/place/text"
            params = {"keywords": kw, "city": city, "output": "json", "offset": page_size, "page": page, "key": api_key}
            success = False
            for attempt in range(retries):
                try:
                    r = requests.get(url, params=params, timeout=timeout)
                    j = r.json()
                    success = True
                    break
                except Exception:
                    time.sleep(1.2*(attempt+1))
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
                all_pois.append({"lng": lng, "lat": lat, "name": poi.get("name",""), "category": kw})
    df = pd.DataFrame(all_pois)
    if not df.empty:
        df = df.drop_duplicates(subset=["lng","lat","name"])
    return df
def generate_candidate_grid(min_lng, max_lng, min_lat, max_lat, size_km=5.0) -> pd.DataFrame:
    step_lat = size_km / KM_PER_DEG
    center_lat = (min_lat + max_lat) / 2.0
    step_lng = size_km / (KM_PER_DEG * math.cos(math.radians(center_lat)) + 1e-12)
    lngs = np.arange(min_lng, max_lng + 1e-9, step_lng)
    lats = np.arange(min_lat, max_lat + 1e-9, step_lat)
    pts = []
    for lng in lngs:
        for lat in lats:
            pts.append({"lng": float(lng + step_lng/2.0), "lat": float(lat + step_lat/2.0)})
    return pd.DataFrame(pts)
class GAPlanner:
    def __init__(self,
                 city: str,
                 api_key: str,
                 keywords=DEFAULT_KEYWORDS,
                 grid_size_km: float = 5.0,
                 num_primaries: int = 10,
                 city_center: Tuple[float,float]=None,
                 verbose: bool=False,
                 seed: Optional[int]=None):
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
        self.pois: pd.DataFrame = pd.DataFrame()
        self.candidates: pd.DataFrame = pd.DataFrame()
        self.cand_coords = None
        self.bounds = None
    def prepare(self, page_size=25, max_pages=3):
        """Fetch POIs and generate candidate grid. Must be called before run()."""
        if self.verbose:
            print(f"[GAPlanner] fetching POIs for {self.city} ...")
        self.pois = fetch_pois_amap(self.city, self.api_key, keywords=self.keywords,
                                    page_size=page_size, max_pages=max_pages)
        if self.pois.empty:
            lat, lng = self._geocode_city(self.city)
            if lat is None:
                raise RuntimeError("无法获取 POI，也无法通过地理编码回退。")
            self.pois = pd.DataFrame([{"lng": lng, "lat": lat, "name": self.city, "category":"geocode_fallback"}])
        min_lng, max_lng = float(self.pois["lng"].min()), float(self.pois["lng"].max())
        min_lat, max_lat = float(self.pois["lat"].min()), float(self.pois["lat"].max())
        pad_lng = (max_lng - min_lng) * 0.08 + 0.01
        pad_lat = (max_lat - min_lat) * 0.08 + 0.01
        self.bounds = (min_lng - pad_lng, max_lng + pad_lng, min_lat - pad_lat, max_lat + pad_lat)
        if self.verbose:
            print(f"[GAPlanner] bbox: {self.bounds}")
        self.candidates = generate_candidate_grid(self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3],
                                                  size_km=self.grid_size_km)
        max_candidates = 800
        if len(self.candidates) > max_candidates:
            self.candidates = self.candidates.sample(max_candidates, random_state=self.seed).reset_index(drop=True)
        self.cand_coords = np.array(self.candidates[["lat","lng"]].to_numpy())
        if self.verbose:
            print(f"[GAPlanner] fetched {len(self.pois)} POIs, {len(self.candidates)} candidates")
    def _geocode_city(self, city_name):
        try:
            url = "https://restapi.amap.com/v3/geocode/geo"
            params = {"address": city_name, "key": self.api_key}
            r = requests.get(url, params=params, timeout=10)
            j = r.json()
            if j.get("status") == "1" and j.get("geocodes"):
                loc = j["geocodes"][0].get("location")
                if loc:
                    lng, lat = map(float, loc.split(","))
                    return lat, lng
        except Exception:
            return None, None
        return None, None
    def _random_chromosome(self) -> np.ndarray:
        n = len(self.candidates)
        if self.num_primaries >= n:
            return np.arange(n, dtype=int)
        return np.array(random.sample(range(n), self.num_primaries), dtype=int)
    def _chromosome_to_coords(self, chrom: np.ndarray) -> np.ndarray:
        return self.cand_coords[chrom]
    def fitness(self, chrom: np.ndarray, weight_repulsion=50.0) -> float:
        centers = self._chromosome_to_coords(chrom)
        poi_coords = self.pois[["lat","lng"]].to_numpy()
        lat_p = poi_coords[:,0][:,None]
        lng_p = poi_coords[:,1][:,None]
        lat_c = centers[:,0][None,:]
        lng_c = centers[:,1][None,:]
        R = 6371.0
        dlat = np.radians(lat_c - lat_p)
        dlon = np.radians(lng_c - lng_p)
        a = np.sin(dlat/2.0)**2 + np.cos(np.radians(lat_p)) * np.cos(np.radians(lat_c)) * np.sin(dlon/2.0)**2
        c = 2 * np.arctan2(np.sqrt(a + 1e-12), np.sqrt(1 - a + 1e-12))
        dmat = R * c
        best = np.min(dmat, axis=1)
        part1 = np.sum(best**2)
        if centers.shape[0] >= 2:
            lat_c_arr = centers[:,0]
            lng_c_arr = centers[:,1]
            idxs = np.arange(len(centers))
            rep_pen = 0.0
            for i in range(len(centers)):
                di = _haversine_km_array(lat_c_arr[i], lng_c_arr[i], lat_c_arr, lng_c_arr)
                di[i] = 1e9
                threshold = max(self.grid_size_km * 0.5, 1.0)
                close = di < threshold
                if np.any(close):
                    rep_pen += np.sum((threshold - di[close])**2)
            part2 = weight_repulsion * rep_pen
        else:
            part2 = 0.0
        return float(part1 + part2)
    def _select_parents(self, population: List[np.ndarray], fitnesses: List[float], tournament_size=3):
        pop_indices = list(range(len(population)))
        contestants = random.sample(pop_indices, min(tournament_size, len(population)))
        best = min(contestants, key=lambda i: fitnesses[i])
        return population[best]
    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        set1 = set(p1.tolist())
        set2 = set(p2.tolist())
        child_set = set()
        n_take = max(1, len(p1)//2)
        take_from_p1 = random.sample(list(set1), n_take)
        child_set.update(take_from_p1)
        for idx in p2.tolist():
            if len(child_set) >= len(p1):
                break
            child_set.add(idx)
        all_idx = set(range(len(self.candidates)))
        remaining = list(all_idx - child_set)
        while len(child_set) < len(p1):
            child_set.add(random.choice(remaining))
            remaining = list(all_idx - child_set)
        child = np.array(sorted(child_set), dtype=int)
        return child
    def _mutate(self, chrom: np.ndarray, mutation_rate=0.15):
        n = len(chrom)
        for i in range(n):
            if random.random() < mutation_rate:
                all_idx = set(range(len(self.candidates)))
                current = set(chrom.tolist())
                choices = list(all_idx - current)
                if not choices:
                    continue
                chrom[i] = random.choice(choices)
        uniq = list(dict.fromkeys(chrom.tolist()))
        if len(uniq) < n:
            remaining = list(set(range(len(self.candidates))) - set(uniq))
            while len(uniq) < n:
                uniq.append(random.choice(remaining))
        return np.array(uniq, dtype=int)
    def run(self,
            pop_size: int = 60,
            generations: int = 40,
            elite_frac: float = 0.12,
            mutation_rate: float = 0.15,
            weight_repulsion: float = 50.0,
            tournament_size: int = 3,
            verbose: bool = True,
            page_size=25,
            max_pages=3) -> Tuple[folium.Map, Dict[str,Any]]:
        start_time = time.time()
        self.prepare(page_size=page_size, max_pages=max_pages)
        if verbose:
            print(f"[GAPlanner] Running GA: pop={pop_size}, gen={generations}, primaries={self.num_primaries}")
        if pop_size < 8:
            pop_size = max(8, pop_size)
        population = [self._random_chromosome() for _ in range(pop_size)]
        fitnesses = [self.fitness(ch, weight_repulsion=weight_repulsion) for ch in population]
        elite_n = max(1, int(pop_size * elite_frac))
        best_chrom = population[int(np.argmin(fitnesses))]
        best_fit = min(fitnesses)
        history = {"best": [], "mean": []}
        for gen in range(generations):
            new_pop = []
            sorted_idx = np.argsort(fitnesses)
            elites = [population[i] for i in sorted_idx[:elite_n]]
            new_pop.extend(elites)
            while len(new_pop) < pop_size:
                p1 = self._select_parents(population, fitnesses, tournament_size=tournament_size)
                p2 = self._select_parents(population, fitnesses, tournament_size=tournament_size)
                child = self._crossover(p1, p2)
                child = self._mutate(child, mutation_rate)
                new_pop.append(child)
            population = new_pop
            fitnesses = [self.fitness(ch, weight_repulsion=weight_repulsion) for ch in population]
            gen_best_idx = int(np.argmin(fitnesses))
            gen_best_fit = fitnesses[gen_best_idx]
            gen_mean_fit = float(np.mean(fitnesses))
            history["best"].append(gen_best_fit)
            history["mean"].append(gen_mean_fit)
            if verbose and (gen % max(1, generations//10) == 0 or gen == generations-1):
                print(f"[GA] gen {gen+1}/{generations} best={gen_best_fit:.3f} mean={gen_mean_fit:.3f}")
            if gen_best_fit < best_fit:
                best_fit = gen_best_fit
                best_chrom = population[gen_best_idx]
        centers_coords = self._chromosome_to_coords(best_chrom)
        secondaries = []
        sec_per_primary = max(3, int(round(len(self.pois)/200)))
        for c in centers_coords:
            latc, lngc = float(c[0]), float(c[1])
            for i in range(sec_per_primary):
                angle = random.uniform(0, 2*math.pi)
                dist_km = random.uniform(self.grid_size_km*0.6, self.grid_size_km*1.4)
                dlat = (dist_km / KM_PER_DEG) * math.cos(angle)
                dlng = (dist_km / (KM_PER_DEG * math.cos(math.radians(latc)) + 1e-12)) * math.sin(angle)
                lat2 = latc + dlat
                lng2 = lngc + dlng
                secondaries.append({"lat": lat2, "lng": lng2, "parent_lat": latc, "parent_lng": lngc})
        sec_df = pd.DataFrame(secondaries)
        if not sec_df.empty:
            try:
                ksec = min(len(sec_df), max(1, int(len(centers_coords)*sec_per_primary/1.5)))
                km = KMeans(n_clusters=ksec, random_state=self.seed, n_init=10)
                labs = km.fit_predict(sec_df[["lat","lng"]].values)
                sec_df["cluster"] = labs
                merged_secs = []
                for lab in np.unique(labs):
                    grp = sec_df[sec_df["cluster"] == lab]
                    merged_secs.append({"lat": float(grp["lat"].mean()), "lng": float(grp["lng"].mean()), "count": len(grp)})
                secondaries = merged_secs
            except Exception:
                secondaries = sec_df[["lat","lng"]].to_dict(orient="records")
        else:
            secondaries = []
        map_center = [float(self.pois["lat"].mean()), float(self.pois["lng"].mean())]
        m = folium.Map(location=map_center, zoom_start=11, tiles=None)
        folium.TileLayer(tiles=AMAP_TILE_URL, attr="高德地图", name="高德矢量图", overlay=False, control=False).add_to(m)
        for _, r in self.pois.sample(min(len(self.pois), 400)).iterrows():
            folium.CircleMarker([r["lat"], r["lng"]], radius=2, color="#999999", fill=True, fill_opacity=0.5).add_to(m)
        primaries = []
        for i, c in enumerate(centers_coords, start=1):
            folium.CircleMarker([float(c[0]), float(c[1])], radius=7, color="red", fill=True, fill_opacity=0.9,
                                popup=f"一级站 P{i}").add_to(m)
            primaries.append({"id": f"P{i}", "lat": float(c[0]), "lng": float(c[1])})
        if len(centers_coords) >= 3:
            pts = np.array([[p[1], p[0]] for p in centers_coords])  # [lng,lat]
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                hull_path = [(float(p[1]), float(p[0])) for p in hull_pts]
                hull_path.append(hull_path[0])
                folium.PolyLine(hull_path, color="red", weight=3, opacity=0.7).add_to(m)
            except Exception:
                pass
        for j, s in enumerate(secondaries, start=1):
            folium.CircleMarker([s["lat"], s["lng"]], radius=4, color="blue", fill=True, fill_opacity=0.8,
                                popup=f"二级站 S{j}").add_to(m)
            if primaries:
                dists = [ _haversine_km_array(p["lat"], p["lng"], np.array([s["lat"]]), np.array([s["lng"]]))[0] for p in primaries ]
                nearest_idx = int(np.argmin(dists))
                folium.PolyLine([[s["lat"], s["lng"]], [primaries[nearest_idx]["lat"], primaries[nearest_idx]["lng"]]],
                                color="green", weight=1.2, opacity=0.6).add_to(m)
        info = {
            "city": self.city,
            "poi_count": int(len(self.pois)),
            "candidate_count": int(len(self.candidates)),
            "num_primaries": int(len(centers_coords)),
            "num_secondaries": int(len(secondaries)),
            "best_fitness": float(best_fit),
            "history": history
        }
        if verbose:
            print(f"[GAPlanner] done in {time.time()-start_time:.1f}s, primaries={len(centers_coords)}, secondaries={len(secondaries)}")
        return m, info
