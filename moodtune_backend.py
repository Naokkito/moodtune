import os
import sys
import io
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("moodtune")

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError("please install sentence-transformers: pip install sentence-transformers") from e

try:
    import chromadb
except Exception:
    chromadb = None
    logger.warning("ChromaDB not available - falling back to local vector storage")

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except Exception:
    spotipy = None
    logger.warning("Spotipy not available - spotify integration disabled")

class TextEncoder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"initialized text encoder: {model_name}")

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

class VectorStore:
    def __init__(self, persist_directory="./chroma_db", collection_name="moodtune_songs"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.local_vectors = []
        if chromadb is not None:
            try:
                self.client = chromadb.PersistentClient(path=persist_directory)
                self.collection = self.client.get_or_create_collection(name=collection_name)
                logger.info("connected to chromadb")
            except Exception as e:
                logger.warning(f"chromadb init failed: {e}")
                self.client = None
        if self.client is None:
            logger.info("using local vector storage")

    def upsert(self, ids: List[str], embeddings: List[np.ndarray], metadatas: List[Dict]):
        if self.collection is not None:
            self.collection.upsert(ids=ids, embeddings=[emb.tolist() for emb in embeddings], metadatas=metadatas)
        else:
            for idx, emb, meta in zip(ids, embeddings, metadatas):
                self.local_vectors.append((idx, emb, meta))
        logger.info(f"upserted {len(ids)} vectors")

    def query(self, query_embedding: np.ndarray, n_results: int = 10) -> List[Dict]:
        if self.collection is not None:
            results = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=n_results)
            hits = []
            if results.get('ids') and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    hits.append({
                        'id': results['ids'][0][i],
                        'score': results['distances'][0][i] if results.get('distances') else 0,
                        'metadata': results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else {}
                    })
            return hits
        else:
            if not self.local_vectors:
                return []
            qn = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
            sims = []
            for vec_id, vector, metadata in self.local_vectors:
                vn = vector / (np.linalg.norm(vector) + 1e-9)
                sims.append((vec_id, float(np.dot(vn, qn)), metadata))
            sims.sort(key=lambda x: x[1], reverse=True)
            return [{'id': s[0], 'score': s[1], 'metadata': s[2]} for s in sims[:n_results]]

class SpotifyIntegration:
    def __init__(self, client_id=None, client_secret=None):
        self.client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
        self.spotify_client = None
        if spotipy and self.client_id and self.client_secret:
            try:
                auth = SpotifyClientCredentials(client_id=self.client_id, client_secret=self.client_secret)
                self.spotify_client = spotipy.Spotify(auth_manager=auth)
                logger.info("spotify client initialized")
            except Exception as e:
                logger.warning(f"spotify init failed: {e}")

    def get_track_details(self, track_ids: List[str]) -> List[Dict]:
        if not self.spotify_client:
            return self._mock_track_details(track_ids)
        try:
            results = self.spotify_client.tracks(track_ids)
            tracks = []
            for t in results.get('tracks', []):
                if t:
                    tracks.append({
                        'id': t.get('id'),
                        'name': t.get('name'),
                        'artists': [a['name'] for a in t.get('artists', [])],
                        'album': t.get('album', {}).get('name'),
                        'preview_url': t.get('preview_url'),
                        'external_url': t.get('external_urls', {}).get('spotify'),
                        'popularity': t.get('popularity', 0),
                        'duration_ms': t.get('duration_ms', 180000)
                    })
            return tracks
        except Exception as e:
            logger.error(f"spotify fetch error: {e}")
            return self._mock_track_details(track_ids)

    def _mock_track_details(self, track_ids: List[str]) -> List[Dict]:
        mood_map = {
            'happy': ['Sunshine Pop', 'Summer Vibes', 'Joyful Melodies', 'Happy Beats'],
            'chill': ['Relaxing Beats', 'Calm Atmosphere', 'Peaceful Sounds', 'Chill Vibes'],
            'energy': ['Workout Anthem', 'Power Boost', 'Energetic Rhythm', 'Energy Flow'],
            'focus': ['Deep Focus', 'Study Session', 'Concentration', 'Productive Mood'],
            'party': ['Party Starter', 'Celebration', 'Festival Vibes', 'Dance Floor'],
            'romantic': ['Love Songs', 'Romantic Evening', 'Heartfelt', 'Intimate Moments'],
            'neutral': ['Ambient Track', 'Instrumental Piece']
        }
        tracks = []
        for tid in track_ids:
            mood = tid.split('_')[0] if '_' in tid else 'chill'
            name = f"{np.random.choice(mood_map.get(mood, ['Track']))} {np.random.randint(1,100)}"
            tracks.append({
                'id': tid,
                'name': name,
                'artists': [f"Artist {np.random.randint(1,50)}"],
                'album': f"Album {np.random.randint(1,20)}",
                'preview_url': None,
                'external_url': f"https://open.spotify.com/track/{tid}",
                'popularity': int(np.random.randint(30,100)),
                'duration_ms': int(np.random.choice([180000,200000,220000,240000,300000]))
            })
        return tracks

class CSVDataProcessor:
    def __init__(self):
        self.text_encoder = TextEncoder()
        logger.info("csv processor initialized")

    def process_spotify_csv(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            df_clean = self._clean_data(df.copy())
            df_clean = self._create_text_embeddings(df_clean)
            df_clean = self._generate_audio_features(df_clean)
            df_clean = self._infer_moods(df_clean)
            return {'success': True, 'data': df_clean, 'stats': self._get_dataset_stats(df_clean)}
        except Exception as e:
            logger.error(f"csv processing error: {e}")
            return {'success': False, 'error': str(e)}

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = ['Spotify Streams', 'Spotify Playlist Count', 'Spotify Popularity', 'YouTube Views', 'YouTube Likes', 'TikTok Views']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        text_columns = ['Track', 'Album Name', 'Artist', 'ISRC']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().fillna('Unknown')
        df = df.fillna({'Release Date': 'Unknown', 'All Time Rank': 9999, 'Track Score': 0})
        return df

    def _create_text_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        texts = []
        for _, r in df.iterrows():
            texts.append(f"{r.get('Track','')} {r.get('Artist','')} {r.get('Album Name','')}".strip())
        if texts:
            embeddings = self.text_encoder.encode(texts)
            df['text_embedding'] = list(map(lambda x: x.tolist(), embeddings))
        else:
            df['text_embedding'] = None
        return df

    def _generate_audio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature in ['danceability', 'energy', 'valence', 'acousticness']:
            if feature not in df.columns:
                if 'Spotify Popularity' in df.columns:
                    popularity_norm = df['Spotify Popularity'] / 100
                    if feature == 'energy':
                        df[feature] = np.clip(popularity_norm * np.random.uniform(0.7, 1.0, len(df)), 0, 1)
                    elif feature == 'danceability':
                        df[feature] = np.clip(popularity_norm * np.random.uniform(0.6, 0.95, len(df)), 0, 1)
                    else:
                        df[feature] = np.random.uniform(0, 1, len(df))
                else:
                    df[feature] = np.random.uniform(0, 1, len(df))
        if 'tempo' not in df.columns:
            df['tempo'] = np.random.uniform(60, 180, len(df))
        return df

    def _infer_moods(self, df: pd.DataFrame) -> pd.DataFrame:
        moods = []
        for _, r in df.iterrows():
            combined = f"{str(r.get('Track','')).lower()} {str(r.get('Artist','')).lower()}"
            if any(w in combined for w in ['chill', 'calm', 'relax', 'sleep', 'ambient', 'lo-fi']):
                moods.append('chill')
            elif any(w in combined for w in ['energy', 'workout', 'power', 'pump', 'intense', 'hard']):
                moods.append('energy')
            elif any(w in combined for w in ['happy', 'sunshine', 'joy', 'summer', 'fun', 'bright']):
                moods.append('happy')
            elif any(w in combined for w in ['focus', 'study', 'concentrate', 'productive']):
                moods.append('focus')
            elif any(w in combined for w in ['party', 'celebration', 'festival', 'dance', 'club']):
                moods.append('party')
            elif any(w in combined for w in ['romantic', 'love', 'heart', 'ballad', 'slow']):
                moods.append('romantic')
            elif any(w in combined for w in ['sad', 'melancholy', 'tear', 'heartbreak']):
                moods.append('sad')
            else:
                energy = float(r.get('energy', 0.5))
                valence = float(r.get('valence', 0.5))
                if energy > 0.7 and valence > 0.6:
                    moods.append('happy')
                elif energy > 0.7 and valence <= 0.6:
                    moods.append('energy')
                elif energy <= 0.4 and valence > 0.6:
                    moods.append('chill')
                elif energy <= 0.4 and valence <= 0.4:
                    moods.append('sad')
                else:
                    moods.append('neutral')
        df['mood'] = moods
        logger.info(f"mood distribution: {pd.Series(moods).value_counts().to_dict()}")
        return df

    def _get_dataset_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        stats = {
            'total_tracks': len(df),
            'total_artists': int(df['Artist'].nunique()) if 'Artist' in df.columns else 0,
            'total_albums': int(df['Album Name'].nunique()) if 'Album Name' in df.columns else 0,
            'mood_distribution': df['mood'].value_counts().to_dict() if 'mood' in df.columns else {},
            'avg_popularity': float(df['Spotify Popularity'].mean()) if 'Spotify Popularity' in df.columns else 0,
            'total_streams': int(df['Spotify Streams'].sum()) if 'Spotify Streams' in df.columns else 0
        }
        if 'Artist' in df.columns:
            stats['top_artists'] = df['Artist'].value_counts().head(5).to_dict()
        return stats

class MoodTuneEngine:
    def __init__(self):
        self.text_encoder = TextEncoder()
        self.vector_store = VectorStore()
        self.spotify = SpotifyIntegration()
        self.csv_processor = CSVDataProcessor()
        self.current_dataset: Optional[pd.DataFrame] = None
        self.using_real_data = False
        logger.info("moodtune engine initialized")

    def create_sample_dataset(self, num_tracks=1000):
        moods = ['happy', 'chill', 'energy', 'focus', 'party', 'romantic', 'neutral', 'sad']
        artists = [f"Artist {i}" for i in range(1, 101)]
        tracks = []
        for i in range(num_tracks):
            mood = np.random.choice(moods)
            tracks.append({
                'id': f"{mood}_track_{i}",
                'name': f"{mood.capitalize()} Song {i}",
                'artist': np.random.choice(artists),
                'mood': mood,
                'popularity': int(np.random.randint(0, 100)),
                'duration_ms': int(np.random.randint(120000, 300000))
            })
        return pd.DataFrame(tracks)

    def initialize_with_spotify_data(self, csv_path='Most Streamed Spotify Songs 2024.csv'):
        try:
            df = pd.read_csv(csv_path)
            processed = self.csv_processor.process_spotify_csv(df)
            if processed.get('success'):
                self.current_dataset = processed['data']
                self.using_real_data = True
                self.index_tracks(self.current_dataset)
                stats = processed['stats']
                logger.info(f"loaded spotify dataset: {stats.get('total_tracks',0)} tracks")
                return self.current_dataset
            else:
                logger.error("failed to process spotify csv, falling back")
                return self.initialize_with_sample_data(200)
        except FileNotFoundError:
            logger.warning("spotify csv not found, using sample")
            return self.initialize_with_sample_data(200)
        except Exception as e:
            logger.error(f"error loading spotify dataset: {e}")
            return self.initialize_with_sample_data(200)

    def initialize_with_sample_data(self, num_tracks=200):
        sample = self.create_sample_dataset(num_tracks)
        self.using_real_data = False
        self.index_tracks(sample)
        return sample

    def index_tracks(self, tracks_df):
        track_texts = []
        ids = []
        metadatas = []
        for _, row in tracks_df.iterrows():
            if 'Track' in tracks_df.columns:
                track_name = row.get('Track', 'Unknown Track')
                artist = row.get('Artist', 'Unknown Artist')
                album = row.get('Album Name', 'Unknown Album')
                track_id = row.get('ISRC', f"spotify_{len(ids)}")
                mood = row.get('mood', 'neutral')
                popularity = int(row.get('Spotify Popularity', 50))
                streams = int(row.get('Spotify Streams', 0))
                release_date = row.get('Release Date', 'Unknown')
                text_description = f"{track_name} by {artist} {album} {mood} {release_date}"
                metadata = {
                    'name': track_name, 'artist': artist, 'album': album, 'mood': mood,
                    'popularity': popularity, 'streams': streams, 'source': 'spotify_dataset',
                    'isrc': track_id, 'release_date': release_date,
                    'danceability': float(row.get('danceability', 0.5)),
                    'energy': float(row.get('energy', 0.5)),
                    'valence': float(row.get('valence', 0.5)),
                    'tempo': float(row.get('tempo', 120))
                }
            else:
                track_name = row['name']
                artist = row['artist']
                mood = row['mood']
                track_id = row['id']
                text_description = f"{track_name} by {artist} {mood}"
                metadata = {
                    'name': track_name, 'artist': artist, 'album': 'Sample Album', 'mood': mood,
                    'popularity': int(row.get('popularity', 50)), 'streams': 0, 'source': 'sample_data',
                    'duration_ms': int(row.get('duration_ms', 180000)),
                    'danceability': float(np.random.uniform(0, 1)),
                    'energy': float(np.random.uniform(0, 1)),
                    'valence': float(np.random.uniform(0, 1)),
                    'tempo': float(np.random.uniform(60, 180))
                }
            track_texts.append(text_description)
            ids.append(track_id)
            metadatas.append(metadata)
        if track_texts:
            embeddings = self.text_encoder.encode(track_texts)
            self.vector_store.upsert(ids, embeddings, metadatas)
            logger.info("tracks indexed")

    def get_recommendations_with_songs(self, query: str, limit: int = 10) -> Dict[str, Any]:
        query_embedding = self.text_encoder.encode([query])[0]
        results = self.vector_store.query(query_embedding, limit)
        if not results:
            return {'recommendations': [], 'top_songs': []}
        recs = []
        for r in results:
            m = r.get('metadata', {})
            danceability = float(m.get('danceability', 0.5))
            energy = float(m.get('energy', 0.5))
            valence = float(m.get('valence', 0.5))
            tempo = float(m.get('tempo', 120))
            score = float(r.get('score', 0))
            match_percentage = min(int(score * 125), 125)
            recs.append({
                'id': r.get('id', 'unknown'),
                'name': m.get('name', 'Unknown Track'),
                'artist': m.get('artist', 'Unknown Artist'),
                'album': m.get('album', 'Unknown Album'),
                'popularity': int(m.get('popularity', 50)),
                'match_percentage': match_percentage,
                'mood': m.get('mood', 'neutral'),
                'streams': int(m.get('streams', 0)),
                'audio_features': {'danceability': danceability, 'energy': energy, 'valence': valence, 'tempo': tempo},
                'why_it_matches': self._generate_match_explanation(query, m.get('mood', 'neutral'), danceability, energy, valence, tempo)
            })
        recs.sort(key=lambda x: x['match_percentage'], reverse=True)
        vibe = self._detect_vibe_from_query(query)
        top_songs = self._get_top_songs_for_vibe(vibe, top_n=5)
        return {'recommendations': recs, 'top_songs': top_songs}

    def _detect_vibe_from_query(self, query: str) -> str:
        q = query.lower()
        mapping = {
            'focus': ['focus', 'study', 'concentrate', 'work', 'productiv', 'deep work'],
            'chill': ['chill', 'relax', 'calm', 'peaceful', 'unwind', 'mellow'],
            'energy': ['energy', 'workout', 'exercise', 'pump', 'intense', 'power', 'boost'],
            'happy': ['happy', 'joy', 'fun', 'summer', 'sunny', 'bright', 'upbeat'],
            'party': ['party', 'celebration', 'dance', 'club', 'festival', 'night'],
            'romantic': ['romantic', 'love', 'intimate', 'heart', 'slow', 'ballad'],
            'sad': ['sad', 'melancholy', 'blue', 'heartbreak', 'emotional', 'tear']
        }
        for vibe, kws in mapping.items():
            if any(k in q for k in kws):
                return vibe
        return 'mixed'

    def _get_top_songs_for_vibe(self, vibe: str, top_n: int = 5) -> List[Dict]:
        if self.current_dataset is None:
            return self._get_fallback_top_songs(vibe, top_n)
        try:
            if 'mood' in self.current_dataset.columns:
                subset = self.current_dataset[self.current_dataset['mood'] == vibe].copy()
            else:
                subset = self.current_dataset.copy()
            if subset.empty:
                return self._get_fallback_top_songs(vibe, top_n)
            if 'Spotify Popularity' in subset.columns:
                subset = subset.sort_values('Spotify Popularity', ascending=False)
            elif 'Spotify Streams' in subset.columns:
                subset = subset.sort_values('Spotify Streams', ascending=False)
            top = []
            for i, (_, r) in enumerate(subset.head(top_n).iterrows()):
                song_name = r.get('Track', r.get('name', 'Unknown Track'))
                artist_name = r.get('Artist', r.get('artist', 'Unknown Artist'))
                album_name = r.get('Album Name', r.get('album', 'Unknown Album'))
                top.append({
                    'rank': i + 1,
                    'name': song_name,
                    'artist': artist_name,
                    'album': album_name,
                    'mood': vibe,
                    'popularity': int(r.get('Spotify Popularity', r.get('popularity', 50))),
                    'streams': int(r.get('Spotify Streams', r.get('streams', 0))),
                    'match_percentage': 125 - (i * 5)
                })
            return top
        except Exception as e:
            logger.error(f"top songs error: {e}")
            return self._get_fallback_top_songs(vibe, top_n)

    def _get_fallback_top_songs(self, vibe: str, top_n: int = 5) -> List[Dict]:
        fallback = []
        for i in range(top_n):
            fallback.append({
                'rank': i + 1,
                'name': f'{vibe.capitalize()} Song {i+1}',
                'artist': f'Artist {i+1}',
                'album': f'{vibe.capitalize()} Album',
                'mood': vibe,
                'popularity': 80 - (i * 5),
                'streams': max(0, 1000000 - (i * 100000)),
                'match_percentage': 125 - (i * 5)
            })
        return fallback

    def _generate_match_explanation(self, query: str, mood: str, danceability: float, energy: float, valence: float, tempo: float) -> str:
        q = query.lower()
        explanations = []
        if any(w in q for w in ['focus', 'study', 'work', 'concentrate']):
            if energy < 0.4:
                explanations.append("low energy perfect for deep concentration")
            elif energy < 0.7:
                explanations.append("balanced energy that maintains focus without distraction")
            if tempo < 100:
                explanations.append("moderate tempo supports attention")
        elif any(w in q for w in ['energy', 'workout', 'exercise', 'pump']):
            if energy > 0.7:
                explanations.append("high energy that boosts motivation")
            if tempo > 120:
                explanations.append("fast tempo matches workout rhythm")
            if valence > 0.6:
                explanations.append("positive mood enhances exercise")
        elif any(w in q for w in ['chill', 'relax', 'calm', 'peaceful']):
            if energy < 0.5:
                explanations.append("low energy creates relaxing atmosphere")
            if 0.4 < valence < 0.8:
                explanations.append("balanced mood promotes relaxation")
            if tempo < 110:
                explanations.append("gentle tempo induces calm")
        elif any(w in q for w in ['happy', 'joy', 'fun', 'summer']):
            if valence > 0.7:
                explanations.append("high positivity matches happy mood")
            if energy > 0.6:
                explanations.append("energetic feel enhances joy")
            if danceability > 0.6:
                explanations.append("danceable rhythm adds fun")
        if energy > 0.8:
            explanations.append("high energy level")
        elif energy < 0.3:
            explanations.append("calm energy level")
        if valence > 0.7:
            explanations.append("positive emotional tone")
        elif valence < 0.3:
            explanations.append("mellow emotional tone")
        if danceability > 0.7:
            explanations.append("high danceability")
        elif danceability < 0.3:
            explanations.append("subtle rhythm")
        if not explanations:
            explanations.append(f"matches the {mood} vibe")
        return " • ".join(explanations)

    def get_dataset_info(self) -> Dict[str, Any]:
        if self.using_real_data and self.current_dataset is not None:
            stats = self.csv_processor._get_dataset_stats(self.current_dataset)
            return {
                'dataset_type': 'spotify_2024',
                'total_tracks': stats.get('total_tracks', 0),
                'total_artists': stats.get('total_artists', 0),
                'data_source': 'Spotify Most Streamed 2024',
                'mood_distribution': stats.get('mood_distribution', {}),
                'avg_popularity': stats.get('avg_popularity', 0),
                'total_streams': stats.get('total_streams', 0),
                'top_artists': stats.get('top_artists', {})
            }
        else:
            return {
                'dataset_type': 'sample_data',
                'total_tracks': 200,
                'total_artists': 100,
                'data_source': 'synthetic_sample',
                'status': 'fallback_mode',
                'mood_distribution': {},
                'avg_popularity': 0,
                'total_streams': 0
            }

class RecommendRequest(BaseModel):
    text: str
    limit: int = 10
    user_id: Optional[str] = None

class UploadResponse(BaseModel):
    success: bool
    message: str
    stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

app = FastAPI(title="MoodTune Music Recommendation API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
moodtune_engine = MoodTuneEngine()

@app.on_event("startup")
async def startup_event():
    moodtune_engine.initialize_with_spotify_data()
    info = moodtune_engine.get_dataset_info()
    if moodtune_engine.using_real_data:
        logger.info(f"api ready with {info['total_tracks']} real tracks")
    else:
        logger.info("api ready with sample data (fallback)")

@app.get("/")
async def root():
    return {
        "message": "MoodTune Music Recommendation API",
        "version": "2.0.0",
        "status": "running",
        "dataset_info": moodtune_engine.get_dataset_info(),
        "endpoints": {
            "recommend_with_songs": "/recommend-with-songs (POST)",
            "health": "/health (GET)",
            "dataset_info": "/dataset-info (GET)",
            "test": "/test-connection (GET)"
        }
    }

@app.get("/test-connection")
async def test_connection():
    return {
        "status": "success",
        "message": "Backend is running",
        "dataset": moodtune_engine.get_dataset_info().get('dataset_type'),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "MoodTune API", "dataset": moodtune_engine.get_dataset_info()}

@app.get("/dataset-info")
async def get_dataset_info():
    try:
        return {"success": True, "dataset_info": moodtune_engine.get_dataset_info(), "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"dataset info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend-with-songs")
async def get_recommendations_with_real_songs(request: RecommendRequest):
    try:
        result = moodtune_engine.get_recommendations_with_songs(request.text, request.limit)
        transformed_recommendations = []
        for rec in result.get('recommendations', []):
            transformed_recommendations.append({
                'song_name': rec.get('name', ''),
                'artist_name': rec.get('artist', ''),
                'album': rec.get('album', ''),
                'match_percentage': rec.get('match_percentage', 0),
                'popularity': rec.get('popularity', 0),
                'why_it_matches': rec.get('why_it_matches', ''),
                'audio_features': rec.get('audio_features', {})
            })
        transformed_top_songs = []
        for s in result.get('top_songs', []):
            transformed_top_songs.append({
                'rank': s.get('rank', 0),
                'song_name': s.get('name', ''),
                'artist_name': s.get('artist', ''),
                'album': s.get('album', ''),
                'match_percentage': s.get('match_percentage', 0),
                'popularity': s.get('popularity', 0),
                'streams': s.get('streams', 0)
            })
        return {
            "query": request.text,
            "personalized_recommendations": transformed_recommendations,
            "top_5_songs_for_this_vibe": transformed_top_songs,
            "total_recommendations": len(transformed_recommendations)
        }
    except Exception as e:
        logger.error(f"recommend error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        processed = moodtune_engine.csv_processor.process_spotify_csv(df)
        if processed.get('success'):
            moodtune_engine.current_dataset = processed['data']
            moodtune_engine.using_real_data = True
            moodtune_engine.index_tracks(moodtune_engine.current_dataset)
            stats = processed['stats']
            return UploadResponse(success=True, message="CSV uploaded and processed", stats=stats)
        return UploadResponse(success=False, message="processing failed", error=processed.get('error'))
    except Exception as e:
        logger.error(f"upload error: {e}")
        return UploadResponse(success=False, message="upload failed", error=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
