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
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import re
import sqlite3
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta

# ========== UTF-8 FIX ==========
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ========== LOGGER ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("moodtune")

# ========== LIBRARY CHECKS ==========
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError("please install sentence-transformers: pip install sentence-transformers") from e

try:
    import chromadb
except Exception:
    chromadb = None
    logger.warning("ChromaDB not available - falling back to local vector storage")

# ========== SECURITY CONFIG ==========
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()


# ========== DATABASE SETUP ==========
def init_db():
    conn = sqlite3.connect('moodtune.db')
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Favorites table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            track_id TEXT NOT NULL,
            song_name TEXT NOT NULL,
            artist_name TEXT NOT NULL,
            album TEXT,
            match_percentage INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            UNIQUE(user_id, track_id)
        )
    ''')

    conn.commit()
    conn.close()
    logger.info("Database initialized")


# ========== PASSWORD HASHING ==========
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password(plain_password) == hashed_password


# ========== JWT TOKEN FUNCTIONS ==========
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


# ========== USER MANAGEMENT ==========
class UserManager:
    def __init__(self):
        init_db()

    def create_user(self, username: str, email: str, password: str):
        conn = sqlite3.connect('moodtune.db')
        cursor = conn.cursor()

        try:
            # Check if user already exists
            cursor.execute("SELECT id FROM users WHERE email = ? OR username = ?", (email, username))
            existing_user = cursor.fetchone()
            if existing_user:
                conn.close()
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username or email already exists"
                )

            password_hash = hash_password(password)
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, password_hash)
            )
            conn.commit()
            user_id = cursor.lastrowid
            logger.info(f"Created new user: {username} ({email})")
        except sqlite3.IntegrityError as e:
            conn.close()
            logger.error(f"Integrity error creating user: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already exists"
            )
        except Exception as e:
            conn.close()
            logger.error(f"Error creating user: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )

        conn.close()
        return user_id

    def authenticate_user(self, email: str, password: str):
        conn = sqlite3.connect('moodtune.db')
        cursor = conn.cursor()

        try:
            cursor.execute(
                "SELECT id, username, email, password_hash FROM users WHERE email = ?",
                (email,)
            )
            user = cursor.fetchone()
            conn.close()

            if not user:
                return None

            if not verify_password(password, user[3]):
                return None

            return {
                "id": user[0],
                "username": user[1],
                "email": user[2]
            }
        except Exception as e:
            conn.close()
            logger.error(f"Error authenticating user: {e}")
            return None

    def get_user_by_id(self, user_id: int):
        conn = sqlite3.connect('moodtune.db')
        cursor = conn.cursor()

        try:
            cursor.execute(
                "SELECT id, username, email FROM users WHERE id = ?",
                (user_id,)
            )
            user = cursor.fetchone()
            conn.close()

            if not user:
                return None

            return {
                "id": user[0],
                "username": user[1],
                "email": user[2]
            }
        except Exception as e:
            conn.close()
            logger.error(f"Error getting user by ID: {e}")
            return None


# ========== FAVORITES MANAGEMENT ==========
class FavoritesManager:
    def __init__(self):
        init_db()

    def get_user_favorites(self, user_id: int):
        conn = sqlite3.connect('moodtune.db')
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT track_id, song_name, artist_name, album, match_percentage, created_at
                FROM favorites 
                WHERE user_id = ?
                ORDER BY created_at DESC
            ''', (user_id,))

            favorites = []
            for row in cursor.fetchall():
                favorites.append({
                    "track_id": row[0],
                    "song_name": row[1],
                    "artist_name": row[2],
                    "album": row[3],
                    "match_percentage": row[4],
                    "created_at": row[5]
                })

            conn.close()
            return favorites
        except Exception as e:
            conn.close()
            logger.error(f"Error getting user favorites: {e}")
            return []

    def add_favorite(self, user_id: int, track_data: dict):
        conn = sqlite3.connect('moodtune.db')
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR IGNORE INTO favorites (user_id, track_id, song_name, artist_name, album, match_percentage)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                track_data["track_id"],
                track_data["song_name"],
                track_data["artist_name"],
                track_data.get("album"),
                track_data.get("match_percentage")
            ))
            conn.commit()
            logger.info(f"Added favorite for user {user_id}: {track_data['song_name']}")
        except Exception as e:
            logger.error(f"Error adding favorite: {e}")
        finally:
            conn.close()

    def remove_favorite(self, user_id: int, track_id: str):
        conn = sqlite3.connect('moodtune.db')
        cursor = conn.cursor()

        try:
            cursor.execute(
                "DELETE FROM favorites WHERE user_id = ? AND track_id = ?",
                (user_id, track_id)
            )
            conn.commit()
            logger.info(f"Removed favorite for user {user_id}: {track_id}")
        except Exception as e:
            logger.error(f"Error removing favorite: {e}")
        finally:
            conn.close()


# ========== LIGHTWEIGHT MOOD DETECTOR ==========
class MoodDetector:
    def __init__(self):
        self.mood_keywords = {
            'happy': ['happy', 'joy', 'glad', 'excited', 'love', 'great', 'good', 'amazing', 'wonderful'],
            'sad': ['sad', 'lonely', 'depressed', 'cry', 'heartbroken', 'miserable', 'tears', 'broken'],
            'energetic': ['energetic', 'powerful', 'motivated', 'pumped', 'hyped', 'workout', 'exercise', 'energy'],
            'chill': ['relax', 'calm', 'peaceful', 'chill', 'rest', 'sleepy', 'lofi', 'quiet', 'meditation'],
            'focus': ['study', 'focus', 'concentrate', 'work', 'productive', 'thinking', 'coding', 'reading'],
            'romantic': ['romantic', 'love', 'heart', 'kiss', 'beautiful', 'sweet', 'relationship', 'date'],
            'party': ['party', 'dance', 'club', 'festival', 'celebration', 'night', 'fun', 'friends'],
            'angry': ['angry', 'mad', 'frustrated', 'rage', 'hate', 'annoyed', 'pissed'],
            'melancholic': ['melancholy', 'nostalgic', 'bittersweet', 'thoughtful', 'reflective']
        }

    def detect(self, text: str) -> str:
        text = text.lower()
        scores = {mood: 0 for mood in self.mood_keywords}

        for mood, words in self.mood_keywords.items():
            for w in words:
                if re.search(r'\b' + re.escape(w) + r'\b', text):
                    scores[mood] += 1

        # Only return a mood if we have reasonable confidence
        best_mood, best_score = max(scores.items(), key=lambda x: x[1])
        return best_mood if best_score > 0 else 'neutral'


# ========== TEXT ENCODER ==========
class TextEncoder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized text encoder: {model_name}")

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms


# ========== VECTOR STORAGE ==========
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
                logger.info("Connected to ChromaDB")
            except Exception as e:
                logger.warning(f"ChromaDB init failed: {e}")
                self.client = None
        if self.client is None:
            logger.info("Using local vector storage")

    def upsert(self, ids: List[str], embeddings: List[np.ndarray], metadatas: List[Dict]):
        if self.collection is not None:
            self.collection.upsert(ids=ids, embeddings=[emb.tolist() for emb in embeddings], metadatas=metadatas)
        else:
            for idx, emb, meta in zip(ids, embeddings, metadatas):
                self.local_vectors.append((idx, emb, meta))
        logger.info(f"Upserted {len(ids)} vectors")

    def query(self, query_embedding: np.ndarray, n_results: int = 10) -> List[Dict]:
        if self.collection is not None:
            results = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=n_results)
            hits = []
            if results.get('ids') and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    hits.append({
                        'id': results['ids'][0][i],
                        'score': results['distances'][0][i] if results.get('distances') else 0,
                        'metadata': results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][
                            0] else {}
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


# ========== CSV CLEANER ==========
def clean_spotify_csv(path: str) -> str:
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        # Read CSV with error handling
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")

        if df.empty:
            raise ValueError("CSV file is empty")

        logger.info(f"Original CSV shape: {df.shape}")

        # Clean column names
        original_columns = list(df.columns)
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        logger.info(f"Cleaned column names: {list(df.columns)}")

        # ========== DEFENSIVE COLUMN CHECKS ==========

        # 1. Check for absolutely required columns
        critical_columns = ['track_name', 'track_artist']
        missing_critical = [col for col in critical_columns if col not in df.columns]
        if missing_critical:
            raise ValueError(
                f"Missing critical columns: {missing_critical}. File must contain at least track names and artists.")

        # 2. Check for expected columns and log warnings
        expected_columns = {
            'high_importance': ['track_name', 'track_artist', 'track_album_name', 'track_popularity'],
            'medium_importance': ['playlist_genre', 'danceability', 'energy', 'valence', 'tempo'],
            'low_importance': ['duration_ms', 'key', 'loudness', 'mode', 'speechiness', 'acousticness']
        }

        # Log missing columns by importance level
        for importance_level, columns in expected_columns.items():
            missing = [col for col in columns if col not in df.columns]
            if missing:
                logger.warning(f"Missing {importance_level} columns: {missing}")

        # 3. Check for completely empty or useless columns
        empty_columns = []
        for col in df.columns:
            if df[col].isna().all() or df[col].nunique() <= 1:
                empty_columns.append(col)
                logger.warning(f"Column '{col}' is mostly empty or has no variation")

        # 4. Log available columns for debugging
        available_high = [col for col in expected_columns['high_importance'] if col in df.columns]
        available_medium = [col for col in expected_columns['medium_importance'] if col in df.columns]
        available_low = [col for col in expected_columns['low_importance'] if col in df.columns]

        logger.info(f"Available high importance columns: {available_high}")
        logger.info(f"Available medium importance columns: {available_medium}")
        logger.info(f"Available low importance columns: {available_low}")

        # ========== COLUMN RENAMING WITH VALIDATION ==========
        rename_map = {
            'track_name': 'Track',
            'track_artist': 'Artist',
            'track_album_name': 'Album Name',
            'track_popularity': 'Spotify Popularity',
            'playlist_genre': 'Genre',
        }

        # Only rename columns that exist and log what's being renamed
        existing_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        logger.info(f"Renaming columns: {existing_rename_map}")
        df.rename(columns=existing_rename_map, inplace=True)

        # ========== HANDLE MISSING COLUMNS WITH SMART DEFAULTS ==========

        # Add Spotify Streams if missing (with realistic values based on popularity if available)
        if 'Spotify Streams' not in df.columns:
            if 'Spotify Popularity' in df.columns:
                # Generate streams that correlate with popularity
                base_streams = np.random.randint(10000, 1000000, len(df))
                popularity_factor = df['Spotify Popularity'].fillna(50) / 100
                df['Spotify Streams'] = (base_streams * (1 + popularity_factor * 4)).astype(int)
            else:
                df['Spotify Streams'] = np.random.randint(10000, 5000000, len(df))
            logger.info("Added missing 'Spotify Streams' column with realistic values")

        # Add Release Date if missing
        if 'Release Date' not in df.columns:
            # Generate realistic release dates based on popularity (newer songs tend to be more popular)
            if 'Spotify Popularity' in df.columns:
                years = []
                for pop in df['Spotify Popularity']:
                    if pop > 80:  # Very popular - likely recent
                        year = np.random.choice([2022, 2023, 2024])
                    elif pop > 60:  # Moderately popular
                        year = np.random.choice([2019, 2020, 2021, 2022])
                    else:  # Less popular - could be older
                        year = np.random.randint(2010, 2020)
                    years.append(str(year))
                df['Release Date'] = years
            else:
                df['Release Date'] = '2024'
            logger.info("Added missing 'Release Date' column with context-aware values")

        # Add ISRC if missing
        if 'ISRC' not in df.columns:
            df['ISRC'] = [f"custom_{i:06d}" for i in range(len(df))]
            logger.info("Added missing 'ISRC' column with generated IDs")

        # ========== DATA VALIDATION AND CLEANING ==========

        # Ensure critical text columns exist and are properly formatted
        critical_text_columns = ['Track', 'Artist']
        for col in critical_text_columns:
            if col in df.columns:
                # Fill NaN and convert to string
                df[col] = df[col].fillna('Unknown').astype(str)
                # Remove extra whitespace
                df[col] = df[col].str.strip()
                # Check for empty strings after cleaning
                empty_count = (df[col] == '').sum()
                if empty_count > 0:
                    logger.warning(f"Column '{col}' has {empty_count} empty values after cleaning")
                    df[col] = df[col].replace('', 'Unknown')
            else:
                logger.error(f"Critical column '{col}' missing after renaming!")

        # Ensure numeric columns are properly typed with validation
        numeric_columns = ['Spotify Popularity', 'danceability', 'energy', 'valence', 'tempo']
        for col in numeric_columns:
            if col in df.columns:
                original_dtype = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Check for conversion issues
                na_count = df[col].isna().sum()
                if na_count > 0:
                    logger.warning(f"Column '{col}' had {na_count} non-numeric values converted to NaN")

                # Fill NaN with sensible defaults based on column type
                if col == 'Spotify Popularity':
                    df[col] = df[col].fillna(30)  # Low popularity for unknown
                elif col in ['danceability', 'energy', 'valence']:
                    df[col] = df[col].fillna(0.5)  # Middle value for audio features
                elif col == 'tempo':
                    df[col] = df[col].fillna(120)  # Typical tempo

                # Validate ranges
                if col in ['danceability', 'energy', 'valence']:
                    out_of_range = ((df[col] < 0) | (df[col] > 1)).sum()
                    if out_of_range > 0:
                        logger.warning(f"Column '{col}' has {out_of_range} values outside [0,1] range - clipping")
                        df[col] = df[col].clip(0, 1)

                logger.info(f"Processed numeric column '{col}': {original_dtype} -> {df[col].dtype}")

        # Ensure other text columns are properly typed
        optional_text_columns = ['Album Name', 'Genre']
        for col in optional_text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str).str.strip()
                df[col] = df[col].replace('', 'Unknown')

        # ========== FINAL DATA QUALITY CHECKS ==========

        # Remove any completely empty rows
        initial_count = len(df)
        df = df.dropna(how='all')
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df)} completely empty rows")

        # Check for duplicate tracks
        if 'Track' in df.columns and 'Artist' in df.columns:
            duplicates = df.duplicated(subset=['Track', 'Artist']).sum()
            if duplicates > 0:
                logger.warning(f"Found {duplicates} duplicate track-artist combinations")

        # Final dataset quality report
        logger.info(f"Final dataset shape: {df.shape}")
        logger.info(f"Final columns: {list(df.columns)}")

        if df.empty:
            raise ValueError("Dataset is empty after cleaning!")

        # Save cleaned data
        cleaned_path = "cleaned_spotify_data.csv"
        df.to_csv(cleaned_path, index=False)
        logger.info(f"✅ Cleaned CSV saved as {cleaned_path}")

        # Log sample of the cleaned data
        logger.info(f"Sample of cleaned data:\n{df[['Track', 'Artist', 'Spotify Popularity']].head(3)}")

        return cleaned_path

    except Exception as e:
        logger.error(f"CSV cleaning failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        raise


# ========== MAIN ENGINE ==========
class ImprovedMoodTuneEngine:
    def __init__(self):
        self.text_encoder = TextEncoder()
        self.vector_store = VectorStore()
        self.mood_detector = MoodDetector()
        self.current_dataset: Optional[pd.DataFrame] = None
        logger.info("Improved MoodTune Engine initialized")

    def initialize_with_spotify_data(self, csv_path='high_popularity_spotify_data.csv'):
        try:
            clean_path = clean_spotify_csv(csv_path)
            df = pd.read_csv(clean_path)
            df['mood'] = df.apply(lambda r: self._infer_mood(r), axis=1)
            df = df.drop_duplicates(subset=['Track', 'Artist'], keep='first').reset_index(drop=True)
            self.current_dataset = df
            self.index_tracks(df)
            logger.info(f"Loaded dataset: {len(df)} tracks")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return pd.DataFrame()

    def _infer_mood(self, row):
        """More sophisticated mood inference"""
        energy = float(row.get('energy', 0.5))
        valence = float(row.get('valence', 0.5))
        dance = float(row.get('danceability', 0.5))
        tempo = float(row.get('tempo', 120))

        # More nuanced mood mapping
        if energy > 0.8 and valence > 0.8 and dance > 0.7:
            return 'ecstatic'
        elif energy > 0.7 and valence > 0.7:
            return 'happy'
        elif energy > 0.8 and tempo > 120:
            return 'energetic'
        elif energy < 0.3 and valence < 0.3:
            return 'depressed'
        elif energy < 0.4 and valence < 0.5:
            return 'melancholic'
        elif energy < 0.5 and tempo < 100:
            return 'chill'
        elif dance > 0.7 and energy > 0.6:
            return 'dance'
        elif valence > 0.7 and energy < 0.6:
            return 'romantic'
        elif energy > 0.6 and valence < 0.4:
            return 'angry'
        else:
            return 'neutral'

    def index_tracks(self, df: pd.DataFrame):
        """Create richer text representations for better embeddings"""
        track_texts, ids, metas = [], [], []

        for _, row in df.iterrows():
            # Create much richer text representation
            mood = row['mood']
            popularity = row.get('Spotify Popularity', 50)
            energy = row.get('energy', 0.5)
            danceability = row.get('danceability', 0.5)
            valence = row.get('valence', 0.5)

            # Rich description including audio features
            text_parts = [
                f"Song: {row['Track']}",
                f"Artist: {row['Artist']}",
                f"Mood: {mood}",
                f"Genre: {row.get('Genre', 'Various')}",
                f"Energy: {'high' if energy > 0.7 else 'medium' if energy > 0.4 else 'low'}",
                f"Danceability: {'high' if danceability > 0.7 else 'medium' if danceability > 0.4 else 'low'}",
                f"Valence: {'positive' if valence > 0.6 else 'neutral' if valence > 0.4 else 'negative'}",
                f"Tempo: {'fast' if row.get('tempo', 120) > 120 else 'moderate' if row.get('tempo', 120) > 90 else 'slow'}"
            ]

            text = " ".join(text_parts)
            track_id = row.get('ISRC', f"spotify_{_}")

            metas.append({
                'name': row['Track'],
                'artist': row['Artist'],
                'album': row['Album Name'],
                'mood': mood,
                'popularity': popularity,
                'streams': row.get('Spotify Streams', 0),
                'energy': energy,
                'danceability': danceability,
                'valence': valence,
                'tempo': row.get('tempo', 120),
                'genre': row.get('Genre', 'Unknown')
            })

            track_texts.append(text)
            ids.append(track_id)

        embeddings = self.text_encoder.encode(track_texts)
        self.vector_store.upsert(ids, embeddings, metas)
        logger.info(f"Indexed {len(track_texts)} tracks with rich embeddings")

    def get_recommendations_with_songs(self, query: str, limit: int = 20) -> Dict[str, Any]:
        """Improved recommendation with diversity"""
        detected_mood = self.mood_detector.detect(query)
        logger.info(f"Query: '{query}' -> Detected mood: {detected_mood}")

        # Encode query
        query_embedding = self.text_encoder.encode([query])[0]

        # Get more results than needed for diversity
        raw_results = self.vector_store.query(query_embedding, limit * 3)

        if not raw_results:
            return {'recommendations': [], 'top_songs': []}

        # Apply diversity filtering
        recommendations = self._diversify_recommendations(raw_results, detected_mood, limit)

        # Get top songs for the mood (with diversity)
        top_songs = self._get_diverse_top_songs(detected_mood, 5)

        return {
            'recommendations': recommendations,
            'top_songs': top_songs,
            'detected_mood': detected_mood
        }

    def _diversify_recommendations(self, raw_results, detected_mood, limit):
        """Ensure recommendation diversity"""
        # Group by artist to avoid artist repetition
        artist_groups = {}
        for result in raw_results:
            artist = result['metadata']['artist']
            if artist not in artist_groups:
                artist_groups[artist] = []
            artist_groups[artist].append(result)

        # Select best from each artist group
        diversified = []
        for artist, tracks in artist_groups.items():
            # Take the best match from each artist
            best_track = max(tracks, key=lambda x: x['score'])
            diversified.append(best_track)

        # Sort by score and take top results
        diversified.sort(key=lambda x: x['score'], reverse=True)
        final_results = diversified[:limit]

        # Convert to response format
        recommendations = []
        for result in final_results:
            metadata = result['metadata']
            recommendations.append({
                'song_name': metadata['name'],
                'artist_name': metadata['artist'],
                'album': metadata['album'],
                'mood': metadata['mood'],
                'popularity': metadata['popularity'],
                'match_percentage': min(int(result['score'] * 100), 100),
                'energy': metadata.get('energy', 0.5),
                'danceability': metadata.get('danceability', 0.5)
            })

        return recommendations

    def _get_diverse_top_songs(self, vibe, top_n=5):
        """Get top songs with artist diversity"""
        if self.current_dataset is None or 'mood' not in self.current_dataset.columns:
            return []

        subset = self.current_dataset[self.current_dataset['mood'] == vibe]
        if subset.empty:
            return []

        # Remove duplicates and ensure artist diversity
        subset = subset.drop_duplicates(subset=['Track', 'Artist'])

        # Sort by popularity and ensure artist variety
        subset = subset.sort_values('Spotify Popularity', ascending=False)

        top_songs = []
        artists_seen = set()

        for _, row in subset.iterrows():
            if len(top_songs) >= top_n:
                break
            if row['Artist'] not in artists_seen:
                top_songs.append({
                    'rank': len(top_songs) + 1,
                    'song_name': row['Track'],
                    'artist_name': row['Artist'],
                    'album': row['Album Name'],
                    'popularity': int(row['Spotify Popularity']),
                    'mood': vibe
                })
                artists_seen.add(row['Artist'])

        return top_songs


# ========== FASTAPI APP ==========
class UserRegister(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class FavoriteTrack(BaseModel):
    track_id: str
    song_name: str
    artist_name: str
    album: Optional[str] = None
    match_percentage: Optional[int] = None


class RecommendRequest(BaseModel):
    text: str
    limit: int = 10


app = FastAPI(title="MoodTune Music Recommendation API", version="3.0-with-auth")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])

engine = ImprovedMoodTuneEngine()
user_manager = UserManager()
favorites_manager = FavoritesManager()


@app.on_event("startup")
async def startup_event():
    engine.initialize_with_spotify_data("high_popularity_spotify_data.csv")
    logger.info("API ready!")


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "MoodTune API is running"}


# ========== AUTH ENDPOINTS ==========
@app.post("/auth/register")
async def register(user_data: UserRegister):
    try:
        user_id = user_manager.create_user(
            user_data.username,
            user_data.email,
            user_data.password
        )
        return {"message": "User created successfully", "user_id": user_id}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during registration"
        )


@app.post("/auth/login")
async def login(user_data: UserLogin):
    try:
        user = user_manager.authenticate_user(user_data.email, user_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )

        access_token = create_access_token(data={"sub": user["id"]})
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": user
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during login"
        )


@app.get("/auth/me")
async def get_current_user(payload: dict = Depends(verify_token)):
    try:
        user_id = payload.get("sub")
        user = user_manager.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return user
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error getting user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# ========== FAVORITES ENDPOINTS ==========
@app.get("/favorites")
async def get_favorites(payload: dict = Depends(verify_token)):
    try:
        user_id = payload.get("sub")
        favorites = favorites_manager.get_user_favorites(user_id)
        return {"favorites": favorites}
    except Exception as e:
        logger.error(f"Error getting favorites: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get favorites"
        )


@app.post("/favorites")
async def add_favorite(track: FavoriteTrack, payload: dict = Depends(verify_token)):
    try:
        user_id = payload.get("sub")
        favorites_manager.add_favorite(user_id, track.dict())
        return {"message": "Track added to favorites"}
    except Exception as e:
        logger.error(f"Error adding favorite: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add favorite"
        )


@app.delete("/favorites")
async def remove_favorite(track: FavoriteTrack, payload: dict = Depends(verify_token)):
    try:
        user_id = payload.get("sub")
        favorites_manager.remove_favorite(user_id, track.track_id)
        return {"message": "Track removed from favorites"}
    except Exception as e:
        logger.error(f"Error removing favorite: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove favorite"
        )


# ========== RECOMMENDATION ENDPOINTS ==========
@app.post("/recommend-with-songs")
async def recommend(req: RecommendRequest, payload: dict = Depends(verify_token)):
    try:
        result = engine.get_recommendations_with_songs(req.text, req.limit)
        return {
            "personalized_recommendations": result.get("recommendations", []),
            "top_5_songs_for_this_vibe": result.get("top_songs", []),
            "detected_mood": result.get("detected_mood", "neutral")
        }
    except Exception as e:
        logger.error(f"recommend error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend-with-songs")
async def recommend_public(req: RecommendRequest):
    try:
        result = engine.get_recommendations_with_songs(req.text, req.limit)
        return {
            "personalized_recommendations": result.get("recommendations", []),
            "top_5_songs_for_this_vibe": result.get("top_songs", []),
            "detected_mood": result.get("detected_mood", "neutral")
        }
    except Exception as e:
        logger.error(f"recommend error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)