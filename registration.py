"""
registration.py - Face Database & Registration
Stores known face embeddings in a JSON file for persistence between sessions.

Person C is responsible for this module.
"""

import json
import os
import numpy as np
from datetime import datetime
import cv2
import config


class FaceDatabase:
    """Manages persistent storage of registered face identities."""

    def __init__(self, db_path=None):
        self.db_path = db_path or config.DATABASE_PATH
        self.data = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.db_path):
            print("[DB] Starting fresh database")
            return
        try:
            with open(self.db_path, 'r') as f:
                raw = json.load(f)
            for name, entry in raw.items():
                entry['embeddings'] = [np.array(e) for e in entry['embeddings']]
            self.data = raw
            print(f"[DB] Loaded {len(self.data)} person(s)")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[DB] Could not load database: {e}")

    def _save(self):
        out = {}
        for name, entry in self.data.items():
            out[name] = {
                'name': entry['name'],
                'embeddings': [e.tolist() if isinstance(e, np.ndarray) else e
                               for e in entry['embeddings']],
                'model': entry['model'],
                'registered_at': entry['registered_at'],
                'face_image_path': entry.get('face_image_path', ''),
            }
        with open(self.db_path, 'w') as f:
            json.dump(out, f, indent=2)

    def register(self, name, embedding, model_name, face_image=None):
        """Add a person (or add another embedding to an existing person)."""
        key = name.strip().lower()
        if not key or embedding is None:
            return False

        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # save face image for reference
        face_path = ""
        if face_image is not None:
            face_path = os.path.join(config.KNOWN_FACES_DIR,
                                     f"{key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(face_path, face_image)

        if key in self.data:
            self.data[key]['embeddings'].append(embedding)
        else:
            self.data[key] = {
                'name': name.strip(),
                'embeddings': [embedding],
                'model': model_name,
                'registered_at': datetime.now().isoformat(),
                'face_image_path': face_path,
            }

        self._save()
        return True

    def get_all_embeddings(self):
        """Returns list of (name, embedding) tuples for matching."""
        entries = []
        for key, entry in self.data.items():
            for emb in entry['embeddings']:
                if isinstance(emb, list):
                    emb = np.array(emb)
                entries.append((entry['name'], emb))
        return entries

    def list_users(self):
        if not self.data:
            print("No registered users.")
            return
        print(f"\nRegistered users ({len(self.data)}):")
        for i, (key, entry) in enumerate(self.data.items(), 1):
            n_emb = len(entry['embeddings'])
            date = entry.get('registered_at', '?')[:10]
            print(f"  {i}. {entry['name']} - {n_emb} embedding(s) - {entry['model']} - {date}")
        print()

    def remove_user(self, name):
        key = name.strip().lower()
        if key in self.data:
            del self.data[key]
            self._save()
            print(f"Removed: {name}")
            return True
        print(f"Not found: {name}")
        return False

    def is_empty(self):
        return len(self.data) == 0

    def count(self):
        return len(self.data)
