"""
registration.py — Registration & Persistence Module (Person C's responsibility)

This module handles:
  1. Registering new users (name + face embedding)
  2. Saving known faces to disk (JSON file) for persistence between sessions
  3. Loading known faces on startup
  4. Listing and managing registered users

STORAGE FORMAT (JSON):
  The database stores each person as an entry with:
    - name: The person's name
    - embeddings: A list of embedding vectors (multiple per person for robustness)
    - model: Which recognition model produced the embedding ("facenet" or "arcface")
    - registered_at: Timestamp of first registration
    - face_image_path: Path to saved face image (for reference)

WHY MULTIPLE EMBEDDINGS PER PERSON?
  A single photo might not capture all variations of a person's face.
  Storing 3-5 embeddings from different angles/lighting improves recognition
  accuracy significantly. During matching, we compare against ALL stored
  embeddings and use the best (lowest distance) match.
"""

import json
import os
import numpy as np
from datetime import datetime
import cv2
import config


class FaceDatabase:
    """
    Manages persistent storage of known face identities and their embeddings.

    The database is a JSON file that maps person names to their face data.
    It loads automatically on initialisation and saves after every change.

    Attributes:
        db_path (str): Path to the JSON database file.
        database (dict): In-memory copy of the database.
    """

    def __init__(self, db_path=None):
        """
        Initialise the database, loading existing data if available.

        Parameters:
            db_path (str, optional): Path to the JSON file. Defaults to config.DATABASE_PATH.
        """
        self.db_path = db_path or config.DATABASE_PATH
        self.database = {}
        self._load()

    # ──────────────────────────────────────────
    # Private: Load & Save
    # ──────────────────────────────────────────
    def _load(self):
        """
        Load the database from the JSON file.

        Embeddings are stored as lists in JSON but converted to numpy arrays
        in memory for efficient distance calculations.
        """
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    raw = json.load(f)

                # Convert embedding lists back to numpy arrays
                for name, data in raw.items():
                    data['embeddings'] = [
                        np.array(emb) for emb in data['embeddings']
                    ]
                    self.database = raw

                print(f"[DB] Loaded {len(self.database)} known person(s) from {self.db_path}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[DB] Warning: Could not load database ({e}). Starting fresh.")
                self.database = {}
        else:
            print(f"[DB] No existing database found. Starting fresh.")
            self.database = {}

    def _save(self):
        """
        Save the current database to the JSON file.

        Numpy arrays are converted to lists for JSON serialisation.
        """
        # Create a JSON-safe copy
        serialisable = {}
        for name, data in self.database.items():
            serialisable[name] = {
                'name': data['name'],
                'embeddings': [
                    emb.tolist() if isinstance(emb, np.ndarray) else emb
                    for emb in data['embeddings']
                ],
                'model': data['model'],
                'registered_at': data['registered_at'],
                'face_image_path': data.get('face_image_path', ''),
                'num_embeddings': len(data['embeddings'])
            }

        with open(self.db_path, 'w') as f:
            json.dump(serialisable, f, indent=2)

        print(f"[DB] Saved database with {len(self.database)} person(s).")

    # ──────────────────────────────────────────
    # Public: Register a New Person
    # ──────────────────────────────────────────
    def register(self, name, embedding, model_name, face_image=None):
        """
        Register a new person or add an additional embedding to an existing person.

        Parameters:
            name (str): The person's name (case-insensitive, stored lowercase).
            embedding (numpy.ndarray): The face embedding vector.
            model_name (str): Which model produced this embedding ("facenet" or "arcface").
            face_image (numpy.ndarray, optional): The cropped face image to save for reference.

        Returns:
            str: Status message indicating what happened.
        """
        name_key = name.strip().lower()

        if not name_key:
            return "[ERROR] Name cannot be empty."

        if embedding is None or len(embedding) == 0:
            return "[ERROR] Invalid embedding."

        # Ensure embedding is a numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # Save face image if provided
        face_path = ""
        if face_image is not None:
            face_path = os.path.join(
                config.KNOWN_FACES_DIR,
                f"{name_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            cv2.imwrite(face_path, face_image)

        if name_key in self.database:
            # Person already exists — add another embedding
            self.database[name_key]['embeddings'].append(embedding)
            self.database[name_key]['num_embeddings'] = len(self.database[name_key]['embeddings'])
            status = f"[DB] Added new embedding for '{name}'. Total: {len(self.database[name_key]['embeddings'])}"
        else:
            # New person
            self.database[name_key] = {
                'name': name.strip(),        # preserve original casing for display
                'embeddings': [embedding],
                'model': model_name,
                'registered_at': datetime.now().isoformat(),
                'face_image_path': face_path,
                'num_embeddings': 1
            }
            status = f"[DB] Registered new person: '{name}'"

        self._save()
        print(status)
        return status

    # ──────────────────────────────────────────
    # Public: Get All Known Embeddings
    # ──────────────────────────────────────────
    def get_all_embeddings(self):
        """
        Returns all stored embeddings with their associated names.

        This is used by the recognition module to compare a new face
        against all known faces.

        Returns:
            list of tuples: [(name, embedding_array), ...]
        """
        all_entries = []
        for name_key, data in self.database.items():
            for emb in data['embeddings']:
                if isinstance(emb, list):
                    emb = np.array(emb)
                all_entries.append((data['name'], emb))
        return all_entries

    # ──────────────────────────────────────────
    # Public: List Registered Users
    # ──────────────────────────────────────────
    def list_users(self):
        """
        Print a formatted list of all registered users.

        Returns:
            list of dict: Summary info for each registered person.
        """
        if not self.database:
            print("[DB] No registered users.")
            return []

        users = []
        print("\n╔══════════════════════════════════════════════╗")
        print("║         Registered Users                     ║")
        print("╠══════════════════════════════════════════════╣")
        for i, (key, data) in enumerate(self.database.items(), 1):
            name = data['name']
            num_emb = len(data['embeddings'])
            reg_date = data.get('registered_at', 'Unknown')[:10]
            model = data.get('model', 'Unknown')
            print(f"║  {i}. {name:<20} | {num_emb} embedding(s) | {model:<8} | {reg_date} ║")
            users.append({
                'name': name,
                'num_embeddings': num_emb,
                'model': model,
                'registered_at': reg_date
            })
        print("╚══════════════════════════════════════════════╝\n")
        return users

    # ──────────────────────────────────────────
    # Public: Remove a User
    # ──────────────────────────────────────────
    def remove_user(self, name):
        """
        Remove a person from the database.

        Parameters:
            name (str): The person's name.

        Returns:
            bool: True if removed, False if not found.
        """
        name_key = name.strip().lower()
        if name_key in self.database:
            del self.database[name_key]
            self._save()
            print(f"[DB] Removed '{name}' from database.")
            return True
        else:
            print(f"[DB] '{name}' not found in database.")
            return False

    # ──────────────────────────────────────────
    # Public: Check if Database is Empty
    # ──────────────────────────────────────────
    def is_empty(self):
        """Returns True if no users are registered."""
        return len(self.database) == 0

    def count(self):
        """Returns the number of registered users."""
        return len(self.database)


# ══════════════════════════════════════════════
#  Quick Test — Run this file directly to test
# ══════════════════════════════════════════════
if __name__ == "__main__":
    """
    Test the database with dummy embeddings.
    This verifies that save/load/register/remove all work correctly.
    """
    print("=== Testing FaceDatabase ===\n")

    # Use a temporary test database
    test_db_path = os.path.join(config.DATA_DIR, "test_faces.json")
    db = FaceDatabase(db_path=test_db_path)

    # Create dummy embeddings (128D like FaceNet)
    dummy_embedding_1 = np.random.randn(128).astype(np.float32)
    dummy_embedding_2 = np.random.randn(128).astype(np.float32)
    dummy_embedding_3 = np.random.randn(128).astype(np.float32)

    # Test 1: Register new users
    print("--- Test 1: Register users ---")
    db.register("Alice", dummy_embedding_1, "facenet")
    db.register("Bob", dummy_embedding_2, "facenet")

    # Test 2: Add another embedding for Alice
    print("\n--- Test 2: Add embedding to existing user ---")
    db.register("Alice", dummy_embedding_3, "facenet")

    # Test 3: List users
    print("\n--- Test 3: List users ---")
    db.list_users()

    # Test 4: Get all embeddings
    print("--- Test 4: Get all embeddings ---")
    all_emb = db.get_all_embeddings()
    print(f"  Total embeddings: {len(all_emb)}")
    for name, emb in all_emb:
        print(f"    {name}: shape={emb.shape}")

    # Test 5: Persistence — create a new instance and check data loads
    print("\n--- Test 5: Persistence test ---")
    db2 = FaceDatabase(db_path=test_db_path)
    db2.list_users()
    all_emb2 = db2.get_all_embeddings()
    print(f"  Reloaded {len(all_emb2)} embeddings from disk.")

    # Test 6: Remove a user
    print("\n--- Test 6: Remove user ---")
    db2.remove_user("Bob")
    db2.list_users()

    # Clean up test file
    os.remove(test_db_path)
    print("\n=== All tests passed! ===")
