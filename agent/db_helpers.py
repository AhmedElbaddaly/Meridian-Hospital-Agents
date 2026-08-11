
import sqlite3
import os
import re
import sys

# This file's own directory is agent/, NOT the project root -- the real
# schema.sql / seed.sql / meridian_hospital.db all live in db/ at the repo
# root (used by mcp_server/db_helpers.py too). Pointing at agent/ here
# would silently look for agent/schema.sql (which doesn't exist), skip
# initialization entirely, and fall back to a throwaway, Unix-only
# /tmp/meridian_hospital.db with no tables in it -- every query would then
# fail with "no such table". Fixed to point at the SAME real database
# mcp_server/ uses, so this mock server (agent/db__server.py) sees the same
# seeded patients/admissions/beds data instead of an empty parallel copy.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DB_DIR = os.path.join(ROOT_DIR, "db")
DB_PATH = os.environ.get("MERIDIAN_DB_PATH", os.path.join(DB_DIR, "meridian_hospital.db"))

SCHEMA_PATH = os.path.join(DB_DIR, "schema.sql")
SEED_PATH = os.path.join(DB_DIR, "seed.sql")


class _AutoCloseConnection(sqlite3.Connection):
    """See mcp_server/db_helpers.py's _AutoCloseConnection: plain
    sqlite3.Connection's `with conn:` only commits/rolls back, it never
    closes the file handle -- harmless on Linux/macOS, but locks the .db
    file on Windows. This makes every `with get_connection() as conn:`
    below close automatically on every platform."""

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            self.close()


def get_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, factory=_AutoCloseConnection)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def _clean_sql_for_sqlite(sql_content: str) -> str:
    sql_content = re.sub(r'CREATE DATABASE\s+\[?\w+\]?;?', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'USE\s+\[?\w+\]?;?', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'^\s*GO\s*$', '', sql_content, flags=re.IGNORECASE | re.MULTILINE)
    sql_content = re.sub(
        r'INT\s+IDENTITY\(\s*1\s*,\s*1\s*\)\s+PRIMARY\s+KEY',
        'INTEGER PRIMARY KEY AUTOINCREMENT', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'IDENTITY\(\s*1\s*,\s*1\s*\)', 'AUTOINCREMENT', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'GETDATE\(\)', 'CURRENT_TIMESTAMP', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'NVARCHAR\(\w+\)', 'TEXT', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'VARCHAR\(\w+\)', 'TEXT', sql_content, flags=re.IGNORECASE)
    return sql_content

def init_db():
    with get_connection() as conn:
        cursor = conn.cursor()
        if os.path.exists(SCHEMA_PATH):
            with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
                schema_sql = _clean_sql_for_sqlite(f.read())
            try:
                cursor.executescript(schema_sql)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    print(f"[db] schema error: {e}", file=sys.stderr)
        if os.path.exists(SEED_PATH):
            # Only seed if empty
            try:
                n = cursor.execute("SELECT COUNT(*) FROM Patients").fetchone()[0]
            except Exception:
                n = 0
            if n == 0:
                with open(SEED_PATH, 'r', encoding='utf-8') as f:
                    seed_sql = _clean_sql_for_sqlite(f.read())
                try:
                    cursor.executescript(seed_sql)
                except Exception as e:
                    if "unique" not in str(e).lower():
                        print(f"[db] seed error: {e}", file=sys.stderr)
        conn.commit()

# ---------- helpers matching tools ----------

def add_patient(patient_data: dict):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO Patients (name, age, gender, blood_type, diagnosis) VALUES (?,?,?,?,?)",
            (patient_data['name'], patient_data['age'], patient_data['gender'],
             patient_data.get('blood_type'), patient_data.get('diagnosis')))
        conn.commit()
        return cur.lastrowid

def update_patient_status(patient_id: int, status: str):
    with get_connection() as conn:
        conn.execute("UPDATE Patients SET status = ? WHERE patient_id = ?", (status, patient_id))
        conn.commit()

def get_patient(patient_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM Patients WHERE patient_id = ?", (patient_id,)).fetchone()
        return dict(row) if row else None

def add_admission(admission_data: dict):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO Admissions (patient_id, doctor_id, room_id, status) VALUES (?,?,?,?)",
            (admission_data['patient_id'], admission_data['doctor_id'],
             admission_data.get('room_id'), admission_data.get('status', 'Active')))
        conn.commit()
        return cur.lastrowid

def update_room_status(room_id: int, status: str):
    with get_connection() as conn:
        conn.execute("UPDATE Operating_Rooms SET status = ? WHERE room_id = ?", (status, room_id))
        conn.commit()

def update_icu_bed(bed_id: int, patient_id=None):
    with get_connection() as conn:
        status = 'Occupied' if patient_id else 'Available'
        conn.execute(
            "UPDATE ICU_Beds SET status = ?, patient_id = ? WHERE bed_id = ?",
            (status, patient_id, bed_id))
        conn.commit()

def get_free_icu_beds():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM ICU_Beds WHERE status = 'Available'").fetchall()
        return [dict(r) for r in rows]

def get_icu_bed_by_number(bed_number: str):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM ICU_Beds WHERE bed_number = ?", (bed_number,)).fetchone()
        return dict(row) if row else None

def get_icu_bed(bed_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM ICU_Beds WHERE bed_id = ?", (bed_id,)).fetchone()
        return dict(row) if row else None

def count_available_icu_at_hospital():
    """All free beds (single-facility schema — no per-hospital bed link)."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM ICU_Beds WHERE status = 'Available'").fetchone()[0]

def get_available_operating_rooms():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM Operating_Rooms WHERE status = 'Available'").fetchall()
        return [dict(r) for r in rows]

def get_operating_room_by_number(room_number: str):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM Operating_Rooms WHERE room_number = ?", (room_number,)).fetchone()
        return dict(row) if row else None

def get_operating_room(room_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM Operating_Rooms WHERE room_id = ?", (room_id,)).fetchone()
        return dict(row) if row else None

def reserve_operating_room(room_id: int):
    with get_connection() as conn:
        conn.execute(
            "UPDATE Operating_Rooms SET status = 'Occupied' WHERE room_id = ?", (room_id,))
        conn.commit()

def get_hospitals():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM Hospitals").fetchall()
        return [dict(r) for r in rows]

def get_hospital_info(hospital_id: int = 1):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM Hospitals WHERE hospital_id = ?", (hospital_id,)).fetchone()
        return dict(row) if row else None

def get_user(user_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM Users WHERE user_id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

def get_doctor_by_id_str(doctor_id: str):
    """Accept D001 / D003 style or plain integer."""
    with get_connection() as conn:
        if isinstance(doctor_id, str) and doctor_id.upper().startswith("D"):
            try:
                num = int(doctor_id[1:])
            except ValueError:
                return None
            # seed: doctors are user_id 3..6 — map D001->first doctor etc.
            rows = conn.execute(
                "SELECT * FROM Users WHERE role = 'Doctor' ORDER BY user_id").fetchall()
            if 1 <= num <= len(rows):
                return dict(rows[num - 1])
            return None
        try:
            uid = int(doctor_id)
        except (TypeError, ValueError):
            return None
        row = conn.execute("SELECT * FROM Users WHERE user_id = ? AND role = 'Doctor'", (uid,)).fetchone()
        return dict(row) if row else None

def parse_patient_id(patient_id):
    """Accept P001 / 1 / '1'."""
    if isinstance(patient_id, int):
        return patient_id
    s = str(patient_id).strip()
    if s.upper().startswith("P"):
        try:
            return int(s[1:])
        except ValueError:
            return None
    try:
        return int(s)
    except ValueError:
        return None

# Initialize on import
init_db()
