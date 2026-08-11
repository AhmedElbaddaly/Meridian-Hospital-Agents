import sqlite3
import os
import re
import sys
import io

if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr and sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = os.path.dirname(BASE_DIR)

DB_DIR = os.path.join(ROOT_DIR, "db")

DB_PATH = os.path.join(DB_DIR, "meridian_hospital.db")
SCHEMA_PATH = os.path.join(DB_DIR, "schema.sql")
SEED_PATH = os.path.join(DB_DIR, "seed.sql")

class _AutoCloseConnection(sqlite3.Connection):
    """
    sqlite3.Connection's own `with conn:` block only commits or rolls back
    the transaction -- it never closes the underlying file handle. On
    Linux/macOS a still-open handle doesn't stop a file from being
    deleted/replaced, so this goes unnoticed there; on Windows it does,
    and any later attempt to reset or re-open the database file
    (os.remove, another process opening it exclusively, etc.) fails with
    `PermissionError: [WinError 32] ... used by another process`.

    Subclassing sqlite3.Connection and overriding __exit__ to also close()
    means every existing `with get_connection() as conn:` call site below
    is fixed automatically -- no other line in this file needs to change.
    """

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            self.close()


def get_connection():
    """Establish and return a SQLite database connection with row factory configured."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, factory=_AutoCloseConnection)
    conn.row_factory = sqlite3.Row
    return conn

def _clean_sql_for_sqlite(sql_content: str) -> str:
    """Dynamically transform MS SQL Server syntax to SQLite dialect."""
    # 1. Remove CREATE DATABASE, USE, and GO commands
    sql_content = re.sub(r'CREATE DATABASE\s+\[?\w+\]?;?', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'USE\s+\[?\w+\]?;?', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'^\s*GO\s*$', '', sql_content, flags=re.IGNORECASE | re.MULTILINE)

    # 2. Convert INT IDENTITY(1,1) to INTEGER PRIMARY KEY AUTOINCREMENT
    sql_content = re.sub(r'INT\s+IDENTITY\(\s*1\s*,\s*1\s*\)\s+PRIMARY\s+KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'IDENTITY\(\s*1\s*,\s*1\s*\)', 'AUTOINCREMENT', sql_content, flags=re.IGNORECASE)

    # 3. Convert GETDATE() to CURRENT_TIMESTAMP
    sql_content = re.sub(r'GETDATE\(\)', 'CURRENT_TIMESTAMP', sql_content, flags=re.IGNORECASE)

    # 4. Replace MSSQL specific type definitions
    sql_content = re.sub(r'NVARCHAR\(\w+\)', 'TEXT', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'VARCHAR\(\w+\)', 'TEXT', sql_content, flags=re.IGNORECASE)

    return sql_content

def init_db():
    """Read, sanitize, and execute schema.sql and seed.sql files automatically."""
    print(f"Checking Schema File: {SCHEMA_PATH} -> Exists: {os.path.exists(SCHEMA_PATH)}")
    print(f"Checking Seed File: {SEED_PATH} -> Exists: {os.path.exists(SEED_PATH)}")
    
    with get_connection() as conn:
        cursor = conn.cursor()

        # 1. Execute schema.sql
        if os.path.exists(SCHEMA_PATH):
            with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
                schema_sql = _clean_sql_for_sqlite(f.read())
                try:
                    cursor.executescript(schema_sql)
                    print("[OK] Schema executed successfully.")
                except Exception as e:
                    if "already exists" in str(e):
                        print("[INFO] Schema tables already exist, skipping execution.")
                    else:
                        print(f"[ERROR] Error executing schema.sql: {e}")
        else:
            print(f"[WARNING] {SCHEMA_PATH} not found!")

        # 2. Execute seed.sql
        if os.path.exists(SEED_PATH):
            with open(SEED_PATH, 'r', encoding='utf-8') as f:
                seed_sql = _clean_sql_for_sqlite(f.read())
                try:
                    cursor.executescript(seed_sql)
                    print("[OK] Seed executed successfully.")
                except Exception as e:
                    if "UNIQUE constraint failed" in str(e) or "already exists" in str(e):
                        print("[INFO] Seed data already initialized, skipping.")
                    else:
                        print(f"[ERROR] Error executing seed.sql: {e}")

        conn.commit()

# Execute automatic database initialization on import
init_db()

# ======================================================
# Database Helper Functions (Matching MCP.py Imports)
# ======================================================

def add_patient(patient_data: dict):
    """Register a new patient into the Patients table."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Patients (name, age, gender, blood_type, diagnosis)
            VALUES (?, ?, ?, ?, ?)
        """, (patient_data['name'], patient_data['age'], patient_data['gender'], 
              patient_data.get('blood_type'), patient_data.get('diagnosis')))
        conn.commit()
        return cursor.lastrowid

def update_patient_status(patient_id: int, status: str):
    """Update medical triage status for a patient."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE Patients SET status = ? WHERE patient_id = ?", (status, patient_id))
        conn.commit()

def get_patient(patient_id: int):
    """Retrieve patient record by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Patients WHERE patient_id = ?", (patient_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def add_admission(admission_data: dict):
    """Create a new hospital admission record."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Admissions (patient_id, doctor_id, room_id, status)
            VALUES (?, ?, ?, ?)
        """, (admission_data['patient_id'], admission_data['doctor_id'], 
              admission_data.get('room_id'), admission_data.get('status', 'Active')))
        conn.commit()
        return cursor.lastrowid

def update_room_status(room_id: int, status: str):
    """Update operating room availability status."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE Operating_Rooms SET status = ? WHERE room_id = ?", (status, room_id))
        conn.commit()

def update_icu_bed(bed_id: int, patient_id: int = None):
    """Assign an ICU bed to a patient or release it."""
    with get_connection() as conn:
        cursor = conn.cursor()
        status = 'Occupied' if patient_id else 'Available'
        cursor.execute("""
            UPDATE ICU_Beds SET status = ?, patient_id = ? WHERE bed_id = ?
        """, (status, patient_id, bed_id))
        conn.commit()

def get_free_icu_beds():
    """Fetch all available ICU beds."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM ICU_Beds WHERE status = 'Available'")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_hospital_info(hospital_id: int = 1):
    """Retrieve hospital capacity details."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Hospitals WHERE hospital_id = ?", (hospital_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
