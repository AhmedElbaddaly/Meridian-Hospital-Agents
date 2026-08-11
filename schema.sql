-- memory/schema.sql
--
-- Extends the existing db/meridian_hospital.db with long-term memory tables.
-- These tables live in the SAME database file as Patients/Admissions/etc. so
-- the agent can join memory rows against real patient_id values instead of
-- duplicating hospital data. Nothing here touches or redefines the existing
-- tables from db/schema.sql.

-- ---------------------------------------------------------------------------
-- EPISODIC MEMORY
-- One row per meaningful event the agent decided to keep (via promote-or-drop
-- routing in memory/promote_or_drop.py). Never written to directly by
-- consolidation -- consolidation only READS from this table.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS episodic_memory (
    episode_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER,                -- FK -> Patients.patient_id (nullable: not every episode is patient-specific)
    user_id         INTEGER,                -- FK -> Users.user_id (staff member in the conversation)
    session_id      TEXT NOT NULL,          -- which conversation/session this came from
    timestamp       TEXT NOT NULL,          -- ISO-8601 UTC
    event_summary   TEXT NOT NULL,          -- what happened
    context         TEXT,                   -- why / surrounding circumstances
    outcome         TEXT,                   -- what was decided / resulted
    source_role     TEXT,                   -- 'user' | 'assistant' | 'tool' (who produced the raw item)
    raw_item        TEXT,                   -- original short-term memory item, for audit
    routing_reason  TEXT NOT NULL,          -- the router's logged reasoning for promoting this item
    consolidated    INTEGER NOT NULL DEFAULT 0,  -- 0/1: has this episode already been folded into a semantic fact?

    FOREIGN KEY (patient_id) REFERENCES Patients(patient_id),
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
);

CREATE INDEX IF NOT EXISTS idx_episodic_patient ON episodic_memory(patient_id);
CREATE INDEX IF NOT EXISTS idx_episodic_consolidated ON episodic_memory(consolidated);

-- ---------------------------------------------------------------------------
-- SEMANTIC MEMORY (current, "live" facts only)
-- Built ONLY by the consolidation pass (memory/consolidation.py), on a
-- schedule, never at write time and never directly by the promote/drop
-- router. Each fact is keyed by (patient_id, fact_key) so a new value
-- naturally supersedes -- but the OLD value is preserved in
-- semantic_memory_history, never silently dropped.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS semantic_memory (
    patient_id      INTEGER NOT NULL,
    fact_key        TEXT NOT NULL,          -- e.g. 'allergy:penicillin', 'seat_preference'
    fact_value      TEXT NOT NULL,          -- e.g. 'confirmed_allergic', 'window seat'
    confidence      REAL NOT NULL DEFAULT 1.0,
    version         INTEGER NOT NULL DEFAULT 1,
    status          TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'expired', 'contradicted')),
    source_episode_ids TEXT NOT NULL,       -- JSON list of episode_id's this fact was derived from
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    expires_at      TEXT,                   -- NULL = does not expire

    PRIMARY KEY (patient_id, fact_key),
    FOREIGN KEY (patient_id) REFERENCES Patients(patient_id)
);

-- ---------------------------------------------------------------------------
-- SEMANTIC MEMORY HISTORY (append-only, full version trail)
-- Every insert/update/expire/contradiction-resolution to semantic_memory
-- writes a row here FIRST. This is what "versioning, never silently
-- overwritten" means in practice: semantic_memory is a materialized view of
-- "latest active version per fact_key", this table is the ledger.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS semantic_memory_history (
    history_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL,
    fact_key        TEXT NOT NULL,
    version         INTEGER NOT NULL,
    fact_value      TEXT NOT NULL,
    confidence      REAL NOT NULL,
    status          TEXT NOT NULL,          -- 'active' | 'superseded' | 'expired' | 'contradicted'
    change_reason   TEXT NOT NULL,          -- 'initial_write' | 'update' | 'expiration' | 'conflict_resolution'
    resolution_note TEXT,                   -- e.g. "kept newer episode #482 over older #310: more recent + higher confidence"
    source_episode_ids TEXT NOT NULL,
    recorded_at     TEXT NOT NULL,

    FOREIGN KEY (patient_id) REFERENCES Patients(patient_id)
);

CREATE INDEX IF NOT EXISTS idx_semantic_history_lookup ON semantic_memory_history(patient_id, fact_key);

-- ---------------------------------------------------------------------------
-- CONSOLIDATION RUN LOG
-- Evidence that consolidation is a genuinely separate, periodic pass, not
-- something happening inline at write time.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS consolidation_runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT NOT NULL,
    finished_at     TEXT NOT NULL,
    episodes_scanned INTEGER NOT NULL,
    facts_created   INTEGER NOT NULL,
    facts_updated   INTEGER NOT NULL,
    facts_expired   INTEGER NOT NULL,
    conflicts_resolved INTEGER NOT NULL,
    notes           TEXT
);
