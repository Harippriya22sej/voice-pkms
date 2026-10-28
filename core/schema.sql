CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'text' CHECK (source IN ('text', 'voice')),
    audio_path  TEXT,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status      TEXT NOT NULL DEFAULT 'new'
);

CREATE TABLE IF NOT EXISTS fragments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id     INTEGER NOT NULL REFERENCES notes(id),
    category    TEXT,
    intent      TEXT,
    summary     TEXT,
    slots_json  TEXT,
    embedding   BLOB,
    stability   TEXT NOT NULL DEFAULT 'pending'
                CHECK (stability IN ('pending', 'stable', 'absorbed')),
    merged_into INTEGER REFERENCES fragments(id),
    confidence  REAL,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS knowledge (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    category      TEXT,
    title         TEXT NOT NULL,
    content_json  TEXT,
    embedding     BLOB,
    status        TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'superseded')),
    version       INTEGER NOT NULL DEFAULT 1,
    supersedes_id INTEGER REFERENCES knowledge(id),
    updated_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS knowledge_sources (
    knowledge_id INTEGER NOT NULL REFERENCES knowledge(id),
    fragment_id  INTEGER NOT NULL REFERENCES fragments(id),
    PRIMARY KEY (knowledge_id, fragment_id)
);

CREATE TABLE IF NOT EXISTS review_queue (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    type         TEXT NOT NULL CHECK (type IN ('fuse', 'contradict', 'stabilize')),
    payload_json TEXT,
    status       TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'rejected')),
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);