-- SQL
CREATE TABLE batch_log (
  id SERIAL PRIMARY KEY,
  batch INTEGER,
  comment TEXT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE document
  ADD COLUMN event_date TIMESTAMP NULL,
  ADD COLUMN created_at TIMESTAMP NULL;

CREATE TABLE country_lookup (
  id SERIAL PRIMARY KEY,
  country_code TEXT,
  country_m49 TEXT,
  country_name TEXT,
  region TEXT NULL,
  subregion TEXT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE document_country (
  document_id INTEGER REFERENCES document(id),
  country_id INTEGER REFERENCES country_lookup(id),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
