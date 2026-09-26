CREATE TABLE sessions (token_hash text PRIMARY KEY, owner_id text NOT NULL, expires_at timestamptz NOT NULL);
CREATE TABLE conversations (id uuid PRIMARY KEY, owner_id text NOT NULL, revision integer NOT NULL DEFAULT 0);
CREATE TABLE turns (
 conversation_id uuid REFERENCES conversations(id) ON DELETE CASCADE,
 request_id uuid NOT NULL,
 fingerprint text NOT NULL,
 ordinal integer NOT NULL,
 user_text text NOT NULL,
 assistant_text text NOT NULL DEFAULT '',
 provider text NOT NULL,
 status text NOT NULL CHECK (status IN ('running','complete','partial','error','canceled')),
 error_code text,
 lease_until timestamptz NOT NULL,
 PRIMARY KEY (conversation_id, request_id),
 UNIQUE (conversation_id, ordinal)
);
CREATE UNIQUE INDEX one_running_turn ON turns(conversation_id) WHERE status='running';
