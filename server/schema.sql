CREATE SCHEMA IF NOT EXISTS ailearn;
SET search_path TO ailearn, public;
CREATE TABLE IF NOT EXISTS schema_migrations(version integer PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS users(id serial PRIMARY KEY, username text UNIQUE NOT NULL, password_hash text NOT NULL, created_at timestamptz NOT NULL DEFAULT now());
CREATE UNIQUE INDEX IF NOT EXISTS users_lower_username ON users(lower(username));
CREATE TABLE IF NOT EXISTS sessions(token_hash text PRIMARY KEY, user_id integer NOT NULL REFERENCES users ON DELETE CASCADE, expires_at timestamptz NOT NULL);
CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires_at);
CREATE TABLE IF NOT EXISTS articles(slug text PRIMARY KEY, category text NOT NULL, position integer NOT NULL, content jsonb NOT NULL);
CREATE TABLE IF NOT EXISTS cards(id serial PRIMARY KEY, article_slug text NOT NULL REFERENCES articles, content_key text UNIQUE NOT NULL, position integer NOT NULL, front text NOT NULL, back text NOT NULL, UNIQUE(id,article_slug));
CREATE TABLE IF NOT EXISTS user_articles(user_id integer NOT NULL REFERENCES users ON DELETE CASCADE, article_slug text NOT NULL REFERENCES articles, is_read boolean NOT NULL DEFAULT false, bookmarked boolean NOT NULL DEFAULT false, enrolled_at timestamptz, updated_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(user_id,article_slug));
CREATE TABLE IF NOT EXISTS user_cards(user_id integer NOT NULL REFERENCES users ON DELETE CASCADE, card_id integer NOT NULL, article_slug text NOT NULL, state jsonb NOT NULL, due_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(user_id,card_id), FOREIGN KEY(card_id,article_slug) REFERENCES cards(id,article_slug), FOREIGN KEY(user_id,article_slug) REFERENCES user_articles(user_id,article_slug) ON DELETE CASCADE);
CREATE INDEX IF NOT EXISTS user_cards_due ON user_cards(user_id,due_at);
CREATE TABLE IF NOT EXISTS reviews(id bigserial PRIMARY KEY, user_id integer NOT NULL, card_id integer NOT NULL, request_id uuid NOT NULL, quality integer NOT NULL CHECK(quality BETWEEN 0 AND 5), kind text NOT NULL CHECK(kind IN ('scheduled','practice')), response_ms integer NOT NULL CHECK(response_ms BETWEEN 0 AND 86400000), payload jsonb NOT NULL, result jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(user_id,request_id), FOREIGN KEY(user_id,card_id) REFERENCES user_cards(user_id,card_id) ON DELETE CASCADE);
INSERT INTO schema_migrations(version) VALUES(1) ON CONFLICT DO NOTHING;
-- Version 2: cards removed from an article's source are retired, not deleted, so review history survives rewrites.
ALTER TABLE cards ADD COLUMN IF NOT EXISTS retired boolean NOT NULL DEFAULT false;
INSERT INTO schema_migrations(version) VALUES(2) ON CONFLICT DO NOTHING;
