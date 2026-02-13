-- Migration: Simplify agent_variant to 'remote' / 'local' only
-- Run this against an existing database to update variant names.
--
-- This migration:
-- 1. Migrates legacy variant names ('premium', 'cheap') to 'remote'
-- 2. Drops the existing CHECK constraint on agent_variant
-- 3. Adds a new CHECK constraint that allows only 'remote' and 'local'
-- 4. Updates the default value to 'remote' for new users
--
-- Usage:
--   psql -h localhost -U postgres -d sujbot -f scripts/migrate_add_cheap_variant.sql

BEGIN;

-- Migrate legacy variant names to 'remote'
UPDATE auth.users SET agent_variant = 'remote' WHERE agent_variant IN ('premium', 'cheap');

-- Drop existing constraint
ALTER TABLE auth.users DROP CONSTRAINT IF EXISTS users_agent_variant_check;

-- Add new constraint with only 'remote' and 'local'
ALTER TABLE auth.users ADD CONSTRAINT users_agent_variant_check
    CHECK (agent_variant IN ('remote', 'local'));

-- Update default to 'remote' for new users
ALTER TABLE auth.users ALTER COLUMN agent_variant SET DEFAULT 'remote';

COMMIT;

-- Verify the change
SELECT
    column_name,
    column_default,
    is_nullable
FROM information_schema.columns
WHERE table_schema = 'auth'
  AND table_name = 'users'
  AND column_name = 'agent_variant';
