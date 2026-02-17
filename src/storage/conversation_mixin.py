"""
Conversation storage mixin — shared conversation CRUD methods.

Both PostgresVectorStoreAdapter and PostgreSQLStorageAdapter use identical
conversation management logic. This mixin provides the shared implementation.

Requires the mixing class to have a `pool` attribute (asyncpg.Pool).
"""

import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ConversationStorageMixin:
    """Mixin providing async conversation CRUD against auth.conversations/messages."""

    async def create_conversation(
        self, conversation_id: str, user_id: int, title: Optional[str] = None
    ) -> None:
        """Create new conversation owned by user."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO auth.conversations (id, user_id, title, created_at, updated_at)
                VALUES ($1, $2, $3, NOW(), NOW())
                """,
                conversation_id,
                user_id,
                title,
            )
            logger.debug(f"Created conversation {conversation_id} for user {user_id}")

    async def get_user_conversations(self, user_id: int, limit: int = 50) -> List[Dict]:
        """Get all conversations for user (ordered by most recent)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       COALESCE((SELECT COUNT(*) FROM auth.messages m
                                 WHERE m.conversation_id = c.id), 0)::int as message_count
                FROM auth.conversations c
                WHERE c.user_id = $1
                ORDER BY c.updated_at DESC
                LIMIT $2
                """,
                user_id,
                limit,
            )
            conversations = [dict(row) for row in rows]
            logger.debug(f"Retrieved {len(conversations)} conversations for user {user_id}")
            return conversations

    async def verify_conversation_ownership(self, conversation_id: str, user_id: int) -> bool:
        """Check if user owns conversation."""
        async with self.pool.acquire() as conn:
            owner_id = await conn.fetchval(
                "SELECT user_id FROM auth.conversations WHERE id = $1", conversation_id
            )
            is_owner = owner_id == user_id
            logger.debug(
                f"Ownership check: conversation {conversation_id}, "
                f"user {user_id}, owner {owner_id}, result {is_owner}"
            )
            return is_owner

    async def get_conversation_history(self, conversation_id: str, limit: int = 100) -> List[Dict]:
        """Get message history for conversation."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, role, content, metadata, created_at
                FROM auth.messages
                WHERE conversation_id = $1
                ORDER BY created_at ASC
                LIMIT $2
                """,
                conversation_id,
                limit,
            )
            messages = []
            for row in rows:
                msg = dict(row)
                if msg.get("metadata") and isinstance(msg["metadata"], str):
                    msg["metadata"] = json.loads(msg["metadata"])
                messages.append(msg)

            logger.debug(f"Retrieved {len(messages)} messages for conversation {conversation_id}")
            return messages

    async def append_message(
        self, conversation_id: str, role: str, content: str, metadata: Optional[Dict] = None
    ) -> int:
        """Append message to conversation."""
        async with self.pool.acquire() as conn:
            metadata_json = json.dumps(metadata) if metadata else None

            message_id = await conn.fetchval(
                """
                INSERT INTO auth.messages (conversation_id, role, content, metadata, created_at)
                VALUES ($1, $2, $3, $4::jsonb, NOW())
                RETURNING id
                """,
                conversation_id,
                role,
                content,
                metadata_json,
            )

            await conn.execute(
                "UPDATE auth.conversations SET updated_at = NOW() WHERE id = $1", conversation_id
            )

            logger.debug(
                f"Appended {role} message (id={message_id}) to conversation {conversation_id}"
            )
            return message_id

    async def delete_conversation(self, conversation_id: str, user_id: int) -> bool:
        """Delete conversation (with ownership check)."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM auth.conversations WHERE id = $1 AND user_id = $2",
                conversation_id,
                user_id,
            )
            deleted = result.split()[-1] == "1"
            logger.debug(f"Delete conversation {conversation_id} for user {user_id}: {deleted}")
            return deleted
