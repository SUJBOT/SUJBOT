#!/usr/bin/env python3
"""
Reset admin user password to a known value.

Usage:
    python scripts/reset_admin_password.py

Or via Docker:
    docker-compose exec backend python scripts/reset_admin_password.py
"""

import asyncio
import os
import sys

import asyncpg
from argon2 import PasswordHasher


async def reset_password():
    """Reset admin password."""

    # Get database URL from environment
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ ERROR: DATABASE_URL environment variable not set")
        return False

    print("🔄 Resetting admin password...")

    conn = None
    try:
        # Connect to database
        conn = await asyncpg.connect(db_url)
        print("✅ Connected to database")

        # Hash new password
        ph = PasswordHasher(
            time_cost=3,
            memory_cost=65536,
            parallelism=4,
            hash_len=32,
            salt_len=16
        )

        new_password = "ChangeThisPassword123!"
        password_hash = ph.hash(new_password)

        # Update admin user
        result = await conn.execute(
            """
            UPDATE auth.users
            SET password_hash = $1, updated_at = NOW()
            WHERE email = 'admin@sujbot.local'
            """,
            password_hash
        )

        # asyncpg returns string like "UPDATE 1" on success
        if result != "UPDATE 1":
            print(f"\n❌ Password reset failed: No user found with email 'admin@sujbot.local'")
            print(f"   Database returned: {result}")
            print("   Tip: Run the database seed script to create the admin user")
            return False

        print("\n✅ Password reset successfully!")
        print("   📧 Email: admin@sujbot.local")
        print("   🔑 Password: ChangeThisPassword123!")

        return True

    except Exception as e:
        print(f"\n❌ Password reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Ensure connection is always closed
        if conn is not None:
            await conn.close()


if __name__ == "__main__":
    success = asyncio.run(reset_password())
    sys.exit(0 if success else 1)
