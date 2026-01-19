#!/usr/bin/env python3
"""
Reset any user's password by email.

Usage:
    python scripts/reset_user_password.py <email> <new_password>

Example:
    python scripts/reset_user_password.py user@example.com MyNewPassword123!

Or via Docker:
    docker-compose exec backend python scripts/reset_user_password.py <email> <new_password>
"""

import asyncio
import asyncpg
import os
import sys
from argon2 import PasswordHasher

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def reset_password(email: str, new_password: str):
    """Reset user password by email."""

    # Get database URL from environment
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ ERROR: DATABASE_URL environment variable not set")
        return False

    print(f"🔄 Resetting password for: {email}")

    conn = None
    try:
        # Connect to database
        conn = await asyncpg.connect(db_url)
        print("✅ Connected to database")

        # Hash new password with Argon2id (same params as AuthManager)
        ph = PasswordHasher(
            time_cost=3,
            memory_cost=65536,
            parallelism=4,
            hash_len=32,
            salt_len=16
        )

        password_hash = ph.hash(new_password)

        # Update user password
        result = await conn.execute(
            """
            UPDATE auth.users
            SET password_hash = $1, updated_at = NOW()
            WHERE email = $2
            """,
            password_hash,
            email
        )

        # Verify that exactly one row was updated
        if result != "UPDATE 1":
            print(f"\n❌ Password reset failed: No user found with email '{email}'")
            print(f"   Database returned: {result}")
            return False

        print("\n✅ Password reset successfully!")
        print(f"   📧 Email: {email}")
        print(f"   🔑 New password set")

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


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/reset_user_password.py <email> <new_password>")
        print("Example: python scripts/reset_user_password.py user@example.com MyNewPassword123!")
        sys.exit(1)

    email = sys.argv[1]
    new_password = sys.argv[2]

    # Basic validation
    if "@" not in email:
        print("❌ ERROR: Invalid email format")
        sys.exit(1)

    if len(new_password) < 8:
        print("⚠️  WARNING: Password is shorter than 8 characters (may not meet security requirements)")

    success = asyncio.run(reset_password(email, new_password))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
