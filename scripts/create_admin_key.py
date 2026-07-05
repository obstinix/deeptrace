"""
scripts/create_admin_key.py

First-run bootstrap: create the initial admin API key.
Run once before starting the server for the first time.

    python scripts/create_admin_key.py --name "admin"
    python scripts/create_admin_key.py --name "ci-bot" --tier pro
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.auth.keys import init_db, create_key, list_keys


async def main():
    p = argparse.ArgumentParser(description="Create a DeepTrace API key")
    p.add_argument("--name", required=True, help="Human-readable label for the key")
    p.add_argument("--tier", default="admin", choices=["free", "pro", "admin"])
    p.add_argument("--notes", default="", help="Optional notes")
    args = p.parse_args()

    await init_db()

    # Warn if admin key already exists
    existing = await list_keys()
    admin_keys = [k for k in existing if k["tier"] == "admin"]
    if admin_keys and args.tier == "admin":
        print(f"[warn] {len(admin_keys)} admin key(s) already exist:")
        for k in admin_keys:
            print(f"  {k['id'][:8]}… {k['name']}")
        ans = input("Create another? [y/N] ").strip().lower()
        if ans != "y":
            sys.exit(0)

    result = await create_key(
        name=args.name,
        tier=args.tier,
        notes=args.notes,
    )

    print("\n" + "-" * 60)
    print(f"  Name:   {result['name']}")
    print(f"  Tier:   {result['tier']}")
    print(f"  Key ID: {result['id']}")
    print(f"\n  [WARNING] RAW KEY (save this now - not stored):")
    print(f"\n  {result['raw_key']}\n")
    print("-" * 60)
    print(f"\nUsage:\n  curl -H 'X-API-Key: {result['raw_key']}' http://localhost:8000/api/health")


if __name__ == "__main__":
    asyncio.run(main())
