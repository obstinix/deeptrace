"""
scripts/verify_auth.py

Auto-verification suite for API Key Auth & Rate Limiting.
Creates temporary test keys, performs API calls against a running server,
and asserts correct behavior (200, 401, 403, 429).
"""
import asyncio
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

# Add root directory to path to import api
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.auth.keys import init_db, create_key, revoke_key

BASE_URL = "http://localhost:8000"


def make_request(path: str, key: str = None, method: str = "GET", data: dict = None) -> tuple:
    url = f"{BASE_URL}{path}"
    headers = {}
    if key:
        headers["X-API-Key"] = key
    
    req_data = None
    if data is not None:
        req_data = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, headers=headers, method=method, data=req_data)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = response.read().decode("utf-8")
            try:
                res_json = json.loads(res_data) if res_data else {}
            except Exception as pe:
                print(f"DEBUG: Parse failed. Status: {response.status}, Content: {repr(res_data)}")
                raise pe
            return response.status, res_json, dict(response.headers)
    except urllib.error.HTTPError as e:
        res_data = e.read().decode("utf-8")
        try:
            res_json = json.loads(res_data) if res_data else {}
        except Exception as pe:
            print(f"DEBUG: Error parse failed. Status: {e.code}, Content: {repr(res_data)}")
            raise pe
        return e.code, res_json, dict(e.headers)
    except Exception as e:
        print(f"Connection failed: {e}")
        return 0, {}, {}


async def main():
    print("Initializing Auth Database...")
    await init_db()

    print("Creating temporary verification keys...")
    admin_key_info = await create_key(name="verify-admin", tier="admin", notes="Temp key for verification")
    pro_key_info = await create_key(name="verify-pro", tier="pro", notes="Temp key for verification")
    free_key_info = await create_key(name="verify-free", tier="free", notes="Temp key for verification")

    admin_key = admin_key_info["raw_key"]
    pro_key = pro_key_info["raw_key"]
    free_key = free_key_info["raw_key"]

    failures = 0

    print("\n--- Test 1: Public endpoint /api/health ---")
    status, body, headers = make_request("/api/health")
    if status == 200:
        print("PASS: /api/health accessible without key")
    else:
        print(f"FAIL: /api/health returned status {status}")
        failures += 1

    print("\n--- Test 2: Gated endpoint without API key ---")
    status, body, headers = make_request("/api/keys/me/usage")
    if status == 401:
        print("PASS: Missing API key returns 401")
        if "www-authenticate" in headers:
            print("PASS: WWW-Authenticate header found")
        else:
            print("FAIL: WWW-Authenticate header missing")
            failures += 1
    else:
        print(f"FAIL: Gated endpoint returned status {status} (expected 401)")
        failures += 1

    print("\n--- Test 3: Gated endpoint with invalid API key ---")
    status, body, headers = make_request("/api/keys/me/usage", key="dt_invalidkeyhere12345")
    if status == 401:
        print("PASS: Invalid API key returns 401")
    else:
        print(f"FAIL: Invalid API key returned status {status} (expected 401)")
        failures += 1

    print("\n--- Test 4: Gated endpoint with valid API key ---")
    status, body, headers = make_request("/api/keys/me/usage", key=free_key)
    if status == 200:
        print("PASS: Valid key returns 200")
        # Check rate limit headers
        if "x-ratelimit-minute-limit" in headers:
            print(f"PASS: X-RateLimit headers present (Limit: {headers.get('x-ratelimit-minute-limit')})")
        else:
            print("FAIL: X-RateLimit headers missing")
            failures += 1
    else:
        print(f"FAIL: Valid key returned status {status}")
        failures += 1

    print("\n--- Test 5: Admin Gating ---")
    # Try admin endpoint /api/keys with free key
    status, body, headers = make_request("/api/keys", key=free_key)
    if status == 403:
        print("PASS: Free key cannot access admin endpoints (403)")
    else:
        print(f"FAIL: Free key on admin endpoint returned {status} (expected 403)")
        failures += 1

    # Try with admin key
    status, body, headers = make_request("/api/keys", key=admin_key)
    if status == 200:
        print("PASS: Admin key can access admin endpoints (200)")
    else:
        print(f"FAIL: Admin key on admin endpoint returned {status}")
        failures += 1

    print("\n--- Test 6: Rate Limiting (Free Key limit 10/min) ---")
    # Create a fresh key to start with a clean rate limit bucket
    limit_key_info = await create_key(name="verify-limit", tier="free", notes="Temp key for limit check")
    limit_key = limit_key_info["raw_key"]

    success_count = 0
    throttled = False
    for i in range(11):
        status, body, headers = make_request("/api/keys/me/usage", key=limit_key)
        if status == 200:
            success_count += 1
        elif status == 429:
            throttled = True
            print(f"PASS: Request {i+1} got 429. Resets in {body.get('detail')}")
            if "retry-after" in headers:
                print(f"PASS: Retry-After header present: {headers.get('retry-after')}s")
            else:
                print("FAIL: Retry-After header missing on 429 response")
                failures += 1
            break
        else:
            print(f"Unexpected status code {status} at request {i+1}")
            break
    
    if success_count == 10 and throttled:
        print("PASS: Successfully rate limited after 10 requests")
    else:
        print(f"FAIL: Throttling failed. Success count: {success_count}, Throttled: {throttled}")
        failures += 1

    print("\n--- Cleaning up temporary keys... ---")
    await revoke_key(admin_key_info["id"])
    await revoke_key(pro_key_info["id"])
    await revoke_key(free_key_info["id"])
    await revoke_key(limit_key_info["id"])
    print("Cleanup done.")

    if failures == 0:
        print("\nALL AUTH VERIFICATION TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print(f"\nAUTH VERIFICATION FAILED: {failures} test(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
