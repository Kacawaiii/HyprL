#!/usr/bin/env python3
"""
Test Alpaca connection and account status.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load env
config_dir = Path(__file__).parent.parent / "configs" / "runtime"
for env_file in [".env.normal", ".env.aggressive", ".env.mix"]:
    env_path = config_dir / env_file
    if env_path.exists():
        load_dotenv(env_path)
        break

def test_connection():
    """Test Alpaca API connection."""
    import requests
    
    api_key = os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("APCA_API_SECRET_KEY")
    base_url = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    
    if not api_key or not api_secret:
        print("❌ API keys not configured")
        print("   Copy configs/runtime/.env.example to .env.normal and add your keys")
        return False
    
    print("Testing Alpaca connection...")
    print(f"  API Key: {api_key[:8]}...")
    print(f"  Base URL: {base_url}")
    
    try:
        response = requests.get(
            f"{base_url}/v2/account",
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Connection successful!")
            print(f"   Account: {data['account_number']}")
            print(f"   Equity: ${float(data['equity']):,.2f}")
            print(f"   Cash: ${float(data['cash']):,.2f}")
            print(f"   Status: {data['status']}")
            return True
        else:
            print(f"\n❌ Connection failed: {response.status_code}")
            print(f"   {response.text}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
