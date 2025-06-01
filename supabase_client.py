from supabase import create_client, Client
from dotenv import load_dotenv
import os

# Load environment variables from .end file
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY belum diset di .env atau environment variabel.")
print("SUPABASE_URL:", SUPABASE_URL)
print("SUPABASE_KEY:", SUPABASE_KEY)

# Membuat klien Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)