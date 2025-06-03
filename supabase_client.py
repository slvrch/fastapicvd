from supabase import create_client, Client
from dotenv import load_dotenv
import os

# Load environment variables from .end file
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Validasi env vars
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY belum diset di environment variables.")

# Membuat klien Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)