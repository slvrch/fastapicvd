from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase_client import supabase
from datetime import datetime, timezone
from typing import Dict, Any


app = FastAPI()

class User(BaseModel):
    nama : str
    email: str
    no_tlp: str

@app.post("/register", response_model=Dict[str, Any])
def register_user(user: User) -> Dict[str, Any]:
    try:
        result = supabase.table("users").insert({
            "nama": user.nama,
            "email": user.email,
            "no_tlp": user.no_tlp,
            "registered_at": datetime.now(timezone.utc).isoformat()
        }).execute()
        
        if getattr(result, "error", None) is None:
            return {
                "status": "success",
                "user_id": result.data[0]["id"]
            }
        else:
            raise HTTPException(status_code=500, detail="Gagal menyimpan data pengguna.")        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    