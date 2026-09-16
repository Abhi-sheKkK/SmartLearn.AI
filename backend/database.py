from supabase import create_client, Client
from backend.config import get_settings

settings = get_settings()

try:
    if not settings.SUPABASE_URL or not settings.SUPABASE_KEY or settings.SUPABASE_URL == "your-supabase-project-url":
        print("Warning: Supabase credentials are not configured. Using local in-memory/disk database fallback.")
        client = None
    else:
        client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
except Exception as e:
    print(f"Warning: Failed to initialize Supabase client ({e}). Using local database fallback.")
    client = None

# Local fallbacks when Supabase is not configured
_local_sessions = {}
_local_messages = {}
_local_test_results = {}

def get_db():
    return client

def insert_session(data: dict) -> dict:
    if not client:
        session_id = data.get("id")
        _local_sessions[session_id] = {**data}
        return _local_sessions[session_id]
    response = client.table('sessions').insert(data).execute()
    return response.data[0] if response.data else {}

def get_session(session_id: str) -> dict:
    if not client:
        return _local_sessions.get(session_id, {})
    response = client.table('sessions').select('*').eq('id', session_id).execute()
    return response.data[0] if response.data else {}

def get_completed_session_by_video_id(video_id: str) -> dict | None:
    """Find an existing completed session for the given video_id."""
    if not client:
        for s in _local_sessions.values():
            if s.get("video_id") == video_id and s.get("status") in ("completed", "ready"):
                return s
        return None
    try:
        response = client.table('sessions').select('*').eq('video_id', video_id).in_('status', ['completed', 'ready']).execute()
        return response.data[0] if response.data else None
    except Exception:
        return None

_local_video_materials = {}

def save_video_material(video_id: str, material_type: str, data: any):
    """Save generated material (notes_markdown, test_data, etc.) for a video_id."""
    if not video_id:
        return
    if video_id not in _local_video_materials:
        _local_video_materials[video_id] = {}
    _local_video_materials[video_id][material_type] = data

    if client and material_type == "notes_markdown":
        try:
            client.table('sessions').update({"notes_markdown": data}).eq('video_id', video_id).execute()
        except Exception as e:
            print(f"Error saving video material to Supabase: {e}")

def get_video_material(video_id: str, material_type: str) -> any:
    """Retrieve saved material for a video_id across any session."""
    if not video_id:
        return None
    if video_id in _local_video_materials and material_type in _local_video_materials[video_id]:
        return _local_video_materials[video_id][material_type]
    
    existing = get_completed_session_by_video_id(video_id)
    if existing and existing.get(material_type):
        return existing.get(material_type)
    return None

def update_session(session_id: str, data: dict) -> dict:
    if not client:
        if session_id in _local_sessions:
            _local_sessions[session_id].update(data)
            return _local_sessions[session_id]
        return {"id": session_id, **data}
    response = client.table('sessions').update(data).eq('id', session_id).execute()
    return response.data[0] if response.data else {}

def insert_message(data: dict) -> dict:
    if not client:
        sid = data.get("session_id")
        if sid not in _local_messages:
            _local_messages[sid] = []
        _local_messages[sid].append(data)
        return data
    response = client.table('messages').insert(data).execute()
    return response.data[0] if response.data else {}

def get_messages(session_id: str, agent_type: str) -> list:
    if not client:
        messages = _local_messages.get(session_id, [])
        return [m for m in messages if m.get("agent_type") == agent_type]
    response = client.table('messages').select('*').eq('session_id', session_id).eq('agent_type', agent_type).execute()
    return response.data

def insert_test_result(data: dict) -> dict:
    if not client:
        sid = data.get("session_id")
        if sid not in _local_test_results:
            _local_test_results[sid] = []
        _local_test_results[sid].append(data)
        return data
    response = client.table('test_results').insert(data).execute()
    return response.data[0] if response.data else {}

def get_test_results(session_id: str) -> list:
    if not client:
        return _local_test_results.get(session_id, [])
    response = client.table('test_results').select('*').eq('session_id', session_id).execute()
    return response.data

def upload_frame(file_path: str, video_id: str) -> str:
    """Upload a frame image, stored/keyed by video_id so re-processing the same video reuses it."""
    if not client:
        import os
        file_name = os.path.basename(file_path)
        return f"/storage/{video_id}/unique_frames/{file_name}"
    import os
    file_name = os.path.basename(file_path)
    storage_path = f"{video_id}/{file_name}"
    with open(file_path, "rb") as f:
        client.storage.from_(settings.SUPABASE_STORAGE_BUCKET).upload(storage_path, f)
    public_url = client.storage.from_(settings.SUPABASE_STORAGE_BUCKET).get_public_url(storage_path)
    return public_url

def get_frame_url(file_path: str, video_id: str) -> str:
    """Build the public URL for an already-uploaded frame without re-uploading it."""
    import os
    file_name = os.path.basename(file_path)
    if not client:
        return f"/storage/{video_id}/unique_frames/{file_name}"
    storage_path = f"{video_id}/{file_name}"
    return client.storage.from_(settings.SUPABASE_STORAGE_BUCKET).get_public_url(storage_path)
