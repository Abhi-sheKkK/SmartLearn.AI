-- Supabase PostgreSQL Schema for SmartLearn.AI

-- 1. Sessions Table
CREATE TABLE IF NOT EXISTS public.sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    youtube_url TEXT NOT NULL,
    video_id VARCHAR(50) NOT NULL,
    title TEXT,
    status VARCHAR(50) DEFAULT 'created',
    transcript_text TEXT,
    frame_urls JSONB DEFAULT '[]'::jsonb,
    notes_markdown TEXT,
    viva_score JSONB DEFAULT '{"asked": 0, "correct": 0}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 2. Messages Table (for Doubt Chat and Viva interactions)
CREATE TABLE IF NOT EXISTS public.messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES public.sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    agent_type VARCHAR(20) DEFAULT 'doubt', -- 'doubt', 'viva', 'notes'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 3. Test Results Table
CREATE TABLE IF NOT EXISTS public.test_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES public.sessions(id) ON DELETE CASCADE,
    questions JSONB NOT NULL,
    user_answers JSONB NOT NULL,
    score INTEGER NOT NULL,
    total INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 4. Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON public.messages(session_id);
CREATE INDEX IF NOT EXISTS idx_test_results_session_id ON public.test_results(session_id);

-- Enable Row Level Security (RLS) - Allow public access for anon key
ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.test_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public read/write access to sessions" ON public.sessions FOR ALL USING (true);
CREATE POLICY "Allow public read/write access to messages" ON public.messages FOR ALL USING (true);
CREATE POLICY "Allow public read/write access to test_results" ON public.test_results FOR ALL USING (true);
