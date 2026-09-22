const API = {
    baseUrl: '/api',

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            headers: { 'Content-Type': 'application/json' },
            ...options,
        };
        try {
            const response = await fetch(url, config);
            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(error.detail || 'Request failed');
            }
            return await response.json();
        } catch (error) {
            console.error(`API Error [${endpoint}]:`, error);
            throw error;
        }
    },

    async createSession(youtubeUrl) {
        return this.request('/sessions', {
            method: 'POST',
            body: JSON.stringify({ youtube_url: youtubeUrl }),
        });
    },

    async getSession(sessionId) {
        return this.request(`/sessions/${sessionId}`);
    },

    async getSessionStatus(sessionId) {
        return this.request(`/sessions/${sessionId}/status`);
    },

    async processVideo(sessionId) {
        return this.request(`/sessions/${sessionId}/process`, { method: 'POST' });
    },

    async getFrames(sessionId) {
        return this.request(`/sessions/${sessionId}/frames`);
    },

    // ---- Unified agent endpoint: POST /sessions/{id}/agent -------------------------------------

    /** Run one turn and wait for the complete JSON response (no streaming). */
    async agentOnce(sessionId, body) {
        return this.request(`/sessions/${sessionId}/agent`, {
            method: 'POST',
            body: JSON.stringify({ ...body, stream: false }),
        });
    },

    /**
     * Run one turn as a Server-Sent-Events stream. `handlers` maps event types to callbacks:
     *   start | route | token | tool_start | tool_end | retry | audio | notes_progress | interrupt | error | final
     * Resolves with the `final` event's data (rejects on a stream `error` event or an HTTP error).
     */
    async agentStream(sessionId, body, handlers = {}, signal) {
        const response = await fetch(`${this.baseUrl}/sessions/${sessionId}/agent`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
            body: JSON.stringify({ ...body, stream: true }),
            signal,
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(typeof err.detail === 'string' ? err.detail : 'Request failed');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let finalEvent = null;

        const dispatch = (raw) => {
            let type = 'message';
            const data = [];
            for (const line of raw.split('\n')) {
                if (line.startsWith('event:')) type = line.slice(6).trim();
                else if (line.startsWith('data:')) data.push(line.slice(5).trim());
            }
            if (type === 'done') return;
            const payload = data.length ? JSON.parse(data.join('\n')) : {};
            if (type === 'final') finalEvent = payload;
            if (type === 'error') throw new Error(payload.message || 'Agent run failed');
            if (handlers[type]) handlers[type](payload);
        };

        for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            let idx;
            while ((idx = buffer.indexOf('\n\n')) !== -1) {
                const raw = buffer.slice(0, idx);
                buffer = buffer.slice(idx + 2);
                if (raw.trim()) dispatch(raw);
            }
        }
        if (buffer.trim()) dispatch(buffer);
        if (!finalEvent) throw new Error('The connection closed before the response completed.');
        return finalEvent;
    },

    /** Checkpointed state: viva_score, quiz (no answer key), test_result, notes_draft, interrupt, ... */
    async getAgentState(sessionId) {
        return this.request(`/sessions/${sessionId}/agent/state`);
    },

    async getAgentHistory(sessionId, agent) {
        const qs = agent ? `?agent=${encodeURIComponent(agent)}` : '';
        return this.request(`/sessions/${sessionId}/agent/history${qs}`);
    },
};
