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

    async generateNotes(sessionId) {
        return this.request(`/sessions/${sessionId}/notes`, { method: 'POST' });
    },

    async getNotes(sessionId) {
        return this.request(`/sessions/${sessionId}/notes`);
    },

    async sendChat(sessionId, message, agentType = 'doubt') {
        return this.request(`/sessions/${sessionId}/chat`, {
            method: 'POST',
            body: JSON.stringify({ message, agent_type: agentType }),
        });
    },

    async getChatHistory(sessionId, agentType = 'doubt') {
        return this.request(`/sessions/${sessionId}/chat/history?agent_type=${agentType}`);
    },

    async generateTest(sessionId, numQuestions = 10, difficulty = 'medium') {
        return this.request(`/sessions/${sessionId}/test/generate`, {
            method: 'POST',
            body: JSON.stringify({ num_questions: numQuestions, difficulty }),
        });
    },

    async submitTest(sessionId, answers) {
        return this.request(`/sessions/${sessionId}/test/submit`, {
            method: 'POST',
            body: JSON.stringify({ answers }),
        });
    },

    async getTestResults(sessionId) {
        return this.request(`/sessions/${sessionId}/test/results`);
    },
};
