window.ChatModule = (() => {
    let currentSessionId = null;
    const msgContainer = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send-btn');

    const MATH_DELIMITERS = [
        {left: '$$', right: '$$', display: true},
        {left: '$', right: '$', display: false},
        {left: '\\(', right: '\\)', display: false},
        {left: '\\[', right: '\\]', display: true}
    ];

    const scrollToBottom = () => {
        msgContainer.scrollTop = msgContainer.scrollHeight;
    };

    // Markdown + math + code highlighting for a message body.
    const renderInto = (el, text) => {
        el.innerHTML = MD.render(text);
        if (window.renderMathInElement) {
            window.renderMathInElement(el, { delimiters: MATH_DELIMITERS, throwOnError: false });
        }
        el.querySelectorAll('pre code').forEach((block) => hljs.highlightElement(block));
    };

    const createMessage = (role) => {
        const div = document.createElement('div');
        div.className = `message ${role}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? 'U' : 'AI';

        const content = document.createElement('div');
        content.className = 'message-content';

        div.appendChild(avatar);
        div.appendChild(content);
        msgContainer.appendChild(div);
        return { div, content };
    };

    const appendMessage = (role, text) => {
        const { content } = createMessage(role);
        renderInto(content, text);
        scrollToBottom();
    };

    const typingHtml = `
        <div class="typing-indicator" style="padding:0">
            <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
        </div>`;

    const handleSend = async () => {
        const text = chatInput.value.trim();
        if (!text) return;

        chatInput.value = '';
        chatInput.style.height = 'auto'; // reset height

        appendMessage('user', text);
        const { content } = createMessage('assistant');
        content.innerHTML = typingHtml;
        scrollToBottom();
        sendBtn.disabled = true;

        // Re-render at most once per animation frame while tokens stream in.
        let streamed = '';
        let frame = null;
        const paint = () => {
            frame = null;
            renderInto(content, streamed);
            scrollToBottom();
        };
        const status = (msg) => {
            if (!streamed) content.innerHTML = `<p class="stream-status">${msg}</p>` + typingHtml;
        };

        try {
            const final = await API.agentStream(currentSessionId, { message: text, agent: 'doubt' }, {
                token: (d) => {
                    streamed += d.text;
                    if (frame === null) frame = requestAnimationFrame(paint);
                },
                tool_start: (d) => status(d.name === 'python_repl' ? 'Running code to verify...' : 'Searching the web...'),
                retry: (d) => status(d.fallback ? 'Switching to a backup model...' : 'Retrying...'),
            });
            if (frame !== null) cancelAnimationFrame(frame);
            // The final event carries the complete text (including any notes appended after streaming).
            renderInto(content, final.content || streamed || '_No response._');
        } catch (err) {
            if (frame !== null) cancelAnimationFrame(frame);
            renderInto(content, `**Error:** Failed to get response. ${err.message}`);
        } finally {
            scrollToBottom();
            sendBtn.disabled = false;
            chatInput.focus();
        }
    };

    const loadHistory = async () => {
        try {
            const res = await API.getAgentHistory(currentSessionId, 'doubt');
            const history = res.messages || [];
            if (history.length > 0) {
                msgContainer.innerHTML = ''; // clear the default greeting
                history.forEach((msg) => appendMessage(msg.role, msg.content));
            }
        } catch (err) {
            console.error('Failed to load chat history', err);
        }
    };

    return {
        init: (sessionId) => {
            currentSessionId = sessionId;

            sendBtn.addEventListener('click', handleSend);

            chatInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                }
            });

            // auto-resize textarea
            chatInput.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = (this.scrollHeight) + 'px';
            });

            loadHistory();
        }
    };
})();
