window.ChatModule = (() => {
    let currentSessionId = null;
    const msgContainer = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send-btn');

    const scrollToBottom = () => {
        msgContainer.scrollTop = msgContainer.scrollHeight;
    };

    const appendMessage = (role, text) => {
        const div = document.createElement('div');
        div.className = `message ${role}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? 'U' : 'AI';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        content.innerHTML = marked.parse(text);

        // Render math in message
        if (window.renderMathInElement) {
            window.renderMathInElement(content, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\(', right: '\\)', display: false},
                    {left: '\\[', right: '\\]', display: true}
                ],
                throwOnError : false
            });
        }
        
        // Highlight code
        content.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });

        div.appendChild(avatar);
        div.appendChild(content);
        msgContainer.appendChild(div);
        scrollToBottom();
    };

    const showTyping = () => {
        const div = document.createElement('div');
        div.className = `message assistant typing-indicator-msg`;
        div.innerHTML = `
            <div class="message-avatar">AI</div>
            <div class="message-content" style="padding: 12px 16px;">
                <div class="typing-indicator" style="padding:0">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        msgContainer.appendChild(div);
        scrollToBottom();
        return div;
    };

    const removeTyping = (el) => {
        if (el && el.parentNode) {
            el.parentNode.removeChild(el);
        }
    };

    const handleSend = async () => {
        const text = chatInput.value.trim();
        if (!text) return;

        chatInput.value = '';
        chatInput.style.height = 'auto'; // reset height

        appendMessage('user', text);
        const typingEl = showTyping();
        sendBtn.disabled = true;

        try {
            const res = await API.sendChat(currentSessionId, text, 'doubt');
            removeTyping(typingEl);
            appendMessage('assistant', res.content || res.response || '');
        } catch (err) {
            removeTyping(typingEl);
            appendMessage('assistant', `**Error:** Failed to get response. ${err.message}`);
        } finally {
            sendBtn.disabled = false;
            chatInput.focus();
        }
    };

    const loadHistory = async () => {
        try {
            const res = await API.getChatHistory(currentSessionId, 'doubt');
            const history = res.messages || res;
            if (history && history.length > 0) {
                // clear default msg
                msgContainer.innerHTML = '';
                history.forEach(msg => {
                    appendMessage(msg.role, msg.content);
                });
            }
        } catch (err) {
            console.error("Failed to load chat history", err);
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
