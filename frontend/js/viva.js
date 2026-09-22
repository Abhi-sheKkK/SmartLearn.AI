window.VivaModule = (() => {
    let currentSessionId = null;
    // 'answering' -> student is answering a question; 'clarifying' -> the scoring gate is open and the
    // student may elaborate before the score is finalized; 'reviewing' -> evaluation shown, next question loaded.
    let mode = 'answering';

    const startView = document.getElementById('viva-start-view');
    const activeView = document.getElementById('viva-active-view');
    const startBtn = document.getElementById('start-viva-btn');

    const qCountDisplay = document.getElementById('viva-q-count');
    const questionText = document.getElementById('viva-question-text');
    const answerInput = document.getElementById('viva-answer-input');
    const submitBtn = document.getElementById('viva-submit-btn');
    const nextBtn = document.getElementById('viva-next-btn');
    const skipBtn = document.getElementById('viva-skip-btn');
    const endBtn = document.getElementById('viva-end-btn');
    const feedbackArea = document.getElementById('viva-feedback-area');
    const audioEl = document.getElementById('viva-audio');

    const MATH_DELIMITERS = [
        {left: '$$', right: '$$', display: true},
        {left: '$', right: '$', display: false},
        {left: '\\(', right: '\\)', display: false},
        {left: '\\[', right: '\\]', display: true}
    ];

    const renderText = (container, text) => {
        container.innerHTML = MD.render(text);
        if (window.renderMathInElement) {
            window.renderMathInElement(container, { delimiters: MATH_DELIMITERS, throwOnError: false });
        }
    };

    // Streams tokens into a container, repainting at most once per animation frame.
    const makeStreamer = (container) => {
        let text = '';
        let frame = null;
        return {
            push(t) {
                text += t;
                if (frame === null) frame = requestAnimationFrame(() => { frame = null; renderText(container, text); });
            },
            stop() { if (frame !== null) { cancelAnimationFrame(frame); frame = null; } },
            get text() { return text; },
        };
    };

    const setAudio = (url) => {
        if (url) {
            audioEl.src = url;
            audioEl.style.display = 'block';
        } else {
            audioEl.removeAttribute('src');
            audioEl.style.display = 'none';
        }
    };

    const setMode = (next) => {
        mode = next;
        const clarifying = next === 'clarifying';
        const reviewing = next === 'reviewing';
        submitBtn.style.display = reviewing ? 'none' : 'inline-flex';
        submitBtn.textContent = clarifying ? 'Submit Clarification' : 'Submit Answer';
        skipBtn.style.display = clarifying ? 'inline-flex' : 'none';
        nextBtn.style.display = reviewing ? 'inline-flex' : 'none';
        answerInput.placeholder = clarifying ? 'Add more detail or clarify your answer (optional)...' : 'Type your answer here...';
        answerInput.disabled = reviewing;
        submitBtn.disabled = false;
        skipBtn.disabled = false;
    };

    // Apply the authoritative post-turn state to the UI.
    const applyFinal = (final) => {
        const viva = (final.state && final.state.viva_score) || {};
        qCountDisplay.textContent = (viva.asked || 0) + (viva.active && viva.pending_question ? 1 : 0) || 1;

        if (final.interrupted && final.interrupt) {
            const ev = final.interrupt.evaluation || {};
            const parts = [];
            if (ev.feedback) parts.push(`**Provisional evaluation:** ${ev.feedback}`);
            parts.push(`**Before I score this:** ${final.interrupt.prompt}`);
            feedbackArea.style.display = 'block';
            renderText(feedbackArea, parts.join('\n\n'));
            setMode('clarifying');
            answerInput.value = '';
            answerInput.focus();
            return;
        }

        const assistant = (final.messages || []).filter((m) => m.role === 'assistant');
        const scoredThisTurn = final.agent === 'viva' && assistant.length > 1;
        if (scoredThisTurn) {
            // [evaluation, next question]: feedback above, the new question in the question card.
            feedbackArea.style.display = 'block';
            renderText(feedbackArea, assistant.slice(0, -1).map((m) => m.content).join('\n\n'));
            if (viva.pending_question) renderText(questionText, viva.pending_question);
            setAudio(final.state.audio_url);
            setMode('reviewing');
        } else if (final.agent === 'viva' && viva.pending_question) {
            // First question of a session.
            feedbackArea.style.display = 'none';
            renderText(questionText, viva.pending_question);
            setAudio(final.state.audio_url);
            setMode('answering');
        } else {
            // A doubt answered mid-viva (or an error): show it, keep the current question, keep answering.
            feedbackArea.style.display = 'block';
            renderText(feedbackArea, final.content || '');
            setMode('answering');
        }
    };

    const stream = async (body, container) => {
        const streamer = makeStreamer(container);
        try {
            const final = await API.agentStream(currentSessionId, body, {
                token: (d) => streamer.push(d.text),
                retry: (d) => { if (!streamer.text) renderText(container, d.fallback ? '_Switching to a backup model..._' : '_Retrying..._'); },
            });
            streamer.stop();
            return final;
        } catch (err) {
            streamer.stop();
            throw err;
        }
    };

    const handleStart = async () => {
        startBtn.disabled = true;
        startBtn.textContent = 'Starting...';
        try {
            startView.style.display = 'none';
            activeView.style.display = 'block';
            feedbackArea.style.display = 'none';
            answerInput.style.display = '';
            questionText.style.display = '';
            endBtn.style.display = '';
            endBtn.disabled = false;
            endBtn.textContent = 'End Session';
            answerInput.value = '';
            renderText(questionText, '_Preparing your first question..._');
            setAudio('');
            const final = await stream({ action: 'viva_start' }, questionText);
            applyFinal(final);
        } catch (err) {
            alert('Failed to start viva: ' + err.message);
            activeView.style.display = 'none';
            startView.style.display = 'block';
        } finally {
            startBtn.disabled = false;
            startBtn.textContent = 'Start Viva';
        }
    };

    const submit = async (body, busyLabel) => {
        submitBtn.disabled = true;
        skipBtn.disabled = true;
        answerInput.disabled = true;
        submitBtn.textContent = busyLabel;
        feedbackArea.style.display = 'block';
        renderText(feedbackArea, '_Evaluating..._');
        try {
            const final = await stream(body, feedbackArea);
            answerInput.disabled = false;
            applyFinal(final);
            if (mode === 'answering') answerInput.value = '';
        } catch (err) {
            alert('Evaluation failed: ' + err.message);
            answerInput.disabled = false;
            setMode(mode);
        }
    };

    const handleSubmit = async () => {
        const text = answerInput.value.trim();
        if (mode === 'clarifying') {
            // Empty text is allowed here: it finalizes the score as it stands.
            await submit({ resume: { elaboration: text } }, 'Scoring...');
            return;
        }
        if (!text) return;
        await submit({ message: text, agent: 'viva' }, 'Evaluating...');
    };

    const handleSkip = async () => {
        await submit({ resume: { elaboration: '' } }, 'Scoring...');
    };

    const handleNext = () => {
        answerInput.value = '';
        feedbackArea.style.display = 'none';
        setMode('answering');
        answerInput.focus();
    };

    const handleEnd = async () => {
        endBtn.disabled = true;
        endBtn.textContent = 'Ending...';
        try {
            feedbackArea.style.display = 'block';
            const final = await stream({ action: 'viva_end' }, feedbackArea);
            renderText(feedbackArea, final.content || '');

            answerInput.style.display = 'none';
            submitBtn.style.display = 'none';
            nextBtn.style.display = 'none';
            skipBtn.style.display = 'none';
            endBtn.style.display = 'none';
            questionText.style.display = 'none';
            setAudio('');
        } catch (err) {
            alert('Failed to end viva: ' + err.message);
            endBtn.disabled = false;
            endBtn.textContent = 'End Session';
        }
    };

    // If the page is reloaded mid-viva, pick the session back up from the checkpoint.
    const restore = async () => {
        try {
            const state = await API.getAgentState(currentSessionId);
            const viva = state.viva_score || {};
            if (!viva.active || !viva.pending_question) return;
            startView.style.display = 'none';
            activeView.style.display = 'block';
            renderText(questionText, viva.pending_question);
            qCountDisplay.textContent = (viva.asked || 0) + 1;
            setAudio(state.audio_url);
            if (state.interrupt) {
                applyFinal({ interrupted: true, interrupt: state.interrupt, state, messages: [] });
            } else {
                setMode('answering');
            }
        } catch (err) {
            // No checkpoint yet: nothing to restore.
        }
    };

    return {
        init: (sessionId) => {
            currentSessionId = sessionId;
            startBtn.addEventListener('click', handleStart);
            submitBtn.addEventListener('click', handleSubmit);
            skipBtn.addEventListener('click', handleSkip);
            nextBtn.addEventListener('click', handleNext);
            endBtn.addEventListener('click', handleEnd);
            restore();
        }
    };
})();
