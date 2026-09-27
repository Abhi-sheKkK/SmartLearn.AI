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
    const micBtn = document.getElementById('viva-mic-btn');
    const micStatus = document.getElementById('viva-mic-status');

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

    // ---- Speech-to-text answering (Web Speech API) --------------------------------------------
    // Client-side only: no backend involvement. Falls back to hidden (text-only) where unsupported
    // (Firefox, Safari < 17 desktop, most non-Chromium browsers).
    const SpeechRecognitionImpl = window.SpeechRecognition || window.webkitSpeechRecognition;
    const micSupported = !!SpeechRecognitionImpl;
    let recognizer = null;
    let isRecording = false;
    let baseAnswerText = ''; // textarea contents captured when recording started; speech is appended after it

    const setMicUI = (recording) => {
        isRecording = recording;
        micBtn.classList.toggle('recording', recording);
        micBtn.title = recording ? 'Stop recording' : 'Answer by speech';
        micStatus.style.display = recording ? 'block' : 'none';
        if (recording) micStatus.textContent = 'Listening... click the mic to stop.';
    };

    const stopRecording = () => {
        if (recognizer && isRecording) recognizer.stop(); // onend finishes the UI reset
    };

    const initSpeechRecognition = () => {
        if (!SpeechRecognitionImpl) return; // leave micBtn hidden

        recognizer = new SpeechRecognitionImpl();
        recognizer.continuous = true;
        recognizer.interimResults = true;
        recognizer.lang = navigator.language || 'en-US';

        recognizer.onresult = (event) => {
            let finalChunk = '';
            let interimChunk = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) finalChunk += transcript;
                else interimChunk += transcript;
            }
            if (finalChunk) {
                baseAnswerText = (baseAnswerText.trim() + ' ' + finalChunk.trim()).trim() + ' ';
            }
            answerInput.value = (baseAnswerText + interimChunk).trim();
        };

        recognizer.onerror = (event) => {
            const messages = {
                'not-allowed': 'Microphone access was denied. Allow microphone access in your browser to use speech input.',
                'no-speech': "Didn't catch that -- no speech detected.",
                'audio-capture': 'No microphone was found.',
            };
            micStatus.style.display = 'block';
            micStatus.textContent = messages[event.error] || `Speech recognition error: ${event.error}`;
            micStatus.style.color = 'var(--error)';
        };

        recognizer.onend = () => {
            setMicUI(false);
            micStatus.style.color = '';
            answerInput.focus();
        };

        micBtn.addEventListener('click', () => {
            if (isRecording) {
                stopRecording();
                return;
            }
            baseAnswerText = answerInput.value.trim() ? answerInput.value.trim() + ' ' : '';
            micStatus.style.color = '';
            try {
                recognizer.start();
                setMicUI(true);
            } catch (err) {
                // start() throws if a recognition session is already active (e.g. double-click race).
            }
        });
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
        micBtn.disabled = reviewing;
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
        stopRecording();
        micStatus.style.display = 'none';
        startBtn.disabled = true;
        startBtn.textContent = 'Starting...';
        try {
            startView.style.display = 'none';
            activeView.style.display = 'block';
            feedbackArea.style.display = 'none';
            answerInput.style.display = '';
            questionText.style.display = '';
            micBtn.style.display = micSupported ? '' : 'none';
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
        stopRecording();
        submitBtn.disabled = true;
        skipBtn.disabled = true;
        micBtn.disabled = true;
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
        stopRecording();
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
            micBtn.style.display = 'none';
            micStatus.style.display = 'none';
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
            micBtn.style.display = micSupported ? '' : 'none';
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
            initSpeechRecognition();
            startBtn.addEventListener('click', handleStart);
            submitBtn.addEventListener('click', handleSubmit);
            skipBtn.addEventListener('click', handleSkip);
            nextBtn.addEventListener('click', handleNext);
            endBtn.addEventListener('click', handleEnd);
            restore();
        }
    };
})();
