window.VivaModule = (() => {
    let currentSessionId = null;
    let questionCount = 0;
    
    const startView = document.getElementById('viva-start-view');
    const activeView = document.getElementById('viva-active-view');
    const startBtn = document.getElementById('start-viva-btn');
    
    const qCountDisplay = document.getElementById('viva-q-count');
    const questionText = document.getElementById('viva-question-text');
    const answerInput = document.getElementById('viva-answer-input');
    const submitBtn = document.getElementById('viva-submit-btn');
    const nextBtn = document.getElementById('viva-next-btn');
    const endBtn = document.getElementById('viva-end-btn');
    const feedbackArea = document.getElementById('viva-feedback-area');

    const renderText = (container, text) => {
        container.innerHTML = marked.parse(text);
        if (window.renderMathInElement) {
            window.renderMathInElement(container, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\(', right: '\\)', display: false},
                    {left: '\\[', right: '\\]', display: true}
                ],
                throwOnError : false
            });
        }
    };

    const handleStart = async () => {
        startBtn.disabled = true;
        startBtn.textContent = 'Starting...';
        
        try {
            const res = await API.sendChat(currentSessionId, "Start a viva session", 'viva');
            startView.style.display = 'none';
            activeView.style.display = 'block';
            questionCount = 1;
            qCountDisplay.textContent = questionCount;
            renderText(questionText, res.content || res.response || '');
        } catch (err) {
            alert('Failed to start viva: ' + err.message);
            startBtn.disabled = false;
            startBtn.textContent = 'Start Viva';
        }
    };

    const handleSubmit = async () => {
        const answer = answerInput.value.trim();
        if (!answer) return;

        submitBtn.disabled = true;
        submitBtn.textContent = 'Evaluating...';
        answerInput.disabled = true;

        try {
            const res = await API.sendChat(currentSessionId, answer, 'viva');
            
            feedbackArea.style.display = 'block';
            renderText(feedbackArea, res.content || res.response || '');
            
            submitBtn.style.display = 'none';
            nextBtn.style.display = 'block';
            
        } catch (err) {
            alert('Evaluation failed: ' + err.message);
            submitBtn.disabled = false;
            submitBtn.textContent = 'Submit Answer';
            answerInput.disabled = false;
        }
    };

    const handleNext = () => {
        // Clear UI for next input
        answerInput.value = '';
        answerInput.disabled = false;
        feedbackArea.style.display = 'none';
        
        submitBtn.style.display = 'block';
        submitBtn.disabled = false;
        submitBtn.textContent = 'Submit Answer';
        nextBtn.style.display = 'none';
        
        questionCount++;
        qCountDisplay.textContent = questionCount;
    };

    const handleEnd = async () => {
        endBtn.disabled = true;
        endBtn.textContent = 'Ending...';
        try {
            const res = await API.sendChat(currentSessionId, "End viva session and give me a summary", 'viva');
            feedbackArea.style.display = 'block';
            renderText(feedbackArea, res.content || res.response || '');
            
            answerInput.style.display = 'none';
            submitBtn.style.display = 'none';
            nextBtn.style.display = 'none';
            endBtn.style.display = 'none';
            questionText.style.display = 'none';
            
        } catch (err) {
            alert('Failed to end viva: ' + err.message);
            endBtn.disabled = false;
            endBtn.textContent = 'End Session';
        }
    };

    return {
        init: (sessionId) => {
            currentSessionId = sessionId;
            startBtn.addEventListener('click', handleStart);
            submitBtn.addEventListener('click', handleSubmit);
            nextBtn.addEventListener('click', handleNext);
            endBtn.addEventListener('click', handleEnd);
        }
    };
})();
