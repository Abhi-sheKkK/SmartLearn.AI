window.TestModule = (() => {
    let currentSessionId = null;
    let questionsData = [];
    let userAnswers = {};

    const configView = document.getElementById('test-config-view');
    const activeView = document.getElementById('test-active-view');
    const resultsView = document.getElementById('test-results-view');
    
    const qCountRange = document.getElementById('test-q-count');
    const qCountDisplay = document.getElementById('test-q-count-display');
    const diffSelect = document.getElementById('test-difficulty');
    
    const generateBtn = document.getElementById('generate-test-btn');
    const qContainer = document.getElementById('test-questions-container');
    const progressBar = document.getElementById('test-progress-bar');
    const submitBtn = document.getElementById('submit-test-btn');
    const retakeBtn = document.getElementById('retake-test-btn');

    qCountRange.addEventListener('input', (e) => {
        qCountDisplay.textContent = e.target.value;
    });

    const updateProgress = () => {
        const answered = Object.keys(userAnswers).length;
        const total = questionsData.length;
        const percent = total > 0 ? (answered / total) * 100 : 0;
        progressBar.style.width = `${percent}%`;
    };

    const renderQuestions = (questions) => {
        qContainer.innerHTML = '';
        questionsData = questions;
        userAnswers = {};
        updateProgress();

        questions.forEach((q, index) => {
            const card = document.createElement('div');
            card.className = 'test-question-card glass-card';
            card.id = `q-card-${index}`;
            
            const qNum = document.createElement('div');
            qNum.className = 'question-number';
            qNum.textContent = `Question ${index + 1} of ${questions.length}`;
            
            const qText = document.createElement('div');
            qText.className = 'question-text';
            qText.textContent = q.question; // could use marked if needed

            const optsDiv = document.createElement('div');
            optsDiv.className = 'test-options';

            q.options.forEach((optStr, optIndex) => {
                const optLabel = document.createElement('label');
                optLabel.className = 'test-option';
                
                const radio = document.createElement('input');
                radio.type = 'radio';
                radio.name = `q-${index}`;
                radio.value = optStr;
                
                radio.addEventListener('change', () => {
                    userAnswers[index] = optStr;
                    // styling
                    card.querySelectorAll('.test-option').forEach(el => el.classList.remove('selected'));
                    optLabel.classList.add('selected');
                    updateProgress();
                });

                const textSpan = document.createElement('span');
                textSpan.textContent = optStr;

                optLabel.appendChild(radio);
                optLabel.appendChild(textSpan);
                optsDiv.appendChild(optLabel);
            });

            card.appendChild(qNum);
            card.appendChild(qText);
            card.appendChild(optsDiv);
            qContainer.appendChild(card);
        });
    };

    const handleGenerate = async () => {
        generateBtn.disabled = true;
        generateBtn.textContent = 'Generating...';
        
        const count = parseInt(qCountRange.value, 10);
        const diff = diffSelect.value;
        
        try {
            const res = await API.generateTest(currentSessionId, count, diff);
            const questions = res.questions || res.test_questions || [];
            renderQuestions(questions);
            configView.style.display = 'none';
            activeView.style.display = 'block';
        } catch (err) {
            alert('Failed to generate test: ' + err.message);
        } finally {
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate Test';
        }
    };

    const handleSubmit = async () => {
        if (Object.keys(userAnswers).length < questionsData.length) {
            alert('Please answer all questions before submitting.');
            return;
        }

        submitBtn.disabled = true;
        submitBtn.textContent = 'Submitting...';

        try {
            // Map selected answer strings to question ID -> index string
            const answersMap = {};
            questionsData.forEach((q, index) => {
                const selectedText = userAnswers[index];
                const optIndex = q.options.indexOf(selectedText);
                answersMap[q.id] = String(optIndex);
            });

            const res = await API.submitTest(currentSessionId, answersMap);
            renderResults(res);
            
            activeView.style.display = 'none';
            resultsView.style.display = 'block';
        } catch (err) {
            alert('Submit failed: ' + err.message);
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Submit Test';
        }
    };

    const renderResults = (resultsObj) => {
        const scoreDisplay = document.getElementById('final-test-score');
        const detailedContainer = document.getElementById('test-detailed-results');
        
        const scorePercent = resultsObj.percentage !== undefined ? resultsObj.percentage : Math.round((resultsObj.score / resultsObj.total) * 100);
        scoreDisplay.textContent = `${scorePercent}%`;
        
        detailedContainer.innerHTML = '';
        const itemList = resultsObj.results || resultsObj.detailed_results || [];
        
        itemList.forEach((item, index) => {
            const div = document.createElement('div');
            div.className = `result-item ${item.is_correct || item.correct ? 'correct' : 'incorrect'}`;
            
            let html = `<strong>Q${index + 1}: ${item.question}</strong><br>`;
            const userChoice = item.options && item.user_answer !== null && item.user_answer !== undefined ? item.options[item.user_answer] : 'Not answered';
            html += `Your answer: <em>${userChoice}</em><br>`;
            
            if (!item.is_correct && !item.correct) {
                const correctChoice = item.options && item.correct_answer !== undefined ? item.options[item.correct_answer] : item.correct_answer;
                html += `Correct answer: <strong style="color:var(--success)">${correctChoice}</strong><br>`;
            }
            
            if (item.explanation) {
                html += `<div class="result-explanation">${item.explanation}</div>`;
            }
            
            div.innerHTML = html;
            detailedContainer.appendChild(div);
        });
    };

    const handleRetake = () => {
        resultsView.style.display = 'none';
        configView.style.display = 'block';
        userAnswers = {};
        questionsData = [];
    };

    return {
        init: (sessionId) => {
            currentSessionId = sessionId;
            generateBtn.addEventListener('click', handleGenerate);
            submitBtn.addEventListener('click', handleSubmit);
            retakeBtn.addEventListener('click', handleRetake);
        }
    };
})();
