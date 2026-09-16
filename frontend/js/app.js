document.addEventListener('DOMContentLoaded', () => {
    const urlInput = document.getElementById('youtube-url-input');
    const submitBtn = document.getElementById('submit-url-btn');
    const errorMessage = document.getElementById('error-message');
    const overlay = document.getElementById('processing-overlay');
    const progressSteps = document.querySelectorAll('.progress-step');

    const ytRegex = /^(https?:\/\/)?(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/)[\w-]+/;

    // Intersection Observer for features animation
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.feature-card').forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transitionDelay = `${index * 0.1}s`;
        observer.observe(card);
    });

    const showError = (msg) => {
        errorMessage.textContent = msg;
        errorMessage.style.display = 'block';
        setTimeout(() => { errorMessage.style.display = 'none'; }, 5000);
    };

    const updateProgress = (statusStr) => {
        let activeIndex = 0;
        if (['pending', 'extracting_id', 'downloading_transcript', 'extracting_transcript'].includes(statusStr)) activeIndex = 0;
        else if (['chunking_transcript', 'processing_cues', 'analyzing'].includes(statusStr)) activeIndex = 1;
        else if (['extracting_frames', 'deduplicating_frames', 'uploading_frames'].includes(statusStr)) activeIndex = 2;
        else if (['ready', 'completed'].includes(statusStr)) activeIndex = 3;

        progressSteps.forEach((step, index) => {
            step.classList.remove('active', 'complete');
            if (index < activeIndex) {
                step.classList.add('complete');
            } else if (index === activeIndex) {
                step.classList.add('active');
            }
        });
    };

    const pollStatus = async (sessionId) => {
        try {
            const statusRes = await API.getSessionStatus(sessionId);
            updateProgress(statusRes.status);

            if (statusRes.status === 'ready' || statusRes.status === 'completed') {
                updateProgress('completed');
                setTimeout(() => {
                    window.location.href = `/dashboard.html?session=${sessionId}`;
                }, 1000);
            } else if (statusRes.status === 'error') {
                throw new Error('Processing failed on the server.');
            } else {
                setTimeout(() => pollStatus(sessionId), 3000);
            }
        } catch (err) {
            overlay.classList.remove('active');
            submitBtn.classList.remove('loading');
            showError(err.message || 'Error checking status');
        }
    };

    const handleStart = async () => {
        const url = urlInput.value.trim();
        if (!url) {
            showError('Please enter a YouTube URL');
            return;
        }
        if (!ytRegex.test(url)) {
            showError('Please enter a valid YouTube video URL');
            return;
        }

        submitBtn.classList.add('loading');
        
        try {
            const session = await API.createSession(url);
            
            // Start processing
            await API.processVideo(session.id);
            
            overlay.classList.add('active');
            updateProgress('extracting_transcript');
            
            pollStatus(session.id);
        } catch (error) {
            submitBtn.classList.remove('loading');
            showError(error.message || 'Failed to create session');
        }
    };

    submitBtn.addEventListener('click', handleStart);
    urlInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleStart();
    });
});
