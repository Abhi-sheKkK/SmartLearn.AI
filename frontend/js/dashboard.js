document.addEventListener('DOMContentLoaded', async () => {
    const params = new URLSearchParams(window.location.search);
    const sessionId = params.get('session');

    if (!sessionId) {
        window.location.href = '/';
        return;
    }

    const titleDisplay = document.getElementById('session-title-display');
    const thumbnail = document.getElementById('lecture-thumbnail');
    const videoLink = document.getElementById('lecture-video-link');
    const sourceLink = document.getElementById('lecture-source-link');
    const navItems = document.querySelectorAll('.nav-item');
    const panels = document.querySelectorAll('.content-panel');

    // Load session info
    try {
        const session = await API.getSession(sessionId);
        titleDisplay.textContent = session.title || 'Untitled Lecture';
        titleDisplay.title = session.title || 'Untitled Lecture';

        if (session.video_id && thumbnail) {
            thumbnail.src = `https://img.youtube.com/vi/${session.video_id}/mqdefault.jpg`;
            thumbnail.alt = session.title || 'Lecture thumbnail';
        }
        if (session.youtube_url) {
            if (videoLink) videoLink.href = session.youtube_url;
            if (sourceLink) sourceLink.href = session.youtube_url;
        }
    } catch (err) {
        console.error("Failed to load session info", err);
        titleDisplay.textContent = "Session Error";
    }

    // Tab switching logic
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetPanelId = item.getAttribute('data-target');
            
            // Update nav active state
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            
            // Update panel visibility
            panels.forEach(panel => {
                panel.classList.remove('active');
                if (panel.id === targetPanelId) {
                    panel.classList.add('active');
                }
            });
        });
    });

    // Initialize modules
    if (window.NotesModule) window.NotesModule.init(sessionId);
    if (window.ChatModule) window.ChatModule.init(sessionId);
    if (window.VivaModule) window.VivaModule.init(sessionId);
    if (window.TestModule) window.TestModule.init(sessionId);
});
