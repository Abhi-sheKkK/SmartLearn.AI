window.NotesModule = (() => {
    let currentSessionId = null;
    const contentArea = document.getElementById('notes-content-area');
    const generateBtn = document.getElementById('generate-notes-btn');
    const downloadBtn = document.getElementById('download-pdf-btn');

    // Custom marked renderer for images
    const renderer = new marked.Renderer();
    renderer.image = function(href, title, text) {
        return `<img src="${href}" alt="${text}" class="frame-image" title="${title || ''}">`;
    };
    marked.setOptions({ renderer });

    const showSkeleton = () => {
        contentArea.innerHTML = `
            <div class="notes-skeleton title"></div>
            <div class="notes-skeleton"></div>
            <div class="notes-skeleton"></div>
            <div class="notes-skeleton"></div>
            <div class="notes-skeleton img"></div>
            <div class="notes-skeleton"></div>
            <div class="notes-skeleton"></div>
        `;
    };

    const renderNotes = (markdownText) => {
        const html = marked.parse(markdownText);
        contentArea.innerHTML = html;
        
        // Render math
        if (window.renderMathInElement) {
            window.renderMathInElement(contentArea, {
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
        contentArea.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });

        generateBtn.textContent = 'Regenerate Notes';
        downloadBtn.style.display = 'inline-block';
    };

    const handleGenerate = async () => {
        showSkeleton();
        generateBtn.disabled = true;
        try {
            const res = await API.generateNotes(currentSessionId);
            renderNotes(res.markdown || res.notes || '');
        } catch (err) {
            contentArea.innerHTML = `<div class="empty-state"><p class="empty-state-text" style="color:var(--error)">Failed to generate notes: ${err.message}</p></div>`;
        } finally {
            generateBtn.disabled = false;
        }
    };

    const handleDownload = () => {
        const printWindow = window.open('', '', 'height=800,width=800');
        printWindow.document.write('<html><head><title>Lecture Notes</title>');
        printWindow.document.write('<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;line-height:1.6;padding:2rem;color:#1a1a2e;} img.frame-image, img{display:block;max-width:85%;margin:1.5rem auto;border-radius:10px;border:1px solid #ddd;page-break-inside:avoid;} pre{background:#f4f4f8;padding:1.25rem;border-radius:8px;overflow-x:auto;} blockquote{border-left:4px solid #6c5ce7;padding-left:1rem;color:#555;margin:1rem 0;} h1,h2,h3{page-break-after:avoid;color:#0d0d20;margin-top:1.5rem;}</style>');
        // include katex css for print
        printWindow.document.write('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">');
        printWindow.document.write('</head><body>');
        printWindow.document.write(contentArea.innerHTML);
        printWindow.document.write('</body></html>');
        printWindow.document.close();
        
        // Give time for images/katex to load
        setTimeout(() => {
            printWindow.print();
        }, 1000);
    };

    const loadExisting = async () => {
        try {
            const res = await API.getNotes(currentSessionId);
            if (res && (res.markdown || res.notes)) {
                renderNotes(res.markdown || res.notes);
            }
        } catch (err) {
            // No notes exist yet, do nothing (show empty state)
        }
    };

    return {
        init: (sessionId) => {
            currentSessionId = sessionId;
            generateBtn.addEventListener('click', handleGenerate);
            downloadBtn.addEventListener('click', handleDownload);
            loadExisting();
        }
    };
})();
