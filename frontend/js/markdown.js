// Markdown -> HTML that leaves math alone.
// marked runs before KaTeX, and would otherwise treat `_`, `*` and `\` inside $...$ as Markdown
// (so `$x_i + y_j$` turns into italics). Math spans are swapped for placeholders while marked runs.
window.MD = (() => {
    const MATH = /\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$(?!\s)[^$\n]+?(?<!\s)\$/g;

    const escapeHtml = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

    return {
        render(text) {
            const stash = [];
            const shielded = text.replace(MATH, (m) => {
                stash.push(m);
                return `@@MATH${stash.length - 1}@@`;
            });
            return marked.parse(shielded).replace(/@@MATH(\d+)@@/g, (_, i) => escapeHtml(stash[Number(i)]));
        },
    };
})();
