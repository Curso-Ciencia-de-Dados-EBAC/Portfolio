/* By edsaac (https://github.com/streamlit/streamlit/issues/5251#issuecomment-2755219516) */

// Access the main app window
var main_app = window.parent.document

// Start the MathJax configuration before loading from the CDN
main_app.defaultView.MathJax = {
    tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']]
    },
    svg: {
        fontCache: 'global'
    }
};

(function () {
    var script = main_app.createElement('script');
    // Load the MathJax version that Plotly bundles when exporting to HTML
    script.type = 'text/javascript';
    script.async = true;
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG';
    main_app.head.appendChild(script);
})();