// Simple theme helper
(function() {
    console.log("HyprL Theme Loaded");
    
    // Observer to force cleanups if Streamlit re-renders parts of the DOM
    const observer = new MutationObserver((mutations) => {
        // Potential logic to re-apply classes if Streamlit wipes them
        // For now, CSS !important rules handle most things.
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
})();
