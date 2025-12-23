// Theme toggle functionality
(function() {
    'use strict';

    // Get saved theme or default to light
    function getTheme() {
        return localStorage.getItem('theme') || 'light';
    }

    // Apply theme to document
    function applyTheme(theme) {
        if (theme === 'dark') {
            document.documentElement.setAttribute('data-theme', 'dark');
        } else {
            document.documentElement.removeAttribute('data-theme');
        }
    }

    // Save theme preference
    function saveTheme(theme) {
        localStorage.setItem('theme', theme);
    }

    // Toggle between light and dark
    function toggleTheme() {
        const currentTheme = getTheme();
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        applyTheme(newTheme);
        saveTheme(newTheme);
    }

    // Create and inject toggle button into navbar
    function createToggleButton() {
        const navbarEnd = document.querySelector('.navbar-end');
        if (!navbarEnd) return;

        // Check if button already exists
        if (document.querySelector('.theme-toggle')) return;

        const toggleBtn = document.createElement('a');
        toggleBtn.className = 'navbar-item theme-toggle';
        toggleBtn.title = 'Toggle Dark Mode';
        toggleBtn.innerHTML = '<i class="fas fa-sun icon-sun"></i><i class="fas fa-moon icon-moon"></i>';
        toggleBtn.addEventListener('click', function(e) {
            e.preventDefault();
            toggleTheme();
        });

        // Insert before the search button
        const searchBtn = navbarEnd.querySelector('.search');
        if (searchBtn) {
            navbarEnd.insertBefore(toggleBtn, searchBtn);
        } else {
            navbarEnd.appendChild(toggleBtn);
        }
    }

    // Initialize on page load
    function init() {
        // Apply saved theme immediately
        applyTheme(getTheme());

        // Create toggle button when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', createToggleButton);
        } else {
            createToggleButton();
        }
    }

    // Apply theme immediately to prevent flash
    applyTheme(getTheme());

    // Run init when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Re-create button after pjax navigation (Icarus uses pjax)
    document.addEventListener('pjax:complete', createToggleButton);
})();
