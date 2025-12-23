// Inject custom CSS and theme toggle JS
hexo.extend.filter.register('after_render:html', function(str){
    // Inject CSS in head
    str = str.replace('</head>', '<link rel="stylesheet" href="/css/custom.css"></head>');
    // Inject theme toggle script at end of body
    str = str.replace('</body>', '<script src="/js/theme-toggle.js"></script></body>');
    return str;
});
