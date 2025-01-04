const form = document.getElementById('articleForm');
const textarea = form.querySelector('textarea[name="article_text"]');
const errorDiv = document.getElementById('error-message');

form.addEventListener('submit', function(event) {
    // if textarea text is too long
    if (textarea.value.length > 2490) {
        event.preventDefault(); 
        errorDiv.textContent = 'Text too long, 2490 characters or less';
        errorDiv.style.display = 'block';
    } else {
        errorDiv.style.display = 'none';
        errorDiv.textContent = '';
    }
});
