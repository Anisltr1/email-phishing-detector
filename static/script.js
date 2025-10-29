// DOM Elements
const emailTextarea = document.getElementById('emailText');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const retryBtn = document.getElementById('retryBtn');

// State Elements
const loadingState = document.getElementById('loadingState');
const resultsSection = document.getElementById('resultsSection');
const errorState = document.getElementById('errorState');

// Result Elements
const resultIcon = document.getElementById('resultIcon');
const resultTitle = document.getElementById('resultTitle');
const resultDescription = document.getElementById('resultDescription');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceProgress = document.getElementById('confidenceProgress');
const rawScore = document.getElementById('rawScore');
const classification = document.getElementById('classification');
const textLength = document.getElementById('textLength');
const errorMessage = document.getElementById('errorMessage');

// Application State
let isAnalyzing = false;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    checkServerHealth();
});

// Event Listeners
function setupEventListeners() {
    analyzeBtn.addEventListener('click', analyzeEmail);
    clearBtn.addEventListener('click', clearInput);
    retryBtn.addEventListener('click', analyzeEmail);
    
    // Enable Enter key to submit (Ctrl+Enter for new line)
    emailTextarea.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            e.preventDefault();
            analyzeEmail();
        }
    });
    
    // Auto-resize textarea
    emailTextarea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 400) + 'px';
        
        // Update button state based on content
        updateButtonState();
    });
    
    // Initial button state
    updateButtonState();
}

// Update button states based on input
function updateButtonState() {
    const hasText = emailTextarea.value.trim().length > 0;
    const hasMinimumText = emailTextarea.value.trim().length >= 10;
    
    clearBtn.disabled = !hasText;
    analyzeBtn.disabled = !hasMinimumText || isAnalyzing;
    
    if (!hasMinimumText && hasText) {
        analyzeBtn.title = 'Please enter at least 10 characters';
    } else {
        analyzeBtn.title = '';
    }
}

// Check server health
async function checkServerHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (!data.model_loaded || !data.tokenizer_loaded) {
            showError('AI model is not properly loaded. Please refresh the page.');
        }
    } catch (error) {
        console.warn('Health check failed:', error);
    }
}

// Clear input
function clearInput() {
    emailTextarea.value = '';
    emailTextarea.style.height = 'auto';
    hideAllStates();
    updateButtonState();
    emailTextarea.focus();
}

// Analyze email
async function analyzeEmail() {
    const emailText = emailTextarea.value.trim();
    
    if (!emailText) {
        showError('Please enter some email content to analyze.');
        return;
    }
    
    if (emailText.length < 10) {
        showError('Please enter at least 10 characters for meaningful analysis.');
        return;
    }
    
    setAnalyzingState(true);
    showLoadingState();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                email_text: emailText
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        showResults(data);
        
    } catch (error) {
        console.error('Analysis error:', error);
        showError(error.message || 'Failed to analyze email. Please try again.');
    } finally {
        setAnalyzingState(false);
    }
}

// Set analyzing state
function setAnalyzingState(analyzing) {
    isAnalyzing = analyzing;
    updateButtonState();
    
    if (analyzing) {
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    } else {
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Email';
    }
}

// Show loading state
function showLoadingState() {
    hideAllStates();
    loadingState.classList.remove('hidden');
}

// Show results
function showResults(data) {
    hideAllStates();
    
    const isPhishing = data.label === 1;
    const confidence = Math.round(data.confidence);
    const score = data.score;
    
    // Update result icon and info
    resultIcon.className = `result-icon ${isPhishing ? 'phishing' : 'safe'}`;
    resultIcon.innerHTML = `<i class="fas ${isPhishing ? 'fa-exclamation-triangle' : 'fa-shield-alt'}"></i>`;
    
    // Update result info
    const resultInfo = resultIcon.parentElement.querySelector('.result-info');
    resultInfo.className = `result-info ${isPhishing ? 'phishing' : 'safe'}`;
    
    resultTitle.textContent = isPhishing ? 'Potential Phishing Detected' : 'Email Appears Safe';
    resultDescription.textContent = isPhishing 
        ? 'This email shows characteristics commonly found in phishing attempts. Exercise caution.'
        : 'This email appears to be legitimate based on our analysis.';
    
    // Update confidence meter
    confidenceValue.textContent = `${confidence}%`;
    confidenceProgress.className = `confidence-progress ${isPhishing ? 'phishing' : 'safe'}`;
    confidenceProgress.style.width = `${confidence}%`;
    
    // Update technical details
    rawScore.textContent = score.toFixed(4);
    classification.textContent = data.prediction;
    textLength.textContent = `${data.cleaned_text.length} characters`;
    
    // Show results with animation
    resultsSection.classList.remove('hidden');
    
    // Animate confidence bar
    setTimeout(() => {
        confidenceProgress.style.width = `${confidence}%`;
    }, 100);
    
    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'nearest' 
        });
    }, 200);
}

// Show error state
function showError(message) {
    hideAllStates();
    errorMessage.textContent = message;
    errorState.classList.remove('hidden');
}

// Hide all states
function hideAllStates() {
    loadingState.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorState.classList.add('hidden');
}

// Utility function to format numbers
function formatNumber(num, decimals = 2) {
    return Number(num).toFixed(decimals);
}

// Add some example emails for quick testing
function addExampleEmails() {
    const examples = [
        {
            title: "Safe Email Example",
            content: "Hi mom, I will be home for dinner tonight, see you soon. Love you!"
        },
        {
            title: "Phishing Email Example", 
            content: "URGENT: Your bank account is locked! Click verify NOW to avoid permanent suspension. Act immediately!"
        },
        {
            title: "Business Email Example",
            content: "Dear team, please review the quarterly report attached. The meeting is scheduled for tomorrow at 2 PM in conference room A."
        }
    ];
    
    // You could add buttons to load these examples if desired
    window.loadExample = function(index) {
        if (examples[index]) {
            emailTextarea.value = examples[index].content;
            emailTextarea.style.height = 'auto';
            emailTextarea.style.height = Math.min(emailTextarea.scrollHeight, 400) + 'px';
            updateButtonState();
            hideAllStates();
        }
    };
}

// Initialize examples
addExampleEmails();

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to analyze
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (!isAnalyzing && emailTextarea.value.trim().length >= 10) {
            analyzeEmail();
        }
    }
    
    // Escape to clear
    if (e.key === 'Escape') {
        clearInput();
    }
});

// Add visual feedback for form validation
emailTextarea.addEventListener('blur', function() {
    const text = this.value.trim();
    if (text.length > 0 && text.length < 10) {
        this.style.borderColor = '#f56565';
    } else {
        this.style.borderColor = '#e2e8f0';
    }
});

emailTextarea.addEventListener('focus', function() {
    this.style.borderColor = '#667eea';
});

// Performance optimization: debounce input events
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Debounced version of updateButtonState for better performance
const debouncedUpdateButtonState = debounce(updateButtonState, 100);

// Replace the direct call in the input event listener
emailTextarea.removeEventListener('input', updateButtonState);
emailTextarea.addEventListener('input', debouncedUpdateButtonState);