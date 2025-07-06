// Import Hugging Face Transformers.js
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.6.1';

// Configure environment settings to prevent common errors
env.allowLocalModels = false;
env.useBrowserCache = false;
env.allowRemoteModels = true;

// Global state management
class StoryGenerator {
    constructor() {
        this.generator = null;
        this.currentModel = null;
        this.isGenerating = false;
        this.isModelLoading = false;
        
        // DOM elements
        this.elements = {
            titleInput: document.getElementById('story-title'),
            descriptionInput: document.getElementById('story-description'),
            modelSelect: document.getElementById('model-select'),
            wordCountSelect: document.getElementById('word-count'),
            generateBtn: document.getElementById('generate-btn'),
            retryBtn: document.getElementById('retry-btn'),
            copyBtn: document.getElementById('copy-btn'),
            loadingContainer: document.getElementById('loading-container'),
            loadingText: document.getElementById('loading-text'),
            progressBar: document.getElementById('progress-bar'),
            errorContainer: document.getElementById('error-container'),
            errorMessage: document.getElementById('error-message'),
            storyOutput: document.getElementById('story-output'),
            storyContent: document.getElementById('story-content'),
            generationInfo: document.getElementById('generation-info')
        };
        
        this.initializeEventListeners();
        this.loadExamplePrompts();
        this.checkBrowserCompatibility();
    }

    checkBrowserCompatibility() {
        const isCompatible = 'serviceWorker' in navigator && 
                           'fetch' in window && 
                           'Promise' in window;
        
        if (!isCompatible) {
            this.showError('Your browser may not be fully compatible with this application. Please use a modern browser like Chrome, Firefox, or Safari.');
        }
    }

    initializeEventListeners() {
        this.elements.generateBtn.addEventListener('click', () => this.generateStory());
        this.elements.retryBtn.addEventListener('click', () => this.generateStory());
        this.elements.copyBtn.addEventListener('click', () => this.copyToClipboard());
        
        // Model selection change handler
        this.elements.modelSelect.addEventListener('change', () => {
            this.generator = null; // Reset generator when model changes
            this.currentModel = null;
        });
        
        // Input validation
        this.elements.titleInput.addEventListener('input', () => this.validateInputs());
        this.elements.descriptionInput.addEventListener('input', () => this.validateInputs());
    }

    loadExamplePrompts() {
        const exampleCards = document.querySelectorAll('.example-card');
        exampleCards.forEach(card => {
            card.addEventListener('click', () => {
                const title = card.dataset.title;
                const description = card.dataset.description;
                this.elements.titleInput.value = title;
                this.elements.descriptionInput.value = description;
                this.validateInputs();
            });
        });
    }

    validateInputs() {
        const title = this.elements.titleInput.value.trim();
        const description = this.elements.descriptionInput.value.trim();
        const isValid = title.length > 0 && description.length > 0;
        
        this.elements.generateBtn.disabled = !isValid || this.isGenerating;
        return isValid;
    }

    async loadModel(modelName) {
        if (this.generator && this.currentModel === modelName) {
            return this.generator;
        }

        if (this.isModelLoading) {
            throw new Error('Model is already loading. Please wait.');
        }

        this.isModelLoading = true;
        this.updateLoadingState('Loading model...', 10);

        try {
            // Progress simulation for model loading
            const progressInterval = setInterval(() => {
                const currentProgress = parseInt(this.elements.progressBar.style.width) || 10;
                if (currentProgress < 70) {
                    this.updateProgress(currentProgress + 10);
                }
            }, 1000);

            this.generator = await pipeline('text-generation', modelName, {
                quantized: true,
                progress_callback: (progress) => {
                    if (progress.status === 'downloading') {
                        const percent = Math.round((progress.loaded / progress.total) * 100);
                        this.updateLoadingState(`Downloading model: ${percent}%`, Math.min(percent, 70));
                    } else if (progress.status === 'loading') {
                        this.updateLoadingState('Initializing model...', 80);
                    }
                }
            });

            clearInterval(progressInterval);
            this.currentModel = modelName;
            this.updateLoadingState('Model loaded successfully!', 100);
            
            return this.generator;

        } catch (error) {
            this.isModelLoading = false;
            throw new Error(`Failed to load model: ${error.message}`);
        } finally {
            this.isModelLoading = false;
        }
    }

    async generateStory() {
        if (!this.validateInputs()) {
            this.showError('Please fill in both the title and description fields.');
            return;
        }

        if (this.isGenerating) {
            return;
        }

        this.isGenerating = true;
        this.hideAllSections();
        this.showLoadingState();
        
        const startTime = Date.now();
        
        try {
            // Get form values
            const title = this.elements.titleInput.value.trim();
            const description = this.elements.descriptionInput.value.trim();
            const modelName = this.elements.modelSelect.value;
            const maxTokens = parseInt(this.elements.wordCountSelect.value);
            
            // Load model
            const generator = await this.loadModel(modelName);
            
            // Prepare prompt
            const prompt = `Title: ${title}\nDescription: ${description}\nStory:\n\n`;
            
            this.updateLoadingState('Generating your story...', 90);
            
            // Generate story with proper parameters
            const result = await generator(prompt, {
                max_new_tokens: maxTokens,
                temperature: 0.7,
                do_sample: true,
                top_k: 50,
                top_p: 0.9,
                repetition_penalty: 1.1,
                return_full_text: false
            });
            
            this.updateLoadingState('Story generated!', 100);
            
            // Process the result
            let generatedText = '';
            if (Array.isArray(result)) {
                generatedText = result[0]?.generated_text || '';
            } else if (result?.generated_text) {
                generatedText = result.generated_text;
            } else {
                throw new Error('Invalid response format from model');
            }
            
            // Clean up the generated text
            generatedText = generatedText.replace(prompt, '').trim();
            
            if (!generatedText) {
                throw new Error('No story was generated. Please try again with different parameters.');
            }
            
            // Show the generated story
            const endTime = Date.now();
            const generationTime = ((endTime - startTime) / 1000).toFixed(1);
            
            this.showStory(generatedText, {
                model: modelName,
                wordCount: this.countWords(generatedText),
                generationTime: generationTime
            });
            
        } catch (error) {
            console.error('Story generation error:', error);
            this.showError(this.getErrorMessage(error));
        } finally {
            this.isGenerating = false;
            this.elements.generateBtn.disabled = false;
            this.elements.generateBtn.classList.remove('btn--loading');
        }
    }

    getErrorMessage(error) {
        const errorMsg = error.message.toLowerCase();
        
        if (errorMsg.includes('network') || errorMsg.includes('fetch')) {
            return 'Network error: Please check your internet connection and try again.';
        } else if (errorMsg.includes('model') || errorMsg.includes('load')) {
            return 'Model loading failed: The AI model could not be loaded. Please try a different model or refresh the page.';
        } else if (errorMsg.includes('memory') || errorMsg.includes('quota')) {
            return 'Memory error: The model is too large for your device. Please try the SmolLM model instead.';
        } else if (errorMsg.includes('timeout')) {
            return 'Request timeout: The generation is taking too long. Please try again with a shorter word count.';
        } else {
            return `Generation error: ${error.message}. Please try again or contact support if the problem persists.`;
        }
    }

    countWords(text) {
        return text.trim().split(/\s+/).filter(word => word.length > 0).length;
    }

    updateLoadingState(message, progress) {
        this.elements.loadingText.textContent = message;
        this.updateProgress(progress);
    }

    updateProgress(percent) {
        this.elements.progressBar.style.width = `${Math.min(percent, 100)}%`;
    }

    showLoadingState() {
        this.elements.generateBtn.disabled = true;
        this.elements.generateBtn.classList.add('btn--loading');
        this.elements.loadingContainer.style.display = 'block';
        this.updateProgress(0);
    }

    hideAllSections() {
        this.elements.loadingContainer.style.display = 'none';
        this.elements.errorContainer.style.display = 'none';
        this.elements.storyOutput.style.display = 'none';
    }

    showError(message) {
        this.hideAllSections();
        this.elements.errorMessage.textContent = message;
        this.elements.errorContainer.style.display = 'block';
        this.elements.generateBtn.disabled = false;
        this.elements.generateBtn.classList.remove('btn--loading');
    }

    showStory(content, info) {
        this.hideAllSections();
        this.elements.storyContent.textContent = content;
        this.elements.generationInfo.innerHTML = `
            <span>Model: ${info.model}</span>
            <span>Words: ${info.wordCount}</span>
            <span>Time: ${info.generationTime}s</span>
        `;
        this.elements.storyOutput.style.display = 'block';
    }

    async copyToClipboard() {
        try {
            const storyText = this.elements.storyContent.textContent;
            await navigator.clipboard.writeText(storyText);
            
            // Show visual feedback
            const originalText = this.elements.copyBtn.textContent;
            this.elements.copyBtn.textContent = 'Copied!';
            this.elements.copyBtn.classList.add('btn--copied');
            
            setTimeout(() => {
                this.elements.copyBtn.textContent = originalText;
                this.elements.copyBtn.classList.remove('btn--copied');
            }, 2000);
            
        } catch (error) {
            console.error('Copy failed:', error);
            // Fallback for browsers that don't support clipboard API
            this.fallbackCopyToClipboard(this.elements.storyContent.textContent);
        }
    }

    fallbackCopyToClipboard(text) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            const originalText = this.elements.copyBtn.textContent;
            this.elements.copyBtn.textContent = 'Copied!';
            this.elements.copyBtn.classList.add('btn--copied');
            
            setTimeout(() => {
                this.elements.copyBtn.textContent = originalText;
                this.elements.copyBtn.classList.remove('btn--copied');
            }, 2000);
        } catch (error) {
            console.error('Fallback copy failed:', error);
        } finally {
            document.body.removeChild(textArea);
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    try {
        new StoryGenerator();
    } catch (error) {
        console.error('Failed to initialize Story Generator:', error);
        
        // Show a basic error message if initialization fails
        const errorContainer = document.getElementById('error-container');
        const errorMessage = document.getElementById('error-message');
        
        if (errorContainer && errorMessage) {
            errorMessage.textContent = 'Failed to initialize the application. Please refresh the page and try again.';
            errorContainer.style.display = 'block';
        }
    }
});

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    event.preventDefault();
});

// Export for potential testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = StoryGenerator;
}