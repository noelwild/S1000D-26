// Aquila S1000D-AI Frontend Application
class AquilaApp {
    constructor() {
        this.currentDocument = null;
        this.currentModule = null;
        this.isSTEView = true;
        this.documents = [];
        this.modules = [];
        this.icns = [];
        this.ws = null;
        
        this.initializeApp();
    }

    initializeApp() {
        this.setupEventListeners();
        this.setupWebSocket();
        this.loadDocuments();
        this.startPeriodicRefresh();
    }

    setupEventListeners() {
        // Upload buttons
        document.getElementById('uploadBtn').addEventListener('click', () => this.showUploadModal());
        document.getElementById('uploadBtn2').addEventListener('click', () => this.showUploadModal());
        
        // Modal controls
        document.getElementById('cancelUpload').addEventListener('click', () => this.hideUploadModal());
        document.getElementById('confirmUpload').addEventListener('click', () => this.handleUpload());
        
        // View toggle
        document.getElementById('steBtn').addEventListener('click', () => this.switchToSTE());
        document.getElementById('verbatimBtn').addEventListener('click', () => this.switchToVerbatim());
        
        // Document selection
        document.getElementById('documentSelect').addEventListener('change', (e) => {
            this.currentDocument = e.target.value;
            this.loadDataModules();
        });
        
        // File input
        document.getElementById('fileInput').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                document.getElementById('confirmUpload').disabled = false;
            }
        });
    }

    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleProgressUpdate(data);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket connection closed. Attempting to reconnect...');
            setTimeout(() => this.setupWebSocket(), 5000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    handleProgressUpdate(data) {
        if (data.type === 'progress') {
            this.showProgress(data.phase, data.detail, data.processing_type, data.current_text, data.progress_section);
            
            // Update progress bar based on phase
            const progressPercent = this.getProgressPercent(data.phase);
            document.getElementById('progressBar').style.width = `${progressPercent}%`;
            document.getElementById('progressPercent').textContent = `${progressPercent}%`;
            
            if (data.phase === 'finished') {
                setTimeout(() => {
                    this.hideProgress();
                    this.loadDocuments();
                }, 3000);
            }
        }
    }

    getProgressPercent(phase) {
        const phases = {
            'upload_complete': 10,
            'text_extraction': 15,
            'text_extracted': 20,
            'classification': 35,
            'ste_conversion': 50,
            'module_creation': 60,
            'modules_created': 70,
            'images_processing': 80,
            'image_analysis': 90,
            'finished': 100
        };
        return phases[phase] || 0;
    }

    showProgress(phase, detail, processingType, currentText, progressSection) {
        const container = document.getElementById('progressContainer');
        const phaseElement = document.getElementById('progressPhase');
        const detailElement = document.getElementById('progressDetail');
        const processingTypeElement = document.getElementById('processingType');
        const currentTextElement = document.getElementById('currentText');
        const progressSectionElement = document.getElementById('progressSection');
        
        container.classList.remove('hidden');
        phaseElement.textContent = this.formatPhase(phase);
        detailElement.textContent = detail;
        
        // Update processing type
        if (processingType) {
            processingTypeElement.textContent = processingType;
            processingTypeElement.classList.remove('hidden');
        } else {
            processingTypeElement.classList.add('hidden');
        }
        
        // Update current text being processed
        if (currentText) {
            currentTextElement.textContent = currentText;
            currentTextElement.classList.remove('hidden');
        } else {
            currentTextElement.classList.add('hidden');
        }
        
        // Update progress section
        if (progressSection) {
            progressSectionElement.textContent = `Section ${progressSection}`;
            progressSectionElement.classList.remove('hidden');
        } else {
            progressSectionElement.classList.add('hidden');
        }
    }

    hideProgress() {
        document.getElementById('progressContainer').classList.add('hidden');
    }

    formatPhase(phase) {
        const phases = {
            'upload_complete': 'Upload Complete',
            'text_extraction': 'Extracting Text',
            'text_extracted': 'Text Extracted',
            'classification': 'AI Classification',
            'ste_conversion': 'STE Conversion',
            'module_creation': 'Creating Module',
            'modules_created': 'Modules Created',
            'images_processing': 'Processing Images',
            'image_analysis': 'Image Analysis',
            'finished': 'Complete'
        };
        return phases[phase] || 'Processing';
    }

    showUploadModal() {
        document.getElementById('uploadModal').classList.remove('hidden');
        document.getElementById('uploadModal').classList.add('flex');
    }

    hideUploadModal() {
        document.getElementById('uploadModal').classList.add('hidden');
        document.getElementById('uploadModal').classList.remove('flex');
        document.getElementById('fileInput').value = '';
        document.getElementById('confirmUpload').disabled = true;
    }

    async handleUpload() {
        const fileInput = document.getElementById('fileInput');
        const operationalContext = document.getElementById('operationalContext').value;
        const file = fileInput.files[0];
        
        if (!file) {
            alert('Please select a PDF file');
            return;
        }
        
        // Hide the modal immediately when upload starts
        this.hideUploadModal();
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('operational_context', operationalContext);
        
        try {
            const response = await fetch('/api/documents/upload', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                this.showProgress('upload_complete', 'Upload started');
                console.log('Upload successful:', result);
            } else {
                const error = await response.json();
                alert(`Upload failed: ${error.detail}`);
            }
        } catch (error) {
            console.error('Upload error:', error);
            alert('Upload failed. Please try again.');
        }
    }

    async loadDocuments() {
        try {
            const response = await fetch('/api/documents');
            if (response.ok) {
                this.documents = await response.json();
                this.updateDocumentSelect();
            }
        } catch (error) {
            console.error('Error loading documents:', error);
        }
    }

    updateDocumentSelect() {
        const select = document.getElementById('documentSelect');
        select.innerHTML = '<option value="">Select document...</option>';
        
        this.documents.forEach(doc => {
            const option = document.createElement('option');
            option.value = doc.id;
            option.textContent = `${doc.filename} (${doc.status})`;
            select.appendChild(option);
        });
    }

    async loadDataModules() {
        if (!this.currentDocument) {
            this.updateModulesList([]);
            return;
        }
        
        try {
            const response = await fetch(`/api/data-modules?document_id=${this.currentDocument}`);
            if (response.ok) {
                this.modules = await response.json();
                this.updateModulesList(this.modules);
            }
        } catch (error) {
            console.error('Error loading data modules:', error);
        }
    }

    updateModulesList(modules) {
        const container = document.getElementById('modulesList');
        
        if (modules.length === 0) {
            container.innerHTML = `
                <div class="text-gray-400 text-sm text-center py-8">
                    No data modules found
                </div>
            `;
            return;
        }
        
        container.innerHTML = '';
        
        modules.forEach((module, index) => {
            const moduleElement = document.createElement('div');
            moduleElement.className = 'module-item';
            moduleElement.innerHTML = `
                <div class="dmc">${module.dmc}</div>
                <div class="title">${module.title}</div>
                <div class="type">${module.type}</div>
            `;
            
            moduleElement.addEventListener('click', () => this.selectModule(module, moduleElement));
            container.appendChild(moduleElement);
        });
    }

    selectModule(module, element) {
        // Update active state
        document.querySelectorAll('.module-item').forEach(item => {
            item.classList.remove('active');
        });
        element.classList.add('active');
        
        this.currentModule = module;
        this.updateContentArea();
        this.updateModuleInfo();
    }

    updateContentArea() {
        const contentArea = document.getElementById('contentArea');
        
        if (!this.currentModule) {
            contentArea.innerHTML = `
                <div class="h-full flex items-center justify-center text-gray-400">
                    <div class="text-center">
                        <div class="text-6xl mb-4">📋</div>
                        <h3 class="text-xl font-semibold mb-2">Select a Data Module</h3>
                        <p class="text-gray-500">Choose a module from the sidebar to view its content</p>
                    </div>
                </div>
            `;
            return;
        }
        
        const content = this.isSTEView ? this.currentModule.ste_content : this.currentModule.verbatim_content;
        
        contentArea.innerHTML = `
            <div class="content-editor">
                <div class="prose prose-invert max-w-none">
                    ${this.formatContent(content)}
                </div>
            </div>
        `;
    }

    formatContent(content) {
        // Simple content formatting
        return content
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/^/, '<p>')
            .replace(/$/, '</p>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>');
    }

    updateModuleInfo() {
        const infoElement = document.getElementById('moduleInfo');
        const titleElement = document.getElementById('moduleTitle');
        
        if (!this.currentModule) {
            infoElement.textContent = '';
            titleElement.textContent = 'Select a data module';
            return;
        }
        
        titleElement.textContent = this.currentModule.title;
        infoElement.innerHTML = `
            <div class="flex items-center space-x-4">
                <span>DMC: <code class="bg-gray-700 px-2 py-1 rounded text-sm">${this.currentModule.dmc}</code></span>
                <span>Type: ${this.currentModule.type}</span>
                <span>Info Code: ${this.currentModule.info_code}</span>
            </div>
        `;
    }

    switchToSTE() {
        this.isSTEView = true;
        document.getElementById('steBtn').className = 'px-3 py-1 text-sm rounded bg-blue-600 text-white';
        document.getElementById('verbatimBtn').className = 'px-3 py-1 text-sm rounded text-gray-300 hover:text-white';
        this.updateContentArea();
    }

    switchToVerbatim() {
        this.isSTEView = false;
        document.getElementById('verbatimBtn').className = 'px-3 py-1 text-sm rounded bg-blue-600 text-white';
        document.getElementById('steBtn').className = 'px-3 py-1 text-sm rounded text-gray-300 hover:text-white';
        this.updateContentArea();
    }

    startPeriodicRefresh() {
        // Refresh documents every 10 seconds
        setInterval(() => {
            this.loadDocuments();
            if (this.currentDocument) {
                this.loadDataModules();
            }
        }, 10000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AquilaApp();
});