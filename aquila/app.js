// Aquila S1000D-AI Frontend Application with Project Management
class AquilaApp {
    constructor() {
        this.currentDocument = null;
        this.currentModule = null;
        this.currentProject = null;
        this.isSTEView = true;
        this.documents = [];
        this.modules = [];
        this.icns = [];
        this.ws = null;
        this.projects = [];
        
        this.initializeApp();
    }

    async initializeApp() {
        // Set up event listeners first, before checking current project
        this.setupEventListeners();
        
        // First check if we have a current project
        await this.checkCurrentProject();
        
        // If no project is selected, show project selection
        if (!this.currentProject) {
            this.showProjectSelection();
        } else {
            this.initializeMainApp();
        }
    }

    async checkCurrentProject() {
        try {
            const response = await fetch('/api/projects/current');
            if (response.ok) {
                const result = await response.json();
                this.currentProject = result.current_project;
                this.updateProjectDisplay();
            }
        } catch (error) {
            console.error('Error checking current project:', error);
        }
    }

    initializeMainApp() {
        this.setupWebSocket();
        this.loadDocuments();
        this.startPeriodicRefresh();
        this.showMainInterface();
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

        // Project management
        document.getElementById('projectBtn').addEventListener('click', () => this.showProjectSelection());
        document.getElementById('newProjectBtn').addEventListener('click', () => this.showNewProjectModal());
        document.getElementById('cancelNewProject').addEventListener('click', () => this.hideNewProjectModal());
        document.getElementById('confirmNewProject').addEventListener('click', () => this.handleCreateProject());
        document.getElementById('cancelProjectSelection').addEventListener('click', () => this.hideProjectSelection());
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
            'planning': 30,
            'planning_complete': 40,
            'population': 50,
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
            'text_extracted': 'Text Cleaned',
            'planning': 'Planning Modules',
            'planning_complete': 'Planning Complete',
            'population': 'Populating Modules',
            'modules_created': 'Modules Created',
            'images_processing': 'Processing Images',
            'image_analysis': 'Image Analysis',
            'finished': 'Complete'
        };
        return phases[phase] || 'Processing';
    }

    // Project Management Methods
    async showProjectSelection() {
        await this.loadProjects();
        this.hideMainInterface();
        document.getElementById('projectSelectionModal').classList.remove('hidden');
        document.getElementById('projectSelectionModal').classList.add('flex');
    }

    hideProjectSelection() {
        document.getElementById('projectSelectionModal').classList.add('hidden');
        document.getElementById('projectSelectionModal').classList.remove('flex');
        if (this.currentProject) {
            this.showMainInterface();
        }
    }

    showNewProjectModal() {
        document.getElementById('newProjectModal').classList.remove('hidden');
        document.getElementById('newProjectModal').classList.add('flex');
        document.getElementById('projectName').value = '';
        document.getElementById('projectDescription').value = '';
    }

    hideNewProjectModal() {
        document.getElementById('newProjectModal').classList.add('hidden');
        document.getElementById('newProjectModal').classList.remove('flex');
    }

    async loadProjects() {
        try {
            console.log('Loading projects...');
            const response = await fetch('/api/projects');
            if (response.ok) {
                this.projects = await response.json();
                console.log('Projects loaded:', this.projects);
                this.updateProjectsList();
            } else {
                console.error('Failed to load projects:', response.status);
            }
        } catch (error) {
            console.error('Error loading projects:', error);
        }
    }

    updateProjectsList() {
        const container = document.getElementById('projectsList');
        console.log('Updating projects list with', this.projects.length, 'projects');
        
        if (this.projects.length === 0) {
            container.innerHTML = `
                <div class="text-gray-400 text-center py-8">
                    <div class="text-4xl mb-4">📁</div>
                    <p>No projects found</p>
                    <p class="text-sm mt-2">Create your first project to get started</p>
                </div>
            `;
            return;
        }
        
        container.innerHTML = '';
        
        this.projects.forEach(project => {
            const projectElement = document.createElement('div');
            projectElement.className = 'bg-gray-700 p-4 rounded-lg cursor-pointer hover:bg-gray-600 transition-colors';
            projectElement.innerHTML = `
                <div class="flex items-center justify-between">
                    <div>
                        <h3 class="text-lg font-semibold text-white">${project.name}</h3>
                        <p class="text-gray-300 text-sm">${project.description || 'No description'}</p>
                        <p class="text-gray-400 text-xs mt-1">Created: ${new Date(project.created_at).toLocaleDateString()}</p>
                    </div>
                    <div class="flex items-center space-x-2">
                        <button class="select-project-btn bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded text-sm transition-colors">
                            Select
                        </button>
                        <button class="delete-project-btn bg-red-600 hover:bg-red-700 px-3 py-1 rounded text-sm transition-colors">
                            Delete
                        </button>
                    </div>
                </div>
            `;
            
            const selectBtn = projectElement.querySelector('.select-project-btn');
            const deleteBtn = projectElement.querySelector('.delete-project-btn');
            
            selectBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.selectProject(project.id);
            });
            
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteProject(project.id, project.name);
            });
            
            container.appendChild(projectElement);
        });
    }

    async handleCreateProject() {
        const name = document.getElementById('projectName').value.trim();
        const description = document.getElementById('projectDescription').value.trim();
        
        if (!name) {
            alert('Please enter a project name');
            return;
        }
        
        try {
            const response = await fetch('/api/projects', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `name=${encodeURIComponent(name)}&description=${encodeURIComponent(description)}`
            });
            
            if (response.ok) {
                const project = await response.json();
                this.hideNewProjectModal();
                await this.loadProjects();
                // Auto-select the new project
                await this.selectProject(project.id);
            } else {
                const error = await response.json();
                alert(`Failed to create project: ${error.detail}`);
            }
        } catch (error) {
            console.error('Error creating project:', error);
            alert('Failed to create project. Please try again.');
        }
    }

    async selectProject(projectId) {
        try {
            const response = await fetch(`/api/projects/${projectId}/select`, {
                method: 'POST'
            });
            
            if (response.ok) {
                await this.checkCurrentProject();
                this.hideProjectSelection();
                this.initializeMainApp();
            } else {
                const error = await response.json();
                alert(`Failed to select project: ${error.detail}`);
            }
        } catch (error) {
            console.error('Error selecting project:', error);
            alert('Failed to select project. Please try again.');
        }
    }

    async deleteProject(projectId, projectName) {
        if (!confirm(`Are you sure you want to delete project "${projectName}"? This action cannot be undone.`)) {
            return;
        }
        
        try {
            const response = await fetch(`/api/projects/${projectId}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                await this.loadProjects();
                // If we deleted the current project, reset current project
                if (this.currentProject && this.currentProject.id === projectId) {
                    this.currentProject = null;
                    this.updateProjectDisplay();
                }
            } else {
                const error = await response.json();
                alert(`Failed to delete project: ${error.detail}`);
            }
        } catch (error) {
            console.error('Error deleting project:', error);
            alert('Failed to delete project. Please try again.');
        }
    }

    updateProjectDisplay() {
        const projectNameElement = document.getElementById('currentProjectName');
        if (this.currentProject) {
            projectNameElement.textContent = this.currentProject.name;
        } else {
            projectNameElement.textContent = 'No Project Selected';
        }
    }

    showMainInterface() {
        document.getElementById('mainInterface').classList.remove('hidden');
    }

    hideMainInterface() {
        document.getElementById('mainInterface').classList.add('hidden');
    }

    showUploadModal() {
        if (!this.currentProject) {
            alert('Please select a project first');
            return;
        }
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
        
        if (!this.currentProject) {
            alert('Please select a project first');
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
        if (!this.currentProject) {
            return;
        }
        
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
        if (!this.currentDocument || !this.currentProject) {
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
            if (this.currentProject) {
                this.loadDocuments();
                if (this.currentDocument) {
                    this.loadDataModules();
                }
            }
        }, 10000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new AquilaApp();
});