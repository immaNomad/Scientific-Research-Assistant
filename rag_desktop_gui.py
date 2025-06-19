#!/usr/bin/env python3
"""
RAG Research Assistant - Desktop GUI Application
Modern desktop interface using PyQt5
"""

import sys
import os
import subprocess
import threading
import json
from pathlib import Path

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QGridLayout, QLabel, QTextEdit, 
                                QLineEdit, QPushButton, QCheckBox, QComboBox,
                                QProgressBar, QTextBrowser, QFileDialog, 
                                QMessageBox, QGroupBox, QSplitter, QTabWidget)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QPixmap
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("❌ PyQt5 not available. Please install with: pip install PyQt5")
    sys.exit(1)

class SearchWorker(QThread):
    """Worker thread for running searches without blocking the UI"""
    finished = pyqtSignal(bool, str, str)  # success, output, error
    progress = pyqtSignal(str)  # status message
    
    def __init__(self, command):
        super().__init__()
        self.command = command
    
    def run(self):
        try:
            self.progress.emit("Starting search...")
            
            # Execute the command
            result = subprocess.run(
                self.command, 
                capture_output=True, 
                text=True, 
                timeout=300
            )
            
            if result.returncode == 0:
                self.finished.emit(True, result.stdout, "")
            else:
                self.finished.emit(False, "", result.stderr)
                
        except subprocess.TimeoutExpired:
            self.finished.emit(False, "", "Search timed out (5 minutes)")
        except Exception as e:
            self.finished.emit(False, "", str(e))

class RAGDesktopGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.exe_path = Path("dist/RAG-Research-Assistant")
        self.worker = None
        
        self.init_ui()
        self.check_executable()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Scientific Research Assistant")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)
        
        # Set Klein blue and white theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
                color: #000000;
            }
            QWidget {
                background-color: #ffffff;
                color: #000000;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #002FA7;
                border-radius: 8px;
                margin: 5px;
                padding-top: 15px;
                background-color: #ffffff;
                color: #000000;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #002FA7;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #002FA7;
                color: #ffffff;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
                min-width: 120px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #0033CC;
            }
            QPushButton:pressed {
                background-color: #001F80;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
            QLineEdit, QTextEdit {
                background-color: #ffffff;
                border: 2px solid #002FA7;
                border-radius: 6px;
                padding: 10px;
                font-size: 12px;
                color: #000000;
                selection-background-color: #002FA7;
                selection-color: #ffffff;
            }
            QLineEdit:focus, QTextEdit:focus {
                border-color: #0033CC;
                background-color: #f8f9ff;
            }
            QTextBrowser {
                background-color: #ffffff;
                border: 2px solid #002FA7;
                border-radius: 6px;
                padding: 10px;
                color: #000000;
                selection-background-color: #002FA7;
                selection-color: #ffffff;
            }
            QProgressBar {
                border: 2px solid #002FA7;
                border-radius: 6px;
                text-align: center;
                background-color: #ffffff;
                color: #000000;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #002FA7;
                border-radius: 4px;
            }
            QCheckBox {
                color: #000000;
                font-size: 11px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 3px;
                border: 2px solid #002FA7;
                background-color: #ffffff;
            }
            QCheckBox::indicator:checked {
                background-color: #002FA7;
                border-color: #002FA7;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #0033CC;
            }
            QCheckBox::indicator:hover {
                border-color: #0033CC;
            }
            QLabel {
                color: #000000;
            }
            QTabWidget::pane {
                border: 2px solid #002FA7;
                background-color: #ffffff;
                border-radius: 6px;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                color: #000000;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
                border: 1px solid #002FA7;
            }
            QTabBar::tab:selected {
                background-color: #002FA7;
                color: #ffffff;
            }
            QTabBar::tab:hover {
                background-color: #e6e9ff;
            }
            QSplitter::handle {
                background-color: #002FA7;
                height: 3px;
            }
            QSplitter::handle:hover {
                background-color: #0033CC;
            }
            QStatusBar {
                background-color: #f8f9ff;
                color: #000000;
                border-top: 1px solid #002FA7;
            }
            QComboBox {
                background-color: #ffffff;
                border: 2px solid #002FA7;
                border-radius: 6px;
                padding: 8px;
                color: #000000;
                min-width: 120px;
            }
            QComboBox:hover {
                border-color: #0033CC;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #000000;
                margin-right: 5px;
            }
            QScrollBar:vertical {
                background-color: #f0f0f0;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #002FA7;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #0033CC;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Header
        self.create_header(main_layout)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # Input section
        input_widget = self.create_input_section()
        splitter.addWidget(input_widget)
        
        # Results section
        results_widget = self.create_results_section()
        splitter.addWidget(results_widget)
        
        # Set splitter proportions - give more space to results
        splitter.setSizes([300, 500])
        
        # Status bar
        self.statusBar().showMessage("Ready - RAG Research Assistant loaded")
    
    def create_header(self, layout):
        """Create the header section"""
        # Header removed - no titles needed
        pass
    
    def create_input_section(self):
        """Create the input section"""
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        
        # Query section
        query_group = QGroupBox("🔍 Research Query")
        query_layout = QVBoxLayout(query_group)
        
        self.query_text = QTextEdit()
        self.query_text.setPlaceholderText("Enter your research question or keywords here...\n\nExample: 'machine learning transformers natural language processing'")
        self.query_text.setMaximumHeight(80)
        self.query_text.setMinimumHeight(60)
        query_layout.addWidget(self.query_text)
        
        # Sources section removed - using defaults
        
        # AI Models section removed
        
        # Output section removed - using defaults
        
        # Buttons section (compact)
        buttons_group = QGroupBox("🚀 Actions")
        buttons_layout = QHBoxLayout(buttons_group)
        buttons_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        
        self.search_button = QPushButton("🔍 Search Papers")
        self.search_button.clicked.connect(self.search_papers)
        
        self.recent_button = QPushButton("📄 Recent Papers")
        self.recent_button.clicked.connect(self.recent_papers)
        
        self.full_button = QPushButton("🔬 Full Analysis")
        self.full_button.clicked.connect(self.full_analysis)
        
        self.clear_button = QPushButton("🗑️ Clear")
        self.clear_button.clicked.connect(self.clear_all)
        self.clear_button.setStyleSheet("QPushButton { background-color: #8B0000; color: #ffffff; } QPushButton:hover { background-color: #A52A2A; } QPushButton:pressed { background-color: #660000; }")
        
        buttons_layout.addWidget(self.search_button)
        buttons_layout.addWidget(self.recent_button)
        buttons_layout.addWidget(self.full_button)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.clear_button)
        
        # Status section removed
        
        # Add all groups to input layout
        input_layout.addWidget(query_group)
        input_layout.addWidget(buttons_group)
        
        return input_widget
    
    def create_results_section(self):
        """Create the results section"""
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        
        # Results display
        results_group = QGroupBox("📋 Search Results")
        results_group_layout = QVBoxLayout(results_group)
        
        # Results display - no tabs needed
        self.results_text = QTextBrowser()
        self.results_text.setFont(QFont("Consolas", 11))
        self.results_text.setPlaceholderText("Search results will appear here...\n\nClick 'Search Papers' to start finding relevant scientific literature.")
        self.results_text.setMinimumHeight(400)
        self.results_text.setLineWrapMode(QTextBrowser.WidgetWidth)
        self.results_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.results_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Show More button
        self.show_more_button = QPushButton("📄 Show More Details")
        self.show_more_button.clicked.connect(self.show_more_details)
        self.show_more_button.setVisible(False)  # Hidden by default
        
        results_group_layout.addWidget(self.results_text)
        results_group_layout.addWidget(self.show_more_button)
        results_layout.addWidget(results_group)
        
        # Store full and truncated versions
        self.full_results = ""
        self.truncated_results = ""
        self.showing_full = False
        
        return results_widget
    
    def check_executable(self):
        """Check if the executable exists"""
        if self.exe_path.exists():
            self.statusBar().showMessage("Ready to search scientific literature")
        else:
            self.statusBar().showMessage("ERROR: Executable not found")
            
            # Show error dialog
            QMessageBox.warning(
                self, 
                "Executable Not Found",
                f"RAG Research Assistant executable not found at:\n{self.exe_path}\n\n"
                "Please ensure the executable is built and in the correct location."
            )
        
        # LLM status check removed since AI models section was removed
    
    def get_sources(self):
        """Get default literature sources"""
        # Return all sources by default since UI controls were removed
        return ["arxiv", "semantic_scholar", "pubmed"]
    
    def generate_enhanced_analysis(self, raw_output):
        """Generate enhanced analysis with abstract and hypothesis using LLM (including free options)"""
        query = self.query_text.toPlainText().strip()
        
        # Try multiple LLM options in order of preference
        enhanced_content = None
        
        # Option 1: Try OpenAI if available
        try:
            from src.models.llm_client import llm_manager
            
            enhancement_prompt = f"""
Based on the following research query and search results, please provide a comprehensive analysis:

1. COMPREHENSIVE ABSTRACT (300-500 words): Write a detailed abstract that synthesizes the key findings, methodologies, current state of research, gaps, and future implications from all the research papers found. Include statistical trends, major authors/institutions, and emerging themes.

2. RESEARCH HYPOTHESES (3-5 hypotheses): Generate specific, testable research hypotheses that could advance this field. Each hypothesis should be novel, measurable, and based on gaps identified in the current literature.

RESEARCH QUERY: {query}

SEARCH RESULTS:
{raw_output}

Please format your response EXACTLY as follows:

=== COMPREHENSIVE ABSTRACT ===
[Write a comprehensive 300-500 word abstract synthesizing all findings, methodologies, trends, gaps, and implications from the research papers. Include quantitative insights where possible.]

=== RESEARCH HYPOTHESES ===
H1: [Specific, testable hypothesis addressing a research gap]
H2: [Another specific, testable hypothesis]
H3: [Third specific, testable hypothesis]
H4: [Fourth specific, testable hypothesis - optional]
H5: [Fifth specific, testable hypothesis - optional]

=== ORIGINAL SEARCH RESULTS ===
"""

            import asyncio
            response = asyncio.run(llm_manager.generate(enhancement_prompt, max_tokens=800, temperature=0.7))
            enhanced_content = response.content
            print("✅ Used OpenAI for enhanced analysis")
            
        except Exception as openai_error:
            print(f"OpenAI LLM failed: {openai_error}")
            
            # Option 2: Try free Hugging Face API
            try:
                from src.models.free_llm_client import free_llm_client
                
                # Shorter prompt for free API with limited context
                truncated_results = raw_output[:2000] if len(raw_output) > 2000 else raw_output
                free_prompt = f"""Research Analysis for: {query}

Based on these research results, provide:
1. Summary of key findings
2. Research gaps identified  
3. Future research directions

Results: {truncated_results}"""
                
                import asyncio
                response = asyncio.run(free_llm_client.generate(
                    free_prompt,
                    max_tokens=400,
                    temperature=0.7
                ))
                
                enhanced_content = f"=== FREE AI ANALYSIS ===\n\n{response.content}\n\n=== ORIGINAL SEARCH RESULTS ==="
                print("✅ Used free Hugging Face API for analysis")
                
            except Exception as free_error:
                print(f"Free LLM failed: {free_error}")
                
                # Option 3: Enhanced template fallback
                try:
                    from src.models.free_llm_client import FreeLLMClient
                    template_client = FreeLLMClient()
                    enhanced_content = template_client._generate_enhanced_template(f"analysis for {query}")
                    enhanced_content = f"=== ENHANCED TEMPLATE ANALYSIS ===\n\n{enhanced_content}\n\n=== ORIGINAL SEARCH RESULTS ==="
                    print("✅ Used enhanced template analysis")
                    
                except Exception as template_error:
                    print(f"Template analysis failed: {template_error}")
                    enhanced_content = f"=== ENHANCED ANALYSIS UNAVAILABLE ===\nAll AI analysis methods failed\n\n=== ORIGINAL SEARCH RESULTS ==="
        
        # Return enhanced content + original results
        if enhanced_content:
            return enhanced_content + "\n\n" + raw_output
        else:
            return f"=== ENHANCED ANALYSIS UNAVAILABLE ===\nNo AI analysis available\n\n=== ORIGINAL SEARCH RESULTS ===\n{raw_output}"
    
    def build_command(self, command_type, query=None):
        """Build the command to execute"""
        if not self.exe_path.exists():
            return None
        
        cmd = [str(self.exe_path), command_type]
        
        if query:
            cmd.append(query)
        
        sources = self.get_sources()
        if sources:
            cmd.extend(["--sources"] + sources)
        
        # Output file removed - using default behavior
        
        return cmd
    
    def execute_search(self, command_type, query=None):
        """Execute a search command"""
        if not self.exe_path.exists():
            QMessageBox.critical(self, "Error", "Executable not found!")
            return
        
        sources = self.get_sources()
        if not sources:
            QMessageBox.warning(self, "No Sources Selected", "Please select at least one literature source.")
            return
        
        if command_type == "search" and not query:
            QMessageBox.warning(self, "No Query", "Please enter a search query.")
            return
        
        # Build command
        cmd = self.build_command(command_type, query)
        if not cmd:
            return
        
        # Disable buttons and show searching message
        self.set_ui_enabled(False)
        
        # Show searching message in results
        self.results_text.setPlainText("Searching....")
        
        # Start worker thread
        self.worker = SearchWorker(cmd)
        self.worker.finished.connect(self.on_search_finished)
        self.worker.progress.connect(self.on_search_progress)
        self.worker.start()
    
    def set_ui_enabled(self, enabled):
        """Enable/disable UI elements"""
        self.search_button.setEnabled(enabled)
        self.recent_button.setEnabled(enabled)
        self.full_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)
    
    def on_search_progress(self, message):
        """Handle search progress updates"""
        # Progress updates removed - just show searching
    
    def on_search_finished(self, success, output, error):
        """Handle search completion"""
        # Re-enable UI
        self.set_ui_enabled(True)
        
        if success:
            # Check if we have actual results
            if "No papers found" in output or len(output.strip()) < 50:
                # Display raw output without enhancement if no papers found
                self.results_text.setPlainText(output)
                self.statusBar().showMessage("Search completed - No papers found")
                self.show_more_button.setVisible(False)
            else:
                self.statusBar().showMessage("Generating enhanced analysis...")
                
                # Generate enhanced output with abstract and hypothesis
                enhanced_output = self.generate_enhanced_analysis(output)
                
                # Store full results and create truncated version
                self.full_results = enhanced_output
                self.truncated_results = self.create_truncated_results(enhanced_output)
                
                # Display truncated results initially
                self.results_text.setPlainText(self.truncated_results)
                self.show_more_button.setVisible(True)
                self.show_more_button.setText("📄 Show More Details")
                self.showing_full = False
                
                self.statusBar().showMessage("Search completed successfully")
            
            # Show success message
            QMessageBox.information(
                self, 
                "Search Completed", 
                "Search completed successfully!\n\nCheck the results below."
            )
        else:
            self.statusBar().showMessage("Search failed")
            
            # Display error
            self.results_text.setPlainText(f"ERROR: {error}")
            
            # Show error message
            QMessageBox.critical(self, "Search Failed", f"Search failed with error:\n\n{error}")
    
    def search_papers(self):
        """Handle search papers button"""
        query = self.query_text.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "No Query", "Please enter a search query.")
            return
        
        self.execute_search("search", query)
    
    def recent_papers(self):
        """Handle recent papers button"""
        self.execute_search("recent")
    
    def full_analysis(self):
        """Handle full analysis button"""
        query = self.query_text.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "No Query", "Please enter a search query for full analysis.")
            return
        
        self.execute_search("full", query)
    
    def show_more_details(self):
        """Toggle between full and truncated results"""
        if self.showing_full:
            # Show truncated version
            self.results_text.setPlainText(self.truncated_results)
            self.show_more_button.setText("📄 Show More Details")
            self.showing_full = False
        else:
            # Show full version
            self.results_text.setPlainText(self.full_results)
            self.show_more_button.setText("📄 Show Less Details")
            self.showing_full = True
    
    def create_truncated_results(self, full_text):
        """Create a truncated version of the results for initial display"""
        lines = full_text.split('\n')
        truncated_lines = []
        
        for line in lines:
            if line.strip().startswith('Abstract:') and len(line) > 200:
                # Truncate abstracts to 200 characters
                truncated_lines.append(line[:200] + "... [truncated]")
            else:
                truncated_lines.append(line)
                
        return '\n'.join(truncated_lines)
    
    def clear_all(self):
        """Clear all fields"""
        self.query_text.clear()
        self.results_text.clear()
        self.show_more_button.setVisible(False)
        self.full_results = ""
        self.truncated_results = ""
        self.showing_full = False
        self.statusBar().showMessage("Fields cleared - Ready for new search")

def main():
    """Main application entry point"""
    if not PYQT_AVAILABLE:
        print("❌ PyQt5 is required for the desktop GUI")
        print("📦 Install with: pip install PyQt5")
        return 1
    
    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("Scientific Research Assistant")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Group 4 - Mapúa University")
    
    # Set application icon (if available)
    try:
        app.setWindowIcon(QIcon("icon.png"))
    except:
        pass
    
    # Create and show main window
    window = RAGDesktopGUI()
    window.show()
    
    # Start event loop
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())