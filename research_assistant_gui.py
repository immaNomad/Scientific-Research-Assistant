#!/usr/bin/env python3
"""
Research Assistant - Desktop GUI Application
Beautiful chatbox interface with white background and Klein Blue accents
Features: RAG (Retrieval-Augmented Generation) + RL (Reinforcement Learning) optimization
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import asyncio
import threading
import sys
import os
import json
from datetime import datetime
from typing import Optional

# Set up API keys from centralized config
try:
    from config.api_keys import GOOGLE_GEMINI_API_KEY
    if GOOGLE_GEMINI_API_KEY:
        os.environ['GOOGLE_GEMINI_API_KEY'] = GOOGLE_GEMINI_API_KEY
        print("✅ API key loaded from config/api_keys.py")
    else:
        print("⚠️  Warning: No API key found in config/api_keys.py")
        print("Please edit config/api_keys.py and add your API key")
except ImportError:
    print("⚠️  Warning: config/api_keys.py not found")
    print("Copy config/api_keys.example.py to config/api_keys.py and add your API key")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.rl.rl_optimizer import RLEnhancedRAG, ResearchAnalysis
except ImportError:
    print("Error: Could not import RL modules. Please ensure src/rl/rl_optimizer.py exists.")
    sys.exit(1)

# Klein Blue color palette
KLEIN_BLUE = "#002FA7"
LIGHT_KLEIN_BLUE = "#4D7FBF"
ULTRA_LIGHT_KLEIN_BLUE = "#E6EFFF"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F5F5F5"
DARK_GRAY = "#333333"
SUCCESS_GREEN = "#00C851"
WARNING_ORANGE = "#FF8800"
ERROR_RED = "#FF4444"

class ChatMessage:
    """Represents a chat message in the interface"""
    def __init__(self, sender: str, content: str, message_type: str = "text", metadata: dict = None):
        self.sender = sender
        self.content = content
        self.message_type = message_type  # "text", "analysis", "papers", "error", "success"
        self.metadata = metadata or {}
        self.timestamp = datetime.now().strftime("%H:%M:%S")

class ResearchAssistantGUI:
    """Main GUI application for Research Assistant (RAG + RL)"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.rl_rag = None
        self.last_analysis = None
        self.chat_messages = []
        
        self.setup_window()
        self.setup_styles()
        self.create_widgets()
        self.initialize_rl_system()
        
        # Start with welcome message
        self.add_system_message("Welcome to Scientific Research Assistant!")
        self.add_system_message("Ask me anything about research papers and I'll learn to optimize my responses!")
    
    def setup_window(self):
        """Configure the main window"""
        self.root.title("Scientific Research Assistant")
        self.root.geometry("1200x800")
        self.root.configure(bg=WHITE)
        
        # Set minimum size
        self.root.minsize(800, 600)
        
        # Center window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1200 // 2)
        y = (self.root.winfo_screenheight() // 2) - (800 // 2)
        self.root.geometry(f"1200x800+{x}+{y}")
        
        # Set icon (if available)
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
    
    def setup_styles(self):
        """Configure ttk styles"""
        self.style = ttk.Style()
        
        # Configure button styles
        self.style.configure("Klein.TButton",
                           background=KLEIN_BLUE,
                           foreground=WHITE,
                           borderwidth=0,
                           focuscolor='none',
                           padding=(10, 5))
        
        self.style.map("Klein.TButton",
                      background=[('active', LIGHT_KLEIN_BLUE),
                                ('pressed', KLEIN_BLUE)])
        
        # Configure frame styles
        self.style.configure("Klein.TFrame",
                           background=WHITE,
                           relief="solid",
                           borderwidth=1)
        
        # Configure label styles
        self.style.configure("Title.TLabel",
                           background=WHITE,
                           foreground=KLEIN_BLUE,
                           font=("Lucida Sans", 18, "bold"))
        
        self.style.configure("Subtitle.TLabel",
                           background=WHITE,
                           foreground=DARK_GRAY,
                           font=("Lucida Sans", 12))
        
        self.style.configure("Stats.TLabel",
                           background=ULTRA_LIGHT_KLEIN_BLUE,
                           foreground=KLEIN_BLUE,
                           font=("Lucida Sans", 11),
                           padding=(5, 2))
    
    def create_widgets(self):
        """Create and arrange GUI widgets"""
        
        # Main container with Klein Blue border
        main_frame = tk.Frame(self.root, bg=WHITE, relief="solid", bd=2, highlightbackground=KLEIN_BLUE, highlightthickness=2)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main content area
        content_frame = tk.Frame(main_frame, bg=WHITE)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left panel (chat)
        self.create_chat_panel(content_frame)
        
        # Right panel (info and controls)
        self.create_info_panel(content_frame)
        
        # Bottom Klein Blue separator
        bottom_sep = tk.Frame(main_frame, height=2, bg=KLEIN_BLUE)
        bottom_sep.pack(fill=tk.X, pady=(10, 5))
        
        # Bottom input area
        self.create_input_area(main_frame)
    

    
    def create_chat_panel(self, parent):
        """Create the main chat interface"""
        chat_frame = tk.Frame(parent, bg=WHITE)
        chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Chat display area with Klein Blue border
        chat_container = tk.Frame(chat_frame, bg=KLEIN_BLUE, relief="solid", bd=2)
        chat_container.pack(fill=tk.BOTH, expand=True)
        
        self.chat_text = scrolledtext.ScrolledText(
            chat_container,
            wrap=tk.WORD,
            bg=WHITE,
            fg=DARK_GRAY,
            font=("Lucida Sans", 13),
            relief="flat",
            borderwidth=0,
            padx=15,
            pady=15
        )
        self.chat_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Configure text tags for styling
        self.chat_text.tag_configure("user", foreground=KLEIN_BLUE, font=("Lucida Sans", 13, "bold"))
        self.chat_text.tag_configure("system", foreground=SUCCESS_GREEN, font=("Lucida Sans", 13, "bold"))
        self.chat_text.tag_configure("error", foreground=ERROR_RED, font=("Lucida Sans", 13, "bold"))
        self.chat_text.tag_configure("timestamp", foreground=LIGHT_KLEIN_BLUE, font=("Lucida Sans", 11))
        self.chat_text.tag_configure("separator", foreground=KLEIN_BLUE)
        self.chat_text.tag_configure("paper_title", foreground=KLEIN_BLUE, font=("Lucida Sans", 13, "bold"))
        self.chat_text.tag_configure("metadata", foreground=DARK_GRAY, font=("Lucida Sans", 11, "italic"))
        self.chat_text.tag_configure("rl_stats", foreground=KLEIN_BLUE, font=("Lucida Sans", 12, "bold"))
    
    def create_info_panel(self, parent):
        """Create the information and controls panel"""
        info_frame = tk.Frame(parent, bg=WHITE)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10))
        info_frame.configure(width=320)
        
        # RL Statistics
        self.create_rl_stats_section(info_frame)
        
        # Controls
        self.create_controls_section(info_frame)
    
    def create_rl_stats_section(self, parent):
        """Create RL statistics display"""
        stats_container = tk.Frame(parent, bg=KLEIN_BLUE, relief="solid", bd=2)
        stats_container.pack(fill=tk.X, pady=(0, 15))
        
        stats_frame = tk.Frame(stats_container, bg=ULTRA_LIGHT_KLEIN_BLUE)
        stats_frame.pack(fill=tk.BOTH, padx=2, pady=2)
        
        # Title
        title_label = tk.Label(stats_frame, text="🧠 RL Learning Stats", 
                              bg=ULTRA_LIGHT_KLEIN_BLUE, fg=KLEIN_BLUE, 
                              font=("Lucida Sans", 14, "bold"))
        title_label.pack(pady=(10, 5))
        
        # Stats labels
        self.q_table_label = tk.Label(stats_frame, text="Q-table: Loading...", 
                                     bg=ULTRA_LIGHT_KLEIN_BLUE, fg=KLEIN_BLUE, 
                                     font=("Lucida Sans", 12))
        self.q_table_label.pack(fill=tk.X, pady=2)
        
        self.epsilon_label = tk.Label(stats_frame, text="Exploration: Loading...", 
                                     bg=ULTRA_LIGHT_KLEIN_BLUE, fg=KLEIN_BLUE, 
                                     font=("Lucida Sans", 12))
        self.epsilon_label.pack(fill=tk.X, pady=2)
        
        self.experiences_label = tk.Label(stats_frame, text="Experiences: Loading...", 
                                         bg=ULTRA_LIGHT_KLEIN_BLUE, fg=KLEIN_BLUE, 
                                         font=("Lucida Sans", 12))
        self.experiences_label.pack(fill=tk.X, pady=2)
        
        self.reward_label = tk.Label(stats_frame, text="Last Reward: -", 
                                    bg=ULTRA_LIGHT_KLEIN_BLUE, fg=KLEIN_BLUE, 
                                    font=("Lucida Sans", 12))
        self.reward_label.pack(fill=tk.X, pady=(2, 10))
    
    def create_controls_section(self, parent):
        """Create control buttons"""
        controls_container = tk.Frame(parent, bg=KLEIN_BLUE, relief="solid", bd=2)
        controls_container.pack(fill=tk.X, pady=(0, 15))
        
        controls_frame = tk.Frame(controls_container, bg=WHITE)
        controls_frame.pack(fill=tk.BOTH, padx=2, pady=2)
        
        # Title
        title_label = tk.Label(controls_frame, text="🎮 Controls", 
                              bg=WHITE, fg=KLEIN_BLUE, 
                              font=("Lucida Sans", 14, "bold"))
        title_label.pack(pady=(10, 5))
        
        # RL iterations selector
        iter_frame = tk.Frame(controls_frame, bg=WHITE)
        iter_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(iter_frame, text="RL Iterations:", bg=WHITE, fg=DARK_GRAY, 
                font=("Lucida Sans", 12)).pack(anchor=tk.W)
        
        self.iterations_var = tk.StringVar(value="3")
        iterations_combo = ttk.Combobox(iter_frame, textvariable=self.iterations_var,
                                      values=["1", "2", "3", "4", "5"], state="readonly", width=15)
        iterations_combo.pack(fill=tk.X, pady=(2, 0))
        
        # Control buttons with Klein Blue styling
        button_frame = tk.Frame(controls_frame, bg=WHITE)
        button_frame.pack(fill=tk.X, padx=10, pady=(10, 15))
        
        buttons = [
            ("📊 View Stats", self.show_detailed_stats),
            ("💾 Export Analysis", self.export_analysis),
            ("🔄 Reset RL", self.reset_rl),
            ("🧹 Clear Chat", self.clear_chat)
        ]
        
        for text, command in buttons:
            btn = tk.Button(button_frame, text=text, command=command,
                           bg=KLEIN_BLUE, fg=WHITE, font=("Lucida Sans", 12),
                           relief="flat", bd=0, pady=5,
                           activebackground=LIGHT_KLEIN_BLUE, activeforeground=WHITE)
            btn.pack(fill=tk.X, pady=2)
    

    
    def create_input_area(self, parent):
        """Create the input area at the bottom"""
        input_frame = tk.Frame(parent, bg=WHITE)
        input_frame.pack(fill=tk.X, pady=(5, 10))
        
        # Input label
        input_label = tk.Label(input_frame, text="💭 Research Query:", 
                              bg=WHITE, fg=KLEIN_BLUE, font=("Lucida Sans", 14, "bold"))
        input_label.pack(anchor=tk.W, padx=10)
        
        # Input area with Klein Blue border
        input_container = tk.Frame(input_frame, bg=WHITE)
        input_container.pack(fill=tk.X, padx=10, pady=(5, 0))
        
        # Text input with Klein Blue border
        text_container = tk.Frame(input_container, bg=KLEIN_BLUE, relief="solid", bd=2)
        text_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.query_entry = tk.Text(text_container, 
                                 height=3,
                                 bg=WHITE,
                                 fg=DARK_GRAY,
                                 font=("Lucida Sans", 13),
                                 relief="flat",
                                 bd=0,
                                 padx=10,
                                 pady=8,
                                 wrap=tk.WORD)
        self.query_entry.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Send button with Klein Blue styling
        self.send_button = tk.Button(input_container, 
                                   text="🚀 Analyze\n(Ctrl+Enter)",
                                   command=self.send_query,
                                   bg=KLEIN_BLUE, fg=WHITE,
                                   font=("Lucida Sans", 13, "bold"),
                                   relief="flat", bd=0,
                                   padx=20, pady=15,
                                   activebackground=LIGHT_KLEIN_BLUE,
                                   activeforeground=WHITE)
        self.send_button.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind keyboard shortcuts
        self.query_entry.bind("<Control-Return>", lambda e: self.send_query())
        self.query_entry.bind("<Control-Key-Return>", lambda e: self.send_query())
        
        # Focus on input
        self.query_entry.focus_set()
    
    def initialize_rl_system(self):
        """Initialize the RL system in a separate thread"""
        def init_rl():
            try:
                # Ensure API key is set
                if not os.getenv('GOOGLE_GEMINI_API_KEY'):
                    os.environ['GOOGLE_GEMINI_API_KEY'] = 'AIzaSyCoPpl_gokRRupcMFd5B1SpAZinQxWmEpQ'
                
                self.rl_rag = RLEnhancedRAG()
                self.root.after(0, self.update_rl_stats)
                self.root.after(0, lambda: self.add_system_message("RL system initialized and ready! 🎉"))
                self.root.after(0, lambda: self.add_system_message("✅ Gemini API configured and working"))
            except Exception as e:
                error_msg = f"Failed to initialize RL system: {e}"
                self.root.after(0, lambda msg=error_msg: self.add_error_message(msg))
                print(f"RL init error: {e}")  # Also print to console for debugging
        
        threading.Thread(target=init_rl, daemon=True).start()
    
    def add_message(self, message: ChatMessage):
        """Add a message to the chat"""
        self.chat_messages.append(message)
        
        # Format and display message
        self.chat_text.configure(state=tk.NORMAL)
        
        # Add Klein Blue separator line
        self.chat_text.insert(tk.END, "─" * 60 + "\n", "separator")
        
        # Add timestamp and sender
        timestamp_text = f"[{message.timestamp}] "
        self.chat_text.insert(tk.END, timestamp_text, "timestamp")
        
        if message.sender == "System":
            self.chat_text.insert(tk.END, f"🤖 {message.sender}: ", "system")
        elif message.sender == "User":
            self.chat_text.insert(tk.END, f"👤 {message.sender}: ", "user")
        elif message.sender == "Error":
            self.chat_text.insert(tk.END, f"❌ {message.sender}: ", "error")
        
        # Add content based on type
        if message.message_type == "analysis":
            self.format_analysis_message(message)
        elif message.message_type == "papers":
            self.format_papers_message(message)
        else:
            self.chat_text.insert(tk.END, f"{message.content}\n\n")
        
        self.chat_text.configure(state=tk.DISABLED)
        self.chat_text.see(tk.END)
    
    def format_analysis_message(self, message: ChatMessage):
        """Format an analysis result message"""
        analysis = message.metadata.get('analysis')
        if not analysis:
            self.chat_text.insert(tk.END, f"{message.content}\n\n")
            return
        
        # Query
        self.chat_text.insert(tk.END, "📋 Analysis Complete!\n\n", "system")
        self.chat_text.insert(tk.END, f"🔍 Query: {analysis.query}\n\n", "paper_title")
        
        # RL Stats with Klein Blue highlighting
        if analysis.processing_metadata.get('rl_enabled'):
            iterations = analysis.processing_metadata.get('rl_iterations', 0)
            reward = analysis.processing_metadata.get('best_reward', 0)
            self.chat_text.insert(tk.END, f"🤖 RL Optimization: ", "rl_stats")
            self.chat_text.insert(tk.END, f"{iterations} iterations, best reward: {reward:.3f}\n", "metadata")
        
        # Papers found count only
        paper_count = len(analysis.papers)
        self.chat_text.insert(tk.END, f"📚 Papers Found: ", "rl_stats")
        self.chat_text.insert(tk.END, f"{paper_count}\n\n", "metadata")
        
        # Show detailed papers list right after RL stats
        if paper_count > 0:
            self.chat_text.insert(tk.END, f"📚 Found {paper_count} papers:\n\n", "paper_title")
            for i, paper in enumerate(analysis.papers, 1):  # Show all papers
                self.chat_text.insert(tk.END, f"{i}. {paper.title}\n", "metadata")
                if paper.authors:
                    authors = ", ".join(paper.authors[:3])
                    if len(paper.authors) > 3:
                        authors += " et al."
                    self.chat_text.insert(tk.END, f"   Authors: {authors}\n", "metadata")
                if paper.doi:
                    self.chat_text.insert(tk.END, f"   DOI: {paper.doi}\n", "metadata")
                self.chat_text.insert(tk.END, f"   Source: {paper.source.upper()}\n", "metadata")
                self.chat_text.insert(tk.END, "\n")
        
        # Complete summarized findings - show all content
        if analysis.summarized_findings:
            self.chat_text.insert(tk.END, f"\n📝 Detailed Summarized Findings:\n", "paper_title")
            self.chat_text.insert(tk.END, "─" * 60 + "\n", "metadata")
            self.chat_text.insert(tk.END, f"{analysis.summarized_findings}\n\n", "metadata")
        
        # Complete research hypothesis - show all content
        if analysis.hypothesis:
            self.chat_text.insert(tk.END, f"🔬 Research Hypothesis:\n", "paper_title")
            self.chat_text.insert(tk.END, "─" * 60 + "\n", "metadata")
            self.chat_text.insert(tk.END, f"{analysis.hypothesis}\n\n", "metadata")
        
        # Processing time
        proc_time = analysis.processing_metadata.get('processing_time_seconds', 0)
        self.chat_text.insert(tk.END, f"⏱️ Processing Time: {proc_time:.2f} seconds\n\n", "metadata")
        
        # Auto-scroll to bottom to show complete content
        self.chat_text.see(tk.END)
    
    def format_papers_message(self, message: ChatMessage):
        """Format a papers list message"""
        papers = message.metadata.get('papers', [])
        
        self.chat_text.insert(tk.END, f"📚 Found {len(papers)} papers:\n\n")
        
        for i, paper in enumerate(papers, 1):
            self.chat_text.insert(tk.END, f"{i}. ", "metadata")
            self.chat_text.insert(tk.END, f"{paper.title}\n", "paper_title")
            
            authors = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            self.chat_text.insert(tk.END, f"   Authors: {authors}\n", "metadata")
            
            if paper.doi:
                self.chat_text.insert(tk.END, f"   DOI: {paper.doi}\n", "metadata")
            
            self.chat_text.insert(tk.END, f"   Source: {paper.source.upper()}\n", "metadata")
            self.chat_text.insert(tk.END, "\n")
        
        self.chat_text.insert(tk.END, "\n")
    
    def add_user_message(self, content: str):
        """Add a user message"""
        message = ChatMessage("User", content, "text")
        self.add_message(message)
    
    def add_system_message(self, content: str):
        """Add a system message"""
        message = ChatMessage("System", content, "text")
        self.add_message(message)
    
    def add_error_message(self, content: str):
        """Add an error message"""
        message = ChatMessage("Error", content, "error")
        self.add_message(message)
    
    def send_query(self):
        """Process and send the user query"""
        query = self.query_entry.get("1.0", tk.END).strip()
        if not query:
            return
        
        if not self.rl_rag:
            self.add_error_message("RL system not initialized yet. Please wait...")
            return
        
        # Clear input
        self.query_entry.delete("1.0", tk.END)
        
        # Add user message
        self.add_user_message(query)
        

        
        # Disable send button and show progress
        self.send_button.configure(state="disabled", text="🔄 Analyzing...", bg=WARNING_ORANGE)
        
        # Process query in background
        def process_query():
            try:
                iterations = int(self.iterations_var.get())
                
                # Run the RL analysis
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                analysis = loop.run_until_complete(
                    self.rl_rag.research_and_analyze_with_rl(query, max_iterations=iterations)
                )
                loop.close()
                
                self.last_analysis = analysis
                
                # Update UI in main thread
                self.root.after(0, lambda: self.display_analysis_result(analysis))
                self.root.after(0, self.update_rl_stats)
                
            except Exception as exc:
                error_msg = f"Analysis failed: {str(exc)}"
                self.root.after(0, lambda: self.add_error_message(error_msg))
            finally:
                self.root.after(0, lambda: self.send_button.configure(
                    state="normal", text="🚀 Analyze\n(Ctrl+Enter)", bg=KLEIN_BLUE))
        
        threading.Thread(target=process_query, daemon=True).start()
    
    def display_analysis_result(self, analysis):
        """Display the analysis result in chat"""
        message = ChatMessage("System", "Analysis complete!", "analysis", {"analysis": analysis})
        self.add_message(message)
        
        # Update reward display
        if analysis.processing_metadata.get('best_reward'):
            reward = analysis.processing_metadata['best_reward']
            self.reward_label.configure(text=f"Last Reward: {reward:.3f}")
    
    def update_rl_stats(self):
        """Update RL statistics display"""
        if not self.rl_rag:
            return
        
        try:
            stats = self.rl_rag.get_rl_stats()
            
            self.q_table_label.configure(text=f"Q-table: {stats['q_table_size']} states")
            self.epsilon_label.configure(text=f"Exploration (ε): {stats['epsilon']:.3f}")
            self.experiences_label.configure(text=f"Experiences: {stats['total_experiences']}")
            
        except Exception as e:
            print(f"Error updating RL stats: {e}")
    
    def show_detailed_stats(self):
        """Show detailed RL statistics in a popup"""
        if not self.rl_rag:
            messagebox.showwarning("Warning", "RL system not initialized yet.")
            return
        
        try:
            stats = self.rl_rag.get_rl_stats()
            
            stats_text = f"""🧠 RL Learning Statistics

Q-table Size: {stats['q_table_size']} learned states
Exploration Rate (ε): {stats['epsilon']:.3f}
Learning Rate: {stats['learning_rate']:.3f}
Total Experiences: {stats['total_experiences']}

🎯 Learning Progress:
The RL agent has learned strategies for {stats['q_table_size']} different query situations.
Current exploration rate of {stats['epsilon']:.3f} means {stats['epsilon']*100:.1f}% chance of trying new approaches.
"""
            
            messagebox.showinfo("RL Statistics", stats_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get RL statistics: {e}")
    
    def export_analysis(self):
        """Export the last analysis to JSON"""
        if not self.last_analysis:
            messagebox.showwarning("Warning", "No analysis to export yet.")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Export Analysis"
            )
            
            if filename:
                # Convert analysis to export format
                export_data = {
                    "query": self.last_analysis.query,
                    "timestamp": self.last_analysis.timestamp,
                    "rl_optimization": {
                        "enabled": self.last_analysis.processing_metadata.get('rl_enabled', False),
                        "iterations": self.last_analysis.processing_metadata.get('rl_iterations', 0),
                        "best_reward": self.last_analysis.processing_metadata.get('best_reward', 0.0),
                        "learning_stats": self.rl_rag.get_rl_stats() if self.rl_rag else {}
                    },
                    "papers": [
                        {
                            "title": paper.title,
                            "doi": paper.doi,
                            "authors": paper.authors,
                            "abstract": paper.abstract,
                            "source": paper.source,
                            "url": paper.url,
                            "published_date": paper.published_date,
                            "citation_count": paper.citation_count,
                            "venue": paper.venue
                        }
                        for paper in self.last_analysis.papers
                    ],
                    "summarized_findings": self.last_analysis.summarized_findings,
                    "hypothesis": self.last_analysis.hypothesis,
                    "metadata": self.last_analysis.processing_metadata
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                self.add_system_message(f"✅ Analysis exported to: {os.path.basename(filename)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export analysis: {e}")
    
    def reset_rl(self):
        """Reset RL learning"""
        if not self.rl_rag:
            messagebox.showwarning("Warning", "RL system not initialized yet.")
            return
        
        result = messagebox.askyesno("Reset RL Learning", 
                                   "Are you sure you want to reset all RL learning?\nThis will clear the Q-table and start fresh.")
        
        if result:
            try:
                self.rl_rag.rl_optimizer.q_table = {}
                self.rl_rag.rl_optimizer.epsilon = 0.3
                self.update_rl_stats()
                self.add_system_message("🔄 RL learning reset successfully! Starting fresh with exploration.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to reset RL: {e}")
    
    def clear_chat(self):
        """Clear the chat history"""
        result = messagebox.askyesno("Clear Chat", "Are you sure you want to clear the chat history?")
        
        if result:
            self.chat_text.configure(state=tk.NORMAL)
            self.chat_text.delete(1.0, tk.END)
            self.chat_text.configure(state=tk.DISABLED)
            self.chat_messages = []
            self.add_system_message("Chat cleared! Ready for new queries. 🧹")
    

    
    def run(self):
        """Start the GUI application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        except Exception as e:
            print(f"Application error: {e}")

def main():
    """Main function to run the desktop application"""
    try:
        app = ResearchAssistantGUI()
        app.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application: {e}")

if __name__ == "__main__":
    main() 