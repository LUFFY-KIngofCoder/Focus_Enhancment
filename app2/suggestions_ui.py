from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTextEdit, QMessageBox, QLineEdit, QScrollArea)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv
import requests  # For Gemini API
import json

from database import Database
from sklearn.preprocessing import LabelEncoder

# Load environment variables
load_dotenv()

# Dark theme colors (matching main app)
DARK_PRIMARY = "#1e1e1e"
DARK_SECONDARY = "#2d2d2d"
DARK_TERTIARY = "#3e3e3e"
DARK_TEXT = "#ffffff"
ACCENT_COLOR = "#3daee9"
WARNING_COLOR = "#f39c12"
ERROR_COLOR = "#e74c3c"
SUCCESS_COLOR = "#2ecc71"

class ChatMessage:
    """Class to represent a single chat message"""
    def __init__(self, text, is_user=False):
        self.text = text
        self.is_user = is_user
        self.timestamp = datetime.now()
    
    def format_html(self):
        """Format the message as HTML for display"""
        align = "right" if self.is_user else "left"
        # Dark theme message colors
        bg_color = ACCENT_COLOR if self.is_user else DARK_TERTIARY
        text_color = DARK_TEXT
        sender = "You" if self.is_user else "Focus AI"
        time = self.timestamp.strftime("%H:%M")
        
        return f"""
        <div style="text-align: {align}; margin: 10px;">
            <div style="display: inline-block; background-color: {bg_color}; color: {text_color}; padding: 10px; border-radius: 10px; max-width: 80%;">
                <b>{sender}</b>
                <p style="color: {text_color};">{self.text.replace('\n', '<br>')}</p>
                <div style="font-size: 8pt; color: #cccccc; text-align: right;">{time}</div>
            </div>
        </div>
        """

class SuggestionsUI(QWidget):
    def __init__(self, db, user_id):
        super().__init__()
        self.db = db
        self.user_id = user_id
        self.chat_history = []
        self.predictions = None
        self.user_patterns = {}
        self.session_history = []
        self.productivity_trends = {}
        self.gemini_conversation_history = []  # Track conversation for Gemini
        
        # Initialize pattern detection
        self.init_pattern_detection()
        
        # Load the models
        self.load_models()
        
        # Create refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.check_for_new_sessions)
        self.refresh_timer.start(60000)  # Check every minute
        
        # Store last session ID to track new sessions
        self.last_session_id = self.get_last_session_id()
        
        self.init_ui()
    
    def get_last_session_id(self):
        """Get the ID of the last session from the database"""
        sessions = self.db.get_user_sessions(self.user_id, limit=1)
        if sessions:
            return sessions[0][0]  # First column is session_id
        return None
    
    def check_for_new_sessions(self):
        """Check for new sessions and update recommendations"""
        current_last_id = self.get_last_session_id()
        if current_last_id and current_last_id != self.last_session_id:
            # New session detected
            sessions = self.db.get_user_sessions(self.user_id, limit=1)
            if sessions:
                session = sessions[0]
                # Convert session tuple to dictionary
                session_data = {
                    'Date': session[1],
                    'Day': session[2],
                    'Start Time': session[3],
                    'End Time': session[4],
                    'Task Type': session[5],
                    'App Switch Count': session[6],
                    'Distraction Duration (mins)': session[7],
                    'Total Focus Duration (mins)': session[8],
                    'Focus Score (0-10)': session[9],
                    'Productivity %': session[10],
                    'Break Duration': session[11]
                }
                # Update models and display
                self.update_models(session_data)
                self.last_session_id = current_last_id
                
                # Add a notification message
                self.add_message("🔄 Recommendations updated based on your latest session!")
    
    def connect_to_pomodoro(self, pomodoro_widget):
        """Connect to pomodoro widget signals"""
        if hasattr(pomodoro_widget, 'session_ended'):
            pomodoro_widget.session_ended.connect(self.on_session_ended)
    
    def on_session_ended(self, *args):
        """Handle session ended signal from pomodoro timer"""
        # Force an immediate check for new sessions
        self.check_for_new_sessions()
        
        # Generate immediate feedback
        current_time = datetime.now()
        optimal = self.predict_optimal_session(current_time)
        
        message = "Session Complete! Here's what I suggest for your next session:\n\n"
        message += f"• Recommended Duration: {optimal['session_length']} minutes\n"
        message += f"• Suggested Break: {optimal['break_duration']} minutes\n"
        if optimal['suggested_tasks']:
            message += "• Consider working on: " + ", ".join(optimal['suggested_tasks'])
        
        self.add_message(message)

    def init_pattern_detection(self):
        """Initialize pattern detection for user behavior"""
        self.pattern_metrics = {
            'focus_by_time': {},      # Track focus scores by time
            'breaks_taken': {},       # Track break patterns
            'task_completion': {},    # Track task completion rates
            'distraction_sources': {} # Track common distractions
        }
        
    def update_user_patterns(self, session_data):
        """Update user patterns with new session data"""
        hour = pd.to_datetime(session_data['Start Time']).hour
        
        # Update focus patterns by time
        if hour not in self.pattern_metrics['focus_by_time']:
            self.pattern_metrics['focus_by_time'][hour] = []
        self.pattern_metrics['focus_by_time'][hour].append(session_data['Focus Score (0-10)'])
        
        # Update break patterns
        break_duration = session_data.get('Break Duration', 0)
        session_length = session_data['Total Focus Duration (mins)']
        # Round session length to nearest 5 minutes for better grouping
        session_length = round(session_length / 5) * 5
        if session_length not in self.pattern_metrics['breaks_taken']:
            self.pattern_metrics['breaks_taken'][session_length] = []
        self.pattern_metrics['breaks_taken'][session_length].append(break_duration)
        
        # Update task completion tracking
        task_type = session_data['Task Type']
        if task_type not in self.pattern_metrics['task_completion']:
            self.pattern_metrics['task_completion'][task_type] = {
                'completed': 0,
                'total': 0
            }
        self.pattern_metrics['task_completion'][task_type]['total'] += 1
        if session_data['Productivity %'] >= 75:
            self.pattern_metrics['task_completion'][task_type]['completed'] += 1

    def get_personalized_recommendations(self):
        """Generate personalized recommendations based on user patterns"""
        recommendations = []
        
        # Analyze focus patterns by time
        if self.pattern_metrics['focus_by_time']:
            best_hours = []
            for hour, scores in self.pattern_metrics['focus_by_time'].items():
                avg_score = sum(scores) / len(scores)
                if avg_score >= 7:  # High focus threshold
                    best_hours.append((hour, avg_score))
            if best_hours:
                best_hours.sort(key=lambda x: x[1], reverse=True)
                hour_str = f"{best_hours[0][0]:02d}:00"
                recommendations.append(f"You tend to be most focused around {hour_str}")

        # Analyze break patterns
        if self.pattern_metrics['breaks_taken']:
            optimal_breaks = {}
            for session_length, breaks in self.pattern_metrics['breaks_taken'].items():
                avg_break = sum(breaks) / len(breaks)
                optimal_breaks[session_length] = avg_break
            if optimal_breaks:
                best_ratio = max(optimal_breaks.items(), key=lambda x: x[1])
                # Round break duration to nearest minute
                break_mins = round(best_ratio[1])
                recommendations.append(
                    f"For {int(best_ratio[0])}-minute sessions, taking {break_mins}-minute breaks works best for you"
                )

        # Analyze task completion rates
        if self.pattern_metrics['task_completion']:
            task_success = {}
            for task, stats in self.pattern_metrics['task_completion'].items():
                if stats['total'] > 0:
                    success_rate = (stats['completed'] / stats['total']) * 100
                    task_success[task] = success_rate
            if task_success:
                best_task = max(task_success.items(), key=lambda x: x[1])
                recommendations.append(
                    f"You're most effective at {best_task[0]} tasks ({int(best_task[1])}% success rate)"
                )

        return recommendations

    def predict_optimal_session(self, current_time=None):
        """Predict optimal session parameters based on current time and patterns"""
        if current_time is None:
            current_time = datetime.now()
        
        hour = current_time.hour
        day = current_time.strftime("%A").lower()
        
        # Get base predictions
        best_length = self.predict_best_length(self.prepare_current_data())
        
        # Adjust based on time of day
        if hour in self.pattern_metrics['focus_by_time']:
            recent_scores = self.pattern_metrics['focus_by_time'][hour][-5:]  # Last 5 sessions
            if recent_scores:
                avg_recent_score = sum(recent_scores) / len(recent_scores)
                if avg_recent_score < 5:  # If recent performance is poor
                    best_length = min(best_length, 25)  # Suggest shorter sessions
                elif avg_recent_score > 8:  # If recent performance is excellent
                    best_length = max(best_length, 45)  # Allow longer sessions
        
        # Round best_length to nearest 5 minutes
        best_length = 5 * round(best_length / 5)
        
        # Get optimal break duration
        optimal_break = 5  # Default break duration
        if best_length in self.pattern_metrics['breaks_taken']:
            breaks = self.pattern_metrics['breaks_taken'][best_length]
            if breaks:
                optimal_break = sum(breaks) / len(breaks)
        
        # Round the break duration to nearest minute
        optimal_break = round(optimal_break)
        
        # Ensure break is at least 1 minute
        if optimal_break < 1:
            optimal_break = 5  # Default minimum break
        
        return {
            'session_length': int(best_length),
            'break_duration': int(optimal_break),
            'suggested_tasks': self.suggest_tasks(hour, day)
        }

    def suggest_tasks(self, hour, day):
        """Suggest tasks based on historical performance"""
        task_suggestions = []
        
        # Check task completion rates for this time and day
        for task, stats in self.pattern_metrics['task_completion'].items():
            if stats['total'] >= 3:  # Minimum sessions for consideration
                success_rate = (stats['completed'] / stats['total']) * 100
                if success_rate >= 70:  # High success rate threshold
                    task_suggestions.append((task, success_rate))
        
        # Sort by success rate
        task_suggestions.sort(key=lambda x: x[1], reverse=True)
        return [task for task, _ in task_suggestions[:3]]  # Return top 3 tasks

    def generate_insights(self):
        """Generate insights about user's focus patterns"""
        insights = []
        
        # Analyze productivity trends
        if self.session_history:
            recent_sessions = self.session_history[-10:]  # Last 10 sessions
            avg_focus = sum(s['Focus Score (0-10)'] for s in recent_sessions) / len(recent_sessions)
            
            if avg_focus > 7:
                insights.append("Your focus has been consistently strong lately!")
            elif avg_focus < 5:
                insights.append("Your focus scores have room for improvement. Consider shorter sessions or more frequent breaks.")
        
        # Analyze break patterns
        if self.pattern_metrics['breaks_taken']:
            most_productive_lengths = []
            for length, breaks in self.pattern_metrics['breaks_taken'].items():
                if len(breaks) >= 2:  # Need at least 2 data points
                    most_productive_lengths.append((length, sum(breaks) / len(breaks)))
            
            if most_productive_lengths:
                # Get the most productive session length
                best_length = int(round(max(most_productive_lengths, key=lambda x: x[1])[0]))
                insights.append(f"You're most productive in {best_length}-minute sessions.")
        
        insights.append("Analysis based on your last 10 focus sessions.")
        return insights

    def update_models(self, new_session_data):
        """Update models with new session data"""
        self.session_history.append(new_session_data)
        self.update_user_patterns(new_session_data)
        
        # Update predictions with latest data (last 10 sessions)
        current_data = self.prepare_current_data()
        self.predictions = self.get_model_predictions(current_data)
        
        # Generate new insights
        insights = self.generate_insights()
        
        # Update UI with new insights
        self.update_recommendations_display(insights)

    def update_recommendations_display(self, insights):
        """Update the UI with new recommendations and insights"""
        message = "Based on your recent sessions:\n\n"
        
        # Add insights
        for insight in insights:
            message += f"• {insight}\n"
        
        # Add personalized recommendations
        recommendations = self.get_personalized_recommendations()
        if recommendations:
            message += "\nPersonalized Recommendations:\n"
            for rec in recommendations:
                message += f"• {rec}\n"
        
        # Add optimal session suggestion
        optimal = self.predict_optimal_session()
        message += f"\nSuggested Next Session:\n"
        message += f"• Duration: {optimal['session_length']} minutes\n"
        message += f"• Break: {optimal['break_duration']} minutes\n"
        if optimal['suggested_tasks']:
            message += "• Recommended tasks: " + ", ".join(optimal['suggested_tasks'])
        
        # Add to chat
        self.add_message(message)
        
    def init_ui(self):
        # Create main layout
        layout = QVBoxLayout(self)
        
        # Create title
        title_label = QLabel("Focus Enhancement Assistant")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"color: {DARK_TEXT};")
        layout.addWidget(title_label)
        
        # Create chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumHeight(300)
        self.chat_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: {DARK_PRIMARY};
                color: {DARK_TEXT};
                border-radius: 10px;
                padding: 10px;
                font-family: Arial;
                font-size: 10pt;
                border: 1px solid {DARK_TERTIARY};
            }}
        """)
        layout.addWidget(self.chat_display)
        
        # Create input area (horizontal layout)
        input_layout = QHBoxLayout()
        
        # Create text input
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Ask about your focus habits or for productivity tips...")
        self.message_input.returnPressed.connect(self.send_message)
        self.message_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {DARK_SECONDARY};
                color: {DARK_TEXT};
                border-radius: 15px;
                padding: 8px 15px;
                font-size: 10pt;
                border: 1px solid {DARK_TERTIARY};
            }}
            QLineEdit:focus {{
                border: 1px solid {ACCENT_COLOR};
            }}
        """)
        input_layout.addWidget(self.message_input, 5)  # Input takes 5/6 of the space
        
        # Create send button
        send_button = QPushButton("Send")
        send_button.clicked.connect(self.send_message)
        send_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_COLOR};
                color: {DARK_TEXT};
                border-radius: 15px;
                padding: 8px 15px;
                font-weight: bold;
                border: none;
            }}
            QPushButton:hover {{
                background-color: #4CB9FF;
            }}
        """)
        input_layout.addWidget(send_button, 1)  # Button takes 1/6 of the space
        
        layout.addLayout(input_layout)
        
        # Create refresh button
        refresh_button = QPushButton("Start New Conversation")
        refresh_button.clicked.connect(self.refresh_suggestions)
        refresh_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {SUCCESS_COLOR};
                color: {DARK_TEXT};
                border-radius: 15px;
                padding: 8px 15px;
                margin-top: 10px;
                font-weight: bold;
                border: none;
            }}
            QPushButton:hover {{
                background-color: #33d17a;
            }}
        """)
        layout.addWidget(refresh_button)
        
        # Set widget background
        self.setStyleSheet(f"background-color: {DARK_PRIMARY};")
        
        # Initial chat
        self.refresh_suggestions()
    
    def add_message(self, text, is_user=False):
        """Add a message to the chat history and update display"""
        message = ChatMessage(text, is_user)
        self.chat_history.append(message)
        self.update_chat_display()
    
    def update_chat_display(self):
        """Update the chat display with all messages"""
        html_content = ""
        for message in self.chat_history:
            html_content += message.format_html()
        
        self.chat_display.setHtml(html_content)
        
        # Scroll to bottom to show latest message
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def send_message(self):
        """Handle user sending a message"""
        user_text = self.message_input.text().strip()
        if not user_text:
            return
        
        # Add user message to chat
        self.add_message(user_text, is_user=True)
        self.message_input.clear()
        
        # Generate response with context
        self.generate_response(user_text)
    
    def generate_response(self, user_text, context=None):
        """Generate AI response to user message with better context awareness"""
        # Extract question intent
        question_intent = self.classify_question_intent(user_text)
        
        # Build conversation history for context
        conversation_history = []
        if self.gemini_conversation_history:
            # Get the full history, limited to last 5 exchanges for token efficiency
            last_exchanges = self.gemini_conversation_history[-5:] 
            for exchange in last_exchanges:
                conversation_history.append({"role": "user", "parts": [{"text": exchange['user']}]})
                conversation_history.append({"role": "model", "parts": [{"text": exchange['assistant']}]})
        
        # Build app context that includes current recommendations and usage patterns
        app_context = "Focus Enhancement App Context:\n"
        app_context += "- Analysis based on your last 10 focus sessions\n"
        
        if self.predictions:
            app_context += f"- Your optimal focus time: {self.predictions['best_time']}\n"
            app_context += f"- Your optimal session length: {self.predictions['best_length']} minutes\n"
            app_context += f"- Your most productive day: {self.predictions['best_day']}\n"
        
        # Add recent patterns
        if self.session_history and len(self.session_history) >= 2:
            # Get statistics from recent sessions (limit to last 10)
            recent_sessions = self.session_history[-10:]
            avg_focus = sum(s['Focus Score (0-10)'] for s in recent_sessions) / len(recent_sessions)
            avg_productivity = sum(s['Productivity %'] for s in recent_sessions) / len(recent_sessions)
            
            app_context += f"- Recent average focus score: {avg_focus:.1f}/10\n"
            app_context += f"- Recent productivity: {int(avg_productivity)}%\n"
            
            # Add trend information
            recent_trend = recent_sessions[-1]['Focus Score (0-10)'] - recent_sessions[-2]['Focus Score (0-10)']
            if recent_trend > 1:
                app_context += "- Focus has been improving significantly\n"
            elif recent_trend > 0:
                app_context += "- Focus has been slightly improving\n"
            elif recent_trend < -1:
                app_context += "- Focus has decreased significantly\n"
            elif recent_trend < 0:
                app_context += "- Focus has slightly decreased\n"
        
        # Add personalized recommendations to context
        recommendations = self.get_personalized_recommendations()
        if recommendations:
            app_context += "- Personalized recommendations:\n"
            for rec in recommendations:
                app_context += f"  • {rec}\n"
        
        # Build the main prompt
        system_prompt = f"""You are a helpful, friendly AI assistant named Focus AI integrated into a Focus Enhancement app.

Your primary purpose is to help users improve their focus, productivity, and work habits based on their session data.

{app_context}

The Focus Enhancement app includes:
        1. Pomodoro Timer with customizable session lengths and break durations
        2. App tracking that monitors productive vs distracting applications
        3. Focus sessions that record productivity metrics
        4. Statistics dashboard showing focus patterns
        5. Todo list for task management
        
Use a conversational, helpful tone. Be concise but informative. Always maintain context from previous messages.
Respond to the user's specific query while considering their focus patterns and app usage."""
        
        # Try to use Gemini API if we have a key
        if self.gemini_api_key:
            try:
                # Use Gemini 2.0 Flash model
                url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
                
                headers = {
                    "Content-Type": "application/json",
                }
                
                # Create the content array for the Gemini API
                contents = []
                
                # Add system prompt as the first message from the user
                contents.append({
                    "role": "user", 
                    "parts": [{"text": f"You are the Focus AI assistant. Follow these instructions for all your responses: {system_prompt}"}]
                })
                
                # Add a model acknowledgment 
                contents.append({
                    "role": "model", 
                    "parts": [{"text": "I understand. I'm Focus AI, your productivity and focus assistant. I'll help you improve your work habits based on your session data and app usage patterns."}]
                })
                
                # Add conversation history
                if conversation_history:
                    contents.extend(conversation_history)
                
                # Add current user message
                contents.append({"role": "user", "parts": [{"text": user_text}]})
                
                data = {
                    "contents": contents,
                    "generationConfig": {
                        "temperature": 0.7,
                        "topK": 40,
                        "topP": 0.95,
                        "maxOutputTokens": 1024,
                    }
                }
                
                response = requests.post(url, headers=headers, data=json.dumps(data))
                
                if response.status_code == 200:
                    result = response.json()
                    ai_response = result["candidates"][0]["content"]["parts"][0]["text"]
                    
                    # Store the exchange in conversation history
                    self.gemini_conversation_history.append({
                        "user": user_text,
                        "assistant": ai_response
                    })
                    
                    self.add_message(ai_response)
                    return
                else:
                    print(f"Gemini API error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"Gemini API error: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Fallback to local response generation
        self.generate_local_response(user_text, question_intent)
    
    def classify_question_intent(self, user_text):
        """Classify the user's question to better understand their intent"""
        text = user_text.lower()
        
        # Check if user is commenting that responses are repetitive
        repetition_phrases = [
            "same message", "same response", "same answer", "repetitive", 
            "again and again", "keeps repeating", "giving same", "keeps saying",
            "keeps telling", "always say", "always tell", "redundant"
        ]
        
        for phrase in repetition_phrases:
            if phrase in text:
                return "complaint_repetition"
        
        # Define intent categories and their related keywords
        intent_keywords = {
            "time_question": ["when", "time", "morning", "afternoon", "evening", "night", "day time", "hour", "o'clock"],
            "day_question": ["day", "week", "monday", "tuesday", "wednesday", "thursday", "friday", "weekend", "weekday"],
            "duration_question": ["how long", "minutes", "length", "duration", "session", "pomodoro", "timer"],
            "distraction_question": ["distract", "focus", "concentrate", "attention", "app block", "notification", "alert"],
            "app_usage_question": ["how to use", "features", "app", "function", "tracker", "how does", "work"],
            "statistics_question": ["stats", "data", "progress", "improvement", "history", "tracking", "report"],
            "task_question": ["task", "todo", "to-do", "list", "what should I", "what to do", "priority", "work on"],
            "break_question": ["break", "rest", "pause", "interval", "stop", "between"],
            "tech_question": ["problem", "error", "bug", "not working", "issue", "help", "fix", "broken"],
            "personal_question": ["you", "your", "who are", "what are", "chatbot", "assistant"]
        }
        
        # Check for specific intent patterns
        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return intent
        
        # Check for question words
        question_words = ["how", "what", "why", "where", "when", "who", "which", "can", "should", "could", "would"]
        for word in question_words:
            if text.startswith(word) or f" {word} " in text:
                return "general_question"
        
        # Default intent
        return "general_statement"
    
    def generate_local_response(self, user_text, question_intent="general_statement"):
        """Generate a response locally when API is unavailable"""
        user_text_lower = user_text.lower()

        # Handle complaint about repetitive responses
        if question_intent == "complaint_repetition":
            response = """I apologize for being repetitive. I'll try to be more varied and specific in my responses.

What would you like to know about improving your focus? For example:
• Using specific app features?
• Understanding your focus statistics?
• Techniques for handling distractions?
• Setting up effective Pomodoro sessions?"""
            self.add_message(response)
            
            # Store in conversation history
            self.gemini_conversation_history.append({
                "user": user_text,
                "assistant": response
            })
            return
        
        # Handle based on the classified intent of the question
        if question_intent == "time_question":
            if self.predictions:
                response = f"""Your optimal focus time is **{self.predictions['best_time']}** based on your past Pomodoro sessions.

During this time, your focus score averages higher and you experience fewer distractions. To leverage this, schedule your Pomodoro sessions around this time in the Timer tab.

The app will continue learning your patterns as you use it more."""
            else:
                response = """I don't have enough data about your focus patterns yet.

Try using the Pomodoro Timer at different times of day, and I'll analyze when you're most productive. Many users find mornings (8-10am) or mid-afternoons (2-4pm) most effective.

After 5-10 sessions, I can give you personalized timing recommendations."""
                
        elif question_intent == "day_question":
            if self.predictions:
                response = f"""According to your focus session history, **{self.predictions['best_day']}** is your most productive day.

On {self.predictions['best_day']}s, your productivity metrics are consistently higher than other days. Consider scheduling your most challenging tasks on this day.

You can view your day-by-day performance in the Statistics tab."""
            else:
                response = """I need more focus session data to determine your best day.

Use the Pomodoro Timer throughout different days of the week, and I'll identify patterns in your productivity metrics.

The Statistics tab will show you a breakdown by day once you've completed more sessions."""
        
        elif question_intent == "duration_question":
            if self.predictions:
                response = f"""Your focus data shows your optimal session length is **{self.predictions['best_length']} minutes**.

This is when you maintain peak attention before needing a break. You can select this duration directly in the Pomodoro Timer settings.

The app automatically suggests break times based on your session length."""
            else:
                response = """I haven't collected enough data to suggest your ideal session length.

Our Pomodoro Timer offers preset options of 15, 25, 30, 45, and 60 minutes. Try different lengths to see what works best for you.

After several sessions, I'll analyze which duration gives you the highest focus scores."""
        
        elif question_intent == "distraction_question":
            response = """To manage distractions with our app:

1. Use the "Allowed Apps" feature in the Pomodoro Timer - select only productive apps for your session
2. Enable distraction alerts to notify you when you switch to non-productive apps
3. Review your distraction patterns in the Statistics tab
4. Set shorter Pomodoro sessions (25-30 min) if you're easily distracted

What specific distraction issues are you experiencing?"""
        
        elif question_intent == "app_usage_question":
            response = """The Focus Enhancement app helps you optimize productivity through:

1. **Pomodoro Timer**: Set focused work periods with breaks
2. **App Tracking**: Monitor which applications help or hinder focus
3. **Statistics**: View your productivity trends and patterns
4. **Focus AI**: That's me! I analyze your data to provide personalized recommendations

Which specific feature would you like to learn more about?"""
        
        elif question_intent == "statistics_question":
            response = """Your focus statistics are available in the Statistics tab, showing:

• Daily/weekly productivity trends
• Focus scores across different times and days
• Distraction frequency and duration 
• Most productive vs. distracting applications
• Session length effectiveness

The more sessions you complete, the more detailed insights I can provide. Check the charts to see patterns in your focus habits."""
        
        elif question_intent == "task_question":
            if self.predictions:
                response = f"""For maximum productivity, use the Todo List tab to:

1. Add your most important tasks and schedule them for {self.predictions['best_day']}s
2. Set the Pomodoro Timer to {self.predictions['best_length']} minutes when working on them
3. Schedule challenging tasks around {self.predictions['best_time']} when your focus peaks
4. Break large tasks into smaller subtasks for better focus

The app will track which tasks yield your highest productivity."""
            else:
                response = """To manage tasks effectively with our app:

1. Use the Todo List tab to create and prioritize tasks
2. Start a Pomodoro session and select a task to work on
3. The app will track your focus metrics for different task types
4. Complete several sessions so I can analyze which tasks you focus on best

Would you like me to explain how to set up your first task?"""
        
        elif question_intent == "break_question":
            response = """Our app manages breaks based on the Pomodoro technique:

• After each focus session, the app automatically suggests a 5-minute break
• After 4 sessions, it recommends a longer 15-30 minute break
• The break timer counts down just like focus sessions
• Use breaks to stretch, hydrate, or rest your eyes - but avoid digital distractions

Breaks are essential for maintaining focus over longer periods. Would you like to adjust your break settings?"""
        
        elif question_intent == "tech_question":
            response = """If you're experiencing technical issues:

1. Check that your session data is being saved in the Statistics tab
2. Make sure you're ending Pomodoro sessions properly with the Stop button
3. For app tracking issues, verify your allowed apps are correctly selected
4. Try restarting the app if you notice any performance problems

What specific technical problem are you having?"""
        
        elif question_intent == "personal_question":
            response = """I'm your Focus AI assistant built into the Focus Enhancement app. 

I analyze your Pomodoro session data to identify your optimal focus patterns and provide personalized productivity recommendations.

As you use the app more, my suggestions become more tailored to your specific work habits and focus tendencies."""
        
        # General "what should I do" questions
        elif "general_question" in question_intent or any(phrase in user_text_lower for phrase in ["what should i do", "how can i improve", "how to focus", "improve focus", "focus better"]):
            if self.predictions:
                response = """Based on your focus patterns analysis:

• Your optimal focus time is **{self.predictions['best_time']}**
• Your optimal session length is **{self.predictions['best_length']} minutes**
• Your optimal day for deep work is **{self.predictions['best_day']}**

To maximize your productivity:
1. Schedule your most important work during your optimal time and day
2. Set your Pomodoro timer to your optimal session length
3. Use App Tracking to monitor and minimize distractions

Would you like help setting up your next focus session?"""
            else:
                response = """To improve your focus with our app:

1. Start using the Pomodoro Timer regularly - this builds focus stamina
2. Select only productive apps in the Allowed Apps feature
3. Track which tasks give you the highest focus scores
4. Complete 5-10 sessions so I can analyze your optimal focus patterns

Which would you like to try first?"""
        
        # Default response for other inputs
        else:
            if self.predictions and not any(s in user_text_lower for s in ["thank", "thanks", "ok", "okay", "got it"]):
                response = f"""I'm not sure I understand your question. Based on your focus data:

• Your optimal focus time is **{self.predictions['best_time']}**
• Your optimal session length is **{self.predictions['best_length']} minutes**
• Your optimal day for deep work is **{self.predictions['best_day']}**

What specific aspect of your focus would you like help with?"""
            else:
                response = """I'm here to help you improve your focus and productivity using our app's features. I can answer questions about:

• Using the Pomodoro Timer effectively
• Managing distractions with the app
• Understanding your focus statistics
• Optimizing your task management
• Finding your best focus times and durations

What specifically would you like help with?"""
        
        # Add the response to the chat display
        self.add_message(response)
        
        # Store in conversation history for context in future responses
        self.gemini_conversation_history.append({
            "user": user_text,
            "assistant": response
        })

    def get_user_sessions(self):
        """Get the last 10 sessions for the user"""
        sessions = self.db.get_user_sessions(self.user_id, limit=10)  # Ensure we only get last 10
        return sessions
    
    def prepare_data_for_models(self, sessions):
        """Prepare the session data for model input"""
        if not sessions or len(sessions) == 0:
            print("No session data available for model preparation")
            return None, None
            
        data = []
        for session in sessions:
            # Unpack session data 
            session_id, date, day, start_time, end_time, task_type, app_switch_count, distraction_duration, total_focus_duration, focus_score, productivity_percentage, break_duration = session
            
            # Check for any invalid data
            if None in [date, day, start_time, total_focus_duration, focus_score, productivity_percentage]:
                print(f"Warning: Session {session_id} has missing critical data, skipping")
                continue
            
            session_data = {
                'Date': date,
                'Day': day.lower() if isinstance(day, str) else 'monday',  # Ensure lowercase
                'Start Time': start_time,
                'End Time': end_time,
                'Task Type': task_type.lower() if isinstance(task_type, str) else 'studying',  # Ensure lowercase
                'App Switch Count': app_switch_count if app_switch_count is not None else 0,
                'Distraction Duration (mins)': distraction_duration if distraction_duration is not None else 0,
                'Total Focus Duration (mins)': total_focus_duration,
                'Focus Score (0-10)': focus_score,
                'Productivity %': productivity_percentage,
            }
            data.append(session_data)
        
        # Check if we have enough valid sessions
        if len(data) < 3:
            print("Not enough valid sessions for model preparation")
            return None, None
        
        df = pd.DataFrame(data)
        
        # Convert date strings to datetime objects
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
            except Exception as e:
                print(f"Warning: Could not parse dates: {str(e)}")
                # Create a dummy date column to avoid errors
                df['Date'] = pd.to_datetime('2023-01-01')
        
        # Add Hour column
        try:
            df['Hour'] = pd.to_datetime(df['Start Time'], format='%H:%M:%S').dt.hour
        except:
            try:
                df['Hour'] = pd.to_datetime(df['Start Time'], format='%H:%M').dt.hour
            except:
                    df['Hour'] = df['Start Time'].apply(lambda x: 
                        pd.to_datetime(x, format='%H:%M:%S').hour if ':' in str(x) else 9)
        
        # Calculate session duration and length (from Pomodoro notebook)
        df['Session Duration (mins)'] = df['Total Focus Duration (mins)'] + df['Distraction Duration (mins)']
        
        # Round session length to multiples of 15, within 15-90 minute bounds
        df['Session Length'] = df['Session Duration (mins)'].apply(
            lambda x: min(90, max(15, 15 * round(x/15)))
        )
        
        # Preprocessing for day model (from day_time notebook)
        last_date = df['Date'].max()
        df['Days Ago'] = (last_date - df['Date']).dt.days
        df['Weight'] = df['Days Ago'].apply(self.assign_weight)
        df['Weighted Focus Score'] = df['Focus Score (0-10)'] * df['Weight']
        df['Weighted Productivity'] = df['Productivity %'] * df['Weight']
        
        # Encode categorical variables
        day_encoder = LabelEncoder()
        task_encoder = LabelEncoder()
        df['Day_encoded'] = day_encoder.fit_transform(df['Day'])
        df['Task_Type_encoded'] = task_encoder.fit_transform(df['Task Type'])
        
        # Calculate combined score metric
        df['combined_score'] = df['Focus Score (0-10)'] * df['Productivity %'] / 10
        
        # Add productivity label
        df['Productivity_Label'] = df['Productivity %'].apply(
            lambda x: 'High' if x >= 75 else ('Mid' if x >= 50 else 'Low')
        )
        
        print(f"Successfully prepared dataset with {len(df)} sessions")
        return df, day_encoder
    
    def assign_weight(self, days):
        """Assign weight based on how recent the session is"""
        if days <= 7:
            return 1.0
        elif days <= 14:
            return 0.75
        elif days <= 21:
            return 0.5
        else:
            return 0.25
            
    def predict_best_day(self, data_tuple):
        """Predict the best day for productivity using the model"""
        try:
            df, day_encoder = data_tuple
            if df is None:
                return "Monday"  # Default fallback
                
            # Create feature set for day model
            features = ['Day_encoded', 'App Switch Count', 
                       'Distraction Duration (mins)', 'Total Focus Duration (mins)', 
                       'Weighted Focus Score', 'Weighted Productivity']
            
            # Ensure all required features are present
            for feature in features:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Group by day and calculate average productivity
            day_stats = df.groupby('Day').agg({
                'Weighted Focus Score': 'mean',
                'Weighted Productivity': 'mean',
                'Focus Score (0-10)': 'mean',
                'Productivity %': 'mean'
            }).reset_index()
            
            # Find the best day based on combined metrics
            day_stats['overall_score'] = (
                day_stats['Weighted Focus Score'] * 
                day_stats['Weighted Productivity'] * 
                day_stats['Focus Score (0-10)'] * 
                day_stats['Productivity %']
            )
            
            best_day = day_stats.loc[day_stats['overall_score'].idxmax(), 'Day']
            return best_day.capitalize()
            
        except Exception as e:
            print(f"Error predicting best day: {str(e)}")
            return "Monday"  # Default fallback
    
    def predict_best_time(self, data_tuple):
        """Predict the best time for productivity using the model"""
        try:
            df, _ = data_tuple
            if df is None:
                return "9:00 AM"  # Default fallback
                
            # Group by hour and calculate average metrics
            hour_stats = df.groupby('Hour').agg({
                'Focus Score (0-10)': 'mean',
                'Productivity %': 'mean',
                'Weighted Focus Score': 'mean',
                'Weighted Productivity': 'mean'
            }).reset_index()
            
            # Calculate overall score for each hour
            hour_stats['overall_score'] = (
                hour_stats['Focus Score (0-10)'] * 
                hour_stats['Productivity %'] * 
                hour_stats['Weighted Focus Score'] * 
                hour_stats['Weighted Productivity']
            )
            
            # Find the best hour
            best_hour = int(hour_stats.loc[hour_stats['overall_score'].idxmax(), 'Hour'])
            
            # Format as AM/PM time
            if best_hour == 0:
                return "12:00 AM"
            elif best_hour < 12:
                return f"{best_hour}:00 AM"
            elif best_hour == 12:
                return "12:00 PM"
            else:
                return f"{best_hour-12}:00 PM"
                
        except Exception as e:
            print(f"Error predicting best time: {str(e)}")
            return "9:00 AM"  # Default fallback
    
    def predict_best_length(self, data_tuple):
        """Predict the best session length using the pomodoro model"""
        try:
            df, _ = data_tuple
            if df is None:
                print("Warning: Using default session length (25 min) due to missing data")
                return 25  # Default fallback
                
            # Verify Session Length column exists and has valid values
            if 'Session Length' not in df.columns or df['Session Length'].isna().all():
                print("Warning: Session Length column is missing or all values are NaN")
                return 25  # Default fallback
            
            # Check that there are multiple session lengths to compare
            unique_lengths = df['Session Length'].unique()
            if len(unique_lengths) < 2:
                print(f"Warning: Only one session length in data ({unique_lengths[0]})")
                # Still return the single value if it exists and is valid
                if len(unique_lengths) == 1 and 15 <= unique_lengths[0] <= 90:
                    return int(unique_lengths[0])
                return 25  # Default fallback
                
            # Group by session length and calculate average metrics
            length_stats = df.groupby('Session Length').agg({
                'Focus Score (0-10)': 'mean',
                'Productivity %': 'mean',
                'Weighted Focus Score': 'mean',
                'Weighted Productivity': 'mean'
            }).reset_index()
            
            # Calculate overall score for each length
            length_stats['overall_score'] = (
                length_stats['Focus Score (0-10)'] * 
                length_stats['Productivity %'] * 
                length_stats['Weighted Focus Score'] * 
                length_stats['Weighted Productivity']
            )
            
            # Find the best length
            best_length = int(length_stats.loc[length_stats['overall_score'].idxmax(), 'Session Length'])
            
            # Ensure the length is within reasonable bounds and a multiple of 15
            best_length = max(15, min(90, best_length))
            best_length = 15 * round(best_length/15)  # Round to nearest 15 minutes
            
            print(f"Predicted optimal session length: {best_length} minutes")
            return int(best_length)
            
        except Exception as e:
            print(f"Error predicting best length: {str(e)}")
            import traceback
            traceback.print_exc()
            return 25  # Default Pomodoro length
    
    def get_model_predictions(self, data_tuple):
        """Get predictions from all three models"""
        try:
            if data_tuple[0] is None:
                return {
                    'best_time': "9:00 AM",
                    'best_length': 25,
                    'best_day': "Monday"
                }
                
            # Get best day prediction
            best_day = self.predict_best_day(data_tuple)
            
            # Get best time prediction
            best_time = self.predict_best_time(data_tuple)
            
            # Get best session length
            best_length = self.predict_best_length(data_tuple)
            
            return {
                'best_time': best_time,
                'best_length': best_length,
                'best_day': best_day
            }
        except Exception as e:
            print(f"Error getting predictions: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'best_time': "9:00 AM",
                'best_length': 25,
                'best_day': "Monday"
            }
    
    def generate_initial_message(self, predictions, df):
        """Generate the initial welcome message with personalized suggestions"""
        try:
            # Get some statistics to enrich the prompt
            avg_productivity = df['Productivity %'].mean()
            avg_focus_score = df['Focus Score (0-10)'].mean()
            avg_distraction = df['Distraction Duration (mins)'].mean()
            avg_session_length = df['Total Focus Duration (mins)'].mean()
            session_count = len(df)
            
            prompt = f"""You are a focus and productivity AI assistant that helps users improve their focus habits. 
            Based on the user's last {session_count} focus sessions, I've analyzed their patterns and found:
            
            - Best time to focus is: {predictions['best_time']}
            - Best session length to set is: {predictions['best_length']} minutes
            - user can best focus on: {predictions['best_day']}
            
            Current user statistics:
            - Average productivity: {int(avg_productivity)}%
            - Average focus score: {avg_focus_score:.1f}/10
            - Average distraction time: {int(avg_distraction)} minutes
            - Average session length: {int(avg_session_length)} minutes
            
            Write a friendly greeting introducing yourself as a Focus AI assistant, then provide 2-3 personalized suggestions based on their data.
            End by asking how you can help them improve their focus and productivity.
            Keep your response concise, warm and conversational. 
            suggest user something like
             ✨ Your optimal focus time is 
            ⏱️ Your optimal session length is 
            📅 Your optimal day for deep work is 
            (under 150 words)."""

            
            # Try to use Gemini API if we have a key
            if self.gemini_api_key:
                try:
                    # Use Gemini 2.0 Flash model
                    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
                    
                    headers = {
                        "Content-Type": "application/json",
                    }
                    
                    data = {
                        "contents": [
                            {
                                "parts": [
                                    {
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                        "generationConfig": {
                            "temperature": 0.7,
                            "topK": 40,
                            "topP": 0.95,
                            "maxOutputTokens": 1024,
                        }
                    }
                    
                    # Make the API call
                    response = requests.post(url, headers=headers, data=json.dumps(data))
                    
                    if response.status_code == 200:
                        result = response.json()
                        # Extract text from the response
                        return result["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        print(f"Gemini API error: {response.status_code} - {response.text}")
                except Exception as e:
                    print(f"Gemini API error: {str(e)}")
            
            # Default message if API fails
            return self.generate_default_welcome(predictions)
                
        except Exception as e:
            return self.generate_default_welcome(predictions)
    
    def generate_default_welcome(self, predictions):
        """Generate a default welcome message when API is unavailable"""
        best_time = predictions['best_time']
        best_length = predictions['best_length']
        best_day = predictions['best_day']
        
        return f"""👋 Hi there! I'm your Focus AI assistant, here to help you optimize your productivity and focus.

Based on your session data, I've analyzed your optimal conditions:

✨ Your optimal focus time is **{best_time}**
⏱️ Your optimal session length is **{best_length} minutes**
📅 Your optimal day for deep work is **{best_day}**

I can suggest personalized strategies to help you work with your natural rhythms. What would you like to know about improving your focus?"""
    
    def refresh_suggestions(self):
        """Start a new chat with initial suggestions"""
        # Clear chat history
        self.chat_history = []
        self.gemini_conversation_history = []  # Clear Gemini conversation history
        
        # Get last 10 sessions
        sessions = self.db.get_user_sessions(self.user_id, limit=10)
        if not sessions:
            welcome = """👋 Welcome to Focus AI! I don't have enough data about your focus sessions yet.

Start some focus sessions using the Pomodoro Timer, and I'll analyze your patterns to provide personalized suggestions.

In the meantime, I can still answer questions about focus and productivity techniques. What would you like to know?"""
            self.add_message(welcome)
            return
        
        # Prepare data and get predictions
        try:
            data_tuple = self.prepare_data_for_models(sessions)
            self.predictions = self.get_model_predictions(data_tuple)
            
            # Generate initial message with context
            initial_message = self.generate_initial_message(self.predictions, data_tuple[0])  # Pass the DataFrame
            
            # Store initial message in Gemini conversation history
            self.gemini_conversation_history.append({
                "user": "Hi",
                "assistant": initial_message
            })
            
            # Add to chat
            self.add_message(initial_message)
        except Exception as e:
            print(f"Error generating initial message: {str(e)}")
            import traceback
            traceback.print_exc()
            welcome = """👋 Hi there! I'm your Focus AI assistant.

I can help you discover your optimal focus patterns and suggest techniques to improve your productivity.

What would you like to know about focus, productivity, or work habits?"""
            self.add_message(welcome) 

    def load_models(self):
        """Load and initialize the models"""
        try:
            # Load the time model
            with open('finalised/Best_Time.pkl', 'rb') as f:
                self.best_time_model = pickle.load(f)
            
            # Load the length model
            with open('finalised/Best_Length.pkl', 'rb') as f:
                self.best_length_model = pickle.load(f)
            
            # Load the day model
            with open('finalised/Best_Day.pkl', 'rb') as f:
                self.best_day_model = pickle.load(f)
            
            # Initialize model dictionaries if needed
            if isinstance(self.best_day_model, dict) and "MLP" in self.best_day_model:
                self.best_day_model = self.best_day_model["MLP"]
            
            if isinstance(self.best_time_model, dict) and "MLP" in self.best_time_model:
                self.best_time_model = self.best_time_model["MLP"]
            
            if isinstance(self.best_length_model, dict):
                if "model" in self.best_length_model:
                    self.best_length_model = self.best_length_model["model"]
                elif "DecisionTree" in self.best_length_model:
                    self.best_length_model = self.best_length_model["DecisionTree"]
                else:
                    self.best_length_model = next(iter(self.best_length_model.values()))
            
            # Check for Gemini API key
            self.gemini_api_key = os.getenv('GEMINI_API_KEY')
            if not self.gemini_api_key:
                self.gemini_api_key = ""
                QMessageBox.warning(self, "Warning", "Gemini API key not found. Add GEMINI_API_KEY to your .env file for AI-powered suggestions using Gemini 2.0.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load models: {str(e)}")
            import traceback
            traceback.print_exc() 

    def prepare_current_data(self):
        """Get the last 10 sessions and prepare data for predictions"""
        sessions = self.db.get_user_sessions(self.user_id, limit=10)  # Get last 10 sessions
        if not sessions or len(sessions) < 3:  # Require at least 3 sessions for meaningful predictions
            print("Not enough session data for meaningful predictions (need at least 3 sessions)")
            return None, None
            
        # Prepare the data using the existing method
        data_tuple = self.prepare_data_for_models(sessions)
        
        # Verify the data has been properly processed
        df, encoder = data_tuple
        if df is None or df.empty or 'Session Length' not in df.columns:
            print("Error: Dataset was not properly prepared or is missing required columns")
            return None, None
            
        print(f"Successfully prepared data from {len(sessions)} sessions for model prediction")
        return df, encoder

    def predict_best_length(self, data_tuple):
        """Predict the best session length using the pomodoro model"""
        try:
            df, _ = data_tuple
            if df is None:
                print("Warning: Using default session length (25 min) due to missing data")
                return 25  # Default fallback
                
            # Verify Session Length column exists and has valid values
            if 'Session Length' not in df.columns or df['Session Length'].isna().all():
                print("Warning: Session Length column is missing or all values are NaN")
                return 25  # Default fallback
            
            # Check that there are multiple session lengths to compare
            unique_lengths = df['Session Length'].unique()
            if len(unique_lengths) < 2:
                print(f"Warning: Only one session length in data ({unique_lengths[0]})")
                # Still return the single value if it exists and is valid
                if len(unique_lengths) == 1 and 15 <= unique_lengths[0] <= 90:
                    return int(unique_lengths[0])
                return 25  # Default fallback
                
            # Group by session length and calculate average metrics
            length_stats = df.groupby('Session Length').agg({
                'Focus Score (0-10)': 'mean',
                'Productivity %': 'mean',
                'Weighted Focus Score': 'mean',
                'Weighted Productivity': 'mean'
            }).reset_index()
            
            # Calculate overall score for each length
            length_stats['overall_score'] = (
                length_stats['Focus Score (0-10)'] * 
                length_stats['Productivity %'] * 
                length_stats['Weighted Focus Score'] * 
                length_stats['Weighted Productivity']
            )
            
            # Find the best length
            best_length = int(length_stats.loc[length_stats['overall_score'].idxmax(), 'Session Length'])
            
            # Ensure the length is within reasonable bounds and a multiple of 15
            best_length = max(15, min(90, best_length))
            best_length = 15 * round(best_length/15)  # Round to nearest 15 minutes
            
            print(f"Predicted optimal session length: {best_length} minutes")
            return int(best_length)
            
        except Exception as e:
            print(f"Error predicting best length: {str(e)}")
            import traceback
            traceback.print_exc()
            return 25  # Default Pomodoro length

    def prepare_data_for_models(self, sessions):
        """Prepare the session data for model input"""
        if not sessions or len(sessions) == 0:
            print("No session data available for model preparation")
            return None, None
            
        data = []
        for session in sessions:
            # Unpack session data 
            session_id, date, day, start_time, end_time, task_type, app_switch_count, distraction_duration, total_focus_duration, focus_score, productivity_percentage, break_duration = session
            
            # Check for any invalid data
            if None in [date, day, start_time, total_focus_duration, focus_score, productivity_percentage]:
                print(f"Warning: Session {session_id} has missing critical data, skipping")
                continue
                
            session_data = {
                'Date': date,
                'Day': day.lower() if isinstance(day, str) else 'monday',  # Ensure lowercase
                'Start Time': start_time,
                'End Time': end_time,
                'Task Type': task_type.lower() if isinstance(task_type, str) else 'studying',  # Ensure lowercase
                'App Switch Count': app_switch_count if app_switch_count is not None else 0,
                'Distraction Duration (mins)': distraction_duration if distraction_duration is not None else 0,
                'Total Focus Duration (mins)': total_focus_duration,
                'Focus Score (0-10)': focus_score,
                'Productivity %': productivity_percentage,
            }
            data.append(session_data)
        
        # Check if we have enough valid sessions
        if len(data) < 3:
            print("Not enough valid sessions for model preparation")
            return None, None
            
        df = pd.DataFrame(data)
        
        # Convert date strings to datetime objects
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
            except Exception as e:
                print(f"Warning: Could not parse dates: {str(e)}")
                # Create a dummy date column to avoid errors
                df['Date'] = pd.to_datetime('2023-01-01')
        
        # Add Hour column
        try:
            df['Hour'] = pd.to_datetime(df['Start Time'], format='%H:%M:%S').dt.hour
        except:
            try:
                df['Hour'] = pd.to_datetime(df['Start Time'], format='%H:%M').dt.hour
            except:
                df['Hour'] = df['Start Time'].apply(lambda x: 
                    pd.to_datetime(x, format='%H:%M:%S').hour if ':' in str(x) else 9)
        
        # Calculate session duration and length (from Pomodoro notebook)
        df['Session Duration (mins)'] = df['Total Focus Duration (mins)'] + df['Distraction Duration (mins)']
        
        # Round session length to multiples of 15, within 15-90 minute bounds
        df['Session Length'] = df['Session Duration (mins)'].apply(
            lambda x: min(90, max(15, 15 * round(x/15)))
        )
        
        # Preprocessing for day model (from day_time notebook)
        last_date = df['Date'].max()
        df['Days Ago'] = (last_date - df['Date']).dt.days
        df['Weight'] = df['Days Ago'].apply(self.assign_weight)
        df['Weighted Focus Score'] = df['Focus Score (0-10)'] * df['Weight']
        df['Weighted Productivity'] = df['Productivity %'] * df['Weight']
        
        # Encode categorical variables
        day_encoder = LabelEncoder()
        task_encoder = LabelEncoder()
        df['Day_encoded'] = day_encoder.fit_transform(df['Day'])
        df['Task_Type_encoded'] = task_encoder.fit_transform(df['Task Type'])
        
        # Calculate combined score metric
        df['combined_score'] = df['Focus Score (0-10)'] * df['Productivity %'] / 10
        
        # Add productivity label
        df['Productivity_Label'] = df['Productivity %'].apply(
            lambda x: 'High' if x >= 75 else ('Mid' if x >= 50 else 'Low')
        )
        
        print(f"Successfully prepared dataset with {len(df)} sessions")
        return df, day_encoder
    
    def assign_weight(self, days):
        """Assign weight based on how recent the session is"""
        if days <= 7:
            return 1.0
        elif days <= 14:
            return 0.75
        elif days <= 21:
            return 0.5
        else:
            return 0.25 