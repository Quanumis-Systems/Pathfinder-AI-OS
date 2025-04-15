# Multimodal Input Graphical User Interface (MIGUI)
# This module provides a graphical interface for interacting with the Pathfinder AI OS.
# It supports multimodal inputs such as text, voice, and visual interactions.

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QTextEdit, QLabel, QLineEdit, QFontDialog
from PyQt5.QtCore import Qt
from core.adaptive_interface import AdaptiveInterface
# Importing additional libraries for gesture recognition and NLP.
from transformers import pipeline
from PyQt5.QtGui import QFont
import speech_recognition as sr
import cv2

class MIGUI(QMainWindow):
    """Main window for the Multimodal Input Graphical User Interface."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pathfinder AI OS")
        self.setGeometry(100, 100, 800, 600)

        self.interface = AdaptiveInterface()  # Connect to the adaptive interface.

        # Set up the main layout.
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Add components to the GUI.
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface components."""
        self.label = QLabel("Welcome to Pathfinder AI OS")
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter your command here...")
        self.layout.addWidget(self.input_field)

        self.submit_button = QPushButton("Submit Command")
        self.submit_button.clicked.connect(self.process_command)
        self.layout.addWidget(self.submit_button)

        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.layout.addWidget(self.output_area)

        # Add buttons for additional functionalities.
        self.add_task_button = QPushButton("Add Cognitive Task")
        self.add_task_button.clicked.connect(self.add_cognitive_task)
        self.layout.addWidget(self.add_task_button)

        self.view_task_plan_button = QPushButton("View Task Plan")
        self.view_task_plan_button.clicked.connect(self.view_task_plan)
        self.layout.addWidget(self.view_task_plan_button)

        self.manage_load_button = QPushButton("Manage Cognitive Load")
        self.manage_load_button.clicked.connect(self.manage_cognitive_load)
        self.layout.addWidget(self.manage_load_button)

        self.add_user_button = QPushButton("Add User Profile")
        self.add_user_button.clicked.connect(self.add_user_profile)
        self.layout.addWidget(self.add_user_button)

        self.find_matches_button = QPushButton("Find User Matches")
        self.find_matches_button.clicked.connect(self.find_user_matches)
        self.layout.addWidget(self.find_matches_button)

        self.create_project_button = QPushButton("Create Project")
        self.create_project_button.clicked.connect(self.create_project)
        self.layout.addWidget(self.create_project_button)

        self.list_projects_button = QPushButton("List Projects")
        self.list_projects_button.clicked.connect(self.list_projects)
        self.layout.addWidget(self.list_projects_button)

        # Add accessibility options.
        self.accessibility_button = QPushButton("Accessibility Options")
        self.accessibility_button.clicked.connect(self.open_accessibility_options)
        self.layout.addWidget(self.accessibility_button)

        # Add voice input button.
        self.voice_input_button = QPushButton("Voice Input")
        self.voice_input_button.clicked.connect(self.process_voice_input)
        self.layout.addWidget(self.voice_input_button)

        # Add visual input button.
        self.visual_input_button = QPushButton("Visual Input")
        self.visual_input_button.clicked.connect(self.process_visual_input)
        self.layout.addWidget(self.visual_input_button)

        # Add gesture recognition button.
        self.gesture_recognition_button = QPushButton("Gesture Recognition")
        self.gesture_recognition_button.clicked.connect(self.process_gesture_recognition)
        self.layout.addWidget(self.gesture_recognition_button)

    def open_accessibility_options(self):
        """Open a dialog for accessibility options."""
        font, ok = QFontDialog.getFont()
        if ok:
            self.setFont(font)

    def process_gesture_recognition(self):
        """Process gestures using OpenCV with specific detection logic."""
        self.output_area.append("Starting gesture recognition...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.output_area.append("Error: Could not access the camera.")
            return

        hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'aGest.xml')  # Placeholder for a hand gesture cascade.

        while True:
            ret, frame = cap.read()
            if not ret:
                self.output_area.append("Error: Could not read frame from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hands = hand_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in hands:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Hand Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            cv2.imshow("Gesture Recognition", frame)

            # Close the camera feed on pressing 'q'.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.output_area.append("Gesture recognition ended.")

    def process_voice_input(self):
        """Process voice input with advanced NLP parsing."""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            self.output_area.append("Listening for voice input...")
            try:
                audio = recognizer.listen(source)
                command = recognizer.recognize_google(audio)
                self.output_area.append(f"Voice Command: {command}")

                # Use NLP to parse the command.
                nlp = pipeline("text-classification", model="distilbert-base-uncased")
                parsed_command = nlp(command)
                self.output_area.append(f"Parsed Command: {parsed_command}")

                # Route the parsed command to the adaptive interface.
                response = self.interface.process_command(command)
                self.output_area.append(response)
            except sr.UnknownValueError:
                self.output_area.append("Could not understand the voice input.")
            except sr.RequestError as e:
                self.output_area.append(f"Error with speech recognition service: {e}")

    def process_visual_input(self):
        """Process visual input using OpenCV."""
        self.output_area.append("Opening camera for visual input...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.output_area.append("Error: Could not access the camera.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                self.output_area.append("Error: Could not read frame from camera.")
                break

            cv2.imshow("Visual Input", frame)

            # Example: Close the camera feed on pressing 'q'.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.output_area.append("Visual input processing ended.")

    def process_command(self):
        """Process the command entered by the user."""
        command = self.input_field.text()
        if not command:
            self.output_area.append("[Error] Command cannot be empty.")
            return

        # Example: Parse the command and send it to the adaptive interface.
        response = self.interface.process_command(command)
        self.output_area.append(f"> {command}")
        self.output_area.append(response)
        self.input_field.clear()

    def add_cognitive_task(self):
        response = self.interface.process_command("add_cognitive_task", {
            "task_id": "1",
            "description": "Write project report",
            "priority": 1,
            "estimated_load": 2.5
        })
        self.output_area.append(response)

    def view_task_plan(self):
        response = self.interface.process_command("get_cognitive_task_plan")
        self.output_area.append(response)

    def manage_cognitive_load(self):
        response = self.interface.process_command("manage_cognitive_load", {"max_load": 3.0})
        self.output_area.append(response)

    def add_user_profile(self):
        response = self.interface.process_command("add_user_profile", {
            "user_id": "1",
            "interests": ["AI", "Music"],
            "preferences": {"communication": "text"}
        })
        self.output_area.append(response)

    def find_user_matches(self):
        response = self.interface.process_command("find_user_matches", {"user_id": "1"})
        self.output_area.append(response)

    def create_project(self):
        response = self.interface.process_command("create_project", {
            "project_id": "1",
            "name": "AI Research",
            "resources": {"budget": 10000}
        })
        self.output_area.append(response)

    def list_projects(self):
        response = self.interface.process_command("list_projects")
        self.output_area.append(response)

# Entry point for the MIGUI application.
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MIGUI()
    window.show()
    sys.exit(app.exec_())
