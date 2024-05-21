# Finger Angle Calculator for Physiotherapy

This desktop application, built using PyQt and MediaPipe, is designed to help physiotherapists treat patients by calculating the angle between fingers when they are bent. The application offers two modes: "Register" and "Live."

## Features

### Register Mode
- Performs facial recognition to identify the user.
- Scans and stores the user's face in data storage.
- Calculates and stores the angles between the user's index finger and thumb for five different hand positions.

### Live Mode
- Detects the user's hand gestures in real-time.
- Calculates the angles between the index finger and thumb for five different hand positions.
- Data is not stored in this mode.

## Usage

1. Launch the application.
2. Choose either "Register" or "Live" mode.
3. If you choose "Register" mode, the application will scan your face apply facial recognition and store it in the data storage.
4. For both modes, the application will detect your hand gestures and display the calculated angles between your index finger and thumb for five different positions.

## Implementation Details

- **User Interface**: PyQt framework for creating the desktop application's graphical user interface.
- **Hand Detection and Tracking**: MediaPipe library for detecting and tracking hand gestures.
- **Angle Calculation**: Mathematical algorithms to calculate the angles between the index finger and thumb based on the detected hand positions.
- **Facial Recognition** (Register Mode): Facial recognition technology to identify and register users.
- **Data Storage** (Register Mode): Storage of user data, including facial information and calculated finger angles, in an Excel file.

## Dependencies

- Python 3.x
- PyQt5
- MediaPipe
- OpenCV (for image processing)
- Additional libraries for facial recognition and Excel file handling

## Contributing

Contributions to this project are welcome. Please follow the standard GitHub workflow for creating issues and submitting pull requests.
