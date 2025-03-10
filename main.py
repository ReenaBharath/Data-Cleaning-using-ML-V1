"""
Wrapper script to run the data cleaning and machine learning pipeline.
This script simply calls the main script in the src directory.
"""
import os
import sys

if __name__ == "__main__":
    # Add the project root directory to the Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Import and run the main function from the src/main.py file
    from src.main import main
    
    # Run the main function
    main()
