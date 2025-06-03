from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import subprocess
import time
import logging
import sys
import json
from flask_cors import CORS

# Logger configuration
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='.')
CORS(app)  # Allows CORS requests from other domains

# Check if the static folder exists, and if not, create it
if not os.path.exists('static'):
    os.makedirs('static')
    logger.info("Created folder 'static'")

@app.route('/')
def home():
    """Serves the home page."""
    return render_template('strona.html')

@app.route('/favicon.png')
def favicon():
    """Serves a favicon icon."""
    return send_from_directory('.', 'favicon.png')

@app.route('/style.css')
def css():
    """Serves a CSS file."""
    return send_from_directory('.', 'style.css')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serves static files from the static folder."""
    return send_from_directory('static', filename)

@app.route('/run-analysis', methods=['POST'])
def run_analysis():
    """Runs a hydrology analysis script with the given lake name, country, and time range."""
    logger.info("Hydrological analysis started")
    
    try:
        # Get lake name, country and dates from request
        data = request.json
        lake_name = data.get('lake_name', 'Kisajno')  # Defaults to "Kisajno" if not specified
        country_name = data.get('country_name', 'Poland')  # Defaults to "Poland" if not specified
        start_date = data.get('start_date', '2019-02-02')  # Default start date
        end_date = data.get('end_date', '2025-03-15')  # Default End Date
        
        logger.info(f"Analysis for the lake: {lake_name} in the country: {country_name}, time interval: {start_date} to {end_date}")
        
        # Check if the analysis file exists
        if not os.path.exists('analiza.py'):
            logger.error("File analiza.py does not exist")
            return jsonify({
                'success': False,
                'message': 'File analiza.py does not exist'
            }), 404
        
        # Run analysis script as subprocess
        logger.info(f"Running the analiza.py script for the lake: {lake_name} in the country: {country_name}, time interval: {start_date} to {end_date}")
        start_time = time.time()
        
        # Use the specific Python interpreter you use every day
        python_interpreter = sys.executable  # Uses the same interpreter as the current script
        
        # Run Python script as subprocess, passing lake name, country and dates as arguments
        result = subprocess.run(
            [python_interpreter, 'analiza.py', '--lake', lake_name, '--country', country_name, '--start_date', start_date, '--end_date', end_date], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Error while performing analysis: {result.stderr}")
            return jsonify({
                'success': False,
                'message': f"Error while performing analysis: {result.stderr}"
            }), 500
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Check if images were generated
        expected_images = [
            'water_overlay_first.png',
            'water_overlay_last.png',
            'water_levels.png'
        ]
        
        missing_images = []
        for img in expected_images:
            img_path = os.path.join('static', img)
            if not os.path.exists(img_path):
                missing_images.append(img)
        
        if missing_images:
            logger.warning(f"Missing images: {', '.join(missing_images)}")
        
        # Load the contents of extreme_water_levels_info.txt dynamically
        file_content = ""
        try:
            with open('static/extreme_water_levels_info.txt', 'r') as file:
                file_content1 = file.read()
            with open('static/water_level_trend_info.txt', 'r') as file:
                file_content2 = file.read()
            file_content = file_content1 + "<br><br>" + file_content2
        except FileNotFoundError:
            logger.warning("No info files found - using default message")
            file_content = f"Hydrological analysis for the lake {lake_name} in the country {country_name} has been completed."

        logger.info(f"Analysis completed, execution time: {execution_time:.2f} seconds")
        
        return jsonify({
            'success': True,
            'execution_time': f"{execution_time:.2f}",
            'lake_name': lake_name,
            'country_name': country_name,
            'images': [f'/static/{img}' for img in expected_images if img not in missing_images],
            'message': file_content,  # Dynamic text from file
            'output': result.stdout.strip()
        })
        
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        return jsonify({
            'success': False,
            'message': f'An unexpected error occurred: {str(e)}'
        }), 500


if __name__ == '__main__':
    logger.info(f"Python interpreter used: {sys.executable}")
    
    # Check if the analiza.py script exists, if not, create it from the provided code
    if not os.path.exists('analiza.py'):
        try:
            with open('analiza.py', 'w') as f:
                with open('paste.txt', 'r') as source:
                    f.write(source.read())
            logger.info("Created analysis.py file based on paste.txt")
        except Exception as e:
            logger.error(f"Failed to create file analiza.py: {e}")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)