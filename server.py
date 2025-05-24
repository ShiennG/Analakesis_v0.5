from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import subprocess
import time
import logging
import sys
import json
from flask_cors import CORS

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='.')
CORS(app)  # Umożliwia żądania CORS z innych domen

# Sprawdź, czy folder static istnieje, a jeśli nie, to go utwórz
if not os.path.exists('static'):
    os.makedirs('static')
    logger.info("Utworzono folder 'static'")

@app.route('/')
def home():
    """Serwuje stronę główną."""
    return render_template('strona.html')

@app.route('/favicon.png')
def favicon():
    """Serwuje ikonę favicon."""
    return send_from_directory('.', 'favicon.png')

@app.route('/style.css')
def css():
    """Serwuje plik CSS."""
    return send_from_directory('.', 'style.css')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serwuje pliki statyczne z folderu static."""
    return send_from_directory('static', filename)

@app.route('/run-analysis', methods=['POST'])
def run_analysis():
    """Uruchamia skrypt analizy hydrologicznej z podaną nazwą jeziora, krajem i przedziałem czasu."""
    logger.info("Rozpoczęto analizę hydrologiczną")
    
    try:
        # Pobierz nazwę jeziora, kraj i daty z żądania
        data = request.json
        lake_name = data.get('lake_name', 'Kisajno')  # Domyślnie "Kisajno" jeśli nie podano
        country_name = data.get('country_name', 'Poland')  # Domyślnie "Poland" jeśli nie podano
        start_date = data.get('start_date', '2019-02-02')  # Domyślna data początkowa
        end_date = data.get('end_date', '2025-03-15')  # Domyślna data końcowa
        
        logger.info(f"Analiza dla jeziora: {lake_name} w kraju: {country_name}, przedział czasu: {start_date} do {end_date}")
        
        # Sprawdź, czy plik analizy istnieje
        if not os.path.exists('analiza.py'):
            logger.error("Plik analiza.py nie istnieje")
            return jsonify({
                'success': False,
                'message': 'Plik analiza.py nie istnieje'
            }), 404
        
        # Uruchom skrypt analizy jako podproces
        logger.info(f"Uruchamianie skryptu analiza.py dla jeziora: {lake_name} w kraju: {country_name}, okres: {start_date} do {end_date}")
        start_time = time.time()
        
        # Użyj konkretnego interpretera Python, którego używasz na co dzień
        python_interpreter = sys.executable  # Używa tego samego interpretera, co aktualny skrypt
        
        # Uruchom skrypt Pythona jako podproces, przekazując nazwę jeziora, kraj i daty jako argumenty
        result = subprocess.run(
            [python_interpreter, 'analiza.py', '--lake', lake_name, '--country', country_name, '--start_date', start_date, '--end_date', end_date], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Błąd podczas wykonywania analizy: {result.stderr}")
            return jsonify({
                'success': False,
                'message': f"Błąd podczas wykonywania analizy: {result.stderr}"
            }), 500
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Sprawdź, czy obrazy zostały wygenerowane
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
            logger.warning(f"Brakujące obrazy: {', '.join(missing_images)}")
        
        # Wczytaj dynamicznie zawartość pliku extreme_water_levels_info.txt
        file_content = ""
        try:
            with open('static/extreme_water_levels_info.txt', 'r') as file:
                file_content1 = file.read()
            with open('static/water_level_trend_info.txt', 'r') as file:
                file_content2 = file.read()
            file_content = file_content1 + "<br><br>" + file_content2
        except FileNotFoundError:
            logger.warning("Nie znaleziono plików info - używam domyślnej wiadomości")
            file_content = f"Analiza hydrologiczna dla jeziora {lake_name} w kraju {country_name} została zakończona."

        logger.info(f"Analiza zakończona, czas wykonania: {execution_time:.2f} sekund")
        
        return jsonify({
            'success': True,
            'execution_time': f"{execution_time:.2f}",
            'lake_name': lake_name,
            'country_name': country_name,
            'images': [f'/static/{img}' for img in expected_images if img not in missing_images],
            'message': file_content,  # Dynamiczny tekst z pliku
            'output': result.stdout.strip()
        })
        
    except Exception as e:
        logger.exception(f"Wystąpił nieoczekiwany błąd: {e}")
        return jsonify({
            'success': False,
            'message': f'Wystąpił nieoczekiwany błąd: {str(e)}'
        }), 500


if __name__ == '__main__':
    logger.info(f"Używany interpreter Python: {sys.executable}")
    
    # Sprawdź, czy skrypt analiza.py istnieje, jeśli nie, utwórz go z dostarczonego kodu
    if not os.path.exists('analiza.py'):
        try:
            with open('analiza.py', 'w') as f:
                with open('paste.txt', 'r') as source:
                    f.write(source.read())
            logger.info("Utworzono plik analiza.py na podstawie paste.txt")
        except Exception as e:
            logger.error(f"Nie udało się utworzyć pliku analiza.py: {e}")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)