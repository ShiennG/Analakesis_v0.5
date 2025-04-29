from flask import Flask, request, jsonify, send_from_directory
import traceback
import os
import threading

# Importujemy funkcję run_analysis z naszego modułu
from analiza import run_analysis

# Zmieniamy konfigurację Flask, aby używał plików z głównego katalogu
app = Flask(__name__, 
            static_folder='.', # Ustaw folder statyczny na bieżący katalog
            static_url_path='',  # Ustaw pustą ścieżkę URL dla plików statycznych
            template_folder='.') # Ustaw folder szablonów na bieżący katalog

@app.route('/')
def index():
    return send_from_directory('.', 'strona.html')

# Dodajemy ścieżkę dla pliku CSS
@app.route('/style.css')
def css():
    return send_from_directory('.', 'style.css')

@app.route('/run-analysis', methods=['POST'])
def analyze():
    # Pobieramy dane z żądania POST
    data = request.json
    lake_name = data.get('lake_name', 'Kisajno')  # Domyślnie Kisajno, jeśli nie podano nazwy
    
    try:
        # Uruchamiamy analizę w osobnym wątku, aby uniknąć blokowania serwera
        # Przy większych projektach lepiej użyć systemu kolejkowania zadań jak Celery
        result = run_analysis(lake_name)
        return jsonify(result)
    except Exception as e:
        # W przypadku błędu zwracamy informację
        error_message = str(e)
        traceback.print_exc()  # Wydrukuj pełny stack trace do logów serwera
        return jsonify({
            "success": False,
            "lake_name": lake_name,
            "message": f"Wystąpił błąd podczas analizy: {error_message}"
        })

if __name__ == '__main__':
    # Uruchamiamy aplikację w trybie debug
    app.run(debug=True)