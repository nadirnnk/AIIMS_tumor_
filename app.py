from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
import os
import json
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app, origins=["https://aiimstumor.vercel.app", "https://aiimstumor.vercel.app/"])  # Enable CORS for Vercel

# Make sure to set this environment variable before running the app
# os.environ["GEMINI_API_KEY"] = "your_api_key_here"

@app.route('/')
def serve_frontend():
    return send_from_directory('static', 'index.html')

@app.route('/api/extract', methods=['POST'])
def extract_features():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return jsonify({"error": "Server missing API key"}), 500

    genai.configure(api_key=api_key)

    # Initialize the model using the provided API key
    model = genai.GenerativeModel('gemini-2.5-flash')

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if file and file.filename.lower().endswith('.pdf'):
        pdf_bytes = file.read()

        # Prompt Engineering for JSON specific output with PDF input
        prompt = """
        You are an expert radiologist and medical data extractor. 
        Analyze the attached CT scan report (PDF document) and extract the key clinical features, findings, and impressions.
        

        STRICT RULES:
        - Return ONLY valid JSON. No explanation, no extra text.
        - If a field is not explicitly mentioned, return null.
        - Do NOT infer or guess.
        - Use exact text evidence where possible.
        - Normalize values when appropriate (e.g., sizes like "3.2 cm").
        - If multiple values exist, return the most clinically relevant one.

        OUTPUT SCHEMA:
        {
        "patient_gender": "",
        "patient_age": "",
        "tumor_size": "",
        "tumor_location_pole": "",
        "infiltration_present": "",
        "thrombus_present": "",
        "necrosis_present": ""
        }

        FIELD DEFINITIONS:

        - patient_gender:
        Extract gender (Male/Female/Other)

        - patient_age:
        Extract age in years (e.g., "45")

        - tumor_size:
        Largest reported tumor size (include unit, e.g., " 6.3 × 7.1 × 7.3 cm (AP × TR × CC) ")

        - tumor_location_pole:
        One of:
            - upper pole
            - mid pole
            - lower pole
            - interpolar region
        (Extract exact phrase if available)

        - infiltration_present:
        TRUE if report mentions:
            - invasion
            - infiltration
            - extension into surrounding tissue
        Otherwise FALSE if explicitly absent, else null

        - thrombus_present:
        TRUE if:
            - renal vein thrombus
            - IVC thrombus
        FALSE if explicitly absent, else null

        - necrosis_present:
        TRUE if necrosis is mentioned
        FALSE if explicitly absent, else null

        IMPORTANT:
        - Use TRUE / FALSE (uppercase) for boolean fields
        - Use null if information is missing or unclear

        CT REPORT:
        """
        
        try:
            # Send both the prompt and the PDF data
            response = model.generate_content(
                [
                    prompt,
                    {"mime_type": "application/pdf", "data": pdf_bytes}
                ],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1 # Low temperature for more deterministic, factual extraction
                )
            )
            
            # Parse the JSON to ensure it's valid before sending to frontend
            extracted_data = json.loads(response.text)
            return jsonify(extracted_data)
            
        except json.JSONDecodeError as text_err:
            print("Failed to parse JSON:", response.text)
            return jsonify({"error": "Failed to generate valid JSON from the report.", "raw_output": response.text}), 500
        except Exception as e:
            print(f"Error during API call: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Please upload a PDF file."}), 400

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    print("Server running on http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
    
