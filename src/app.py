# Description: This is the main Flask app that serves as the UI for the SMLP.
# It provides a web interface for the user to interact with the SMLP API.
# The user can train models, predict, explore, perform DOE, etc.
import subprocess
from flask import Flask, render_template, request, redirect, url_for, session
import os
import sys
import uuid
from flask_session import Session
# --------------------------------------------------------------------------
# Dynamically add smlp/src directory to sys.path so we can import smlp_py
# # # --------------------------------------------------------------------------
# current_file_path = os.path.abspath(__file__)  # Path to this file (app.py)
# src_dir = os.path.abspath(os.path.join(os.path.dirname(current_file_path), '..'))  # Path to ../ (which should be /smlp/src if your structure is /smlp/src/Ui)
# if src_dir not in sys.path:  
#     sys.path.insert(0, src_dir)

# Now import it
from smlp_py.smlp_flows import SmlpFlows  # Import the real SMLP logic


# from .smlp_py.smlp_flows import SmlpFlows  # Import the real SMLP logic

# --------------------------------------------------------------------------
# Flask App Setup
# --------------------------------------------------------------------------
app = Flask(__name__)
# app.secret_key = "dev_secret_key"  # For production, set via env variable

app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key')  # Use env var for security
app.config['SESSION_TYPE'] = 'filesystem'  # Use filesystem-based session storage
Session(app)


# Example placeholders for file upload folders
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# --------------------------------------------------------------------------
# HELPER: call_smlp_api
# --------------------------------------------------------------------------
def call_smlp_api( argument_list):
    """
    mode: 'train', 'predict', 'verify', 'optimize', etc.
    argument_list: e.g. ["-data", "my_dataset", "-resp", "y1,y2", ...]
    """

    cmd_string = " ".join(argument_list)  # Convert list to string

    # print("DEBUG: Final SMLP Command String ->", cmd_string)
    # print("DEBUG: Final SMLP Command  list ->", argument_list)

    print("\n--- DEBUG INFO ---")
    print("Current Working Directory:", os.getcwd())  # Where Flask is running from
    print("Process ID (PID):", os.getpid())  # Unique identifier for process
    print("Final SMLP Command String ->", cmd_string)
    print("Final SMLP Command List ->", argument_list)
    print("-------------------\n")

    cwd_dir = os.path.abspath("../regr_smlp/code/")
    try:
        # flow = SmlpFlows(argument_list)
        # output = flow.smlp_flow()
        # return output
        result = subprocess.run(
            argument_list,
            capture_output=True,
            text=True,
            cwd=cwd_dir
        )
        full_output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        print("DEBUG: Command output ->", full_output)
        return result.stdout.strip() if result.returncode == 0 else full_output

    except Exception as e:
        return f"Error calling SMLP: {str(e)}"

# --------------------------------------------------------------------------
# 1) HOME / LANDING PAGE
# --------------------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# --------------------------------------------------------------------------
# 2) TRAIN
# --------------------------------------------------------------------------
@app.route('/train', methods=['GET', 'POST'])
def train():
    """
    Provide a form to pick data, model, hyperparams.
    On POST, calls call_smlp_api(mode='train').
    """
    if request.method == 'POST':
        # Gather form inputs
        data_file = request.files.get('data_file')
        model = request.form.get('model')
        resp = request.form.get('resp')
        feat = request.form.get('feat')
        save_model = request.form.get('save_model', 'f')
        model_name = request.form.get('model_name', 'my_model')
        # ... add more as needed (scale_feat, scale_resp, etc.)

        # Save the uploaded data
        dataset_path = None
        if data_file and data_file.filename:
            filename = f"{uuid.uuid4()}_{data_file.filename}"
            dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            data_file.save(dataset_path)

        # Build arguments for SMLP
        arguments = []
        if dataset_path:
            # Possibly remove ".csv" if SMLP requires that
            base_name, ext = os.path.splitext(dataset_path)
            arguments += ["-data", base_name]

        arguments += ["-model", model]
        arguments += ["-resp", resp]
        arguments += ["-feat", feat]

        if save_model == 't':
            arguments += ["-save_model", "t", "-model_name", model_name]

        # Call SMLP
        output = call_smlp_api("train", arguments)
        session['output'] = output
        return redirect(url_for('results'))

    return render_template('train.html')

import smlp_py

# --------------------------------------------------------------------------
# 3) PREDICT
# --------------------------------------------------------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        model_file = request.files.get('model_file')
        new_data_file = request.files.get('new_data_file')
        save_model = request.form.get('save_model', 'f')
        model_name = request.form.get('model_name', 'my_model')

        # Save files
        model_path = None
        if model_file and model_file.filename:
            filename = f"{uuid.uuid4()}_{model_file.filename}"
            model_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            model_file.save(model_path)

        newdata_path = None
        if new_data_file and new_data_file.filename:
            filename = f"{uuid.uuid4()}_{new_data_file.filename}"
            newdata_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            new_data_file.save(newdata_path)

        # Build arguments
        arguments = []
        if model_path:
            base_name, ext = os.path.splitext(model_path)
            arguments += ["-use_model", "t", "-model_name", base_name]

        if newdata_path:
            base_new_name, _ = os.path.splitext(newdata_path)
            arguments += ["-new_data", base_new_name]

        if save_model == 't':
            arguments += ["-save_model", "t", "-model_name", model_name]

        # Call SMLP
        output = call_smlp_api("predict", arguments)
        session['output'] = output
        return redirect(url_for('results'))

    return render_template('predict.html')

# --------------------------------------------------------------------------
# 4) EXPLORATION (certify, query, verify, synthesize, optimize, optsyn)
# --------------------------------------------------------------------------
@app.route('/explore', methods=['GET', 'POST'])
def explore():
    modes_list = ['certify', 'query', 'verify', 'synthesize', 'optimize', 'optsyn']
    
    if request.method == 'POST':
        chosen_mode = request.form.get('explore_mode', '')

        if chosen_mode not in modes_list:
            return "Error: Invalid mode selected", 400

        # Get files from the form
        dataset_path = "../regr_smlp/data/smlp_toy_basic"
        spec_path = "../regr_smlp/specs/smlp_toy_basic.spec"


        # # Save data file (only if provided)
        # if data_file and data_file.filename:
        #     dataset_path = os.path.join(DATA_DIR, data_file.filename)
        #     data_file.save(dataset_path)

        # # Save spec file (only if provided)
        # if spec_file and spec_file.filename:
        #     spec_path = os.path.join(SPEC_DIR, spec_file.filename)
        #     spec_file.save(spec_path)

        # Check if at least one required file is uploaded
        if not dataset_path and not spec_path:
            return "Error: No valid dataset or spec file found", 400

        # Build command arguments list dynamically
        arguments = [
            'python3', os.path.abspath("../src/run_smlp.py"),
            "-data", os.path.abspath(dataset_path),
            "-out_dir", os.path.abspath(request.form.get('out_dir_val', './')),
            "-pref", request.form.get('pref_val', 'Test113'),
            "-mode", chosen_mode, 
            "-pareto", request.form.get('pareto', 't'),
            "-resp", request.form.get('resp_expr', 'y1,y2'),
            "-feat", request.form.get('feat_expr', 'x1,x2,p1,p2'),
            "-model", request.form.get('model_expr', 'dt_sklearn'),
            "-dt_sklearn_max_depth", request.form.get('dt_sklearn_max_depth', '15'),
            "-mrmr_pred", request.form.get('mrmr_pred', '0'),
            "-epsilon", request.form.get('epsilon', '0.05'),
            "-delta_rel", request.form.get('delta_rel', '0.01'),
            "-save_model", request.form.get('save_model', 't'),
            "-model_name", request.form.get('model_name', 'test113_model'),
            "-save_model_config", request.form.get('save_model_config', 't'),
            "-plots", request.form.get('plots_opt', 'f'),
            "-seed", request.form.get('seed_val', '10'),
            "-log_time", request.form.get('log_time', 'f'),
            "-spec", os.path.abspath(spec_path)
        ]

        # Debugging: Print the command before execution
        # print("DEBUG: Final SMLP Command ->", arguments)

        # Call SMLP API
        output = call_smlp_api(arguments)
        # print("DEBUG: EXPLORE SMLP Output ->", output)
        session['output'] = output
        return redirect(url_for('results'))

    return render_template('exploration.html', modes=modes_list)

# --------------------------------------------------------------------------
# 5) DOE
# --------------------------------------------------------------------------
@app.route('/doe', methods=['GET', 'POST'])
def doe():
    """
    Let the user pick DOE options and pass them to SMLP in 'doe' mode.
    """
    if request.method == 'POST':
        doe_spec_file = request.files.get('doe_spec_file')
        doe_algo = request.form.get('doe_algo')
        doe_num_samples = request.form.get('doe_num_samples', '')

        arguments = []
        if doe_spec_file and doe_spec_file.filename:
            specname = f"{uuid.uuid4()}_{doe_spec_file.filename}"
            spec_path = os.path.join(app.config['UPLOAD_FOLDER'], specname)
            doe_spec_file.save(spec_path)
            base_spec, _ = os.path.splitext(spec_path)
            arguments += ["-doe_spec", base_spec]

        if doe_algo:
            arguments += ["-doe_algo", doe_algo]
        if doe_num_samples:
            arguments += ["-doe_num_samples", doe_num_samples]

        output = call_smlp_api("doe", arguments)
        session['output'] = output
        return redirect(url_for('results'))

    return render_template('doe.html')

# --------------------------------------------------------------------------
# 6) RESULTS
# --------------------------------------------------------------------------
@app.route('/results')
def results():
    output = session.get('output', 'No output available yet.')
    print("\n--- RESULTS ---")
    print(output)
    return render_template('results.html', output=output)

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
