from flask import Flask, render_template, request, redirect, url_for, session
import os
import sys
import uuid

# --------------------------------------------------------------------------
# Dynamically add smlp/src directory to sys.path so we can import smlp_py
# --------------------------------------------------------------------------
current_file_path = os.path.abspath(__file__)  # Path to this file (app.py)
src_dir = os.path.abspath(os.path.join(os.path.dirname(current_file_path), '..'))  # Path to ../ (which should be /smlp/src if your structure is /smlp/src/Ui)
if src_dir not in sys.path:  
    sys.path.insert(0, src_dir)

from smlp_py.smlp_flows import SmlpFlows  # Import the real SMLP logic

# --------------------------------------------------------------------------
# Flask App Setup
# --------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "dev_secret_key"  # For production, set via env variable

# Example placeholders for file upload folders
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --------------------------------------------------------------------------
# HELPER: call_smlp_api
# --------------------------------------------------------------------------
def call_smlp_api(mode, argument_list):
    """
    mode: 'train', 'predict', 'verify', 'optimize', etc.
    argument_list: e.g. ["-data", "my_dataset", "-resp", "y1,y2", ...]
    """
    cmd_parts = ["-mode", mode] + argument_list
    print("DEBUG: SMLP arguments:", cmd_parts)

    try:
        flow = SmlpFlows(cmd_parts)
        output = flow.smlp_flow()
        return output
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
print('BABABABABABABABABABABABAB' + smlp_py.__file__)

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
    """
    Let the user pick a mode from [certify, query, verify, synthesize, optimize, optsyn]
    and fill out relevant arguments (data_file, spec_file, etc.). Then call SMLP.
    """
    modes_list = ['certify', 'query', 'verify', 'synthesize', 'optimize', 'optsyn']
    if request.method == 'POST':
        chosen_mode = request.form.get('explore_mode', '')
        data_file = request.files.get('data_file')
        spec_file = request.files.get('spec_file')

        arguments = []

        # Upload data file
        if data_file and data_file.filename:
            filename = f"{uuid.uuid4()}_{data_file.filename}"
            dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            data_file.save(dataset_path)
            base_name, _ = os.path.splitext(dataset_path)
            arguments += ["-data", base_name]

        # Upload spec file
        if spec_file and spec_file.filename:
            specname = f"{uuid.uuid4()}_{spec_file.filename}"
            spec_path = os.path.join(app.config['UPLOAD_FOLDER'], specname)
            spec_file.save(spec_path)
            base_spec, _ = os.path.splitext(spec_path)
            arguments += ["-spec", base_spec]

        # out_dir, pref
        out_dir_val = request.form.get('out_dir_val', '')
        if out_dir_val:
            arguments += ["-out_dir", out_dir_val]

        pref_val = request.form.get('pref_val', '')
        if pref_val:
            arguments += ["-pref", pref_val]

        # pareto
        pareto_val = request.form.get('pareto', 'f')
        if pareto_val == 't':
            arguments += ["-pareto", "t"]

        # resp, feat
        resp_expr = request.form.get('resp_expr', '')
        if resp_expr:
            arguments += ["-resp", resp_expr]

        feat_expr = request.form.get('feat_expr', '')
        if feat_expr:
            arguments += ["-feat", feat_expr]

        # model, dt_sklearn_max_depth
        model_expr = request.form.get('model_expr', '')
        if model_expr:
            arguments += ["-model", model_expr]

        dt_max = request.form.get('dt_sklearn_max_depth', '')
        if dt_max:
            arguments += ["-dt_sklearn_max_depth", dt_max]

        # data_scaler
        data_scaler = request.form.get('data_scaler', '')
        if data_scaler:
            arguments += ["-data_scaler", data_scaler]

        # beta, objv_names, objv_exprs, epsilon, delta_rel
        beta_expr = request.form.get('beta_expr', '')
        if beta_expr:
            arguments += ["-beta", beta_expr]

        objv_names = request.form.get('objv_names', '')
        if objv_names:
            arguments += ["-objv_names", objv_names]

        objv_exprs = request.form.get('objv_exprs', '')
        if objv_exprs:
            arguments += ["-objv_exprs", objv_exprs]

        epsilon = request.form.get('epsilon', '')
        if epsilon:
            arguments += ["-epsilon", epsilon]

        delta_rel = request.form.get('delta_rel', '')
        if delta_rel:
            arguments += ["-delta_rel", delta_rel]

        # save_model_config, mrmr_pred, plots, seed, log_time
        smc = request.form.get('save_model_config', 'f')
        arguments += ["-save_model_config", smc]

        mrmr_pred = request.form.get('mrmr_pred', '')
        if mrmr_pred:
            arguments += ["-mrmr_pred", mrmr_pred]

        plots_opt = request.form.get('plots_opt', 'f')
        arguments += ["-plots", plots_opt]

        seed_val = request.form.get('seed_val', '')
        if seed_val:
            arguments += ["-seed", seed_val]

        log_time = request.form.get('log_time', 'f')
        arguments += ["-log_time", log_time]

        # Finally call SMLP with the chosen mode
        output = call_smlp_api(chosen_mode, arguments)
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
    return render_template('results.html', output=output)

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
