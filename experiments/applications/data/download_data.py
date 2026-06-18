"""
Download and save benchmark results used by the leaderboard rank-interval experiments.

Examples
--------
From the repository root::

    python experiments/applications/data/download_data.py
    python experiments/applications/data/download_data.py tabarena
    python experiments/applications/data/download_data.py mmlu

Outputs (written to ``--output-dir``, default: this script's directory):

- tabarena -> ``tabarena_results.csv``
- mmlu     -> ``mmlu_by_subject.pkl``

TabArena
--------
Clones a pinned ``tabrepo`` revision (https://github.com/autogluon/tabrepo) and
installs AutoGluon inside a disposable virtual environment, so your analysis
environment is not modified. The first run can take several minutes (clone, pip
installs, and result download).

Environment isolation
---------------------
TabArena requires cloning ``tabrepo`` and installing AutoGluon and its
dependencies, which requires downgrading core packages
(numpy/pandas/scipy/scikit-learn/pyarrow). To protect the current environment,
the ``tabarena`` download is performed inside a disposable virtual environment
created just for the build; the actual download runs as a subprocess using that
venv's interpreter. Pass ``--no-cleanup`` to keep the build directory for
debugging.

MMLU
----
Downloads per-subject correctness data from Hugging Face:
``PromptEval/PromptEval_MMLU_correctness``
(https://huggingface.co/datasets/PromptEval/PromptEval_MMLU_correctness).
The download runs in the current environment (needs ``datasets``/``pandas``/
``tqdm`` only) and takes roughly 10 minutes for all 57 subjects. Setting
``HF_TOKEN`` can improve rate limits.

Requirements
------------
- ``git`` on PATH.
- A working C++ compiler (``tabrepo`` compiles a small C++ AUC extension on
  import).
"""

import argparse
import os
import subprocess
import sys
import tempfile
import shutil


BENCHMARK_CHOICES = ['tabarena', 'mmlu']


# ---------------------------------------------------------------------------
# TabArena: isolated build environment
# ---------------------------------------------------------------------------
def _venv_python(venv_dir):
    """Path to the Python interpreter inside a created venv."""
    if os.name == 'nt':
        return os.path.join(venv_dir, 'Scripts', 'python.exe')
    return os.path.join(venv_dir, 'bin', 'python')


def _compiler_env():
    """Return environment variables for the TabArena worker subprocess."""
    env = os.environ.copy()
    if sys.platform != 'darwin': # macOS only
        return env
    try:
        sdk = subprocess.check_output(
            ['xcrun', '--show-sdk-path'], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return env
    sdk_cxx = os.path.join(sdk, 'usr', 'include', 'c++', 'v1')
    if os.path.exists(os.path.join(sdk_cxx, 'cstdint')):
        existing = env.get('CPLUS_INCLUDE_PATH', '')
        env['CPLUS_INCLUDE_PATH'] = (
            sdk_cxx + (os.pathsep + existing if existing else '')
        )
    return env


def _preprocess_tabarena_model_names(model):
    """Shorten TabArena model labels for plots and tables."""
    return (
        model
        .str.replace('REALMLP_GPU', 'REALMLP', regex=False)
        .str.replace('(tuned + ensemble)', '(T+E)', regex=False)
        .str.replace('(tuned)', '(T)', regex=False)
        .str.replace('(default)', '(D)', regex=False)
    )


def _download_tabarena_results(output_dir):
    """Import tabrepo and save TabArena CSV. Must run under the disposable venv interpreter."""
    from tabrepo.nips2025_utils.tabarena_context import TabArenaContext

    tabarena_context = TabArenaContext()
    tabarena_results = tabarena_context.load_results_paper(download_results='auto')

    # The column ``metric_error`` is the test error (per tabarena docs).
    # Baseline methods (no ``config_type``) are removed.
    tabarena_results = tabarena_results.dropna(subset='config_type', ignore_index=True)
    tabarena_results = tabarena_results.rename(
        columns={'method': 'model', 'metric_error': 'performance_score'}
    )
    keep_cols = [
        'dataset', 'model', 'fold', 'problem_type', 'config_type', 'performance_score',
    ]
    tabarena_results = tabarena_results[keep_cols]
    tabarena_results['model'] = _preprocess_tabarena_model_names(tabarena_results['model'])

    output_path = os.path.join(output_dir, 'tabarena_results.csv')
    tabarena_results.to_csv(output_path, index=False)
    print(f'TabArena results saved to {output_path}')


def download_tabarena(output_dir, cleanup=True):
    """Download TabArena paper results using a disposable virtual environment.

    A fresh virtual environment is created, ``tabrepo`` (pinned revision) and its
    runtime dependencies are installed there, and the download runs as a
    subprocess using that venv's interpreter. Imports must not happen in the
    caller's process, or they would install/load tabrepo into the analysis env.

    Parameters
    ----------
    output_dir : str
        Directory to write ``tabarena_results.csv``.
    cleanup : bool, default True
        When True, remove the disposable build environment afterwards.
    """
    build_dir = tempfile.mkdtemp(prefix='tabarena_build_')
    venv_dir = os.path.join(build_dir, 'venv')
    tabrepo_path = os.path.join(build_dir, 'tabrepo')
    # Pinned tabrepo revision that supports load_results_paper(download_results="auto").
    # This is the revision used in the paper.
    tabrepo_repo_url = 'https://github.com/autogluon/tabrepo.git'
    tabrepo_revision = '2ef8a15'
    try:
        print(f'Creating isolated build environment in {build_dir} ...')
        subprocess.check_call([sys.executable, '-m', 'venv', venv_dir])
        vpy = _venv_python(venv_dir)
        subprocess.check_call([vpy, '-m', 'pip', 'install', '--upgrade', 'pip'])

        print('Cloning tabrepo ...')
        subprocess.check_call(['git', 'clone', tabrepo_repo_url, tabrepo_path])
        subprocess.check_call(['git', '-C', tabrepo_path, 'checkout', tabrepo_revision])

        # Install core tabrepo only (the benchmark extra pulls newer torch/tabdpt
        # constraints). This already pulls `autogluon.core`, which provides the
        # `autogluon` namespace needed for the tabarena imports.
        print('Installing tabrepo and runtime dependencies (this can take a while) ...')
        subprocess.check_call([vpy, '-m', 'pip', 'install', '-e', tabrepo_path])

        # Only install the heavy `autogluon.tabular[all]` extra if `autogluon`
        # is still not importable (it pulls torch, which has no wheel on some
        # platforms, e.g. x86-64 macOS).
        if subprocess.call(
            [vpy, '-c', 'import autogluon'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) != 0:
            print('autogluon not available; installing autogluon.tabular[all] ...')
            subprocess.check_call([vpy, '-m', 'pip', 'install', 'autogluon.tabular[all]'])
        else:
            print('autogluon available via tabrepo; skipping autogluon.tabular[all].')

        # tabrepo's import chain pulls in its paper/plotting utilities, which need
        # `seaborn` (and `tueplots`) on top of the matplotlib/pandas/numpy that
        # tabrepo already installs.
        subprocess.check_call([vpy, '-m', 'pip', 'install', 'tueplots', 'seaborn'])

        print('Downloading TabArena results in isolated environment ...')
        worker_env = _compiler_env()
        worker_env['DOWNLOAD_DATA_TABARENA_OUTPUT'] = os.path.abspath(output_dir)
        subprocess.check_call([vpy, os.path.abspath(__file__)], env=worker_env)
    finally:
        if cleanup:
            print('Removing isolated build environment ...')
            shutil.rmtree(build_dir, ignore_errors=True)
        else:
            print(f'Build environment kept at {build_dir}')


# ---------------------------------------------------------------------------
# MMLU: runs in the current environment
# ---------------------------------------------------------------------------
def download_mmlu(output_dir):
    """Download MMLU per-subject correctness data and pickle it by subject.

    Data are fetched from the Hugging Face Hub with ``datasets`` and saved to
    ``mmlu_by_subject.pkl`` under ``output_dir``.

    Parameters
    ----------
    output_dir : str
        Directory to write ``mmlu_by_subject.pkl``.
    """
    import pickle

    import pandas as pd
    from datasets import load_dataset
    from tqdm import tqdm

    subjects = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
        'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
        'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics',
        'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts',
        'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
        'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
        'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
        'high_school_physics', 'high_school_psychology', 'high_school_statistics',
        'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality',
        'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management',
        'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
        'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
        'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
        'sociology', 'us_foreign_policy', 'virology', 'world_religions',
    ]
    model_name_map = {
        'meta_llama_llama_3_8b': 'Llama-3-8B',
        'meta_llama_llama_3_8b_instruct': 'Llama-3-8B-Instruct',
        'meta_llama_llama_3_70b_instruct': 'Llama-3-70B-Instruct',
        'codellama_codellama_34b_instruct': 'CodeLlama-34B-Instruct',
        'google_flan_t5_xl': 'Flan-T5-XL',
        'google_flan_t5_xxl': 'Flan-T5-XXL',
        'google_flan_ul2': 'Flan-UL2',
        'ibm_mistralai_merlinite_7b': 'Merlinite-7B',
        'mistralai_mixtral_8x7b_instruct_v01': 'Mixtral-8x7B-Instruct',
        'mistralai_mistral_7b_instruct_v0_2': 'Mistral-7B-Instruct-v0.2',
        'google_gemma_7b': 'Gemma-7B',
        'google_gemma_7b_it': 'Gemma-7B-IT',
        'tiiuae_falcon_40b': 'Falcon-40B',
        'mistralai_mistral_7b_v0_1': 'Mistral-7B-v0.1',
        'tiiuae_falcon_180b': 'Falcon-180B',
    }
    subject_dfs = {}
    for subject in tqdm(subjects, desc='Subjects'):
        ds = load_dataset('PromptEval/PromptEval_MMLU_correctness', subject)
        subject_dfs[subject] = []
        for model_key in ds.keys():
            df = ds[model_key].to_pandas()
            df.insert(0, 'model', model_name_map[model_key])
            subject_dfs[subject].append(df)

    mmlu_by_subject = {}
    for subject, dfs in subject_dfs.items():
        mmlu_by_subject[subject] = pd.concat(dfs, ignore_index=True).copy()

    output_path = os.path.join(output_dir, 'mmlu_by_subject.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(mmlu_by_subject, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'MMLU results saved to {output_path}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Download and save benchmark results (tabarena and/or mmlu).',
    )
    parser.add_argument(
        'benchmarks',
        nargs='*',
        default=None,
        metavar='{tabarena,mmlu}',
        help='One or more benchmarks to download (default: both tabarena and mmlu).',
    )
    parser.add_argument(
        '--output-dir',
        default=os.path.dirname(os.path.abspath(__file__)),
        help='Directory to write outputs to (default: this script\'s directory).',
    )
    parser.add_argument(
        '--cleanup',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='For tabarena: remove the disposable build environment afterwards '
             '(default: enabled; pass --no-cleanup to keep it for debugging).',
    )
    args = parser.parse_args(argv)

    # Default to all benchmarks when none are given; de-duplicate while preserving order.
    args.benchmarks = list(dict.fromkeys(args.benchmarks or BENCHMARK_CHOICES))
    invalid = [b for b in args.benchmarks if b not in BENCHMARK_CHOICES]
    if invalid:
        parser.error(
            'invalid choice(s): %s (choose from %s)'
            % (', '.join(invalid), ', '.join(BENCHMARK_CHOICES))
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    for benchmark in args.benchmarks:
        if benchmark == 'tabarena':
            download_tabarena(args.output_dir, cleanup=args.cleanup)
        elif benchmark == 'mmlu':
            download_mmlu(args.output_dir)

    print('Done.')


if __name__ == '__main__':
    tabarena_output = os.environ.pop('DOWNLOAD_DATA_TABARENA_OUTPUT', None)
    if tabarena_output is not None:
        _download_tabarena_results(tabarena_output)
    else:
        main()
