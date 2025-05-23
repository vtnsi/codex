import os
import json
import datetime
import glob
import directory_tree
import shutil

from ..modules import combinatorial, output

def codex_env_checks(kwargs):
    try:  
        # requd
        new_dirname = kwargs['name']
    except:
        raise KeyError("In creating a new CODEX directory; requires <name of CODEX directory>.")
    try:
        # not required
        new_parent_dirname = kwargs['parent_dir']
    except KeyError:
        new_parent_dirname = os.path.dirname(os.path.realpath('.'))
        print(f"Field <parent_dir> was unspecified. Creating new CODEX directory parent directory, {new_parent_dirname}.")

    new_parent_dirname = os.path.realpath(new_parent_dirname)
        
    try:
        assertpath = os.path.realpath(os.path.join(os.getcwd(), new_parent_dirname))
        assert os.path.exists(assertpath)
    except AssertionError:
        raise FileNotFoundError(f"Creating template CODEX directory failed. Folder {assertpath} does not exist.")

    try:
        include_templates = (str.lower(kwargs['include_templates']) == 'true')
    except KeyError:
        include_templates = False
    try:
        include_tutorial = (str.lower(kwargs['include_examples']) == 'true')
    except KeyError:
        include_tutorial = False

    return new_dirname, new_parent_dirname, include_templates, include_tutorial


def setup_new_codex_env(dir_name=None, parent_dir="", templates=False, tutorial=False):
    if templates:
        print("CODEX env setup: Adding templates")
    if tutorial:
        print("CODEX env setup: Adding tutorial materials.")


    existing_codex_dirs = glob.glob(os.path.realpath(os.path.join(parent_dir, dir_name+'*')))

    # exec_dir = os.path.dirname(os.path.realpath(__file__)) exec_dir = "../"
    exec_dir = os.path.dirname(os.path.realpath('.'))
    try:
        assert os.path.exists(os.path.join(exec_dir, "resources", "templates"))
    except:
        #exec_dir = './'
        exec_dir = os.path.realpath('.')
    

    if dir_name is None or dir_name == "":
        dir_name = "new_codex_dir"

    if len(existing_codex_dirs)  == 0:
        codex_dir_new = os.path.realpath(os.path.join(parent_dir, f'{dir_name}'))
    else:
        codex_dir_new = os.path.realpath(os.path.join(parent_dir, f'{dir_name}_{len(existing_codex_dirs)}'))
    os.makedirs(codex_dir_new, exist_ok=False)

    for component in ["binning", "configs", "splits", "performance", "datasets", "runs", "universe"]:
        subdir = os.path.join(codex_dir_new, component)
        os.makedirs(subdir, exist_ok=True)

    if templates:
        for filename in os.listdir(os.path.join(exec_dir, "resources", "templates")):
            for subdir in os.listdir(codex_dir_new):
                if str(subdir)[:-1] in str(filename):
                    shutil.copy(os.path.join(exec_dir, "resources", "templates", filename), os.path.join(codex_dir_new, subdir, filename))
    if tutorial:
        for filename in os.listdir(os.path.join(exec_dir, "resources", "tutorial")):
            for subdir in os.listdir(codex_dir_new):
                if str(subdir)[:-1] in str(filename):
                    shutil.copy(os.path.join(exec_dir, "resources", "tutorial", filename), os.path.join(codex_dir_new, subdir, filename))

    directory_tree.DisplayTree(codex_dir_new)
    print(f"Successfully constructed CODEX directory at location {os.path.realpath(codex_dir_new)}.")
    return codex_dir_new

#PIPTEST
def handle_input_file(input):
    if type(input) is str:
        with open(input) as f:
            codex_input = json.load(f)
    elif type(input) is dict:
        codex_input = input
    else:
        raise TypeError("CODEX input can only be input config dictionary or path to input config dictionary (str).")
    return codex_input

def define_experiment_variables(codex_input):
    '''
        Gets universally required CODEX variables and sets modes and switches provided 
        from the input file.
    '''

    #timed = codex_input['timestamp']
    # Required for every codex mode
    # CODEX DIR RELATIVE TO CODEX REPO
    codex_dir = codex_input['codex_directory']
    
    try:
        output_dir = codex_input['output_directory']
    except:
        print("output_dir not found. Defaulting output to codex directory, {}".format(codex_dir))
        output_dir = codex_dir

    if output_dir is None:
        #output_dir = './'
        output_dir = os.path.realpath('.')
    
    config_id = codex_input['config_id']

    output_dir = os.path.abspath(output.make_output_dir_nonexist(output_dir))
    output_dir_config = os.path.realpath(os.path.join(output_dir, config_id))
    
    output_dir_config = output.make_output_dir_nonexist(output_dir_config)

    strengths = codex_input['t']

    combinatorial.labelCentric = (
        codex_input['counting_mode'] == 'label_centric')
    global use_augmented_universe
    use_augmented_universe = codex_input['use_augmented_universe']

    return output_dir_config, strengths


def define_training_variables(codex_input):
    training_dir = os.path.abspath(os.path.join(
        os.getcwd(), codex_input['training_run_directory']))
    training_data_dir = os.path.abspath(os.path.join(
        os.getcwd(), codex_input['training_data_directory']))
    dataset_dir = os.path.abspath(os.path.join(
        os.getcwd(), codex_input['dataset_spec_directory']))

    return training_dir, training_data_dir, dataset_dir


def extract_names(codex_input):
    '''
        Gets dataset and model name.
    '''
    return codex_input['dataset_name'], codex_input['model_name']


def extract_sp_partitioning(codex_input, probe_name='_included', exploit_name='_excluded'):
    '''
        Extracts one split and performance file from input specifications, with
        the added function of partitioning each data structure based on belonging in the
        probe or exploit set.

        Returns:
        split: dict
            Read from a JSON file whose keys are the split partition and values are lists of 
            the sample ID's as presented in the dataset. Or, if given as a list of split files, 
            nested under the split file name it came from.

        performance: dict
            Read from a JSON file containing a model's performance on a test set. Or, if given as a
            list of split files, is nested under the split file name it came from.

        metric: str
            Chosen metric to evaluate performance for the experiment.
    '''
    probe_name = codex_input['probe_test_tag']
    exploit_name = codex_input['exploit_test_tag']

    assert type(codex_input['split_file']) is str and type(
        codex_input['split_file']) is str
    split_filename = codex_input['split_file']
    performance_filename = codex_input['performance_file']

    split, performance, metric = extract_sp(
        codex_input, split_filename, performance_filename)
    if 'test{}'.format(probe_name) in split and 'test{}'.format(exploit_name) in split:
        split_p = {'split_id': split['split_id'],
                   'partition': probe_name,
                   'train': split['train'],
                   'validation': split['validation'] if 'validation' in split else [],
                   'test{}'.format(probe_name): split['test{}'.format(probe_name)]}
        split_e = {'split_id': split['split_id'],
                   'partition': exploit_name,
                   'train': split['train'],
                   'validation': split['validation'] if 'validation' in split else [],
                   'test{}'.format(exploit_name): split['test{}'.format(exploit_name)]}
    else:
        raise KeyError("No partition found from specified subset name!")

    if 'test{}'.format(probe_name) in split and 'test{}'.format(exploit_name) in performance:
        perf_p = {'split_id': performance['split_id'],
                  'partition': probe_name,
                  'test{}'.format(probe_name): performance['test{}'.format(probe_name)]}
        perf_e = {'split_id': performance['split_id'],
                  'partition': exploit_name,
                  'test{}'.format(exploit_name): performance['test{}'.format(exploit_name)]}
    else:
        raise KeyError("No partition found from specified subset name!")

    return split_p, split_e, perf_p, perf_e, metric


def extract_sp(codex_input, split_filename=None, performance_filename=None):
    '''
        Extracts one or more split and performance files from input specifications.

        Returns:
        split: dict
            Read from a JSON file whose keys are the split partition and values are lists of 
            the sample ID's as presented in the dataset. Or, if given as a list of split files, 
            nested under the split file name it came from.

        performance: dict
            Read from a JSON file containing a model's performance on a test set. Or, if given as a
            list of split files, is nested under the split file name it came from.

        metric: str
            Chosen metric to evaluate performance for the experiment.
    '''
    codex_dir = codex_input['codex_directory']
    split_folder = codex_input['split_folder']
    performance_folder = codex_input['performance_folder']
    metric = codex_input['metric']

    # Initial pass
    if split_filename is None and performance_filename is None:
        split_filename = codex_input['split_file']
        performance_filename = codex_input['performance_file']

    # Might be list if multiple, str if single
    if type(split_filename) is list:
        if performance_filename is None:
            performance_filename = [None]*len(split_filename)

        if type(performance_filename) is not list:
            raise ValueError(
                "No corresponding performance files to given split files.")
        else:
            assert len(split_filename) == len(performance_filename)

        num_splits = len(split_filename)

        split = {filename: None for filename in split_filename}
        performance = {filename: None for filename in split_filename}

        for i in range(num_splits):
            split[split_filename[i]], performance[split_filename[i]], metric = extract_sp(
                codex_input, split_filename[i], performance_filename[i])

    elif type(split_filename) is str:
        # Add/return split
        # os.path.join(codex_dir, split_folder, split_filename)) as s:
        with open(os.path.abspath(os.path.join(codex_dir, split_folder, split_filename))) as s:
            split = json.load(s)
            split['split_id'] = split_filename

        if performance_filename is None:
            performance = None
        else:
            with open(os.path.abspath(os.path.join(codex_dir, performance_folder, performance_filename))) as p:
                performance = json.load(p)
                performance['split_id'] = split_filename
    else:
        raise ValueError("Unknown object for split file.")

    return split, performance, metric


def extract_dataset():
    return
