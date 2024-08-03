import fnmatch
import glob
import json
import os



def extract_partial_json_structure(dir_path, json_type, all_json_keys=None):
    if not all_json_keys:
        all_json_keys = set()

    folder_list = [f.path for f in os.scandir(dir_path) if f.is_dir()]
    for folder_name in folder_list:
        json_files = search_json_file(folder_name, json_type)

        for j_file in json_files:
            with open(j_file, encoding="utf-8") as file:
                try:
                    json_data = json.load(file)
                    extract_json_keys(json_data, all_json_keys)
                except json.JSONDecodeError:
                    continue

        extract_partial_json_structure(folder_name, json_type, all_json_keys)
        

    return all_json_keys


def extract_json_keys(json_data, key_set, current_key=""):
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            if current_key:
                nested_key = f"{current_key}.{key}"
            else:
                nested_key = key
            key_set.add(nested_key)

            if isinstance(value, (dict, list)):
                extract_json_keys(value, key_set, nested_key)

    elif isinstance(json_data, list):
        for idx, item in enumerate(json_data):
            if isinstance(item, (dict, list)):
                extract_json_keys(item, key_set, current_key)



def merge_lists(kv_filter, complete_structure):
    result = []

    for item_a in kv_filter:
        if '*' in item_a:
            matched_items = fnmatch.filter(complete_structure, item_a)
            result.extend(matched_items)
        elif item_a in complete_structure:
            result.append(item_a)

    return result



def get_complete_struct(dir_path, json_type):
    var = extract_partial_json_structure(dir_path, json_type)
    filter = get_kv_filter(json_type)
    return merge_lists(filter, var)


def search_json_file(folder_name, j_type):
    file_pattern = {
        'report': "report_*.json",
        'summary': "summary_*.json",
        'out': "*newout.json",
    }

    j_file = os.path.join(folder_name, file_pattern.get(j_type, f"{j_type}_*.json"))
    j_files = glob.glob(j_file)
    return j_files



def append_json_content(json_file, kv_types, kv_filter):
    try:
        with open(json_file, encoding="utf-8") as file:
            data = json.load(file)
    except:
        return []

    if kv_types == 'key':
        return get_json_key(data, kv_filter)
    elif kv_types == 'value':
        return get_json_value(data, kv_filter)
    else:
        return get_json_kv(data, kv_filter)
    


def align_data_with_structure(data, structure, kv_type):
    missing_keys = set(structure).difference(data)
    if kv_type == 'key':
        data.extend(key.split('.')[-1] for key in missing_keys)
    elif kv_type == "value":
        data.extend(['null'] * len(missing_keys))
    else:
        aligned_data = [kv.split('.')[-1] for pair in zip(missing_keys, ['null'] * len(missing_keys)) for kv in pair]
        data.extend(aligned_data)
    return data



def traverse_and_read_json_with_label(dir_path, json_types, kv_types='both', result=0, complete_structure=False):
    folder_list = [f.path for f in os.scandir(dir_path) if f.is_dir()]
    json_data_list = []

    if complete_structure:
        json_structure_dict = {j_type: get_complete_struct(dir_path, j_type) for j_type in json_types}

    for folder_name in folder_list:
        json_data = []
        for j_type in json_types:
            j_files = search_json_file(folder_name, j_type)
            kv_filter = get_kv_filter(j_type)
            for j_file in j_files:
                data = append_json_content(j_file, kv_types, kv_filter)
                if complete_structure:
                    structure = json_structure_dict[j_type]
                    data = align_data_with_structure(data, structure, kv_types)
                json_data.extend(data)
        iter_data = traverse_and_read_json_with_label(folder_name, json_types, kv_types, result, complete_structure)
        if json_data:
            if len(iter_data) == 1:
                json_data_list.append([json_data + iter_data[0][0], result])
            else:
                json_data_list.append([json_data, result])
    return json_data_list



def traverse_and_read_json(dir_path, json_types, kv_types='both', json_data=None):
    folder_list = [f.path for f in os.scandir(dir_path) if f.is_dir()]
    if not json_data:
        json_data = []
    for folder_name in folder_list:
        for j_type in json_types:
            j_files = search_json_file(folder_name, j_type)
            kv_filter = get_kv_filter(j_type)
            for j_file in j_files:
                json_data.extend(append_json_content(j_file, kv_types, kv_filter))
        traverse_and_read_json(folder_name, json_types, kv_types, json_data)
    return json_data



def get_kv_filter(j_type):
    kv_filter_dict = {
        'report': [
                'description', 'num_releases', 'repo', 'repo.url', 'repo.last_activity', 'repo.num_stars', 'repo.num_forks', 'repo.commits', 'repo.branches', 'repo.contributors', 'version', 'version.permissions', 'version.permissions.*', 'composition', 'version.devDependencies', 'version.devDependencies.*'
            ],
        'out': [
                'Calls', 'Calls.Args'
            ],
        'summary': [
                'files', 'network', 'network.info'
            ]
    }
    return kv_filter_dict.get(j_type, [])



def get_json_key(data, kv_filter, parent=''):
    keys = [
        key
        for key in data.keys()
        if not kv_filter or any(fnmatch.fnmatch(f"{parent}.{key}", pattern) for pattern in kv_filter) or key in kv_filter
    ]

    nested_keys = [
        key
        for k, v in data.items()
        if isinstance(v, dict)
        for key in get_json_key(v, kv_filter, parent=f"{parent}.{k}" if parent else k)
    ]

    return keys + nested_keys



def get_json_value(data, kv_filter, parent=''):
    result = []
    if isinstance(data, dict):
        for key, value in data.items():
            nested_key = f"{parent}.{key}" if parent else key
            if not kv_filter or nested_key in kv_filter or any(fnmatch.fnmatch(nested_key, pattern) for pattern in kv_filter):
                if not isinstance(value, (dict, list)):
                    result.append('null' if not value else value)
                else:
                    result.extend(get_json_value(value, kv_filter, nested_key))
    elif isinstance(data, list):
        for item in data:
            result.extend(get_json_value(item, kv_filter, parent))
    else:
        result.append('null' if not data else data)
    return result




def get_json_kv(data, kv_filter, parent=''):
    result = []
    if isinstance(data, dict):
        for key, value in data.items():
            nested_key = f"{parent}.{key}" if parent else key
            if kv_filter and nested_key not in kv_filter and not any(fnmatch.fnmatch(nested_key, pattern) for pattern in kv_filter):
                continue

            result.append(key)
            if not isinstance(value, (dict, list)):
                result.append('null' if not value else value)
            else:
                result.extend(get_json_kv(value, kv_filter, nested_key))
    elif isinstance(data, list):
        for item in data:
            result.extend(get_json_kv(item, kv_filter, parent))
    else:
        result.append('null' if not data else data)
    return result



def save_json_to_file(json_data, output_file):
    with open(output_file, "w") as f:
        for data in json_data:
            f.write(json.dumps(data))



def process_json(input_file, output_file, processing_funcs):
    with open(input_file) as f:
        data = f.read()

    for func in processing_funcs:
        data = func(data)

    with open(output_file, "w") as f:
        f.write(data)