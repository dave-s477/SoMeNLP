import json
import copy

from pathlib import Path

from somenlp.utils import find_type_in_dict, getFromDict, setInDict, get_abbr

class Tuner():
    def __init__(self, config, time):
        self.config = config
        self.values_to_vary = find_type_in_dict(self.config, list)
        self._gen_all_parameter_combinations()
        self._gen_all_configs()

    def _entry_to_combinations(self, entry):
        res = []
        for val in entry['values']:
            res.append([{
                'path': entry['path'],
                'values': val
            }])
        return res

    def _gen_all_parameter_combinations(self):
        old_combinations = self._entry_to_combinations(self.values_to_vary[0])
        for val in self.values_to_vary[1:]:
            #print(self._entry_to_combinations(val))
            new_combinations = []
            for combination in old_combinations:
                for next_param in self._entry_to_combinations(val):
                    comb = combination.copy()
                    comb.extend(next_param)
                    new_combinations.append(comb)
            old_combinations = new_combinations
        self.combinations = old_combinations

    def _gen_config_name(self, parameter_config):
        name = ''
        for parameter in parameter_config:
            for sub_path in parameter['path']:
                name += '{}-'.format(get_abbr(sub_path)) 
            name += '{}_'.format(parameter['values'])
        for char in ['{', '}', ' ', "'", '/', '\\', ':', ',']:
            name = name.replace(char, '')
        return name.rstrip('_')[:200]

    def _gen_all_configs(self):
        self.configs_to_execute = {}
        for combination in self.combinations:
            config = copy.deepcopy(self.config)
            config_name = self._gen_config_name(combination)
            for entry in combination:
                setInDict(config, entry['path'], entry['values'])
            data_c_file = config.pop('data', None)
            if data_c_file is None:
                raise(RuntimeError("No data config was provided in tuning config."))
            data_c_path = Path(data_c_file)
            with data_c_path.open(mode='r') as data_c_json:
                data_conf = json.load(data_c_json)
            self.configs_to_execute[config_name] = {
                'data': data_conf,
                'model': config
            }

    def yield_configs(self):
        for c_name, v in self.configs_to_execute.items():
            yield c_name, v['data'], v['model']