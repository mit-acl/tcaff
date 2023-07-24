import numpy as np

def parse_line(line, identifier, is_list=False):
    '''
    parses line of form: 
        mota: 0.89
    or
        motas: [0.95, 0.96, 0.95]
    line: single line of text
    identifier: first word to remove from line (should include colon if colon 
        present in identifier)
    is_list: True if a list of values instead of a single float
    '''
    try:
        if not is_list:
            return float(line.split(identifier)[1].strip().split()[0])
        else:
            list_str = line.split(identifier)[1].strip()
            list_floats = [float(sub_str.strip()) for sub_str in list_str.split('[')[1].split(']')[0].strip().split(',')]
            return list_floats
    except ValueError as val_err:
        if str(val_err) == "could not convert string to float: 'None'":
            raise Exception('None found in line')
        else:
            raise val_err
    except:
        print('offending line:')
        print(line)
        print(identifier)
        assert False

class RoverOutputParser():
    
    def __init__(self, metric_file):
        self.metric_file = metric_file
        self.metrics = []
        self.multiple_vals = []
        self.parsed = None
        
    def add_metric(self, metric, has_multiple_vals):
        '''
        metric - metric name (ex: 'mota')
        has_multiple_vals - bool whether metric is single float or list of floats
        '''
        self.metrics.append(metric)
        self.multiple_vals.append(has_multiple_vals)
        
    def parse(self, format=list, skip_none_lines=True):
        '''
        parses self.metric_file using added metrics
        '''
        self.parsed = []
        with open(self.metric_file, 'r') as f:
            new_entry = [None for m in self.metrics]
            skip_line = False
            for line in f.readlines():
                for i, (metric, is_list) in enumerate(zip(self.metrics, self.multiple_vals)):
                    if f'{metric}:' in line:
                        assert new_entry[i] is None, f'{metric}: found multiple values'
                        try:
                            new_entry[i] = parse_line(line, f'{metric}:', is_list=is_list)
                        except Exception as ex:
                            if str(ex) == 'None found in line':
                                skip_line = True
                                new_entry[i] = -1
                            else:
                                raise(ex)
                        if None not in new_entry: 
                            if not skip_line:
                                self.parsed.append(new_entry)
                            skip_line = False                                
                            new_entry = [None for m in self.metrics]
        if format==dict:
            for i in range(len(self.parsed)):
                self.parsed[i] = {m: vals for m, vals in zip(self.metrics, self.parsed[i])}
        return self.parsed
    
    def get_array(self, metric):
        if self.parsed is None:
            return None
        return np.array([entry[metric] for entry in self.parsed])
    
    def num_entries(self):
        return len(self.parsed)

