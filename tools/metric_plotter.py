#!/usr/bin/python3

import matplotlib.pyplot as plt
import sys

class MetricHandler:

    def __init__(self):
        self.mota = dict()
        self.motp = dict()
        self.inconsistencies = dict()
        
    def add_trial(self, metric_type, identifier, val):
        metric = self._get_metric_from_type(metric_type)
        if identifier in metric:
            avg, num, all = metric[identifier]
            avg = (avg * num + val) / (num + 1)
            num += 1
            all += [val]
            metric[identifier] = (avg, num, all)
        else:
            metric[identifier] = (val, 1, [val])
            
    def get_metric_along_line(self, metric_type, line_lambda, identifier_x_idx, ret_type='avg'):
        metric = self._get_metric_from_type(metric_type)
        line_x = []
        line_y = []
        for identifier, (avg, _, all) in metric.items():
            if line_lambda(identifier):
                if ret_type == 'avg':
                    line_x.append(identifier[identifier_x_idx])
                    line_y.append(avg)
                elif ret_type == 'all':
                    for el in all:
                        line_x.append(identifier[identifier_x_idx])
                        line_y.append(el)
                elif ret_type == 'box':
                    line_x.append(identifier[identifier_x_idx])
                    line_y.append(all)
        return line_x, line_y
        
    def __str__(self):
        ret = 'mota:\n'
        for identifier, (avg, num, _) in self.mota.items():
            ret += f'\t(sigma_t: {identifier[0]}, sigma_r: {identifier[1]}): {avg}\n'
            ret += f'\t\tn: {num}\n'
        ret += '\nmotp:\n'
        for identifier, (avg, num, _) in self.motp.items():
            ret += f'\t(sigma_t: {identifier[0]}, sigma_r: {identifier[1]}): {avg}\n'
            ret += f'\t\tn: {num}\n'
        return ret
                            
        
    def _get_metric_from_type(self, metric_type):
        if metric_type == 'mota':
            return self.mota
        elif metric_type == 'motp':
            return self.motp
        elif metric_type == 'inconsistencies':
            return self.inconsistencies
    
def parse_metric_file(metric_file, inconsistencies=False):
    mh = MetricHandler()
    with open(metric_file, 'r') as f:
        trial = []
        for line in f.readlines():
            if line.strip() and line.strip()[0] == '#': 
                continue
            elif 'sigma_t' and 'sigma_r' in line:
                assert not trial
                sigma_t = line.split('&')[0].split('=')[1].strip()
                sigma_r = line.split('&')[1].split('=')[1].strip()
                trial.append((float(sigma_t), float(sigma_r)))
            elif 'mota' in line:
                assert len(trial) == 1
                trial.append(float(line.split('mota:')[1].strip()))
            elif 'motp' in line:
                assert len(trial) == 2
                trial.append(float(line.split('motp:')[1].strip()))
                if not inconsistencies:
                    mh.add_trial('mota', trial[0], trial[1])
                    mh.add_trial('motp', trial[0], trial[2])
                    trial = []
            elif inconsistencies and 'inconsistencies' in line:
                assert len(trial) == 3
                trial.append(float(line.split('inconsistencies:')[1].strip()))
                mh.add_trial('mota', trial[0], trial[1])
                mh.add_trial('motp', trial[0], trial[2])
                mh.add_trial('inconsistencies', trial[0], trial[3])
                trial = []
    return mh

if __name__ == '__main__':
    metric_file = sys.argv[1]
    mh = parse_metric_file(metric_file)



    print(mh)
    # x, y = mh.get_metric_along_line('mota', lambda x: x[0] == 0, 1, ret_type='avg')
    # plt.plot(x, y, 'o')
    # plt.show()

    # x, y = mh.get_metric_along_line('mota', lambda x: x[1] == 0.0, 0, ret_type='all')
    # plt.plot(x, y, 'o')
    # plt.title('All Trials MOTA')
    # plt.xlabel('Translation Error Standard Dev (m)')
    # plt.ylabel('MOTA')
    # plt.show()

    x, y = mh.get_metric_along_line('mota', lambda x: x[1] == 0.0, 0, ret_type='avg')
    plt.plot(x, y, 'o')
    plt.title('Average MOTA')
    plt.xlabel('Translation Error Standard Dev (m)')
    plt.ylabel('MOTA')
    plt.show()

    X, Y = mh.get_metric_along_line('mota', lambda x: x[1] == 0.0, 0, ret_type='box')
    Y_sorted = [y for x, y in sorted(zip(X, Y))]
    X_sorted = sorted(X)
    fig, ax1 = plt.subplots()
    ax1.boxplot(Y_sorted[1:], notch=False, vert=True, whis=1.5)
    ax1.set_xticklabels(X_sorted[1:])
    plt.title('MOTA Boxplots')
    plt.xlabel('Translation Error Standard Dev (m)')
    plt.ylabel('MOTA')
    plt.show()