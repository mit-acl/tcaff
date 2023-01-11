#!/usr/bin/python3

import matplotlib.pyplot as plt
import sys

class MetricHandler:

    def __init__(self, metric_types):
        self.data = dict()
        for metric in metric_types:
            self.data[metric] = dict()
        
    def add_trial(self, metric_type, identifier, val):
        metric = self.data[metric_type]
        if identifier in metric:
            avg, num, all = metric[identifier]
            avg = (avg * num + val) / (num + 1)
            num += 1
            all += [val]
            metric[identifier] = (avg, num, all)
        else:
            metric[identifier] = (val, 1, [val])
            
    def get_metric_along_line(self, metric_type, line_lambda, identifier_x_idx, ret_type='avg'):
        metric = self.data[metric_type]
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
        for identifier, (avg, num, _) in self.data['mota'].items():
            ret += f'\t(sigma_t: {identifier[0]}, sigma_r: {identifier[1]}): {avg}\n'
            ret += f'\t\tn: {num}\n'
        return ret
    
def parse_metric_file(metric_file, metric_types=['mota', 'motp', 'inconsistencies', 'fp', 'fn', 'switch']):
    mh = MetricHandler(metric_types=metric_types)
    with open(metric_file, 'r') as f:
        trial = dict()
        for line in f.readlines():
            if line.strip() and line.strip()[0] == '#': 
                continue
            elif 'sigma_t' and 'sigma_r' in line:
                if trial:
                    for metric, val in trial['metrics'].items():
                        mh.add_trial(metric, trial['identifier'], val)
                    trial = dict()

                sigma_t = line.split('&')[0].split('=')[1].strip()
                sigma_r = line.split('&')[1].split('=')[1].strip()
                trial['identifier'] = (float(sigma_t), float(sigma_r))
                trial['metrics'] = dict()                    
            else:
                for i, metric in enumerate(metric_types):
                    if metric in line:
                        trial['metrics'][metric] = float(line.split(f'{metric}:')[1].strip())                            
                        break
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