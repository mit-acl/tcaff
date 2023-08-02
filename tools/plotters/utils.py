def get_list_val(line, identifier):
    # print(line.split(identifier)[1].strip().split('[')[1].split(']')[0])
    input = line.split(identifier)[1].strip().split('[')[1].split(']')[0]
    # float_str = f'[{input}]'
    return [float(idx) for idx in input.split(', ')]