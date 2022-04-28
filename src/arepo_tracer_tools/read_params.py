def read_params(snap_path, params_file="parameters-usedvalues"):
    """Returns the used parameters of an arepo run as a dict"""
    param = {}
    with open(snap_path + "/" + params_file) as f:
        lines = f.readlines()
        i = 1
        for line in lines:
            line = line.strip()
            line = [item.strip() for item in line.split()]
            try:
                param[line[0]] = float(line[1])
            except ValueError:
                param[line[0]] = line[1]
            i += 1
    return param
