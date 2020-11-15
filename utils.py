import os
import json
import time
import shutil
import signal
import subprocess
from datetime import datetime

def kill_processes_on_port(port):
    process = subprocess.Popen(["lsof", "-i", ":{0}".format(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    for process in str(stdout.decode("utf-8")).split("\n")[1:]:       
        data = [x for x in process.split(" ") if x != '']
        if (len(data) <= 1):
            continue
        os.kill(int(data[1]), signal.SIGKILL)

def initialize_dirs_and_files(args, config):
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    exp_path = os.path.join(args.log_dir, args.exp_name+'@'+str(round(datetime.utcnow().timestamp())))
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    else:
        print('Deleting old log {} in 5 sec!'.format(exp_path))
        time.sleep(5)
        shutil.rmtree(exp_path)
        os.mkdir(exp_path)
    config.exp_path = exp_path
    weights_path = os.path.join(config.exp_path, args.weights_dir)
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    config.weights_path = weights_path
    filename = os.path.join(exp_path, 'config.json')
    json.dump(config.toDict(), open(filename, 'w'), default=lambda o: str(o), indent=4)