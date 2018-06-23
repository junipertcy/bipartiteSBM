# computer_trace.py
import os
import sys
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from engines.mcmc import *
from det_k_bisbm.optimalks import *
from det_k_bisbm.ioutils import *

from pymongo import MongoClient
from configparser import ConfigParser
import math
# from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
from bson import ObjectId

from loky import get_reusable_executor
executor = get_reusable_executor(max_workers=16, timeout=10000)


parser = ConfigParser()
parser.read('/home/tzuchi/.config/mongo@aws/dev.ini')
host = parser.get('det_k_bisbm', 'host')
database = parser.get('det_k_bisbm', 'database')
user = parser.get('det_k_bisbm', 'user')
password = parser.get('det_k_bisbm', 'password')
dbClient = MongoClient(host)
db = dbClient[database]
db.authenticate(user, password, mechanism='SCRAM-SHA-1')

mcmc = MCMC(f_engine="../../engines/bipartiteSBM-MCMC/bin/mcmc", 
    n_sweeps=1,
    is_parallel=False,
    n_cores=1,
    mcmc_steps=1e6,
    mcmc_await_steps=1e5,
    mcmc_cooling="abrupt_cool",
    mcmc_epsilon=0.01
)

def func(file):    
    edgelist = get_edgelist("dataset/" + str(file["_id"]) + ".gml.edgelist", " ")
    types= mcmc.gen_types(500, 500)
    oks = OptimalKs(mcmc, edgelist, types)
    oks.set_params(init_ka=15, init_kb=15, i_th=0.1)
    oks.set_adaptive_ratio(0.9)
    oks.set_exist_bookkeeping(True)
    oks.set_logging_level("info")
    
    from itertools import combinations_with_replacement, combinations

    desc_len = [0]
    for k in range(1, 20):
        desc_len_at_k = []
        for case in list(combinations_with_replacement(range(1, 20), 2)):
            if sum(case) == k:
                if case[0] == case[1]:
                    oks.compute_and_update(case[0], case[0], recompute=False)
                    desc_len_at_k += [oks.confident_desc_len[(case[0], case[0])]]
                else:
                    oks.compute_and_update(case[0], case[1], recompute=False)
                    desc_len_at_k += [oks.confident_desc_len[(case[0], case[1])]]
                    oks.compute_and_update(case[1], case[0], recompute=False)
                    desc_len_at_k += [oks.confident_desc_len[(case[1], case[0])]]
        if len(desc_len_at_k) != 0:
            desc_len += [min(desc_len_at_k)]
        else:
            desc_len += [0]
    return (str(file["_id"]), desc_len)


find_schema = {}
projection = {
    "_id": 1
}

res = list(db.new_benches.find(find_schema, projection))
results = list(executor.map(func, res))

for r in results:
    if r is not None:
        file = r[0]
        desc_len = r[1]
        find_schema = { 
            "_id": ObjectId(file)
        }
        
        result = db.new_benches.update_one(
            find_schema,
            {
                "$set": {
                    "trace": ",".join(map(lambda x: str(x), desc_len))
                }
            }
        )
        
        if result is None:
            print("Update failed!")
            raise Exception
        else:
            pass

