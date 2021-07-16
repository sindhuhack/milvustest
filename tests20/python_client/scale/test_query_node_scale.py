import random

from base.collection_wrapper import ApiCollectionWrapper
from scale.helm_env import HelmEnv
from utils.util_log import test_log as log
from common import common_func as cf
from common import common_type as ct
from pymilvus_orm import Index, connections

prefix = "search_scale"
nb = 5000
default_schema = cf.gen_default_collection_schema()
default_search_exp = "int64 >= 0"
default_index_params = {"index_type": "IVF_SQ8", "metric_type": "L2", "params": {"nlist": 64}}


class TestSearchScale:
    def test_search_scale(self):
        release_name = "scale-test"
        env = HelmEnv(release_name=release_name)
        env.helm_install_cluster_milvus()

        # connect
        connections.add_connection(default={"host": '10.98.0.8', "port": 19530})
        connections.connect(alias='default')

        # create
        c_name = "data_scale_one"
        collection_w = ApiCollectionWrapper()
        collection_w.init_collection(name=c_name, schema=cf.gen_default_collection_schema())
        # insert
        data = cf.gen_default_list_data(ct.default_nb)
        mutation_res, _ = collection_w.insert(data)
        assert mutation_res.insert_count == ct.default_nb
        # # create index
        # collection_w.create_index(ct.default_float_vec_field_name, default_index_params)
        # assert collection_w.has_index()
        # assert collection_w.index()[0] == Index(collection_w.collection, ct.default_float_vec_field_name,
        #                                         default_index_params)
        collection_w.load()
        # vectors = [[random.random() for _ in range(ct.default_dim)] for _ in range(5)]
        res1, _ = collection_w.search(data[-1][:5], ct.default_float_vec_field_name,
                                      ct.default_search_params, ct.default_limit)

        # scale queryNode pod
        env.helm_upgrade_cluster_milvus(queryNode=2)

        c_name_2 = "data_scale_two"
        collection_w2 = ApiCollectionWrapper()
        collection_w2.init_collection(name=c_name_2, schema=cf.gen_default_collection_schema())
        collection_w2.insert(data)
        assert collection_w2.num_entities == ct.default_nb
        collection_w2.load()
        res2, _ = collection_w2.search(data[-1][:5], ct.default_float_vec_field_name,
                                       ct.default_search_params, ct.default_limit)

        assert res1[0].ids == res2[0].ids
