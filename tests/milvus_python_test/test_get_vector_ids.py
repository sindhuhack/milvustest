import time
import random
import pdb
import threading
import logging
from multiprocessing import Pool, Process
import pytest
from milvus import IndexType, MetricType
from utils import *

dim = 128
index_file_size = 10
GET_TIMEOUT = 30
nprobe = 1
top_k = 1
epsilon = 0.0001
tag = "1970-01-01"
nb = 6000


class TestGetVectorIdsBase:
    def get_valid_segment_name(self, connect, table):
        vectors = gen_vector(nb, dim)
        status, ids = connect.add_vectors(table, vectors)
        assert status.OK()
        status, info = connect.table_info(table)
        assert status.OK()
        return info.partitions_stat[0].segments_stat[0].segment_name
    """
    ******************************************************************
      The following cases are used to test `get_vector_ids` function
    ******************************************************************
    """
    @pytest.mark.timeout(GET_TIMEOUT)
    def test_get_vector_ids_table_name_None(self, connect, table):
        '''
        target: get vector ids where table name is None
        method: call get_vector_ids with the table_name: None
        expected: status not ok
        '''
        table_name = None
        segment_name = self.get_valid_segment_name(connect, table)
        status, vector_ids = connect.get_vector_ids(table_name, segment_name)
        assert not status.OK()

    @pytest.mark.timeout(GET_TIMEOUT)
    def test_get_vector_ids_table_name_not_existed(self, connect, table):
        '''
        target: get vector ids where table name does not exist
        method: call get_vector_ids with a random table_name, which is not in db
        expected: status not ok
        '''
        table_name = gen_unique_str("not_existed_table")
        segment_name = self.get_valid_segment_name(connect, table)
        status, vector_ids = connect.get_vector_ids(table_name, segment_name)
        assert not status.OK()
    
    @pytest.fixture(
        scope="function",
        params=gen_invalid_table_names()
    )
    def get_table_name(self, request):
        yield request.param

    @pytest.mark.timeout(GET_TIMEOUT)
    def test_get_vector_ids_table_name_invalid(self, connect, table, get_table_name):
        '''
        target: get vector ids where table name is invalid
        method: call get_vector_ids with invalid table_name
        expected: status not ok
        '''
        table_name = get_table_name
        segment_name = self.get_valid_segment_name(connect, table)
        status, vector_ids = connect.get_vector_ids(table_name, segment_name)
        assert not status.OK()