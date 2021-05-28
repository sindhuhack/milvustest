import time

import pandas as pd
import pytest

from base.client_request import ApiReq
from utils.util_log import test_log as log
from common import common_func as cf
from common import common_type as ct
from common.common_type import CaseLabel

prefix = "insert"
default_schema = cf.gen_default_collection_schema()
default_binary_schema = cf.gen_default_binary_collection_schema()


class TestInsertParams(ApiReq):
    """ Test case of Insert interface """

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.xfail(reason="issue #5302")
    def test_insert_dataframe_data(self):
        """
        target: test insert DataFrame data
        method: 1.create 2.insert dataframe data
        expected: assert num entities
        """
        self._connect()
        nb = ct.default_nb
        collection = self._collection()
        df = cf.gen_default_dataframe_data(nb)
        self.collection.insert(data=df)
        assert collection.num_entities == nb

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.xfail(reason="issue #5470")
    def test_insert_list_data(self):
        """
        target: test insert list-like data
        method: 1.create 2.insert list data
        expected: assert num entities
        """
        conn = self._connect()
        nb = ct.default_nb
        collection = self._collection()
        data = cf.gen_default_list_data(nb)
        self.collection.insert(data=data)
        self.connection.connection.get_connection().flush([collection.name])
        assert collection.num_entities == nb

    def test_insert_non_data_type(self):
        """
        target: test insert with non-dataframe, non-list data
        method: insert with data (non-dataframe and non-list type)
        expected: raise exception
        """
        pass

    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_empty_data(self):
        """
        target: test insert empty data
        method: insert empty
        expected: raise exception
        """
        self._collection()
        ex, _ = self.collection.insert(data=[])
        assert "Column cnt not match with schema" in str(ex)

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_empty_dataframe(self):
        """
        target: test insert empty dataframe
        method: insert with empty dataframe
        expected: raise exception
        """
        data = pd.DataFrame()
        self._collection()
        ex, _ = self.collection.insert(data=data)
        assert "Column cnt not match with schema" in str(ex)

    def test_insert_dataframe_only_columns(self):
        """
        target: test insert with dataframe just columns
        method: dataframe just have columns
        expected: num entities is zero
        """
        pass

    def test_insert_invalid_field_name_dataframe(self):
        """
        target: test insert with invalid dataframe data
        method: insert with invalid field name dataframe
        expected: raise exception
        """
        pass

    def test_insert_dataframe_nan_value(self):
        """
        target: test insert dataframe with nan value
        method: insert dataframe with nan value
        expected: todo
        """
        pass

    def test_insert_dataframe_index(self):
        """
        target: test insert dataframe with index
        method: insert dataframe with index
        expected: todo
        """
        pass

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.xfail(reason="issue #5445")
    def test_insert_none(self):
        """
        target: test insert None
        method: data is None
        expected: raise exception
        """
        self._collection()
        ex, _ = self.collection.insert(data=None)
        log.info(str(ex))

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.xfail(reason="issue #5421")
    def test_insert_numpy_data(self):
        """
        target: test insert numpy.ndarray data
        method: 1.create by schema 2.insert data
        expected: assert num_entities
        """
        self._connect()
        nb = 10
        self._collection()
        data = cf.gen_numpy_data(nb)
        ex, _ = self.collection.insert(data=data)
        log.error(str(ex))

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.xfail(reason="issue #5302")
    def test_insert_binary_dataframe(self):
        """
        target: test insert binary dataframe
        method: 1. create by schema 2. insert dataframe
        expected: assert num_entities
        """
        self._connect()
        nb = ct.default_nb
        collection = self._collection(schema=default_binary_schema)
        df = cf.gen_default_binary_dataframe_data(nb)
        self.collection.insert(data=df)
        assert collection.num_entities == nb

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.xfail(reason="issue #5414")
    def test_insert_binary_data(self):
        """
        target: test insert list-like binary data
        method: 1. create by schema 2. insert data
        expected: assert num_entities
        """
        self._connect()
        nb = ct.default_nb
        collection = self._collection(schema=default_binary_schema)
        data = cf.gen_default_binary_list_data(nb)
        self.collection.insert(data=data)
        assert collection.num_entities == nb

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.xfail(reason="issue #5470")
    def test_insert_single(self):
        """
        target: test insert single
        method: insert one entity
        expected: verify num
        """
        conn = self._connect()
        collection = self._collection()
        data = cf.gen_default_list_data(nb=1)
        self.collection.insert(data=data)
        conn.flush([collection.name])
        assert collection.num_entities == 1

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_dim_not_match(self):
        """
        target: test insert with not match dim
        method: insert data dim not equal to schema dim
        expected: raise exception
        """
        self._connect()
        nb = ct.default_nb
        collection = self._collection()
        df = cf.gen_default_dataframe_data(nb, dim=129)
        ex, _ = self.collection.insert(data=df)
        message = "Collection field dim is {},but entities field dim is {}".format(ct.default_dim, 129)
        assert message in str(ex)

    def test_insert_field_name_not_match(self):
        """
        target: test insert field name not match
        method: data field name not match schema
        expected: raise exception
        """
        pass

    def test_insert_field_value_not_match(self):
        """
        target: test insert data value not match
        method: insert data value type not match schema
        expected: raise exception
        """
        pass

    def test_insert_value_less(self):
        """
        target: test insert value less than other
        method: int field value less than vec-field value
        expected: raise exception
        """
        pass

    def test_insert_vector_value_less(self):
        """
        target: test insert vector value less than other
        method: vec field value less than int field
        expected: todo
        """
        pass

    def test_insert_fields_more(self):
        """
        target: test insert with fields more
        method: field more than schema fields
        expected: todo
        """
        pass

    def test_insert_fields_less(self):
        """
        target: test insert with fields less
        method: fields less than schema fields
        expected: raise exception
        """
        pass

    def test_insert_list_order_inconsistent_schema(self):
        """
        target: test insert data fields order inconsistent with schema
        method: insert list data, data fields order inconsistent with schema
        expected: raise exception
        """
        pass

    def test_insert_dataframe_order_inconsistent_schema(self):
        """
        target: test insert with dataframe fields inconsistent with schema
        method: insert dataframe, and fields order inconsistent with schema
        expected: assert num entities
        """
        pass


class TestInsertOperation(ApiReq):
    """
    ******************************************************************
      The following cases are used to test insert interface operations
    ******************************************************************
    """

    def teardown_method(self):
        if self.collection is not None and self.collection.collection is not None:
            self.collection.drop()

    def setup_method(self):
        pass

    def test_insert_without_connection(self):
        """
        target: test insert without connection
        method: insert after remove connection
        expected: raise exception
        """
        self._collection()
        self.connection.remove_connection(ct.default_alias)
        res_list, _ = self.connection.list_connections()
        assert ct.default_alias not in res_list
        data = cf.gen_default_list_data(10)
        ex, check = self.collection.insert(data=data)
        assert "There is no connection with alias '{}'".format(ct.default_alias) in str(ex)
        assert self.collection.is_empty()

    def test_insert_drop_collection(self):
        """
        target: test insert and drop
        method: insert data and drop collection
        expected: verify collection if exist
        """
        collection = self._collection()
        collection_list, _ = self.utility.list_collections()
        assert collection.name in collection_list
        self.collection.drop()
        collection_list, _ = self.utility.list_collections()
        assert collection.name not in collection_list

    def test_insert_create_index(self):
        """
        target: test insert and create index
        method: 1. insert 2. create index
        expected: verify num entities and index
        """
        pass

    def test_insert_after_create_index(self):
        """
        target: test insert after create index
        method: 1. create index 2. insert data
        expected: verify index and num entities
        """
        pass

    def test_insert_binary_after_index(self):
        """
        target: test insert binary after index
        method: 1.create index 2.insert binary data
        expected: 1.index ok 2.num entities correct
        """
        pass

    def test_insert_search(self):
        """
        target: test insert and search
        method: 1.insert data 2.search
        expected: verify search result
        """
        pass

    def test_insert_binary_search(self):
        """
        target: test insert and search
        method: 1.insert binary data 2.search
        expected: search result correct
        """
        pass

    def test_insert_ids(self):
        """
        target: test insert with ids
        method: insert with ids field value
        expected: 1.verify num entities 2.verify ids
        """
        schema = cf.gen_default_collection_schema(primary_field=ct.default_int64_field_name)
        collection = self._collection(schema=schema)
        assert not collection.auto_id
        assert collection.primary_field.name == ct.default_int64_field_name
        data = cf.gen_default_list_data(ct.default_nb)
        self.collection.insert(data=data)
        time.sleep(1)
        assert collection.num_entities == ct.default_nb
        # TODO assert ids

    def test_insert_ids_without_value(self):
        """
        target: test insert ids value not match
        method: insert without ids field value
        expected: raise exception
        """
        pass

    def test_insert_same_ids(self):
        """
        target: test insert ids field
        method: insert with same ids
        expected: num entities equal to nb
        """
        pass

    def test_insert_invalid_type_ids(self):
        """
        target: test insert with non-int64 ids
        method: insert ids field with non-int64 value
        expected: raise exception
        """
        pass

    def test_insert_multi_threading(self):
        """
        target: test concurrent insert
        method: multi threads insert
        expected: verify num entities
        """
        pass

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.xfail(reason="issue #5470")
    def test_insert_multi_times(self):
        """
        target: test insert multi times
        method: insert data multi times
        expected: verify num entities
        """
        conn = self._connect()
        collection = self._collection()
        for _ in range(ct.default_nb):
            df = cf.gen_default_dataframe_data(1)
            self.collection.insert(data=df)
        self.connection.connection.get_connection().flush([collection.name])
        # conn.flush([collection.name])
        assert collection.num_entities == ct.default_nb


class TestInsertAsync(ApiReq):
    """
    ******************************************************************
      The following cases are used to test insert async
    ******************************************************************
    """
    def test_insert_sync(self):
        """
        target: test async insert
        method: insert with async=True
        expected: todo
        """
        pass

    def test_insert_async_false(self):
        """
        target: test insert with false async
        method: async = false
        expected: no exception
        """
        pass
