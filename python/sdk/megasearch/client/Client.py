import logging
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol, TCompactProtocol, TJSONProtocol
from thrift.Thrift import TException, TApplicationException

from ..thrift import MegasearchService
from ..thrift import ttypes
from .Abstract import (
    ConnectIntf, TableSchema,
    IndexType, ColumnType,
    ConnectParam, Column,
    VectorColumn, Range,
    CreateTablePartitionParam,
    DeleteTablePartitionParam,
    RowRecord, QueryRecord,
    QueryResult, TopKQueryResult
)

from .Status import Status
from .Exceptions import (
    RepeatingConnectError, ConnectParamMissingError,
    DisconnectNotConnectedClientError,
    ParamError
)
# TODO, status more specific

LOGGER = logging.getLogger(__name__)

__VERSION__ = '0.0.1'
__NAME__ = 'Client'


class Prepare(object):
    @classmethod
    def column(cls, name, type):
        """
        # TODO Is a must
        Table column param

        :param type: ColumnType, type of the column, default=ColumnType.INVALID
        :param name: str, name of the column

        :return Column
        """
        # TODO type in Thrift
        temp_column = Column(name=name, type=type)
        return ttypes.Column(name=temp_column.name, type=temp_column.type)

    @classmethod
    def vector_column(cls, name, dimension,
                      index_type=IndexType.RAW,
                      store_raw_vector=False):
        """
        Table vector column description

        :param dimension: int64, vector dimension
        :param index_type: IndexType
        :param store_raw_vector: Bool, Is vector self stored in the table

        `Column`:
            :param name: Name of the column
            :param type: Default type is ColumnType.VECTOR, can't change

        :return VectorColumn
        """
        temp = VectorColumn(name=name, dimension=dimension,
                            index_type=index_type, store_raw_vector=store_raw_vector)
        return ttypes.VectorColumn(base=temp.name, dimension=temp.dimension,
                                   store_raw_vector=temp.store_raw_vector,
                                   index_type=temp.index_type)

    @classmethod
    def table_schema(cls, table_name,
                     vector_columns,
                     attribute_columns,
                     partition_column_names):
        """

        :param table_name: Name of the table
        :param vector_columns: List of VectorColumns

            `VectorColumn`:
                - dimension: int, default = 0
                        Dimension of the vector, different vector_columns'
                        dimension may vary
                - index_type: (optional) IndexType, default=IndexType.RAW
                        Vector's index type
                - store_raw_vector : (optional) bool, default=False
                # TODO Is it a MUST
                - name: str
                        Name of the column
                - type: ColumnType, default=ColumnType.VECTOR, can't change

        :param attribute_columns: List of Columns. Attribute
                columns are Columns whose type aren't ColumnType.VECTOR

            `Column`:
                - name: str
                - type: ColumnType, default=ColumnType.INVALID

        :param partition_column_names: List of str.

              Partition columns name
                indicates which attribute columns is used for partition, can
                have lots of partition columns as long as:
                -> No. partition_column_names <= No. attribute_columns
                -> partition_column_names IN attribute_column_names

        :return: TableSchema
        """
        temp = TableSchema(table_name,vector_columns,
                           attribute_columns,partition_column_names)

        return ttypes.TableSchema(temp.table_name,temp.vector_columns,
                                  temp.attribute_columns, temp.partition_column_names)

    @classmethod
    def range(cls, start, end):
        """
        :param start: Partition range start value
        :param end: Partition range end value

        :return Range
        """
        temp = Range(start=start, end=end)
        return ttypes.Range(start_value=temp.start, end_value=temp.end)

    @classmethod
    def create_table_partition_param(cls,
                                     table_name,
                                     partition_name,
                                     column_name_to_range):
        """
        Create table partition parameters
        :param table_name: str, Table name,
            VECTOR/FLOAT32/FLOAT64 ColumnType is not allowed for partition
        :param partition_name: str partition name, created partition name
        :param column_name_to_range: dict, column name to partition range dictionary

        :return CreateTablePartitionParam
        """
        temp = CreateTablePartitionParam(table_name=table_name, partition_name=partition_name,
                                         column_name_to_range=column_name_to_range)
        return ttypes.CreateTablePartitionParam(table_name=temp.table_name,
                                                partition_name=temp.partition_name,
                                                range_map=temp.column_name_to_range)

    @classmethod
    def delete_table_partition_param(cls, table_name, partition_names):
        """
        Delete table partition parameters
        :param table_name: Table name
        :param partition_names: List of partition names

        :return DeleteTablePartitionParam
        """
        temp = DeleteTablePartitionParam(table_name=table_name, partition_names=partition_names)
        return ttypes.DeleteTablePartitionParam(table_name=table_name,
                                                partition_name_array=partition_names)

    @classmethod
    def row_record(cls, column_name_to_vector, column_name_to_attribute):
        """
        Record inserted

        :param  column_name_to_vector: dict{str : list[float]}
                Column name to vector map

        :param  column_name_to_attribute: dict{str: str}
                Other attribute columns
        """
        temp = RowRecord(column_name_to_vector=column_name_to_vector,
                         column_name_to_attribute=column_name_to_attribute)
        return ttypes.RowRecord(vector_map=temp.column_name_to_vector,
                                attribute_map=temp.column_name_to_attribute)

    @classmethod
    def query_record(cls, column_name_to_vector,
                     selected_columns, name_to_partition_ranges):
        """
        :param column_name_to_vector: dict{str : list[float]}
                Query vectors, column name to vector map

        :param selected_columns: list[str_column_name]
                List of Output columns

        :param name_to_partition_ranges: dict{str : list[Range]}
                Partition Range used to search

            `Range`:
                :param start: Partition range start value
                :param end: Partition range end value

        :return QueryRecord
        """
        temp = QueryRecord(column_name_to_vector=column_name_to_vector,
                           selected_columns=selected_columns,
                           name_to_partition_ranges=name_to_partition_ranges)
        return ttypes.QueryRecord(vector_map=temp.column_name_to_vector,
                                  selected_column_array=temp.selected_columns,
                                  partition_filter_column_map=name_to_partition_ranges)

    @classmethod
    def query_result(cls, id, score, column_name_to_value):
        """
        Query result

        :type  id: int
        :param id: Output result

        :type  score: float
        :param score: Vector similarity 0 <= score <= 100

        :type  column_name_to_value: dict{str : str}
        :param column_name_to_value: Other columns

        """
        # TODO
        pass

    @classmethod
    def top_k_query_result(cls, query_results):
        """
        TopK query results

        :type  query_results: list[QueryResult]
        :param query_results: TopK query results

        """
        # TODO


class Connection(ConnectIntf):

    def __init__(self):
        self.transport = None
        self.client = None
        self.connect_status = None

    def __repr__(self):
        return '{}'.format(self.connect_status)

    @staticmethod
    def create():
        # TODO in python, maybe this method is useless
        return Connection()

    @staticmethod
    def destroy(connection):
        """Destroy the connection instance"""
        # TODO in python, maybe this method is useless

        pass

    def connect(self, param=None, uri=None):
        # TODO URI
        if self.connect_status and self.connect_status == Status(message='Connected'):
            raise RepeatingConnectError("You have already connected!")

        if not param and not uri:
            raise ConnectParamMissingError('Need host/port or uri param to connect with!')

        try:
            transport = TSocket.TSocket(*param)
            LOGGER.info('transport: {}, {}'.format(transport.host, transport.port))
            self.transport = TTransport.TBufferedTransport(transport)
            protocol = TJSONProtocol.TJSONProtocol(transport)
            self.client = MegasearchService.Client(protocol)
            transport.open()
            self.connect_status = Status(Status.OK, 'Connected')
            LOGGER.info('Connected!')

        except TException as e:
            self.connect_status = Status(Status.CONNECT_FAILED, message=str(e))
            LOGGER.error(self.connect_status)
        finally:
            return self.connect_status

    @property
    def connected(self):
        return self.connect_status == Status()

    def disconnect(self):

        if self.connect_status != Status.OK:
            raise DisconnectNotConnectedClientError("Client is not connected")

        try:

            self.transport.close()
            LOGGER.info('Client Disconnected!')
            self.connect_status = None

        except TException as e:
            return Status(Status.INVALID, str(e))
        return Status(Status.OK, 'Disconnected')

    def create_table(self, param):
        """Create table

        :param param: Provide table information to be created,

                `Please use Prepare.table_schema generate param`

        :return: Status, indicate if connect is successful
        """
        if not isinstance(param, ttypes.TableSchema):
            raise ParamError('Param wrong, you need to prepare table_schema first!')
        try:
            self.client.CreateTable(param)
        except TException as e:
            LOGGER.error('Unable to create table using param:\n{}'.format(param))
            return Status(Status.INVALID, str(e))
        return Status('Table {} created!'.format(param.table_name))

    def delete_table(self, table_name):
        """Delete table

        :param table_name: Name of the table being deleted

        :return: Status, indicate if connect is successful
        """
        try:
            self.client.DeleteTable(table_name)
        except TApplicationException as e:
            LOGGER.error('Unable to delete table {}'.format(table_name))
            return Status(Status.INVALID, str(e))
        return Status('Table {} deleted!'.format(table_name))

    def create_table_partition(self, param):
        """
        Create table partition

        :type  param: CreateTablePartitionParam, provide partition information

                `Please use Prepare.create_table_partition_param generate param`

        :return: Status, indicate if table partition is created successfully
        """
        try:
            self.client.CreateTablePartition(param)
        except TApplicationException as e:
            return Status(Status.INVALID, str(e))
        return Status('Table partition created successfully!')

    def delete_table_partition(self, param):
        """
        Delete table partition

        :type  param: DeleteTablePartitionParam
        :param param: provide partition information to be deleted

                `Please use Prepare.delete_table_partition_param generate param`

        :return: Status, indicate if partition is deleted successfully
        """
        try:
            self.client.DeleteTablePartition(param)
        except TApplicationException as e:
            return Status(Status.INVALID, str(e))
        return Status('Table partition deleted successfully!')

    def add_vector(self, table_name, records):
        """
        Add vectors to table

        :param table_name: table name been inserted
        :param records: List[RowRecord], list of vectors been inserted

                `Please use Prepare.row_record generate records`

        :returns:
            Status : indicate if vectors inserted successfully
            ids :list of id, after inserted every vector is given a id
        """
        try:
            ids = self.client.AddVector(table_name=table_name, record_array=records)
        except TApplicationException as e:
            return Status(Status.INVALID, str(e)), None
        return Status('Vector added successfully!'), ids

    def search_vector(self, table_name, query_records, top_k):
        """
        Query vectors in a table

        :param table_name: str, table name been queried
        :param query_records: list[QueryRecord], all vectors going to be queried

                `Please use Prepare.query_record generate QueryRecord`

        :param top_k: int, how many similar vectors will be searched

        :returns:
            Status:  indicate if query is successful
            query_results: list[TopKQueryResult], return when operation is successful
        """
        try:
            topk_query_results = self.client.SearchVector(
                table_name=table_name, query_record_array=query_records, topk=top_k)

        except TApplicationException as e:
            return Status(Status.INVALID, str(e)), None
        return Status('Success!'), topk_query_results

    def describe_table(self, table_name):
        """
        Show table information

        :type  table_name: str
        :param table_name: which table to be shown

        :returns:
            Status: indicate if query is successful
            table_schema: TableSchema, return when operation is successful
        """
        try:
            table_schema = self.client.DescribeTable(table_name)
        except TApplicationException as e:
            return Status(Status.INVALID, str(e)), None

        # TODO Table Schema
        return Status('Success!'), table_schema

    def show_tables(self):
        """
        Show all tables in database

        :return:
            Status: indicate if this operation is successful
            tables: list[str], list of table names, return when operation
                    is successful
        """
        try:
            tables = self.client.ShowTables()
        except TApplicationException as e:
            return Status(Status.INVALID, str(e))
        return Status('Success!'), tables

    def client_version(self):
        """
        Provide client version

        :return: Client version
        """
        return __VERSION__

    def server_version(self):
        """
        Provide server version

        :return: Server version
        """
        # TODO How to get server version
        pass

    def server_status(self, cmd):
        """
        Provide server status

        :return: Server status
        """
        self.client.Ping(cmd)
        pass
