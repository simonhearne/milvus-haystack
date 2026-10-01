import logging

import pymilvus
import pytest
from haystack import Document
from haystack.document_stores.types import DocumentStore
from haystack.testing.document_store import CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest

from src.milvus_haystack import MilvusDocumentStore
from src.milvus_haystack.document_store import MilvusStoreError

logger = logging.getLogger(__name__)

DEFAULT_CONNECTION_ARGS = {
    "uri": "http://localhost:19530",  # This uri works for Milvus docker service
    # "uri": "./milvus_test.db",  # This uri works for Milvus Lite
}


class TestDocumentStore(CountDocumentsTest, WriteDocumentsTest, DeleteDocumentsTest):
    @pytest.fixture
    def document_store(self) -> MilvusDocumentStore:
        return MilvusDocumentStore(
            connection_args=DEFAULT_CONNECTION_ARGS,
            consistency_level="Strong",
            drop_old=True,
        )

    def test_write_documents(self, document_store: DocumentStore):
        return_value = document_store.write_documents(
            [Document(content="test doc 1"), Document(content="test doc 2"), Document(content="test doc 3")]
        )
        assert document_store.count_documents() == 3
        assert return_value == 3

    def test_delete_documents(self, document_store: DocumentStore):
        """
        Test delete_documents() normal behaviour.
        """
        doc = Document(content="test doc")
        document_store.write_documents([doc])
        assert document_store.count_documents() == 1

        document_store.delete_documents([doc.id])
        assert document_store.count_documents() == 0

    @pytest.mark.skip(reason="Milvus does not currently check if entity primary keys are duplicates")
    def test_write_documents_duplicate_fail(self, document_store: DocumentStore): ...

    @pytest.mark.skip(reason="Milvus does not currently check if entity primary keys are duplicates")
    def test_write_documents_duplicate_skip(self, document_store: DocumentStore): ...

    @pytest.mark.skip(reason="Milvus does not currently check if entity primary keys are duplicates")
    def test_write_documents_duplicate_overwrite(self, document_store: DocumentStore): ...

    def test_to_and_from_dict(self, document_store: MilvusDocumentStore):
        document_store_dict = document_store.to_dict()
        expected_dict = {
            "type": "src.milvus_haystack.document_store.MilvusDocumentStore",
            "init_parameters": {
                "collection_name": "HaystackCollection",
                "collection_description": "",
                "collection_properties": None,
                "connection_args": DEFAULT_CONNECTION_ARGS,
                "consistency_level": "Strong",
                "index_params": None,
                "search_params": None,
                "drop_old": True,
                "primary_field": "id",
                "text_field": "text",
                "vector_field": "vector",
                "sparse_vector_field": None,
                "sparse_index_params": None,
                "sparse_search_params": None,
                "builtin_function": [],
                "partition_key_field": None,
                "partition_names": None,
                "replica_number": 1,
                "timeout": None,
            },
        }
        assert document_store_dict == expected_dict
        reconstructed_document_store = MilvusDocumentStore.from_dict(document_store_dict)
        for field in vars(reconstructed_document_store):
            if field.startswith("__") or field in ["alias", "_milvus_client"]:
                continue
            if field == "builtin_function":
                for func, func_reconstructed in zip(
                    getattr(document_store, field),
                    getattr(reconstructed_document_store, field),
                ):
                    for k, v in func.to_dict().items():
                        if k == "function_name":
                            continue
                        assert v == func_reconstructed.to_dict()[k]
            else:
                assert getattr(reconstructed_document_store, field) == getattr(document_store, field)


class TestPymilvusApiUsage:
    @pytest.fixture
    def document_store(self) -> MilvusDocumentStore:
        return MilvusDocumentStore(
            connection_args=DEFAULT_CONNECTION_ARGS,
            consistency_level="Strong",
            index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 8, "efConstruction": 64}},
            drop_old=True,
        )

    def test_get_index_before_and_after_collection_creation(self, document_store: MilvusDocumentStore):
        assert document_store._get_index() is None

        document_store.write_documents([Document(content="test doc", embedding=[0.1, 0.2, 0.3, 0.4])])

        index = document_store._get_index()
        assert index is not None
        assert index["index_type"] == "HNSW"
        assert index["metric_type"] == "COSINE"
        assert document_store.search_params == {"metric_type": "COSINE", "params": {"ef": 10}}

    @pytest.mark.filterwarnings(r"ignore:.*will be removed in PyMilvus 3\.1")
    def test_col_is_deprecated(self, document_store: MilvusDocumentStore):
        document_store.write_documents([Document(content="test doc", embedding=[0.1, 0.2, 0.3, 0.4])])

        with pytest.warns(DeprecationWarning, match="client"):
            col = document_store.col

        assert col is not None
        assert col.name == document_store.collection_name

    def test_col_raises_without_orm_api(self, document_store: MilvusDocumentStore, monkeypatch):
        # Simulates pymilvus 3.1, which removes the ORM `Collection` API.
        monkeypatch.delattr(pymilvus, "Collection")

        with pytest.warns(DeprecationWarning), pytest.raises(MilvusStoreError, match="client"):
            _ = document_store.col

    def test_collection_existence_is_cached_after_creation(self, document_store: MilvusDocumentStore, monkeypatch):
        document_store.write_documents([Document(content="test doc", embedding=[0.1, 0.2, 0.3, 0.4])])

        calls = []
        has_collection = document_store.client.has_collection

        def spy(*args, **kwargs):
            calls.append(args)
            return has_collection(*args, **kwargs)

        monkeypatch.setattr(document_store.client, "has_collection", spy)
        document_store.count_documents()
        document_store.filter_documents()

        assert calls == []

    @pytest.mark.filterwarnings(r"ignore:.*will be removed in PyMilvus 3\.1")
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_col_uses_configured_database(self):
        db_name = "haystack_col_db_test"
        admin = pymilvus.MilvusClient(**DEFAULT_CONNECTION_ARGS)
        if db_name not in admin.list_databases():
            admin.create_database(db_name)
        try:
            document_store = MilvusDocumentStore(
                connection_args={**DEFAULT_CONNECTION_ARGS, "db_name": db_name},
                collection_name="ColDatabaseTest",
                drop_old=True,
            )
            document_store.write_documents([Document(content="test doc", embedding=[0.1, 0.2, 0.3, 0.4])])

            col = document_store.col

            assert col is not None
            assert col.num_entities >= 0
        finally:
            # Not `admin.use_database()`: on pymilvus 2.5 it switches the connection shared with other tests.
            pymilvus.MilvusClient(**{**DEFAULT_CONNECTION_ARGS, "db_name": db_name}).drop_collection("ColDatabaseTest")
            admin.drop_database(db_name)
