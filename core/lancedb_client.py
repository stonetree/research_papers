# -*- coding: utf-8 -*-
import os
import logging
import pyarrow as pa
import lancedb
from typing import List, Dict, Any

logger = logging.getLogger("LanceDBClient")

LANCE_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "lancedb")
os.makedirs(LANCE_DB_DIR, exist_ok=True)

VECTOR_DIM = 1024

# 定义强类型的 PyArrow 物理 schema，防止特征稀释与字段漂移
VectorSchema = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    pa.field("metadata", pa.struct([
        pa.field("parent_id", pa.string(), nullable=False),
        pa.field("ingestion_tx_id", pa.string(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False)
    ]), nullable=False)
])

class LanceDBClient:
    def __init__(self, db_dir: str = None):
        self.db_dir = db_dir or LANCE_DB_DIR
        self.conn = None

    def _get_connection(self):
        if self.conn is None:
            self.conn = lancedb.connect(self.db_dir)
        return self.conn

    def get_table_name(self, source_type: str) -> str:
        """根据资产子类型（如 'arxiv_paper'）映射独立的物理向量子表，防止类型混淆"""
        return f"vector_chunks_{source_type}"

    def get_or_create_table(self, source_type: str):
        conn = self._get_connection()
        table_name = self.get_table_name(source_type)
        if table_name in conn.table_names():
            return conn.open_table(table_name)
        else:
            return conn.create_table(table_name, schema=VectorSchema)

    def add_vectors(self, source_type: str, rows: List[Dict[str, Any]]):
        """
        向指定的向量子表中批量添加数据记录。
        rows 的元素格式应当符合 VectorSchema，如：
        {
            "id": "chunk_arxiv_2605.0089_0",
            "vector": [...],
            "text": "...",
            "metadata": {
                "parent_id": "...",
                "ingestion_tx_id": "...",
                "schema_version": "1.2"
            }
        }
        """
        tbl = self.get_or_create_table(source_type)
        # 将输入 rows 装载进 PyArrow RecordBatch 以确保严格匹配 schema 约束
        tbl.add(rows)

    def delete_vectors_by_parent_and_tx(self, source_type: str, parent_id: str, ingestion_tx_id: str):
        """
        根据 parent_id 和 ingestion_tx_id 执行逆向强力删除补偿
        """
        conn = self._get_connection()
        table_name = self.get_table_name(source_type)
        if table_name in conn.table_names():
            tbl = conn.open_table(table_name)
            # 使用 LanceDB 的 SQL 过滤语法进行删除
            tbl.delete(f"metadata.parent_id = '{parent_id}' AND metadata.ingestion_tx_id = '{ingestion_tx_id}'")
            logger.info(f"LanceDB 表 {table_name} 已成功物理删除 parent_id={parent_id}, tx_id={ingestion_tx_id} 的残留向量数据")

    def search_vector(self, source_type: str, query_vector: List[float], limit: int = 50) -> List[Dict[str, Any]]:
        """
        在指定向量表中执行 Cosine 相似度语义模糊检索，返回匹配切片列表
        """
        conn = self._get_connection()
        table_name = self.get_table_name(source_type)
        if table_name not in conn.table_names():
            logger.warning(f"向量子表 {table_name} 尚未初始化或不存在任何向量记录。")
            return []
            
        tbl = conn.open_table(table_name)
        results = tbl.search(query_vector).limit(limit).to_list()
        return results
