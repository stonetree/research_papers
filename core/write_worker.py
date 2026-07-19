# -*- coding: utf-8 -*-
import asyncio
import logging
import aiosqlite
from typing import List, Tuple, Any

logger = logging.getLogger("WriteWorker")

class WriteTask:
    def __init__(self, sql_queries: List[Tuple[str, Tuple[Any, ...]]], database_path: str):
        self.sql_queries = sql_queries  # List of tuples (sql_string, params_tuple)
        self.database_path = database_path
        self.future = asyncio.get_event_loop().create_future()

class WriteWorker:
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(WriteWorker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, database_path: str = None):
        if self._initialized:
            return
        from .database import DB_PATH
        self.database_path = database_path or DB_PATH
        self.queue = asyncio.Queue()
        self.worker_task = None
        self._initialized = True

    def start(self):
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._processing_loop())
            logger.info("全局统一独占串行写队列消费者已成功启动。")

    async def execute_write(self, sql_queries: List[Tuple[str, Tuple[Any, ...]]]) -> bool:
        """
        将写入任务放入串行写入队列，并等待其执行完成。
        sql_queries: 元组 (SQL语句, 参数) 的列表
        """
        if self.worker_task is None or self.worker_task.done():
            # 自动启动以保证鲁棒性
            self.start()
            
        task = WriteTask(sql_queries, self.database_path)
        await self.queue.put(task)
        return await task.future

    async def _processing_loop(self):
        while True:
            try:
                task = await self.queue.get()
                success = False
                async with aiosqlite.connect(task.database_path) as db:
                    # 强行限定 busy_timeout 延长锁等待时间
                    await db.execute("PRAGMA busy_timeout = 30000;")
                    # 显式开启 WAL 模式以实现“读写非阻塞”
                    await db.execute("PRAGMA journal_mode = WAL;")
                    try:
                        async with db.cursor() as cursor:
                            for sql, params in task.sql_queries:
                                await cursor.execute(sql, params)
                        await db.commit()
                        success = True
                    except Exception as e:
                        await db.rollback()
                        logger.error(f"独占串行写入流水线发生事务性崩溃，已执行自动回滚。错误详情: {str(e)}")
                        task.future.set_exception(e)
                    finally:
                        if not task.future.done():
                            task.future.set_result(success)
                        self.queue.task_done()
            except asyncio.CancelledError:
                logger.info("WriteWorker 任务处理循环被取消。")
                break
            except Exception as loop_err:
                logger.critical(f"WriteWorker 内部处理异常: {str(loop_err)}")
                await asyncio.sleep(1)  # 防止无限重试紧凑死循环

    async def stop(self):
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            self.worker_task = None
            logger.info("全局串行写队列优雅停机。")
