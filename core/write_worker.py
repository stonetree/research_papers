# -*- coding: utf-8 -*-
import asyncio
import logging
import aiosqlite
from typing import List, Tuple, Any, Dict

logger = logging.getLogger("WriteWorker")

class WriteTask:
    def __init__(self, sql_queries: List[Tuple[str, Tuple[Any, ...]]], database_path: str, loop: asyncio.AbstractEventLoop):
        self.sql_queries = sql_queries  # List of tuples (sql_string, params_tuple)
        self.database_path = database_path
        self.future = loop.create_future()

class WriteWorker:
    _instance = None

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
        self._loop_queues: Dict[int, Dict[str, Any]] = {}
        self._initialized = True

    def start(self):
        """向后兼容方法：触发当前 Loop 队列的自动激活"""
        try:
            self._get_queue_for_current_loop()
        except Exception:
            pass

    def _get_queue_for_current_loop(self) -> asyncio.Queue:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError("必须在活跃的 asyncio 事件循环中调用 execute_write")

        loop_id = id(loop)
        if loop_id not in self._loop_queues or self._loop_queues[loop_id]["task"].done():
            q = asyncio.Queue()
            t = loop.create_task(self._processing_loop(q))
            self._loop_queues[loop_id] = {"queue": q, "task": t, "loop": loop}
            logger.info(f"为 EventLoop(id={loop_id}) 激活独占串行写队列消费者。")
        return self._loop_queues[loop_id]["queue"]

    async def execute_write(self, sql_queries: List[Tuple[str, Tuple[Any, ...]]]) -> bool:
        """
        将写入任务放入当前 EventLoop 独占的串行写入队列中，等待提交完成。
        """
        loop = asyncio.get_running_loop()
        queue = self._get_queue_for_current_loop()
        task = WriteTask(sql_queries, self.database_path, loop)
        await queue.put(task)
        return await task.future

    async def _processing_loop(self, queue: asyncio.Queue):
        while True:
            try:
                task = await queue.get()
                success = False
                async with aiosqlite.connect(task.database_path) as db:
                    await db.execute("PRAGMA busy_timeout = 30000;")
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
                        if not task.future.done():
                            task.future.set_exception(e)
                    finally:
                        if not task.future.done():
                            task.future.set_result(success)
                        queue.task_done()
            except asyncio.CancelledError:
                logger.info("WriteWorker 任务处理循环被取消。")
                break
            except Exception as loop_err:
                logger.critical(f"WriteWorker 内部处理异常: {str(loop_err)}")
                await asyncio.sleep(0.5)

    async def stop(self):
        for loop_id, entry in list(self._loop_queues.items()):
            t = entry.get("task")
            if t and not t.done():
                t.cancel()
        self._loop_queues.clear()
        logger.info("全局串行写队列优雅停机。")
