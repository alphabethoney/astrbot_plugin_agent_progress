# -*- coding: utf-8 -*-
"""agent 进度播报插件。

订阅 AstrBot 内部的日志广播器，从里面挑出 trace 事件
（astr_agent_prepare / agent_tool_call / astr_agent_complete 等），
翻译成人话发到指定会话，用来在聊天里实时监督 agent 干活。

特点：
- 全程不调用大模型，零 token 成本；
- 只在任务耗时超过阈值时才开始播报，短回复不会刷屏；
- 播报按固定间隔合并，一个长任务最多几分钟一条。

依赖：把 trace_enable 打开（面板或接口均可），否则收不到事件。
"""
import asyncio
import os
import time

from astrbot.api import logger
from astrbot.api.event import MessageChain
from astrbot.api.star import Context, Star

# 兜底默认值，配置里填了则以配置为准
DEFAULT_SESSION = ""            # 形如 "aiocqhttp:FriendMessage:1234567890"，留空表示不限制会话
DEFAULT_START_AFTER = 30        # 任务跑过这么多秒才开始播报
DEFAULT_INTERVAL = 30           # 两条播报之间至少间隔这么多秒
DEFAULT_SHOW_RESULT = True      # 是否在收尾时补一条汇总

# 工具名翻译，看着顺眼些
TOOL_CN = {
    "astrbot_execute_shell": "执行命令",
    "astrbot_execute_python": "跑代码",
    "astrbot_execute_ipython": "跑代码",
    "astrbot_file_read_tool": "读文件",
    "astrbot_file_write_tool": "写文件",
    "astrbot_file_edit_tool": "改文件",
    "astrbot_grep_tool": "搜文件内容",
    "web_search_tavily": "联网搜索",
    "tavily_extract_web_page": "抓网页",
    "astr_kb_search": "查知识库",
    "recall_long_term_memory": "翻记忆",
    "send_message_to_user": "发消息",
    "future_task": "管理定时任务",
}


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} 秒"
    return f"{seconds // 60} 分 {seconds % 60} 秒"


class AgentProgress(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self._cfg = config
        self._queue = None
        self._task = None
        self._spans = {}   # span_id -> 统计信息

    # ---------------- 配置 ----------------
    def _get(self, key, default):
        try:
            if self._cfg is None:
                return default
            value = self._cfg.get(key, default)
            return default if value in (None, "") else value
        except Exception:
            return default

    @property
    def session_filter(self):
        return str(self._get("target_session", DEFAULT_SESSION))

    @property
    def start_after(self):
        try:
            return max(0, int(self._get("start_after", DEFAULT_START_AFTER)))
        except Exception:
            return DEFAULT_START_AFTER

    @property
    def interval(self):
        try:
            return max(5, int(self._get("interval", DEFAULT_INTERVAL)))
        except Exception:
            return DEFAULT_INTERVAL

    # ---------------- 拿广播器 ----------------
    @staticmethod
    def _find_broker():
        try:
            from astrbot import logger as alogger
            from astrbot.core.log import LogQueueHandler

            for handler in alogger.handlers:
                if isinstance(handler, LogQueueHandler):
                    return handler.log_broker
            from astrbot.core import LogManager

            return getattr(LogManager, "_log_broker", None)
        except Exception:
            logger.warning("[进度播报] 拿不到日志广播器", exc_info=True)
            return None

    # ---------------- 生命周期 ----------------
    async def initialize(self):
        broker = self._find_broker()
        if broker is None:
            logger.warning("[进度播报] 没找到广播器，插件不会工作")
            return
        self._queue = broker.register()
        self._task = asyncio.create_task(self._loop())
        logger.info("[进度播报] 已启动，开始订阅 trace 事件")

    async def terminate(self):
        if self._task:
            self._task.cancel()
            self._task = None
        if self._queue is not None:
            broker = self._find_broker()
            if broker is not None:
                try:
                    broker.unregister(self._queue)
                except Exception:
                    pass
            self._queue = None

    # ---------------- 主循环 ----------------
    async def _loop(self):
        while True:
            entry = await self._queue.get()
            try:
                if not isinstance(entry, dict) or entry.get("type") != "trace":
                    continue
                await self._handle(entry)
            except Exception:
                logger.warning("[进度播报] 处理事件出错", exc_info=True)

    async def _handle(self, entry: dict):
        umo = str(entry.get("umo") or "")
        if self.session_filter and self.session_filter not in umo:
            return

        action = entry.get("action") or ""
        span = entry.get("span_id") or ""
        now = time.time()

        if action == "astr_agent_prepare":
            self._spans[span] = {
                "umo": umo,
                "start": now,
                "steps": 0,
                "last_sent": 0.0,
                "notified": False,
                "last_tool": "",
            }
            return

        st = self._spans.get(span)
        if st is None:
            return

        if action == "agent_tool_call":
            st["steps"] += 1
            fields = entry.get("fields") or {}
            raw = str(fields.get("tool_name") or "")
            st["last_tool"] = TOOL_CN.get(raw, raw)
            elapsed = now - st["start"]
            if elapsed < self.start_after:
                return
            if now - st["last_sent"] < self.interval:
                return
            st["last_sent"] = now
            st["notified"] = True
            await self._say(
                umo,
                f"进度：已执行 {st['steps']} 步，当前在做「{st['last_tool']}」，"
                f"累计 {_fmt(elapsed)}",
            )
            return

        if action == "astr_agent_complete":
            self._spans.pop(span, None)
            if not st["notified"]:
                return
            await self._say(
                umo, f"完成：一共 {st['steps']} 步，用时 {_fmt(now - st['start'])}"
            )

    async def _say(self, umo: str, text: str):
        """回发到触发这条任务的那个会话本身。"""
        if not umo:
            logger.info("[进度播报] 事件里没有会话信息，内容：%s", text)
            return
        try:
            await self.context.send_message(umo, MessageChain().message(text))
        except Exception:
            logger.warning("[进度播报] 发送失败（目标 %s）: %s", umo, text, exc_info=True)
