# -*- coding: utf-8 -*-
"""agent 进度播报插件。

订阅 AstrBot 内部的日志广播器，从里面挑出 trace 事件
（astr_agent_prepare / agent_tool_call / agent_tool_result / astr_agent_complete），
翻译成人话发到指定会话，用来在聊天里实时监督 agent 干活。

播报策略（按事件性质，默认只报这四类，其余时间保持静默）：
1. 阶段切换：工具按用途归类，类别变了才报，同类连续调用不报；
2. 异常：工具返回错误立刻报；
3. 疑似卡住：一段时间没有任何新事件就报一条；
4. 结束：带答案摘要与 Token 用量的汇总（沿用原有逻辑）。

旧的「按时间间隔刷进度」降级为可选，由 enable_interval_report 开关控制，默认关闭。

特点：
- 全程不调用大模型，零 token 成本；
- 默认按事件性质播报，不再按固定时间刷屏；
- 依赖 trace_enable 打开（面板或接口均可），否则收不到事件。
"""
import ast
import asyncio
import time

from astrbot.api import logger
from astrbot.api.event import MessageChain
from astrbot.api.star import Context, Star

# 兜底默认值，配置里填了则以配置为准
DEFAULT_SESSION = ""                 # 形如 "平台名:消息类型:会话号"，留空表示不限制会话
DEFAULT_START_AFTER = 30             # 旧的时间刷：任务跑过这么多秒才开始播报
DEFAULT_INTERVAL = 30                # 旧的时间刷：两条播报之间至少间隔这么多秒
DEFAULT_SHOW_RESULT = True           # 是否在收尾时补一条汇总
DEFAULT_QUIET_MODE = False           # 安静模式：只报异常、卡住、结束，不报阶段切换
DEFAULT_PHASE_MIN_INTERVAL = 5       # 阶段切换判定阈值：距上次播报切换不足这么多秒则不重复报
DEFAULT_STUCK_TIMEOUT = 120          # 超过这么多秒没有任何新事件就报一条疑似卡住
DEFAULT_ENABLE_INTERVAL_REPORT = False  # 是否启用旧的按时间间隔刷进度（默认关闭）
DEFAULT_RESULT_ANSWER_MIN_LEN = 80   # 答案短于这么多字符就不在汇总里附「答案：」一行
DEFAULT_RESULT_MIN_STEPS = 3         # 收尾汇总门槛：步数达到这么多就发
DEFAULT_RESULT_MIN_SECONDS = 30      # 收尾汇总门槛：耗时达到这么多秒就发

# 事件类型 / 动作名 / 字段名，集中维护，降低上游日志结构变更的脆弱性
EVENT_TYPE_TRACE = "trace"
ACTION_PREPARE = "astr_agent_prepare"
ACTION_TOOL_CALL = "agent_tool_call"
ACTION_TOOL_RESULT = "agent_tool_result"
ACTION_COMPLETE = "astr_agent_complete"
FIELD_TYPE = "type"
FIELD_UMO = "umo"
FIELD_ACTION = "action"
FIELD_SPAN_ID = "span_id"
FIELD_FIELDS = "fields"
FIELD_TOOL_NAME = "tool_name"
FIELD_TOOL_RESULT = "tool_result"
FIELD_RESP = "resp"
FIELD_STATS = "stats"
FIELD_TOKEN_USAGE = "token_usage"
FIELD_TOTAL = "total"

# 工具归类表：tool_name -> 类别 key，未列出的工具一律归入 other。
# 阶段切换只看类别，不看具体是哪个工具，所以同类别的工具连续调用不会触发播报。
TOOL_CATEGORY = {
    # 执行命令 / 跑代码
    "astrbot_execute_shell": "command",
    "astrbot_execute_python": "command",
    "astrbot_execute_ipython": "command",
    # 文件读写
    "astrbot_file_read_tool": "file",
    "astrbot_file_write_tool": "file",
    "astrbot_file_edit_tool": "file",
    "astrbot_grep_tool": "file",
    # 联网请求
    "web_search_tavily": "network",
    "tavily_extract_web_page": "network",
}
DEFAULT_CATEGORY = "other"
# 类别 key -> 中文展示名
CATEGORY_CN = {
    "command": "执行命令",
    "file": "文件读写",
    "network": "联网请求",
    "other": "其它",
}
# 兜底工具名翻译，看着顺眼些（只看具体工具名时用）
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

# 判断工具结果是否为错误的关键字（小写匹配）
ERROR_MARKERS = (
    "error:",
    "error：",
    "traceback (most recent call last)",
)

# 完成通知里答案摘要最多贴多少字符
RESULT_PREVIEW_LEN = 120
# 异常播报里错误文本最多贴多少字符
ERROR_PREVIEW_LEN = 160

# span 超过这么久没有任何动静就淘汰，避免 _spans 无界增长
SPAN_TTL = 1800


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} 秒"
    return f"{seconds // 60} 分 {seconds % 60} 秒"


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _tool_name(fields: dict) -> str:
    """从 tool_call 事件里取出工具名。

    上游有时把整个 {id,name,args,ts} dict 塞进 tool_name，这里做兼容。
    """
    raw = fields.get(FIELD_TOOL_NAME)
    if isinstance(raw, dict):
        name = raw.get("name")
        if name:
            raw = name
    return str(raw or "")


def _category_of(raw_tool: str) -> str:
    """工具名 -> 类别 key。"""
    return TOOL_CATEGORY.get(raw_tool, DEFAULT_CATEGORY)


def _category_cn(category: str) -> str:
    return CATEGORY_CN.get(category, CATEGORY_CN[DEFAULT_CATEGORY])


def _tool_cn(raw_tool: str) -> str:
    return TOOL_CN.get(raw_tool, raw_tool or "未知工具")


def _tool_result_text(fields: dict) -> str:
    """取出工具结果文本。

    上游记录的是 Json 组件的 Python 字典字符串，形如
    "{'id': ..., 'result': '...'}"，能解析就只取 result，否则原样返回。
    """
    raw = fields.get(FIELD_TOOL_RESULT)
    text = raw if isinstance(raw, str) else str(raw or "")
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            data = ast.literal_eval(stripped)
            if isinstance(data, dict):
                result = data.get("result")
                if isinstance(result, str):
                    return result
        except Exception:
            pass
    return text


def _looks_like_error(text: str) -> bool:
    low = text.strip().lower()
    if not low:
        return False
    if low.startswith("error"):
        return True
    return any(marker in low for marker in ERROR_MARKERS)


def _answer_raw(fields: dict) -> str:
    """取出答案原文（未截断），用于判断是否达到附摘要的长度门槛。"""
    resp = fields.get(FIELD_RESP)
    if not isinstance(resp, str):
        return ""
    return resp.strip()


def _answer_preview(fields: dict) -> str:
    text = _answer_raw(fields)
    if not text:
        return ""
    if len(text) > RESULT_PREVIEW_LEN:
        return text[:RESULT_PREVIEW_LEN] + "…"
    return text


def _token_total(fields: dict):
    stats = fields.get(FIELD_STATS)
    if not isinstance(stats, dict):
        return None
    usage = stats.get(FIELD_TOKEN_USAGE)
    if isinstance(usage, dict):
        total = usage.get(FIELD_TOTAL)
    else:
        total = getattr(usage, FIELD_TOTAL, None)
    if isinstance(total, bool):
        return None
    if isinstance(total, (int, float)):
        return int(total)
    return None


def _result_summary(st: dict, fields: dict, now: float, answer_min_len: int) -> str:
    parts = [f"完成：一共 {st['steps']} 步，用时 {_fmt(now - st['start'])}"]
    # 答案过短就不附，避免和 AstrBot 真正发出的回复撞车
    if len(_answer_raw(fields)) >= answer_min_len:
        answer = _answer_preview(fields)
        if answer:
            parts.append(f"答案：{answer}")
    usage = _token_total(fields)
    if usage is not None:
        parts.append(f"Token 用量：{usage}")
    return "\n".join(parts)


class AgentProgress(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self._cfg = config
        self._broker = None
        self._queue = None
        self._task = None
        self._watchdog_task = None
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

    @property
    def show_result(self):
        try:
            return bool(self._get("show_result", DEFAULT_SHOW_RESULT))
        except Exception:
            return DEFAULT_SHOW_RESULT

    @property
    def quiet_mode(self):
        """安静模式：只报异常、卡住、结束。"""
        try:
            return bool(self._get("quiet_mode", DEFAULT_QUIET_MODE))
        except Exception:
            return DEFAULT_QUIET_MODE

    @property
    def phase_min_interval(self):
        """阶段切换判定阈值：距上次播报切换不足这么多秒则不重复报。"""
        try:
            return max(0, int(self._get("phase_min_interval", DEFAULT_PHASE_MIN_INTERVAL)))
        except Exception:
            return DEFAULT_PHASE_MIN_INTERVAL

    @property
    def stuck_timeout(self):
        """超过这么多秒没有任何新事件就报一条疑似卡住。"""
        try:
            return max(5, int(self._get("stuck_timeout", DEFAULT_STUCK_TIMEOUT)))
        except Exception:
            return DEFAULT_STUCK_TIMEOUT

    @property
    def enable_interval_report(self):
        """是否启用旧的按时间间隔刷进度。"""
        try:
            return bool(self._get("enable_interval_report", DEFAULT_ENABLE_INTERVAL_REPORT))
        except Exception:
            return DEFAULT_ENABLE_INTERVAL_REPORT

    @property
    def watchdog_every(self):
        """卡住检测的轮询间隔，取超时值的四分之一，限制在 1~10 秒。"""
        return max(1, min(10, self.stuck_timeout // 4))

    @property
    def result_answer_min_len(self):
        """答案短于这么多字符就不在汇总里附「答案：」一行。"""
        try:
            return max(0, int(self._get("result_answer_min_len", DEFAULT_RESULT_ANSWER_MIN_LEN)))
        except Exception:
            return DEFAULT_RESULT_ANSWER_MIN_LEN

    @property
    def result_min_steps(self):
        """收尾汇总门槛：步数达到这么多就发。"""
        try:
            return max(0, int(self._get("result_min_steps", DEFAULT_RESULT_MIN_STEPS)))
        except Exception:
            return DEFAULT_RESULT_MIN_STEPS

    @property
    def result_min_seconds(self):
        """收尾汇总门槛：耗时达到这么多秒就发。"""
        try:
            return max(0, int(self._get("result_min_seconds", DEFAULT_RESULT_MIN_SECONDS)))
        except Exception:
            return DEFAULT_RESULT_MIN_SECONDS

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
        self._broker = self._find_broker()
        if self._broker is None:
            logger.warning("[进度播报] 没找到广播器，插件不会工作")
            return
        self._queue = self._broker.register()
        self._task = asyncio.create_task(self._loop())
        self._watchdog_task = asyncio.create_task(self._watchdog())
        logger.info("[进度播报] 已启动，开始订阅 trace 事件")

    async def terminate(self):
        for attr in ("_task", "_watchdog_task"):
            task = getattr(self, attr)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                setattr(self, attr, None)
        if self._queue is not None:
            if self._broker is not None:
                try:
                    self._broker.unregister(self._queue)
                except Exception:
                    pass
            self._queue = None

    # ---------------- 事件主循环 ----------------
    async def _loop(self):
        while True:
            entry = await self._queue.get()
            try:
                if not isinstance(entry, dict) or entry.get(FIELD_TYPE) != EVENT_TYPE_TRACE:
                    continue
                await self._handle(entry)
            except Exception:
                logger.warning("[进度播报] 处理事件出错", exc_info=True)

    # ---------------- 卡住看门狗 ----------------
    async def _watchdog(self):
        """没有事件进来时也要能发现卡住，所以另起一个定时轮询。"""
        while True:
            await asyncio.sleep(self.watchdog_every)
            try:
                now = time.monotonic()
                self._sweep(now)
                await self._check_stuck(now)
            except Exception:
                logger.warning("[进度播报] 卡住检测出错", exc_info=True)

    async def _check_stuck(self, now: float):
        for st in list(self._spans.values()):
            if st["stuck_reported"]:
                continue
            idle = now - st["last_seen"]
            if idle < self.stuck_timeout:
                continue
            st["stuck_reported"] = True
            st["notified"] = True
            await self._say(
                st["umo"],
                f"疑似卡住：已 {_fmt(idle)} 没有新动静，"
                f"最后在做「{st['last_tool'] or '未知'}」",
            )

    async def _handle(self, entry: dict):
        now = time.monotonic()
        self._sweep(now)
        umo = str(entry.get(FIELD_UMO) or "")
        if self.session_filter and self.session_filter != umo:
            return
        action = entry.get(FIELD_ACTION) or ""
        span = entry.get(FIELD_SPAN_ID) or ""
        if not span:
            return
        if action == ACTION_PREPARE:
            self._spans[span] = self._new_span(umo, now)
            return
        st = self._spans.get(span)
        if st is None:
            return
        # 有任何新事件就刷新活跃时间，并允许再次判定卡住
        st["last_seen"] = now
        st["stuck_reported"] = False
        if action == ACTION_TOOL_CALL:
            await self._report_tool(st, entry, umo, now)
            return
        if action == ACTION_TOOL_RESULT:
            await self._report_error(st, entry, now)
            return
        if action == ACTION_COMPLETE:
            self._spans.pop(span, None)
            if st["notified"] and self.show_result:
                # 汇总门槛：步数或耗时任一达到各自阈值才发，两个都没到就静默收尾。
                # 耗时用单调时钟（now 与 st["start"] 都来自 time.monotonic），不受墙钟跳变影响。
                elapsed = now - st["start"]
                if st["steps"] >= self.result_min_steps or elapsed >= self.result_min_seconds:
                    fields = entry.get(FIELD_FIELDS) or {}
                    await self._say(
                        st["umo"],
                        _result_summary(st, fields, now, self.result_answer_min_len),
                    )

    def _new_span(self, umo: str, now: float) -> dict:
        return {
            "umo": umo,
            "start": now,
            "steps": 0,
            "last_sent": 0.0,
            "notified": False,
            "last_tool": "",
            "last_category": "",
            "last_phase_at": 0.0,
            "last_seen": now,
            "stuck_reported": False,
        }

    # ---------------- 阶段切换 / 可选时间刷 ----------------
    async def _report_tool(self, st: dict, entry: dict, umo: str, now: float):
        st["steps"] += 1
        fields = entry.get(FIELD_FIELDS) or {}
        raw = _tool_name(fields)
        st["last_tool"] = _tool_cn(raw)
        category = _category_of(raw)
        # 先判阶段切换；切换播报了就不再叠一条时间刷，避免同一次调用发两条
        if await self._report_phase(st, umo, category, now):
            return
        await self._report_interval(st, umo, now)

    async def _report_phase(self, st: dict, umo: str, category: str, now: float) -> bool:
        """工具类别发生变化时报一条阶段消息，返回是否真的发了。"""
        if self.quiet_mode:
            return False
        if category == st["last_category"]:
            return False
        prev = st["last_category"]
        st["last_category"] = category
        # 阈值：距上次播报切换太近就不重复报，抑制类别来回横跳
        if now - st["last_phase_at"] < self.phase_min_interval:
            return False
        st["last_phase_at"] = now
        st["notified"] = True
        elapsed = now - st["start"]
        if not prev:
            text = (
                f"阶段：开始「{_category_cn(category)}」，"
                f"当前工具「{st['last_tool']}」，累计 {_fmt(elapsed)}"
            )
        else:
            text = (
                f"阶段：从「{_category_cn(prev)}」切换到「{_category_cn(category)}」，"
                f"当前工具「{st['last_tool']}」，第 {st['steps']} 步，累计 {_fmt(elapsed)}"
            )
        await self._say(umo, text)
        return True

    async def _report_interval(self, st: dict, umo: str, now: float):
        """旧的按时间间隔刷进度，只有开关打开时才走这里。"""
        if not self.enable_interval_report:
            return
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

    # ---------------- 异常 ----------------
    async def _report_error(self, st: dict, entry: dict, now: float):
        fields = entry.get(FIELD_FIELDS) or {}
        text = _tool_result_text(fields)
        if not _looks_like_error(text):
            return
        st["notified"] = True
        await self._say(
            st["umo"],
            f"异常：工具「{st['last_tool'] or '未知'}」返回错误："
            f"{_truncate(text, ERROR_PREVIEW_LEN)}",
        )

    def _sweep(self, now: float):
        stale = [s for s, st in self._spans.items() if now - st["last_seen"] > SPAN_TTL]
        for span in stale:
            self._spans.pop(span, None)

    async def _say(self, umo: str, text: str):
        """回发到触发这条任务的那个会话本身。"""
        if not umo:
            logger.info("[进度播报] 事件里没有会话信息，未播报（内容 %d 字）", len(text))
            return
        try:
            await self.context.send_message(umo, MessageChain().message(text))
        except Exception:
            logger.warning(
                "[进度播报] 发送失败（目标 %s，内容 %d 字）", umo, len(text), exc_info=True
            )
