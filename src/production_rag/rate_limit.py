from threading import Lock
from time import monotonic
from math import ceil


class FixedWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        if max_requests < 1 or window_seconds <= 0:
            raise ValueError("Rate limit settings must be positive.")

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.windows: dict[tuple[str, str], tuple[float, int]] = {}
        self.lock = Lock()

    def allow(self, tenant_id: str, user_id: str) -> bool:
        now = monotonic()
        identity = (tenant_id, user_id)

        with self.lock:
            expired = [
                key
                for key, (started_at, _) in self.windows.items()
                if now - started_at >= self.window_seconds
            ]
            for key in expired:
                del self.windows[key]

            started_at, count = self.windows.get(identity, (now, 0))
            if count >= self.max_requests:
                return False

            self.windows[identity] = (started_at, count + 1)
            return True

    def retry_after_seconds(self, tenant_id: str, user_id: str) -> int:
        identity = (tenant_id, user_id)
        with self.lock:
            window = self.windows.get(identity)
            if window is None:
                return 0
            started_at, _ = window
            remaining = self.window_seconds - (monotonic() - started_at)
            return max(1, ceil(remaining))
