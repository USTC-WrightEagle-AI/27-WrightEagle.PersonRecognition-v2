"""Bounded RGB/depth matching without a ROS dependency."""
from collections import deque
import time


def stamp(message):
    return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9


class FrameSynchronizer:
    def __init__(self, with_depth, *, clock=time.monotonic):
        self.with_depth = with_depth
        self.clock = clock
        self.colors = deque(maxlen=16)
        self.depths = deque(maxlen=16)
        self.counts = {"rgb": 0, "depth": 0}
        self.latest = {}

    def add(self, stream, message):
        now = self.clock()
        self.counts[stream] += 1
        self.latest[stream] = (stamp(message), now)
        queue = self.colors if stream == "rgb" else self.depths
        queue.append((message, now))

    def pop(self):
        now = self.clock()
        # Keep the previous freshness requirement as well as the 50 ms
        # timestamp tolerance. A larger queue must not admit old frames.
        for queue in (self.colors, self.depths):
            while queue and now - queue[0][1] > 0.5:
                queue.popleft()
        if not self.colors:
            return None
        if not self.with_depth:
            color, _ = self.colors[-1]
            self.colors.clear()
            return color, None
        if not self.depths:
            return None
        # Prefer the newest matchable RGB frame, even if a newer RGB frame
        # has already arrived while this one's depth was in transit.
        for ci in range(len(self.colors) - 1, -1, -1):
            color = self.colors[ci][0]
            di = min(range(len(self.depths)),
                     key=lambda i: abs(stamp(self.depths[i][0]) - stamp(color)))
            depth = self.depths[di][0]
            if abs(stamp(depth) - stamp(color)) <= 0.05:
                for _ in range(ci + 1):
                    self.colors.popleft()
                for _ in range(di + 1):
                    self.depths.popleft()
                return color, depth
        return None

    def diagnostics(self):
        now = self.clock()
        parts = []
        for stream in ("rgb", "depth") if self.with_depth else ("rgb",):
            value = self.latest.get(stream)
            age = f"{now - value[1]:.2f}s" if value else "never"
            parts.append(f"{stream}: received={self.counts[stream]}, last_arrival_age={age}")
        if self.with_depth and all(s in self.latest for s in ("rgb", "depth")):
            delta = self.latest["rgb"][0] - self.latest["depth"][0]
            parts.append(f"latest_rgb_minus_depth={delta:+.3f}s (pair tolerance=0.050s)")
        return "; ".join(parts)
