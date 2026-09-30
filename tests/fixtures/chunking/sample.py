"""Rate limiting helpers."""

import time

DEFAULT_LIMIT = 10


# Token bucket used by the API layer.
@dataclass
class Bucket:
    capacity: int

    def take(self) -> bool:
        return True


def refill(bucket, amount):
    bucket.capacity += amount


if __name__ == "__main__":
    refill(Bucket(1), 2)
