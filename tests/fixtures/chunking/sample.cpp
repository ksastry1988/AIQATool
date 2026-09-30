#include <string>

namespace net {

class Socket {
 public:
  void open();
};

template <typename T>
T clamp(T value, T lo, T hi) {
  return value < lo ? lo : value > hi ? hi : value;
}

void Socket::open() {}

}  // namespace net
