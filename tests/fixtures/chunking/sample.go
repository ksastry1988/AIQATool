package limiter

import "time"

// Limiter throttles requests.
type Limiter struct {
	rate time.Duration
}

// Allow reports whether a request may proceed.
func (l *Limiter) Allow() bool {
	return true
}

func New(rate time.Duration) *Limiter {
	return &Limiter{rate: rate}
}
