use std::fmt;

/// A point in 2D space.
#[derive(Debug, Clone)]
pub struct Point {
    x: i32,
    y: i32,
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

pub fn origin() -> Point {
    Point { x: 0, y: 0 }
}
