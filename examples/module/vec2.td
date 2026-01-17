pub struct Vec2 {
    x: i64,
    y: i64,
}

pub fn new(x: i64, y: i64) -> Vec2;

pub fn add(a: *Vec2, b: *Vec2) -> Vec2;

pub fn dot(a: *Vec2, b: *Vec2) -> i64;
