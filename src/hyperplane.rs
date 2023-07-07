use crate::Vector;

pub struct HyperPlane<const N: usize> {
    pub coefficients: Vector<N>,
    pub constant: f32,
}

impl<const N: usize> HyperPlane<N> {
    pub fn point_is_above(&self, point: &Vector<N>) -> bool {
        self.coefficients.dot_product(point) + self.constant >= 0.0
    }
}
