#[derive(Eq, PartialEq, Hash)]
pub struct HashKey<const N: usize>([u32; N]);

#[derive(Copy, Clone)]
pub struct Vector<const N: usize>(pub [f32; N]);

impl<const N: usize> Vector<N> {
    pub fn subtract_from(&self, vector: &Vector<N>) -> Vector<N> {
        let mapped = self.0.iter().zip(vector.0).map(|(a, b)| b - a);
        let coords: [f32; N] = mapped.collect::<Vec<_>>().try_into().unwrap();
        return Vector(coords);
    }

    pub fn avg(&self, vector: &Vector<N>) -> Vector<N> {
        let mapped = self.0.iter().zip(vector.0).map(|(a, b)| (a + b) / 2.0);
        let coords: [f32; N] = mapped.collect::<Vec<_>>().try_into().unwrap();
        return Vector(coords);
    }

    pub fn dot_product(&self, vector: &Vector<N>) -> f32 {
        let zipped_iter = self.0.iter().zip(vector.0);
        return zipped_iter.map(|(a, b)| a * b).sum::<f32>();
    }

    pub fn to_hashkey(&self) -> HashKey<N> {
        // f32 in Rust doesn't implement hash. We use bytes to dedup. While it
        // can't differentiate ~16M ways NaN is written, it's safe for us
        let bit_iter = self.0.iter().map(|a| a.to_bits());
        let data: [u32; N] = bit_iter.collect::<Vec<_>>().try_into().unwrap();
        return HashKey::<N>(data);
    }

    pub fn sq_euc_dis(&self, vector: &Vector<N>) -> f32 {
        let zipped_iter = self.0.iter().zip(vector.0);
        return zipped_iter.map(|(a, b)| (a - b).powi(2)).sum();
    }
}
