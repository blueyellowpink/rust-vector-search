use crate::HyperPlane;

pub enum Node<const N: usize> {
    Inner(Box<InnerNode<N>>),
    Leaf(Box<LeafNode<N>>),
}

pub struct LeafNode<const N: usize>(pub Vec<usize>);

pub struct InnerNode<const N: usize> {
    pub hyperplane: HyperPlane<N>,
    pub left_node: Node<N>,
    pub right_node: Node<N>,
}
