pub mod hyperplane;
pub mod tree;
pub mod vector;

use std::{cmp::min, collections::HashSet};

use dashmap::DashSet;
use rand::prelude::SliceRandom;
use rayon::prelude::*;

use crate::hyperplane::HyperPlane;
use crate::tree::{InnerNode, LeafNode, Node};
use crate::vector::Vector;

pub struct ANNIndex<const N: usize> {
    pub trees: Vec<Node<N>>,
    pub ids: Vec<i32>,
    pub values: Vec<Vector<N>>,
}

impl<const N: usize> ANNIndex<N> {
    fn build_hyperplane(
        indexes: &[usize],
        all_vecs: &[Vector<N>],
    ) -> (HyperPlane<N>, Vec<usize>, Vec<usize>) {
        let sample: Vec<&usize> = indexes
            .choose_multiple(&mut rand::thread_rng(), 2)
            .collect();
        // cartesian eq for hyperplane n * (x - x_0) = 0
        // n (normal vector) is the coefs x_1 to x_n
        let (a, b) = (*sample[0], *sample[1]);
        let coefficients = all_vecs[a].subtract_from(&all_vecs[b]);
        let point_on_plane = all_vecs[a].avg(&all_vecs[b]);
        let constant = -coefficients.dot_product(&point_on_plane);
        let hyperplane = HyperPlane::<N> {
            coefficients,
            constant,
        };
        let (mut above, mut below) = (vec![], vec![]);
        for &id in indexes.iter() {
            if hyperplane.point_is_above(&all_vecs[id]) {
                above.push(id)
            } else {
                below.push(id)
            };
        }
        (hyperplane, above, below)
    }

    fn build_a_tree(max_size: i32, indexes: &[usize], all_vecs: &[Vector<N>]) -> Node<N> {
        if indexes.len() <= (max_size as usize) {
            return Node::Leaf(Box::new(LeafNode::<N>(indexes.to_vec())));
        }
        let (plane, above, below) = Self::build_hyperplane(indexes, all_vecs);
        let node_above = Self::build_a_tree(max_size, &above, all_vecs);
        let node_below = Self::build_a_tree(max_size, &below, all_vecs);
        Node::Inner(Box::new(InnerNode::<N> {
            hyperplane: plane,
            left_node: node_below,
            right_node: node_above,
        }))
    }

    fn deduplicate(
        vectors: &[Vector<N>],
        ids: &[i32],
        dedup_vectors: &mut Vec<Vector<N>>,
        ids_of_dedup_vectors: &mut Vec<i32>,
    ) {
        let mut hashes_seen = HashSet::new();
        for i in 1..vectors.len() {
            let hash_key = vectors[i].to_hashkey();
            if !hashes_seen.contains(&hash_key) {
                hashes_seen.insert(hash_key);
                dedup_vectors.push(vectors[i]);
                ids_of_dedup_vectors.push(ids[i]);
            }
        }
    }

    pub fn build_index(
        num_trees: i32,
        max_size: i32,
        vecs: &[Vector<N>],
        vec_ids: &[i32],
    ) -> ANNIndex<N> {
        let (mut unique_vecs, mut ids) = (vec![], vec![]);
        Self::deduplicate(vecs, vec_ids, &mut unique_vecs, &mut ids);

        // Trees hold an index into the [unique_vecs] list which is not
        // necessarily its id, if duplicates existed
        let all_indexes: Vec<usize> = (0..unique_vecs.len()).collect();
        let trees: Vec<_> = (0..num_trees)
            .into_par_iter()
            .map(|_| Self::build_a_tree(max_size, &all_indexes, &unique_vecs))
            .collect();

        ANNIndex::<N> {
            trees,
            ids,
            values: unique_vecs,
        }
    }

    fn tree_result(query: Vector<N>, n: i32, tree: &Node<N>, candidates: &DashSet<usize>) -> i32 {
        // take everything in node, if still needed, take from alternate subtree
        match tree {
            Node::Leaf(box_leaf) => {
                let leaf_values = &(box_leaf.0);
                let num_candidates_found = min(n as usize, leaf_values.len());
                for i in 0..num_candidates_found {
                    candidates.insert(leaf_values[i]);
                }
                return num_candidates_found as i32;
            }
            Node::Inner(inner) => {
                let above = (*inner).hyperplane.point_is_above(&query);
                let (main, backup) = match above {
                    true => (&(inner.right_node), &(inner.left_node)),
                    false => (&(inner.left_node), &(inner.right_node)),
                };
                match Self::tree_result(query, n, main, candidates) {
                    k if k < n => k + Self::tree_result(query, n - k, backup, candidates),
                    k => k,
                }
            }
        }
    }

    pub fn search_approximate(&self, query: Vector<N>, top_k: i32) -> Vec<(i32, f32)> {
        let candidates = DashSet::new();

        self.trees.par_iter().for_each(|tree| {
            Self::tree_result(query, top_k, tree, &candidates);
        });

        let mut candidates = candidates
            .into_par_iter()
            .map(|idx| (idx, self.values[idx].sq_euc_dis(&query)))
            .collect::<Vec<_>>();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        candidates
            .into_iter()
            .take(top_k as usize)
            .map(|(idx, dis)| (self.ids[idx] as i32, dis))
            .collect()
    }
}
