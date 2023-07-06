use std::{
    cmp::min,
    collections::{HashMap, HashSet},
    io::BufRead,
};

use rand::{prelude::SliceRandom, seq::IteratorRandom};

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

struct HyperPlane<const N: usize> {
    coefficients: Vector<N>,
    constant: f32,
}

impl<const N: usize> HyperPlane<N> {
    pub fn point_is_above(&self, point: &Vector<N>) -> bool {
        self.coefficients.dot_product(point) + self.constant >= 0.0
    }
}

enum Node<const N: usize> {
    Inner(Box<InnerNode<N>>),
    Leaf(Box<LeafNode<N>>),
}

struct LeafNode<const N: usize>(Vec<usize>);

struct InnerNode<const N: usize> {
    hyperplane: HyperPlane<N>,
    left_node: Node<N>,
    right_node: Node<N>,
}

pub struct ANNIndex<const N: usize> {
    trees: Vec<Node<N>>,
    ids: Vec<i32>,
    values: Vec<Vector<N>>,
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
            .map(|_| Self::build_a_tree(max_size, &all_indexes, &unique_vecs))
            .collect();
        ANNIndex::<N> {
            trees,
            ids,
            values: unique_vecs,
        }
    }

    fn tree_result(
        query: Vector<N>,
        n: i32,
        tree: &Node<N>,
        candidates: &mut HashSet<usize>,
    ) -> i32 {
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
        let mut candidates = HashSet::new();
        for tree in self.trees.iter() {
            Self::tree_result(query, top_k, tree, &mut candidates);
        }
        let mut candidates = candidates
            .into_iter()
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

fn load_raw_wiki_data<const N: usize>(
    filename: &str,
    all_data: &mut Vec<Vector<N>>,
    word_to_idx_mapping: &mut HashMap<String, usize>,
    idx_to_word_mapping: &mut HashMap<usize, String>,
) {
    // wiki-news has 999,994 vectors in 300 dimensions
    let file = std::fs::File::open(filename).expect("could not read file");
    let reader = std::io::BufReader::new(file);
    let mut cur_idx: usize = 0;
    // We skip the first line that simply has metadata
    for maybe_line in reader.lines().skip(1) {
        let line = maybe_line.expect("Should decode the line");
        let mut data_on_line_iter = line.split_whitespace();
        let word = data_on_line_iter
            .next()
            .expect("Each line begins with a word");
        // Update the mappings
        word_to_idx_mapping.insert(word.to_owned(), cur_idx);
        idx_to_word_mapping.insert(cur_idx, word.to_owned());
        cur_idx += 1;
        // Parse the vector. Everything except the word on the line is the vector
        let embedding: [f32; N] = data_on_line_iter
            .map(|s| s.parse::<f32>().unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        all_data.push(Vector(embedding));
    }
}

fn build_and_benchmark_index<const N: usize>(
    my_input_data: &[Vector<N>],
    num_trees: i32,
    max_node_size: i32,
    top_k: i32,
    words_to_visualize: &[String],
    word_to_idx_mapping: &HashMap<String, usize>,
    idx_to_word_mapping: &HashMap<usize, String>,
    sample_idx: Option<&Vec<i32>>,
) -> Vec<HashSet<i32>> {
    println!(
        "dimensions={}, num_trees={}, max_node_size={}, top_k={}",
        N, num_trees, max_node_size, top_k
    );
    // Build the index
    let start = std::time::Instant::now();
    let my_ids: Vec<i32> = (0..my_input_data.len() as i32).collect();
    let index = ANNIndex::<N>::build_index(num_trees, max_node_size, &my_input_data, &my_ids);
    let duration = start.elapsed();
    println!("Built ANN index in {}-D in {:?}", N, duration);
    // Benchmark it with 1000 sequential queries
    let benchmark_idx: Vec<i32> =
        (0..my_input_data.len() as i32).choose_multiple(&mut rand::thread_rng(), 1000);
    let mut search_vectors: Vec<Vector<N>> = Vec::new();
    for idx in benchmark_idx {
        search_vectors.push(my_input_data[idx as usize]);
    }
    let start = std::time::Instant::now();
    for i in 0..1000 {
        index.search_approximate(search_vectors[i], top_k);
    }
    let duration = start.elapsed() / 1000;
    println!("Bulk ANN-search in {}-D has average time {:?}", N, duration);
    // Visualize some words
    for word in words_to_visualize.iter() {
        println!("Currently visualizing {}", word);
        let word_index = word_to_idx_mapping[word];
        let embedding = my_input_data[word_index];
        let nearby_idx_and_distance = index.search_approximate(embedding, top_k);
        for &(idx, distance) in nearby_idx_and_distance.iter() {
            println!(
                "{}, distance={}",
                idx_to_word_mapping[&(idx as usize)],
                distance.sqrt()
            );
        }
    }
    // If [sample_idx] provided, only find the top_k neighbours for those
    // and return that data. Otherwise, find it for all vectors in the
    // corpus. When benchmarking other hyper-parameters, we use a smaller
    // [sample_idx] set to control run-time and get efficient estimates
    // of performance metrics.
    let start = std::time::Instant::now();
    let mut subset: Vec<Vector<N>> = Vec::new();
    let sample_from_my_data = match sample_idx {
        Some(sample_indices) => {
            for &idx in sample_indices {
                subset.push(my_input_data[idx as usize]);
            }
            &subset
        }
        None => my_input_data,
    };
    let index_results: Vec<HashSet<i32>> = sample_from_my_data
        .iter()
        .map(|&vector| search_approximate_as_hashset(&index, vector, top_k))
        .collect();
    let duration = start.elapsed();
    println!(
        "Collected {} quality results in {:?}",
        index_results.len(),
        duration
    );
    return index_results;
}

fn search_approximate_as_hashset<const N: usize>(
    index: &ANNIndex<N>,
    vector: Vector<N>,
    top_k: i32,
) -> HashSet<i32> {
    let nearby_idx_and_distance = index.search_approximate(vector, top_k);
    let mut id_hashset = std::collections::HashSet::new();
    for &(idx, _) in nearby_idx_and_distance.iter() {
        id_hashset.insert(idx);
    }
    return id_hashset;
}

fn main() {
    const DIM: usize = 300;
    const TOP_K: i32 = 20;

    let start = std::time::Instant::now();

    let input_vec_path = format!("{}/data/wikidata.vec", env!("CARGO_MANIFEST_DIR"));
    let mut my_input_data: Vec<Vector<DIM>> = Vec::new();
    let mut word_to_idx_mapping: HashMap<String, usize> = HashMap::new();
    let mut idx_to_word_mapping: HashMap<usize, String> = HashMap::new();
    load_raw_wiki_data::<DIM>(
        &input_vec_path,
        &mut my_input_data,
        &mut word_to_idx_mapping,
        &mut idx_to_word_mapping,
    );

    let words_to_visualize: Vec<String> = ["river", "war", "love", "education"]
        .into_iter()
        .map(|x| x.to_owned())
        .collect();
    let index_results = build_and_benchmark_index::<DIM>(
        &my_input_data,
        3,
        15,
        TOP_K,
        &words_to_visualize,
        &word_to_idx_mapping,
        &idx_to_word_mapping,
        None,
    );
    println!("{index_results:?}");

    let duration = start.elapsed();
    println!("Parsed {} vectors in {:?}", my_input_data.len(), duration);
}
