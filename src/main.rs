use std::{
    collections::{HashMap, HashSet},
    io::BufRead,
};

use rand::seq::IteratorRandom;
use rayon::prelude::*;

use rust_vector_search::{vector::Vector, ANNIndex};

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
    let num_sample = 10; // num sample to benchmark

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
        (0..my_input_data.len() as i32).choose_multiple(&mut rand::thread_rng(), num_sample);
    let mut search_vectors: Vec<Vector<N>> = Vec::new();
    for idx in benchmark_idx {
        search_vectors.push(my_input_data[idx as usize]);
    }
    let start = std::time::Instant::now();
    for i in 0..num_sample {
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
        .par_iter()
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
    const TOP_K: i32 = 10;
    const NUM_TREES: i32 = 50;
    const MAX_NODE_SIZE: i32 = 5;

    let start = std::time::Instant::now();

    let input_vec_path = format!("{}/data/test.vec", env!("CARGO_MANIFEST_DIR"));
    let mut my_input_data: Vec<Vector<DIM>> = Vec::new();
    let mut word_to_idx_mapping: HashMap<String, usize> = HashMap::new();
    let mut idx_to_word_mapping: HashMap<usize, String> = HashMap::new();
    load_raw_wiki_data::<DIM>(
        &input_vec_path,
        &mut my_input_data,
        &mut word_to_idx_mapping,
        &mut idx_to_word_mapping,
    );

    let words_to_visualize: Vec<String> = ["do", "now", "because", "here"]
        .into_iter()
        .map(|x| x.to_owned())
        .collect();
    let index_results = build_and_benchmark_index::<DIM>(
        &my_input_data,
        NUM_TREES,
        MAX_NODE_SIZE,
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
