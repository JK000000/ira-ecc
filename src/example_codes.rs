use std::collections::HashMap;





pub fn edge_ratios_to_node_ratios(edge_ratios: HashMap<usize, f64>) -> HashMap<usize, f64> {
    let mut norm = 0.0;

    for (&i, &val) in edge_ratios.iter() {
        norm += val / i as f64;
    }

    HashMap::from_iter(edge_ratios.into_iter().map(|(i, val)| {
        (i, (val / i as f64) / norm)
    }))
}
