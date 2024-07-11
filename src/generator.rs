use std::collections::HashMap;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use crate::IraEccCodeCompressedRepresentation;

fn shuffle(data: &mut [usize], gen: &mut impl Rng) {
    for i in 0..(data.len() - 1) {
        let idx = gen.gen_range(i..data.len());
        if i != idx {
            data.swap(idx, i);
        }
    }
}

pub struct IraCodeGenerator {
    pub message_len: usize,
    pub check_nodes_left_degree: usize,
    pub message_nodes_degree_ratios: HashMap<usize, f64>,
}

impl IraCodeGenerator {
    pub fn generate(&self, seed: u64) -> IraEccCodeCompressedRepresentation {
        assert!((1.0 - self.message_nodes_degree_ratios.iter().map(|(_, &x)| x).sum::<f64>()) < 1e-8);

        let mut rng = ChaChaRng::seed_from_u64(seed);

        let mut message_node_degrees = Vec::new();

        'outer: for (&quant, &ratio) in self.message_nodes_degree_ratios.iter() {
            let num = (self.message_len as f64 * ratio).round() as usize;

            for _ in 0..num {
                if message_node_degrees.len() == self.message_len {
                    break 'outer;
                }
                message_node_degrees.push(quant);
            }
        }

        let avg_node_degree = (message_node_degrees.iter().sum::<usize>() as f64 / message_node_degrees.len() as f64).round() as usize;

        while message_node_degrees.len() < self.message_len {
            message_node_degrees.push(avg_node_degree);
        }

        let degree_sum = message_node_degrees.iter().sum::<usize>();

        shuffle(&mut message_node_degrees, &mut rng);

        let connections_to_add = (self.check_nodes_left_degree - (degree_sum % self.check_nodes_left_degree)) % self.check_nodes_left_degree;

        for _ in 0..connections_to_add {
            let idx = rng.gen_range(0..message_node_degrees.len());
            message_node_degrees[idx] += 1;
        }

        let num_check_nodes = (degree_sum + connections_to_add) / self.check_nodes_left_degree;

        let mut check_nodes_connections = vec![self.message_len, num_check_nodes + self.message_len, self.check_nodes_left_degree];

        for (idx, &quant) in message_node_degrees.iter().enumerate() {
            for _ in 0..quant {
                check_nodes_connections.push(idx);
            }
        }

        shuffle(&mut check_nodes_connections[3..], &mut rng);

        IraEccCodeCompressedRepresentation {
            data: check_nodes_connections
        }
    }
}