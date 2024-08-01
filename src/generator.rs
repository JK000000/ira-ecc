use std::cmp::Reverse;
use std::collections::HashMap;
use priority_queue::PriorityQueue;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaChaRng;
use crate::{CheckNodeType, IraEccCodeCompressedRepresentation};

fn shuffle<T>(data: &mut [T], gen: &mut impl Rng) {
    for i in 0..(data.len() - 1) {
        let idx = gen.gen_range(i..data.len());
        if i != idx {
            data.swap(idx, i);
        }
    }
}

pub struct ClassicIraCodeGenerator {
    pub message_len: usize,
    pub check_nodes_left_degree: usize,
    pub message_nodes_degree_ratios: HashMap<usize, f64>,
}

impl ClassicIraCodeGenerator {
    pub fn generate(&self, seed: u64) -> IraEccCodeCompressedRepresentation {
        assert!((1.0 - self.message_nodes_degree_ratios.iter().map(|(_, &x)| x).sum::<f64>()) < 1e-8);

        let mut act_seed = [0u8; 32];

        act_seed[0..8].copy_from_slice(&seed.to_le_bytes()[..]);

        let mut rng = ChaChaRng::from_seed(act_seed);

        let mut message_node_degrees = Vec::new();

        let mut degree_ratios_sorted: Vec<_> = self.message_nodes_degree_ratios.iter().map(|(&x, &y)| (x, y)).collect();

        degree_ratios_sorted.sort_unstable_by_key(|(x, _)| *x);

        'outer: for (quant, ratio) in degree_ratios_sorted.into_iter() {
            let num = (self.message_len as f64 * ratio).round() as usize;

            for _ in 0..num {
                if message_node_degrees.len() == self.message_len {
                    break 'outer;
                }
                message_node_degrees.push(quant);
            }
        }

        let avg_node_degree = message_node_degrees.iter().sum::<usize>() as f64 / message_node_degrees.len() as f64;
        let avg_node_degree = avg_node_degree.round() as usize;

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


        let mut res = IraEccCodeCompressedRepresentation {
            data: vec![0usize; 3 + num_check_nodes + num_check_nodes * (self.check_nodes_left_degree + 2) - 1]
        };

        res.set_message_len(self.message_len);
        res.set_encoded_len(self.message_len + num_check_nodes);
        res.set_num_check_nodes(num_check_nodes);

        res[0] = num_check_nodes;

        for i in 1..num_check_nodes {
            res[i] = num_check_nodes + (i * (self.check_nodes_left_degree + 2)) - 1;
        }

        let mut pq = PriorityQueue::new();

        for i in 0..num_check_nodes {
            pq.push((i, self.check_nodes_left_degree), (self.check_nodes_left_degree, rng.next_u64()));
        }

        let mut message_node_degrees: Vec<_> = message_node_degrees.into_iter().enumerate().collect();

        shuffle(&mut message_node_degrees, &mut rng);

        message_node_degrees.sort_unstable_by_key(|(_, x)| Reverse(*x));

        for &(idx, quant) in message_node_degrees.iter() {
            let mut used = Vec::with_capacity(quant);
            for _ in 0..quant {
                let ((check_idx, spaces_left), _) = pq.pop().expect("Not enough check nodes left to connect this data node.");
                let offset = res[check_idx];
                res[offset + self.check_nodes_left_degree - spaces_left] = idx;
                if spaces_left > 1 {
                    used.push((check_idx, spaces_left - 1));
                }
            }

            for item in used.into_iter() {
                pq.push(item, (item.1, rng.next_u64()));
            }
        }

        assert!(pq.is_empty());

        for i in 0..num_check_nodes {
            let offset = res[i];

            if i == 0 {
                res[offset + self.check_nodes_left_degree] = self.message_len;
            } else {
                res[offset + self.check_nodes_left_degree] = self.message_len + i - 1;
                res[offset + self.check_nodes_left_degree + 1] = self.message_len + i;
            }
        }

        res
    }
}

pub struct Hamming84IraCodeGenerator {
    pub message_len: usize,
    pub check_nodes_left_degree: usize,
    pub hamming_nodes_min_dist: usize,
    pub hamming_nodes_ratio: f64,
    pub message_nodes_degree_ratios: HashMap<usize, f64>,
}

pub struct Hamming84IraRepresentation {
    pub left_connections: Vec<Vec<usize>>,
    pub occupied_positions: Vec<bool>
}

fn priority_func(connections_left: usize, degree: usize, rng: &mut impl Rng) -> (u64, u64) {
    ((u64::MAX as f64 * (connections_left as f64 / degree as f64)).floor() as u64, rng.next_u64())
}

impl Hamming84IraCodeGenerator {

    fn num_positions(&self, num_check_nodes: usize) -> usize {
        2 * ((num_check_nodes - 1) / (8 * self.hamming_nodes_min_dist))
    }

    fn position_idx(&self, position_num: usize) -> usize {
        1 + (position_num/2) * 8 * self.hamming_nodes_min_dist + (position_num % 2) * self.hamming_nodes_min_dist
    }

    fn approx_avg_check_node_right_degree(&self) -> f64 {
        let base_ratio = 0.25 / self.hamming_nodes_min_dist as f64;
        let ratio = base_ratio * self.hamming_nodes_ratio;

        ratio * 3.0 + (1.0 - ratio) * self.check_nodes_left_degree as f64
    }

    fn generate_initial_message_node_degrees(&self, rng: &mut impl Rng) -> Vec<usize> {
        let mut message_node_degrees = Vec::new();

        let mut degree_ratios_sorted: Vec<_> = self.message_nodes_degree_ratios.iter().map(|(&x, &y)| (x, y)).collect();

        degree_ratios_sorted.sort_unstable_by_key(|(x, _)| *x);

        'outer: for (quant, ratio) in degree_ratios_sorted.into_iter() {
            let num = (self.message_len as f64 * ratio).round() as usize;

            for _ in 0..num {
                if message_node_degrees.len() == self.message_len {
                    break 'outer;
                }
                message_node_degrees.push(quant);
            }
        }

        let avg_node_degree = message_node_degrees.iter().sum::<usize>() as f64 / message_node_degrees.len() as f64;
        let avg_node_degree = avg_node_degree.round() as usize;

        while message_node_degrees.len() < self.message_len {
            message_node_degrees.push(avg_node_degree);
        }

        shuffle(&mut message_node_degrees, rng);

        message_node_degrees
    }

    fn generate_check_node_data(&self, message_nodes_degree_sum: usize, rng: &mut impl Rng) -> (Vec<bool>, usize, usize) {

        let initial_num_check_nodes = (message_nodes_degree_sum as f64 / self.approx_avg_check_node_right_degree()).ceil() as usize;

        let num_positions = self.num_positions(initial_num_check_nodes);

        let num_occupied_positions = (num_positions as f64 * self.hamming_nodes_ratio).floor() as usize;

        let mut check_nodes_right_degree_sum = 3 * num_occupied_positions + self.check_nodes_left_degree * (initial_num_check_nodes - num_occupied_positions);

        let mut num_check_nodes = initial_num_check_nodes;

        while check_nodes_right_degree_sum < message_nodes_degree_sum {
            check_nodes_right_degree_sum += self.check_nodes_left_degree;
            num_check_nodes += 1;
        }

        let num_positions = self.num_positions(num_check_nodes);

        let mut occupied_positions = vec![false; num_positions];

        for i in 0..num_occupied_positions {
            occupied_positions[i] = true;
        }

        shuffle(&mut occupied_positions, rng);

        (occupied_positions, num_check_nodes, check_nodes_right_degree_sum)
    }

    fn match_check_and_message_nodes(&self, message_node_degrees: &[usize], check_node_degrees: &[usize], rng: &mut impl Rng) -> Vec<Vec<usize>> {

        assert_eq!(
            message_node_degrees.iter().sum::<usize>(),
            check_node_degrees.iter().sum::<usize>()
        );

        let mut connections = vec![vec![]; check_node_degrees.len()];

        let mut pq = PriorityQueue::new();

        for (idx, &degree) in check_node_degrees.iter().enumerate() {
            pq.push((idx, degree), priority_func(degree, degree, rng));
        }

        let mut message_node_degrees: Vec<_> = message_node_degrees.iter().map(|x| *x).enumerate().collect();
        message_node_degrees.sort_unstable_by_key(|&(_, deg)| Reverse(deg));

        for (msg_idx, degree) in message_node_degrees.into_iter() {
            let mut used = Vec::with_capacity(degree);

            for _ in 0..degree {
                let ((chk_idx, conn_left), _) = pq.pop().expect("Not enough check nodes left to connect this data node.");

                connections[chk_idx].push(msg_idx);

                if conn_left > 1 {
                    used.push((chk_idx, conn_left - 1));
                }
            }

            for (chk_idx, conn_left) in used.into_iter() {
                pq.push((chk_idx, conn_left), priority_func(conn_left, check_node_degrees[chk_idx], rng));
            }
        }

        connections
    }

    fn generate_internal_representation(&self, rng: &mut impl Rng) -> Hamming84IraRepresentation {

        let mut message_node_degrees = self.generate_initial_message_node_degrees(rng);

        let mut message_node_degree_sum = message_node_degrees.iter().sum::<usize>();

        let (occupied_positions, num_check_nodes, check_nodes_right_degree_sum) = self.generate_check_node_data(
            message_node_degree_sum,
            rng
        );

        while message_node_degree_sum < check_nodes_right_degree_sum {
            let idx = rng.gen_range(0..message_node_degrees.len());
            message_node_degree_sum += 1;
            message_node_degrees[idx] += 1;
        }

        let mut check_node_degrees = vec![self.check_nodes_left_degree; num_check_nodes];

        for i in 0..occupied_positions.len() {
            if occupied_positions[i] {
                check_node_degrees[self.position_idx(i)] = 3;
            }
        }

        let left_connections = self.match_check_and_message_nodes(&message_node_degrees, &check_node_degrees, rng);

        Hamming84IraRepresentation {
            left_connections,
            occupied_positions
        }
    }

    fn get_right_connections(&self, num_message_nodes: usize, num_check_nodes: usize, is_hamming_node: &[bool]) -> Vec<Vec<usize>> {


        let mut connections = vec![vec![]; num_check_nodes];

        let mut additional_connections: HashMap<usize, usize> = HashMap::new();

        let mut data_node_idx = num_message_nodes;

        for i in 0..num_check_nodes {
            if !is_hamming_node[i] {
                if i == 0 {
                    connections[i].push(data_node_idx);
                    data_node_idx += 1;
                } else {
                    connections[i].push(data_node_idx - 1);
                    connections[i].push(data_node_idx);
                    if let Some(additional_connection) = additional_connections.remove(&i) {
                        connections[i].push(additional_connection);
                    }
                    data_node_idx += 1;
                }
            } else {
                assert_ne!(i, 0);

                assert!(i + 6 * self.hamming_nodes_min_dist < num_check_nodes);

                connections[i].push(data_node_idx - 1);
                connections[i].push(data_node_idx + 0);
                connections[i].push(data_node_idx + 1);
                connections[i].push(data_node_idx + 2);
                connections[i].push(data_node_idx + 3);

                additional_connections.insert(i + 2 * self.hamming_nodes_min_dist, data_node_idx);
                additional_connections.insert(i + 4 * self.hamming_nodes_min_dist, data_node_idx + 1);
                additional_connections.insert(i + 6 * self.hamming_nodes_min_dist, data_node_idx + 2);

                data_node_idx += 4;
            }
        }

        connections
    }

    fn convert_to_common_representation(&self, repr: Hamming84IraRepresentation) -> IraEccCodeCompressedRepresentation {

        let num_check_nodes = repr.left_connections.len();

        let mut is_hamming_node = vec![false; num_check_nodes];

        for i in 0..repr.occupied_positions.len() {
            if repr.occupied_positions[i] {
                is_hamming_node[self.position_idx(i)] = true;
            }
        }

        let num_hamming_nodes = repr.occupied_positions.iter().map(|&x| if x { 1 } else { 0 }).sum::<usize>();

        let encoded_len = self.message_len + 4 * num_hamming_nodes + (repr.left_connections.len() - num_hamming_nodes);

        let right_connections = self.get_right_connections(self.message_len, num_check_nodes, &is_hamming_node);

        let connections: Vec<_> = repr.left_connections.into_iter().zip(right_connections.into_iter()).map(|(mut l, mut r)| {
            l.append(&mut r);
            l.sort_unstable();
            l
        }).collect();

        let mut res = IraEccCodeCompressedRepresentation {
            data: vec![0; 3 + connections.len() + connections.iter().map(|x| x.len()).sum::<usize>()]
        };

        res.set_message_len(self.message_len);
        res.set_encoded_len(encoded_len);
        res.set_num_check_nodes(connections.len());

        let mut data_ptr = connections.len();

        for i in 0..connections.len() {

            let type_addition = ((if !is_hamming_node[i] {
                0
            } else {
                CheckNodeType::HAMMING84 as u8
            }) as usize) << 56;

            res[i] = data_ptr + type_addition;

            for &c in connections[i].iter() {
                res[data_ptr] = c;
                data_ptr += 1;
            }
        }

        res
    }

    pub fn generate(&self, seed: u64) -> IraEccCodeCompressedRepresentation {
        assert!((1.0 - self.message_nodes_degree_ratios.iter().map(|(_, &x)| x).sum::<f64>()) < 1e-8);

        let mut act_seed = [0u8; 32];
        act_seed[0..8].copy_from_slice(&seed.to_le_bytes()[..]);
        let mut rng = ChaChaRng::from_seed(act_seed);

        let repr = self.generate_internal_representation(&mut rng);
        self.convert_to_common_representation(repr)
    }
}