pub mod standard_codes;

#[cfg(test)]
mod tests;
pub mod benchmark;
pub mod generator;
pub mod compressed;
mod hamming;
pub mod design;


use std::cmp::min;
use rayon::prelude::*;
use crate::compressed::IraEccCodeCompressedRepresentation;
use crate::hamming::{HAMMING84_CODEWORDS, hamming84_encode};
use crate::NodeType::{CHECK, DATA};

pub type Bit = u8;

#[derive(Clone)]
enum NodeType {
    DATA,
    CHECK,
}

#[repr(u8)]
#[derive(Clone)]
enum CheckNodeType {
    PARITY = 0,
    HAMMING84 = 1,
}

impl TryFrom<u8> for CheckNodeType {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::PARITY),
            1 => Ok(Self::HAMMING84),
            _ => Err(())
        }
    }
}

#[derive(Clone)]
struct NodeData {
    node_type: NodeType,
    check_node_type: CheckNodeType,
    connections: Vec<usize>,
    received_messages: Vec<(f64, usize)>,
    preprocessed_messages: Vec<(usize, f64)>
}

pub struct IraEccCode {
    message_len: usize,
    encoded_len: usize,
    data_nodes: Vec<NodeData>,
    check_nodes: Vec<NodeData>,
}


#[derive(Copy, Clone)]
pub struct DecodingOptions {
    pub min_rounds: u32,
    pub max_rounds: u32,
    pub epsilon: f64,
    pub max_message: f64,
}

impl Default for DecodingOptions {
    fn default() -> Self {
        DecodingOptions {
            min_rounds: 5,
            max_rounds: 50,
            epsilon: 1e-2,
            max_message: 50.0,
        }
    }
}

pub struct DecodingResult {
    pub data: Vec<f64>,
}

impl DecodingResult {
    pub fn correctness_probability(&self) -> f64 {
        self.data.par_iter().map(|&x| x.abs().exp()).map(|x| (x / (1.0 + x)).ln()).sum::<f64>().exp()
    }

    pub fn to_bits(&self, data: &mut [Bit]) {
        assert!(data.len() >= self.data.len());

        data.par_iter_mut().enumerate().for_each(|(idx, x)| *x = if self.data[idx] >= 0.0 { 0 } else { 1 })
    }

    pub fn to_bytes(&self, data: &mut [u8]) {
        assert_eq!(self.data.len() % 8, 0);
        assert!(data.len() >= self.data.len() / 8);

        data.par_iter_mut().enumerate().for_each(|(idx, x)| {
            *x = 0;

            for i in 0..8 {
                let bit: Bit = if self.data[8 * idx + i] >= 0.0 { 0 } else { 1 };
                *x |= bit * (1u8 << i);
            }
        });
    }
}

impl NodeData {

    fn new_data(connections: Vec<usize>) -> NodeData {
        NodeData {
            node_type: NodeType::DATA,
            check_node_type: CheckNodeType::PARITY,
            connections,
            received_messages: vec![],
            preprocessed_messages: vec![],
        }
    }

    fn new_check(connections: Vec<usize>, check_node_type: CheckNodeType) -> NodeData {
        NodeData {
            node_type: NodeType::CHECK,
            check_node_type,
            connections,
            received_messages: vec![],
            preprocessed_messages: vec![],
        }
    }

    fn preprocess_messages(&mut self) {
        match self.node_type {
            // Message from data node to check node represents belief of the data node in its value
            // It is calculated by summing received messages except for the message sent by receiver of current message
            DATA => {
                let sum: f64 = self.received_messages.iter().map(|(val, _)| *val).sum();

                for &(val, sender) in self.received_messages.iter() {
                    self.preprocessed_messages.push((sender, sum - val));
                }
            }
            CHECK => {
                match self.check_node_type {
                    // Message from check node to data node represents belief of the check node in the value of the data node
                    // Here, message is calculated using tanh product rule
                    // Beliefs are log likelihood values log2(p_0 / p_1), so positive belief means 0 is more likely than 1
                    // With such convention, multiplying beliefs behaves similarly to adding values modulo 2
                    CheckNodeType::PARITY => {
                        let mut prod = 1.0;

                        let mut zero_occurred = false;

                        let mut vals = vec![1.0; self.received_messages.len()];

                        for (idx, &(val, _)) in self.received_messages.iter().enumerate() {
                            if val == 0.0 {
                                if !zero_occurred {
                                    zero_occurred = true;
                                } else {
                                    prod = 0.0;
                                    break;
                                }
                            } else {
                                vals[idx] = (0.5 * val).tanh();
                                prod *= vals[idx];
                            }
                        }

                        let atanh_prod = 2.0 * prod.atanh();

                        for (idx, &(val, sender)) in self.received_messages.iter().enumerate() {
                            if val == 0.0 {
                                self.preprocessed_messages.push((sender, atanh_prod));
                            } else {
                                if zero_occurred {
                                    self.preprocessed_messages.push((sender, 0.0));
                                } else {
                                    self.preprocessed_messages.push((sender, 2.0 * (prod / vals[idx]).atanh()));
                                }
                            }
                        }
                    }
                    CheckNodeType::HAMMING84 => {
                        assert_eq!(self.received_messages.len(), 8);
                        let mut p0 = [0.0; 8];
                        let mut p1 = [0.0; 8];

                        for c in HAMMING84_CODEWORDS.iter() {
                            let mut base_p = 1.0;
                            let mut p = [0.0; 8];

                            for i in 0..8 {
                                if c[i] == 0 {
                                    p[i] = ((0.5 * self.received_messages[i].0).tanh() + 1.0) / 2.0;
                                } else {
                                    p[i] = (-(0.5 * self.received_messages[i].0).tanh() + 1.0) / 2.0;
                                }
                            }

                            let mut zero_occurred = false;

                            for i in 0..8 {
                                if p[i] == 0.0 {
                                    if !zero_occurred {
                                        zero_occurred = true;
                                    } else {
                                        base_p = 0.0;
                                        break;
                                    }
                                } else {
                                    base_p *= p[i];
                                }
                            }

                            for i in 0..8 {
                                p[i] = if p[i] == 0.0 {
                                    base_p
                                } else {
                                    if zero_occurred {
                                        0.0
                                    } else {
                                        base_p / p[i]
                                    }
                                };
                            }

                            for i in 0..8 {
                                if c[i] == 0 {
                                    p0[i] += p[i];
                                } else {
                                    p1[i] += p[i];
                                }
                            }
                        }

                        for i in 0..8 {
                            self.preprocessed_messages.push((self.received_messages[i].1, p0[i].ln() - p1[i].ln()))
                        }
                    }
                }
            }
        }
    }

    /// Generates messages for use in belief propagation algorithm
    fn get_message_for(&self, recipient: usize) -> f64 {
        if let Ok(idx) = self.preprocessed_messages.binary_search_by_key(&recipient, |&(r, _)| r) {
            self.preprocessed_messages[idx].1
        } else {
            match self.node_type {
                DATA => {
                    if self.received_messages.len() == 1 && self.received_messages[0].1 == usize::MAX {
                        self.received_messages[0].0
                    } else {
                        panic!()
                    }
                }
                CHECK => {
                    panic!()
                }
            }
        }
    }
}

pub fn bits_to_belief(bits: &[Bit], belief_level: f64) -> Vec<f64> {
    bits.iter().map(|&b| match b {
        0 => belief_level,
        1 => -belief_level,
        _ => 0.0
    }).collect()
}


/// Encoding an IRA is a simple, exact, linear-time operation
/// In contrary, decoding is a hard problem and approximate algorithms are used
/// This implementation uses 'belief propagation' algorithm
/// In consequence, the decoding process operates on beliefs rather than bits
/// Belief is essentially a random variable of the correct (before errors) value of a bit
/// Here beliefs are represented as 64-bit float equal to
/// log2( P( bit=0 ) / P( bit=1 ) )   (so positive belief indicates 0 bit and negative indicates 1)
/// If input beliefs are close to true value of the above, then output beliefs are also
/// reasonable probability values, so the decoding algorithm is able to estimate how
/// successful was the decoding process
pub trait EccCode {
    fn decode(&mut self, data: &[f64], options: DecodingOptions) -> DecodingResult;
    fn encode(&self, data: &mut [Bit]);

    fn message_len(&self) -> usize;
    fn encoded_len(&self) -> usize;

    fn rate(&self) -> f64 {
        self.message_len() as f64 / self.encoded_len() as f64
    }
}

fn stabilizing_func(val: f64, max_message: f64) -> f64 {
    (val / max_message).tanh() * max_message
}


/// Constructor and helper functions
impl IraEccCode {
    pub fn from_representation(data: &IraEccCodeCompressedRepresentation) -> IraEccCode {
        let mut res = IraEccCode {
            message_len: data.message_len(),
            encoded_len: data.encoded_len(),
            data_nodes: vec![NodeData::new_data(vec![]); data.encoded_len()],
            check_nodes: vec![NodeData::new_check(vec![], CheckNodeType::PARITY); data.num_check_nodes()],
        };

        for i in 0..data.num_check_nodes() {
            let dt = data[i];
            let check_node_type = CheckNodeType::try_from((dt >> 56) as u8).unwrap();
            res.check_nodes[i].check_node_type = check_node_type;
            let start_offset = (dt << 8) >> 8;
            let end_offset = if i < data.num_check_nodes() - 1 {
                (data[i + 1] << 8) >> 8
            } else {
                data.data.len() - 3
            };

            res.check_nodes[i].connections.reserve(end_offset - start_offset);

            for j in start_offset..end_offset {
                res.check_nodes[i].connections.push(data[j]);
                res.data_nodes[data[j]].connections.push(i);
            }
        }

        for n in res.data_nodes.iter_mut() {
            n.connections.sort();
            n.connections.shrink_to_fit();
        }

        for n in res.check_nodes.iter_mut() {
            n.connections.sort();
            n.connections.shrink_to_fit();
        }

        res
    }

    pub fn to_representation(&self) -> IraEccCodeCompressedRepresentation {
        assert!(self.check_nodes.len() > 0);

        let mut degree_sum = 0;

        for n in self.check_nodes.iter() {
            degree_sum += n.connections.len();
        }

        let mut res = IraEccCodeCompressedRepresentation {
            data: vec![0usize; 3 + self.check_nodes.len() + degree_sum]
        };

        res.set_message_len(self.message_len);
        res.set_encoded_len(self.encoded_len);
        res.set_num_check_nodes(self.check_nodes.len());

        let mut curr_offset = self.check_nodes.len();

        for (i, node) in self.check_nodes.iter().enumerate() {
            let node_type: u8 = node.check_node_type.clone() as u8;
            res[i] = curr_offset + ((node_type as usize) << 56);

            for (j, &conn) in node.connections.iter().enumerate() {
                res[curr_offset + j] = conn;
            }

            curr_offset += node.connections.len();
        }

        res
    }

    pub fn rate(&self) -> f64 {
        self.message_len as f64 / self.encoded_len as f64
    }

    fn shortest_cycle_recu(&self, node: (NodeType, usize), parent: usize, curr_depth: usize, data_depths: &mut [usize], check_depths: &mut [usize]) -> usize {
        let mut res = usize::MAX;

        match node.0 {
            DATA => {
                if data_depths[node.1] != usize::MAX {
                    return if curr_depth > data_depths[node.1] {
                        curr_depth - data_depths[node.1]
                    } else {
                        usize::MAX
                    };
                }

                data_depths[node.1] = curr_depth;
            }
            CHECK => {
                if check_depths[node.1] != usize::MAX {
                    return if curr_depth > check_depths[node.1] {
                        curr_depth - check_depths[node.1]
                    } else {
                        usize::MAX
                    };
                }

                check_depths[node.1] = curr_depth;
            }
        }

        let neighbors = match node.0 {
            DATA => &self.data_nodes[node.1].connections,
            CHECK => &self.check_nodes[node.1].connections
        };

        let neighbors_type = match node.0 {
            DATA => NodeType::CHECK,
            CHECK => NodeType::DATA
        };

        for &v in neighbors.iter() {
            if v == parent {
                continue;
            }

            res = min(res, self.shortest_cycle_recu((neighbors_type.clone(), v), node.1, curr_depth + 1, data_depths, check_depths));
        }

        res
    }

    pub fn shortest_cycle(&self) -> usize {
        let mut data_depths = vec![usize::MAX; self.data_nodes.len()];
        let mut check_depths = vec![usize::MAX; self.check_nodes.len()];

        let res = self.shortest_cycle_recu((NodeType::DATA, 0), usize::MAX, 1, &mut data_depths, &mut check_depths);

        data_depths.into_iter().for_each(|x| if x == usize::MAX {
            panic!("Graph has more connected components than 1.");
        });

        check_depths.into_iter().for_each(|x| if x == usize::MAX {
            panic!("Graph has more connected components than 1.");
        });

        res
    }
}


/// Implementation of belief propagation algorithm for decoding
impl IraEccCode {
    /// 'External' source of belief is the data being decoded
    fn send_external_nodes(&mut self, data: &[f64]) {
        assert!(data.len() >= self.encoded_len);

        self.data_nodes.par_iter_mut().enumerate().for_each(|(idx, node)| {
            node.received_messages.push((data[idx], usize::MAX));
        });
    }

    /// Sends messages to 'receiver_node_type' nodes from nodes of the other type
    fn send_internal_nodes(&mut self, receiver_node_type: NodeType) {
        match receiver_node_type {
            NodeType::DATA => {
                self.data_nodes.par_iter_mut().enumerate().for_each(|(idx, node)| {
                    for &sender_idx in node.connections.iter() {
                        node.received_messages.push((self.check_nodes[sender_idx].get_message_for(idx), sender_idx));
                    }
                });
            }
            NodeType::CHECK => {
                self.check_nodes.par_iter_mut().enumerate().for_each(|(idx, node)| {
                    for &sender_idx in node.connections.iter() {
                        node.received_messages.push((self.data_nodes[sender_idx].get_message_for(idx), sender_idx));
                    }
                });
            }
        };
    }

    fn preprocess_internal_nodes(&mut self, node_type: NodeType) {
        match node_type {
            NodeType::DATA => { self.data_nodes.par_iter_mut() }
            NodeType::CHECK => { self.check_nodes.par_iter_mut() }
        }.for_each(|node| {
            node.preprocess_messages();
        });
    }

    fn clear_internal_nodes(&mut self, node_type: NodeType) {
        match node_type {
            NodeType::DATA => { self.data_nodes.par_iter_mut() }
            NodeType::CHECK => { self.check_nodes.par_iter_mut() }
        }.for_each(|node| {
            node.received_messages.clear();
            node.preprocessed_messages.clear();
        });
    }

    /// Too large belief values are attenuated for numerical stability
    fn stabilize_internal_nodes(&mut self, node_type: NodeType, max_message: f64) {
        match node_type {
            NodeType::DATA => { self.data_nodes.par_iter_mut() }
            NodeType::CHECK => { self.check_nodes.par_iter_mut() }
        }.for_each(|node| {
            node.received_messages.iter_mut().for_each(|(val, _)| {
                *val = stabilizing_func(*val, max_message);
            });
        });
    }

    /// Updates predictions and returns the larges change that occurred since the previous round
    fn advance_message_predictions(&self, last_predictions: &mut [f64]) -> f64 {
        last_predictions[0..self.message_len].par_iter_mut().enumerate().map(|(idx, pred)| {
            let next: f64 = self.data_nodes[idx].received_messages.iter().map(|(val, _)| *val).sum();
            let diff: f64 = (next - *pred).abs();
            *pred = next;
            diff
        }).reduce(|| 0.0, |a, b| a.max(b))
    }

    fn decoding_round(&mut self, data: &[f64], predictions: &mut [f64], options: &DecodingOptions) -> f64 {
        assert_eq!(data.len(), self.encoded_len);
        assert_eq!(predictions.len(), self.message_len);

        self.clear_internal_nodes(NodeType::CHECK);
        self.preprocess_internal_nodes(NodeType::DATA);
        self.send_internal_nodes(NodeType::CHECK);
        self.stabilize_internal_nodes(NodeType::CHECK, options.max_message);

        self.clear_internal_nodes(NodeType::DATA);
        self.preprocess_internal_nodes(NodeType::CHECK);
        self.send_internal_nodes(NodeType::DATA);
        self.send_external_nodes(data);
        self.stabilize_internal_nodes(NodeType::DATA, options.max_message);

        self.advance_message_predictions(predictions)
    }

    pub fn ira_decode(&mut self, data: &[f64], options: DecodingOptions) -> DecodingResult {
        let mut predictions = vec![0.0; self.message_len];

        self.clear_internal_nodes(NodeType::CHECK);
        self.clear_internal_nodes(NodeType::DATA);
        self.send_external_nodes(data);

        for iter in 0..options.max_rounds {
            let largest_update = self.decoding_round(data, &mut predictions, &options);

            if iter >= options.min_rounds && largest_update < options.epsilon {
                break;
            }
        }

        DecodingResult {
            data: predictions
        }
    }


    pub fn ira_encode(&self, data: &mut [Bit]) {
        assert!(data.len() >= self.encoded_len);

        let mut data_pos = self.message_len;

        for i in 0..self.check_nodes.len() {
            match self.check_nodes[i].check_node_type {
                CheckNodeType::PARITY => {
                    let mut val = 0 as Bit;

                    for j in 0..(self.check_nodes[i].connections.len() - 1) {
                        assert!(self.check_nodes[i].connections[j] < data_pos);
                        val = ((val as u8 + data[self.check_nodes[i].connections[j]]) % 2) as Bit;
                    }

                    assert_eq!(self.check_nodes[i].connections.last().unwrap().clone(), data_pos);

                    data[data_pos] = val;
                    data_pos += 1;
                }
                CheckNodeType::HAMMING84 => {
                    let mut d = [0 as Bit; 4];

                    assert_eq!(self.check_nodes[i].connections.len(), 8);

                    for j in 0..4 {
                        let conn = self.check_nodes[i].connections[j];
                        assert!(conn < data_pos);
                        d[j] = data[conn];
                    }

                    let p = hamming84_encode(&d);

                    for j in 0..4 {
                        assert_eq!(self.check_nodes[i].connections[4+j], data_pos);
                        data[data_pos] = p[j];
                        data_pos += 1;
                    }
                }
            }
        }
    }
}

impl EccCode for IraEccCode {
    fn decode(&mut self, data: &[f64], options: DecodingOptions) -> DecodingResult {
        self.ira_decode(data, options)
    }

    fn encode(&self, data: &mut [Bit]) {
        self.ira_encode(data);
    }

    fn message_len(&self) -> usize {
        self.message_len
    }

    fn encoded_len(&self) -> usize {
        self.encoded_len
    }
}
