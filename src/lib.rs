pub mod standard_codes;

#[cfg(test)]
mod tests;
pub mod benchmark;
pub mod generator;
pub mod compressed;


use std::collections::HashMap;
use rayon::prelude::*;
use rand::Rng;
use crate::compressed::IraEccCodeCompressedRepresentation;
use crate::NodeType::{CHECK, DATA};

type Bit = u8;

#[derive(Clone)]
enum NodeType {
    DATA,
    CHECK,
}

#[derive(Clone)]
struct NodeData {
    node_type: NodeType,
    connections: Vec<usize>,
    received_messages: Vec<(f64, usize)>,
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
            max_message: 200.0,
        }
    }
}

pub struct DecodingResult {
    data: Vec<f64>,
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
    /// Generates message for use in belief propagation algorithm
    fn generate_message_for(&self, recipient: usize) -> f64 {
        match self.node_type {
            // Message from data node to check node represents belief of the data node in its value
            // It is calculated by summing received messages except for the message sent by receiver of current message
            NodeType::DATA => self.received_messages.iter().filter(|(_, sender)| *sender != recipient).map(|(val, _)| *val).sum(),
            // Message from check node to data node represents belief of the check node in the value of the data node
            // Here, message is calculated using tanh product rule
            // Beliefs are log likelihood values log2(p_0 / p_1), so positive belief means 0 is more likely than 1
            // With such convention, multiplying beliefs behaves similarly to adding values modulo 2
            NodeType::CHECK => 2.0 * self.received_messages.iter().filter(|(_, sender)| *sender != recipient).map(|(val, _)| (0.5 * (*val)).tanh()).product::<f64>().atanh()
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


pub trait EccCode {
    fn decode(&mut self, data: &[f64], options: DecodingOptions) -> DecodingResult;
    fn encode(&self, data: &mut [Bit]);

    fn message_len(&self) -> usize;
    fn encoded_len(&self) -> usize;

    fn rate(&self) -> f64 {
        self.message_len() as f64 / self.encoded_len() as f64
    }
}

const MIN_EXTERNAL_MESSAGE: f64 = 1e-2;

fn stabilizing_func(val: f64, max_message: f64) -> f64 {
    (val / max_message).tanh() * max_message
}


/// Constructor and helper functions
impl IraEccCode {
    pub fn from_representation(data: &IraEccCodeCompressedRepresentation) -> IraEccCode {
        let num_check_nodes = data.encoded_len() - data.message_len();

        let mut res = IraEccCode {
            message_len: data.message_len(),
            encoded_len: data.encoded_len(),
            data_nodes: vec![NodeData {
                node_type: DATA,
                connections: vec![],
                received_messages: vec![],
            }; data.encoded_len()],
            check_nodes: vec![NodeData {
                node_type: CHECK,
                connections: vec![],
                received_messages: vec![],
            }; num_check_nodes],
        };

        for i in 0..num_check_nodes {
            for j in 0..data.check_nodes_left_degree() {
                let data_node_idx = data[j + i * data.check_nodes_left_degree()];
                res.check_nodes[i].connections.push(data_node_idx);
                res.data_nodes[data_node_idx].connections.push(i);
            }
        }

        for i in 0..num_check_nodes {
            res.check_nodes[i].connections.push(data.message_len() + i);
            res.data_nodes[data.message_len() + i].connections.push(i);
            if i > 0 {
                res.check_nodes[i].connections.push(data.message_len() + i - 1);
                res.data_nodes[data.message_len() + i - 1].connections.push(i);
            }
        }

        res
    }

    pub fn to_representation(&self) -> IraEccCodeCompressedRepresentation {
        assert!(self.check_nodes.len() > 0);

        let check_nodes_left_degree = self.check_nodes[0].connections.len() - 1;

        let mut res = IraEccCodeCompressedRepresentation {
            data: vec![0usize; 3 + check_nodes_left_degree * self.check_nodes.len()]
        };

        res.set_message_len(self.message_len);
        res.set_encoded_len(self.encoded_len);
        res.set_check_nodes_left_degree(check_nodes_left_degree);

        for i in 0..self.check_nodes.len() {
            for (j, &val) in self.check_nodes[i].connections.iter().filter(|&&x| x < self.message_len).enumerate() {
                res[j + check_nodes_left_degree * i] = val;
            }
        }

        res
    }

    pub fn rate(&self) -> f64 {
        self.message_len as f64 / self.encoded_len as f64
    }
}


/// Implementation of belief propagation algorithm for decoding
impl IraEccCode {

    /// 'External' source of belief is the data being decoded
    fn send_external_nodes(&mut self, data: &[f64]) {
        assert!(data.len() >= self.encoded_len);

        self.data_nodes.par_iter_mut().enumerate().for_each(|(idx, node)| {
            if data[idx].abs() >= MIN_EXTERNAL_MESSAGE {
                node.received_messages.push((data[idx], usize::MAX));
            }
        });
    }

    /// Sends messages to 'receiver_node_type' nodes from the other type nodes
    fn send_internal_nodes(&mut self, receiver_node_type: NodeType) {
        match receiver_node_type {
            NodeType::DATA => {
                self.data_nodes.par_iter_mut().enumerate().for_each(|(idx, node)| {
                    for &sender_idx in node.connections.iter() {
                        node.received_messages.push((self.check_nodes[sender_idx].generate_message_for(idx), sender_idx));
                    }
                });
            }
            NodeType::CHECK => {
                self.check_nodes.par_iter_mut().enumerate().for_each(|(idx, node)| {
                    for &sender_idx in node.connections.iter() {
                        node.received_messages.push((self.data_nodes[sender_idx].generate_message_for(idx), sender_idx));
                    }
                });
            }
        };
    }

    fn clear_internal_nodes(&mut self, node_type: NodeType) {
        match node_type {
            NodeType::DATA => { self.data_nodes.par_iter_mut() }
            NodeType::CHECK => { self.check_nodes.par_iter_mut() }
        }.for_each(|node| {
            node.received_messages.clear();
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
        self.send_internal_nodes(NodeType::CHECK);
        self.stabilize_internal_nodes(NodeType::CHECK, options.max_message);

        self.clear_internal_nodes(NodeType::DATA);
        self.send_external_nodes(data);
        self.send_internal_nodes(NodeType::DATA);
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

        let mut curr: Bit = 0;

        for i in 0..(self.encoded_len - self.message_len) {
            curr = ((self.check_nodes[i].connections.iter().filter(|&x| *x < self.message_len).map(|&x| data[x] as u32).sum::<u32>() + curr as u32) % 2) as Bit;
            data[i + self.message_len] = curr;
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
