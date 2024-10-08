use std::collections::HashMap;
use rand::{random, Rng, thread_rng};
use crate::generator::{ClassicIraCodeGenerator, Hamming84IraCodeGenerator};
use crate::example_codes::edge_ratios_to_node_ratios;
use super::*;

#[test]
fn simple_graph() {
    let mut code = IraEccCode {
        message_len: 3,
        encoded_len: 6,
        data_nodes: vec![
            NodeData::new_data(vec![(0,0), (1,0)]),
            NodeData::new_data(vec![(0,1), (2,0)]),
            NodeData::new_data(vec![(1,1), (2,1)]),
            NodeData::new_data(vec![(0,2), (1,2)]),
            NodeData::new_data(vec![(1,3), (2,2)]),
            NodeData::new_data(vec![(2,3)]),
        ],
        check_nodes: vec![
            NodeData::new_check(vec![(0,0), (1,0), (3,0)], CheckNodeType::PARITY),
            NodeData::new_check(vec![(0,1), (2,0), (3,1), (4,0)], CheckNodeType::PARITY),
            NodeData::new_check(vec![(1,1), (2,1), (4,1), (5,0)], CheckNodeType::PARITY),
        ],
    };

    assert_eq!(code.shortest_cycle(), 4);

    let mut msg = [0, 1, 1, 0, 0, 0];

    code.encode(&mut msg);

    assert_eq!(msg, [0, 1, 1, 1, 0, 0]);

    let encoded_belief = bits_to_belief(&msg, 10.0);

    let decoding_res = code.decode(&encoded_belief, DecodingOptions::default());

    let correctness_prob = decoding_res.correctness_probability();

    assert!(correctness_prob > 0.999);

    let mut decoded_bits = [0, 0, 0];

    decoding_res.to_bits(&mut decoded_bits);

    assert_eq!(decoded_bits, [0, 1, 1]);


    let mut msg = [0, 1, 1, 0, 0, 0];

    code.encode(&mut msg);

    assert_eq!(msg, [0, 1, 1, 1, 0, 0]);

    let mut encoded_belief = bits_to_belief(&msg, 10.0);

    encoded_belief[5] = 0;

    let decoding_res = code.decode(&encoded_belief, DecodingOptions::default());

    let correctness_prob = decoding_res.correctness_probability();

    assert!(correctness_prob > 0.999);

    let mut decoded_bits = [0, 0, 0];

    decoding_res.to_bits(&mut decoded_bits);

    assert_eq!(decoded_bits, [0, 1, 1]);


    let mut msg = [0, 1, 1, 0, 0, 0];

    code.encode(&mut msg);

    assert_eq!(msg, [0, 1, 1, 1, 0, 0]);

    let mut encoded_belief = bits_to_belief(&msg, 10.0);

    encoded_belief[3] = 0;
    encoded_belief[4] = 0;
    encoded_belief[5] = 0;

    let decoding_res = code.decode(&encoded_belief, DecodingOptions::default());

    let correctness_prob = decoding_res.correctness_probability();

    assert!(correctness_prob > 0.99);

    let mut decoded_bits = [0, 0, 0];

    decoding_res.to_bits(&mut decoded_bits);

    assert_eq!(decoded_bits, [0, 1, 1]);


    let mut msg = [0, 1, 1, 0, 0, 0];

    code.encode(&mut msg);

    assert_eq!(msg, [0, 1, 1, 1, 0, 0]);

    let mut encoded_belief = bits_to_belief(&msg, 10.0);

    encoded_belief[2] = 0;
    encoded_belief[3] = 0;
    encoded_belief[4] = 0;
    encoded_belief[5] = 0;

    let decoding_res = code.decode(&encoded_belief, DecodingOptions::default());

    let correctness_prob = decoding_res.correctness_probability();

    assert!(correctness_prob < 0.5);
    assert!(correctness_prob > 0.49);

    let mut decoded_bits = [0, 0, 0];

    decoding_res.to_bits(&mut decoded_bits);

    assert!(decoded_bits == [0, 1, 1] || decoded_bits == [0, 1, 0]);
}

#[test]
fn simple_graph_with_hamming_node() {
    let mut code = IraEccCode {
        message_len: 4,
        encoded_len: 11,
        data_nodes: vec![
            NodeData::new_data(vec![(0,0), (1,0)]),
            NodeData::new_data(vec![(1,1), (2,0), (3,0)]),
            NodeData::new_data(vec![(0,1), (2,1)]),
            NodeData::new_data(vec![(2,2), (3,1)]),
            NodeData::new_data(vec![(0,2), (1,2)]),
            NodeData::new_data(vec![(1,3), (2,3)]),
            NodeData::new_data(vec![(2,4), (3,2)]),
            NodeData::new_data(vec![(2,5), (3,3)]),
            NodeData::new_data(vec![(2,6), (3,4)]),
            NodeData::new_data(vec![(2,7), (3,5)]),
            NodeData::new_data(vec![(3,6)]),
        ],
        check_nodes: vec![
            NodeData::new_check(vec![(0,0), (2,0), (4,0)], CheckNodeType::PARITY),
            NodeData::new_check(vec![(0,1), (1,0), (4,1), (5,0)], CheckNodeType::PARITY),
            NodeData::new_check(vec![(1,1), (2,1), (3,0), (5,1), (6,0), (7,0), (8,0), (9,0)], CheckNodeType::HAMMING84),
            NodeData::new_check(vec![(1,2), (3,1), (6,1), (7,1), (8,1), (9,1), (10,0)], CheckNodeType::PARITY),
        ],
    };

    let mut msg = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0];

    code.encode(&mut msg);

    assert_eq!(msg, [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1]);

    let encoded_belief = bits_to_belief(&msg, 30.0);

    let decoding_res = code.decode(&encoded_belief, DecodingOptions::default());

    let correctness_prob = decoding_res.correctness_probability();

    assert!(correctness_prob > 0.999);

    let mut decoded_bits = [0, 0, 0, 0];

    decoding_res.to_bits(&mut decoded_bits);

    assert_eq!(decoded_bits, [0, 1, 1, 0]);




    let mut msg = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0];

    code.encode(&mut msg);

    assert_eq!(msg, [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1]);

    msg[6] = 1;

    let mut encoded_belief = bits_to_belief(&msg, 4.0);

    encoded_belief[0] = 0;
    encoded_belief[5] = 0;

    let decoding_res = code.decode(&encoded_belief, DecodingOptions::default());

    let correctness_prob = decoding_res.correctness_probability();

    assert!(correctness_prob > 0.999);

    let mut decoded_bits = [0, 0, 0, 0];

    decoding_res.to_bits(&mut decoded_bits);

    assert_eq!(decoded_bits, [0, 1, 1, 0]);


}

#[test]
fn generated_code() {
    let message_len = 4096;
    let flip_probability: f64 = 0.04;

    let generator = ClassicIraCodeGenerator {
        message_len,
        check_nodes_left_degree: 6,
        message_nodes_degree_ratios: HashMap::from([
            (4, 0.14),
            (5, 0.36),
            (6, 0.16),
            (7, 0.1),
            (8, 0.08),
            (9, 0.05),
            (10, 0.03),
            (11, 0.02),
            (12, 0.02),
            (13, 0.02),
            (14, 0.02),
            (15, 0.02)
        ]),
    };

    let representation = generator.generate(42);

    let mut code = IraEccCode::from_representation(&representation);

    println!("Message size: {}, Encoded size: {}, Rate: {:.3}", message_len, code.encoded_len, message_len as f64 / code.encoded_len as f64);

    let mut message = vec![0 as Bit; code.encoded_len];

    for i in 0..message_len {
        message[i] = thread_rng().gen_range(0..2);
    }

    code.encode(&mut message);

    let belief_level = ((1.0 - flip_probability) / flip_probability).ln();

    let mut belief_data = bits_to_belief(&message, belief_level);

    belief_data.iter_mut().for_each(|x| {
        if random::<f64>() < flip_probability {
            *x = -(*x);
        }
    });

    let decoding_res = code.decode(&belief_data, DecodingOptions {
        min_rounds: 5,
        max_rounds: 50,
        epsilon: 1e-2,
        max_message: 50.0,
    });

    println!("Correctness probability: {}", decoding_res.correctness_probability());

    let mut decoded_bits = vec![0 as Bit; message_len];

    decoding_res.to_bits(&mut decoded_bits);

    let mut wrong_bits = 0;

    for i in 0..message_len {
        if decoded_bits[i] != message[i] {
            wrong_bits += 1;
        }
    }

    assert_eq!(wrong_bits, 0);
}

#[test]
fn generated_code_with_hamming_nodes() {
    let message_len = 8192;
    let flip_probability: f64 = 0.1;

    let generator = Hamming84IraCodeGenerator {
        message_len,
        check_nodes_left_degree: 4,
        hamming_nodes_min_dist: 15,
        hamming_nodes_ratio: 0.9,
        message_nodes_degree_ratios: edge_ratios_to_node_ratios(HashMap::from([
            (2, 0.03115),
            (3, 0.14991),
            (6, 0.04630),
            (7, 0.06217),
            (8, 0.08666),
            (10, 0.12644),
            (17, 0.03430),
            (18, 0.01506),
            (26, 0.00228),
            (27, 0.02258),
            (28, 0.21774),
            (29, 0.08021),
            (100, 0.12521)
        ])
        ),
    };

    let repr = generator.generate(43);

    let mut code = IraEccCode::from_representation(&repr);

    println!("Message size: {}, Encoded size: {}, Rate: {:.3}", message_len, code.encoded_len, message_len as f64 / code.encoded_len as f64);

    let mut message = vec![0 as Bit; code.encoded_len];

    for i in 0..message_len {
        message[i] = thread_rng().gen_range(0..2);
    }

    code.encode(&mut message);

    let belief_level = ((1.0 - flip_probability) / flip_probability).ln();

    let mut belief_data = bits_to_belief(&message, belief_level);

    belief_data.iter_mut().for_each(|x| {
        if random::<f64>() < flip_probability {
            *x = -(*x);
        }
    });

    let decoding_res = code.decode(&belief_data, DecodingOptions {
        min_rounds: 5,
        max_rounds: 50,
        epsilon: 1e-2,
        max_message: 50.0,
    });

    let mut decoded_bits = vec![0 as Bit; message_len];

    decoding_res.to_bits(&mut decoded_bits);

    let mut wrong_bits = 0;

    for i in 0..message_len {
        if decoded_bits[i] != message[i] {
            wrong_bits += 1;
        }
    }

    assert_eq!(wrong_bits, 0);
}
