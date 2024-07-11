use rand::{random, thread_rng};
use crate::generator::IraCodeGenerator;
use crate::NodeType::{CHECK, DATA};
use super::*;

#[test]
fn simple_graph() {
    let mut code = IraEccCode {
        message_len: 3,
        encoded_len: 6,
        data_nodes: vec![
            NodeData {
                node_type: DATA,
                connections: vec![0, 1],
                received_messages: vec![],
            },
            NodeData {
                node_type: DATA,
                connections: vec![0, 2],
                received_messages: vec![],
            },
            NodeData {
                node_type: DATA,
                connections: vec![1, 2],
                received_messages: vec![],
            },
            NodeData {
                node_type: DATA,
                connections: vec![0, 1],
                received_messages: vec![],
            },
            NodeData {
                node_type: DATA,
                connections: vec![1, 2],
                received_messages: vec![],
            },
            NodeData {
                node_type: DATA,
                connections: vec![2],
                received_messages: vec![],
            },
        ],
        check_nodes: vec![
            NodeData {
                node_type: CHECK,
                connections: vec![0, 1, 3],
                received_messages: vec![],
            },
            NodeData {
                node_type: CHECK,
                connections: vec![0, 2, 3, 4],
                received_messages: vec![],
            },
            NodeData {
                node_type: CHECK,
                connections: vec![1, 2, 4, 5],
                received_messages: vec![],
            },
        ],
    };

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

    encoded_belief[5] = 0.0;

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

    encoded_belief[3] = 0.0;
    encoded_belief[4] = 0.0;
    encoded_belief[5] = 0.0;

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

    encoded_belief[2] = 0.0;
    encoded_belief[3] = 0.0;
    encoded_belief[4] = 0.0;
    encoded_belief[5] = 0.0;

    let decoding_res = code.decode(&encoded_belief, DecodingOptions::default());

    let correctness_prob = decoding_res.correctness_probability();

    assert!(correctness_prob < 0.5);
    assert!(correctness_prob > 0.49);

    let mut decoded_bits = [0, 0, 0];

    decoding_res.to_bits(&mut decoded_bits);

    assert!(decoded_bits == [0, 1, 1] || decoded_bits == [0, 1, 0]);
}

#[test]
fn generated_code() {
    let message_len = 4096;
    let flip_probability: f64 = 0.01;

    let generator = IraCodeGenerator {
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