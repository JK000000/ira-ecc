use indicatif::ProgressBar;
use rand::{random, Rng, thread_rng};
use rand_distr::Normal;
use crate::{Bit, bits_to_belief, DecodingOptions, EccCode};

#[derive(Debug)]
pub enum BenchmarkChannelType {
    BSC,
    BEC,
    AWGN,
}

pub struct BenchmarkSettings {
    pub channel_type: BenchmarkChannelType,
    pub channel_error_params: Vec<f64>,
    pub max_num_iterations: usize,
    pub min_num_iterations: usize,
    pub break_threshold: usize,
    pub decoding_options: DecodingOptions,
    pub show_progress: bool,
}

impl Default for BenchmarkSettings {
    fn default() -> Self {
        BenchmarkSettings {
            channel_type: BenchmarkChannelType::BSC,
            channel_error_params: vec![0.0, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            max_num_iterations: 1000,
            min_num_iterations: 100,
            break_threshold: 30,
            decoding_options: DecodingOptions::default(),
            show_progress: true,
        }
    }
}

#[derive(Debug)]
pub struct BenchmarkResult {
    pub channel_error_param: f64,
    pub shannon_limit: f64,
    pub shannon_ratio: f64,
    pub accuracy: f64,
    pub accuracy_top_5: f64,
    pub accuracy_top_10: f64,
    pub average_wrong_bits: f64,
    pub probability_assessment_accuracy: f64,
}

pub fn benchmark_code(code: &mut impl EccCode, settings: BenchmarkSettings) -> Vec<BenchmarkResult> {
    let mut results = vec![];

    let mut pb = match settings.show_progress {
        true => Some(ProgressBar::new((settings.channel_error_params.len() * settings.max_num_iterations) as u64)),
        false => None
    };

    for &channel_error_param in settings.channel_error_params.iter() {
        let shannon_limit = match settings.channel_type {
            BenchmarkChannelType::BSC => { channel_error_param * channel_error_param.log2() + (1.0 - channel_error_param) * (1.0 - channel_error_param).log2() + 1.0 }
            BenchmarkChannelType::BEC => { 1.0 - channel_error_param }
            BenchmarkChannelType::AWGN => { 0.5 * (1.0 + 0.25 / channel_error_param.powi(2)).log2() }
        };

        let shannon_ratio = code.rate() / shannon_limit;

        let mut message;
        let mut belief_data;
        let mut decoded_bits = vec![0 as Bit; code.message_len()];

        let mut num_iterations: usize = 0;
        let mut num_correct: usize = 0;
        let mut num_correct_top5: usize = 0;
        let mut num_correct_top10: usize = 0;
        let mut num_probability_correct: usize = 0;
        let mut total_wrong_bits: usize = 0;


        for _ in 0..settings.max_num_iterations {
            message = vec![0 as Bit; code.encoded_len()];

            for i in 0..code.message_len() {
                message[i] = thread_rng().gen_range(0..2);
            }

            code.encode(&mut message);

            match settings.channel_type {
                BenchmarkChannelType::BSC => {
                    let belief_level = ((1.0 - channel_error_param) / channel_error_param).log2();

                    belief_data = bits_to_belief(&message, belief_level);

                    belief_data.iter_mut().for_each(|x| {
                        if random::<f64>() < channel_error_param {
                            *x = -(*x);
                        }
                    });
                }
                BenchmarkChannelType::BEC => {
                    belief_data = bits_to_belief(&message, settings.decoding_options.max_message);

                    belief_data.iter_mut().for_each(|x| {
                        if random::<f64>() < channel_error_param {
                            *x = 0.0;
                        }
                    });
                }
                BenchmarkChannelType::AWGN => {
                    let dist = Normal::new(0.0, channel_error_param).unwrap();
                    let distorted_bits = message.iter().map(|&x| x as f64 + thread_rng().sample(dist));
                    belief_data = distorted_bits.map(|x| {
                        let dist_0 = ((x - 0.0) / channel_error_param).powi(2);
                        let dist_1 = ((x - 1.0) / channel_error_param).powi(2);
                        return -dist_0 + dist_1;
                    }).collect();
                }
            }


            let decoding_res = code.decode(&belief_data, settings.decoding_options);

            let mut sorted_by_uncertainty: Vec<(usize, f64)> = decoding_res.data.iter().enumerate().map(|(idx, &val)| (idx, val.abs())).collect();
            sorted_by_uncertainty.sort_unstable_by(|(_, a), (_, b)| a.total_cmp(b));

            decoding_res.to_bits(&mut decoded_bits);

            let mut wrong_bits = 0;

            let mut wrong_bit_outside_top5 = false;
            let mut wrong_bit_outside_top10 = false;

            for i in 0..code.message_len() {
                if decoded_bits[i] != message[i] {
                    wrong_bits += 1;

                    let mut outside5 = true;
                    let mut outside10 = true;

                    for j in 0..10 {
                        if sorted_by_uncertainty[j].0 == i {
                            outside10 = false;
                            if i < 5 {
                                outside5 = false;
                            }
                            break;
                        }
                    }

                    wrong_bit_outside_top5 |= outside5;
                    wrong_bit_outside_top10 |= outside10;
                }
            }

            num_iterations += 1;

            if wrong_bits == 0 {
                num_correct += 1;
            }

            if !wrong_bit_outside_top5 {
                num_correct_top5 += 1;
            }

            if !wrong_bit_outside_top10 {
                num_correct_top10 += 1;
            }

            let prob = decoding_res.correctness_probability();

            if (prob > 0.5 && wrong_bits == 0) || (prob > 0.2 && prob < 0.7 && wrong_bits == 1) ||
                (prob > 0.01 && prob < 0.5 && wrong_bits == 2) ||
                (prob > 0.0005 && prob < 0.05 && wrong_bits > 2 && wrong_bits <= 10) || (prob < 0.001 && wrong_bits > 10) {
                num_probability_correct += 1;
            }

            total_wrong_bits += wrong_bits;

            if let Some(pb) = &mut pb {
                pb.inc(1);
            }

            if num_iterations >= settings.min_num_iterations {
                if num_correct >= settings.break_threshold && num_correct <= num_iterations - settings.break_threshold &&
                    num_correct_top10 >= settings.break_threshold && num_correct_top10 <= num_iterations - settings.break_threshold {
                    if let Some(pb) = &mut pb {
                        pb.inc((settings.max_num_iterations - num_iterations) as u64);
                    }

                    break;
                }
            }
        }

        results.push(BenchmarkResult {
            channel_error_param,
            shannon_limit,
            shannon_ratio,
            accuracy: num_correct as f64 / num_iterations as f64,
            accuracy_top_5: num_correct_top5 as f64 / num_iterations as f64,
            accuracy_top_10: num_correct_top10 as f64 / num_iterations as f64,
            average_wrong_bits: total_wrong_bits as f64 / num_iterations as f64,
            probability_assessment_accuracy: num_probability_correct as f64 / num_iterations as f64,
        });
    }

    results
}

