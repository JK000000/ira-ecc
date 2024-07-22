use indicatif::ProgressBar;
use ordered_float::OrderedFloat;
use priority_queue::PriorityQueue;
use rand::{random, Rng, thread_rng};
use rand_distr::Normal;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use crate::{Bit, bits_to_belief, DecodingOptions, EccCode};

#[derive(Debug, Clone)]
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


#[derive(Clone)]
pub struct BerEstimationSettings {
    pub channel_type: BenchmarkChannelType,
    pub channel_param_range: (f64, f64),
    pub decoding_options: DecodingOptions,
    pub num_refinements: usize,
    pub statistical_confidence_level: f64,
    pub confidence_interval_radius_db: f64,
    pub confidence_interval_around_zero: f64,
    pub max_iterations_per_point: usize,
    pub min_iterations_per_point: usize,
    pub show_progress: bool,
}

impl Default for BerEstimationSettings {
    fn default() -> Self {
        BerEstimationSettings {
            channel_type: BenchmarkChannelType::BSC,
            channel_param_range: (0.0, 0.5),
            decoding_options: DecodingOptions::default(),
            num_refinements: 40,
            statistical_confidence_level: 0.99,
            confidence_interval_radius_db: 0.01,
            confidence_interval_around_zero: 1e-4,
            max_iterations_per_point: 1000,
            min_iterations_per_point: 10,
            show_progress: true
        }
    }
}

fn ber_estimation_round(channel_error_param: f64, code: &mut impl EccCode, channel_type: &BenchmarkChannelType, decoding_options: &DecodingOptions) -> usize {
    let mut belief_data;
    let mut message = vec![0 as Bit; code.encoded_len()];

    for i in 0..code.message_len() {
        message[i] = thread_rng().gen_range(0..2);
    }

    code.encode(&mut message);

    match channel_type {
        BenchmarkChannelType::BSC => {
            let belief_level = ((1.0 - channel_error_param) / channel_error_param).ln();

            belief_data = bits_to_belief(&message, belief_level);

            belief_data.iter_mut().for_each(|x| {
                if random::<f64>() < channel_error_param {
                    *x = -(*x);
                }
            });
        }
        BenchmarkChannelType::BEC => {
            belief_data = bits_to_belief(&message, decoding_options.max_message);

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


    let decoding_res = code.decode(&belief_data, decoding_options.clone());

    let mut decoded_msg = vec![0 as Bit; code.message_len()];

    decoding_res.to_bits(&mut decoded_msg);

    let mut wrong_bits = 0;

    for i in 0..code.message_len() {
        if decoded_msg[i] != message[i] {
            wrong_bits += 1;
        }
    }

    wrong_bits
}

fn estimate_ber_at_point(code: &mut impl EccCode, settings: &BerEstimationSettings, point: f64) -> f64 {

    let mut sum = 0;
    let mut n = 0;

    for i in 0..settings.max_iterations_per_point {
        n = i;

        if i >= settings.min_iterations_per_point {
            let alpha = 1.0 - settings.statistical_confidence_level;
            let lower;

            if sum == 0 {
                lower = 0.0;
            } else {
                let dist = ChiSquared::new((2*sum) as f64).unwrap();
                lower = dist.inverse_cdf(alpha * 0.5);
            }

            let dist = ChiSquared::new((2*sum+2) as f64).unwrap();
            let upper = dist.inverse_cdf(1.0 - alpha * 0.5);

            let lower_ber = lower / (n as f64 * code.message_len() as f64);
            let upper_ber = upper / (n as f64 * code.message_len() as f64);


            if lower_ber == 0.0 && upper_ber < settings.confidence_interval_around_zero {
                break;
            } else if lower_ber != 0.0 && (lower_ber - upper_ber).abs() < 2.0 * settings.confidence_interval_radius_db {
                break;
            }
        }

        let wrong_bits = ber_estimation_round(point, code, &settings.channel_type, &settings.decoding_options);
        sum += wrong_bits;
    }

    sum as f64 / (n as f64 * code.message_len() as f64)
}

fn priority_func(a: f64, b: f64) -> u64 {
    ((a-b).abs() * 100.0).floor() as u64
}

pub fn estimate_ber(code: &mut impl EccCode, settings: BerEstimationSettings) -> Vec<(f64, f64)> {

    let mut pb = match settings.show_progress {
        true => Some(ProgressBar::new(settings.num_refinements as u64)),
        false => None
    };

    let mut pq = PriorityQueue::new();

    let begin_ber = estimate_ber_at_point(code, &settings, settings.channel_param_range.0);
    let end_ber = estimate_ber_at_point(code, &settings, settings.channel_param_range.1);

    let mut res = Vec::new();

    res.push((settings.channel_param_range.0, begin_ber));
    res.push((settings.channel_param_range.1, end_ber));

    pq.push(((OrderedFloat(settings.channel_param_range.0), OrderedFloat(settings.channel_param_range.1)), (OrderedFloat(begin_ber), OrderedFloat(end_ber))), priority_func(begin_ber, end_ber));

    for _ in 0..settings.num_refinements {
        let (((b,e), (b_ber, e_ber)), _) = pq.pop().unwrap();

        let mid = (b+e) / 2.0;

        let mid_ber = OrderedFloat(estimate_ber_at_point(code, &settings, f64::from(mid)));

        res.push((f64::from(mid), f64::from(mid_ber)));

        pq.push(((b, mid), (b_ber, mid_ber)), priority_func(f64::from(b_ber), f64::from(mid_ber)));
        pq.push(((mid, e), (mid_ber, e_ber)), priority_func(f64::from(mid_ber), f64::from(e_ber)));

        if let Some(pb) = &mut pb {
            pb.inc(1);
        }
    }

    res.sort_unstable_by_key(|(x, _)| (x * 10000.0).floor() as i64);

    res
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
                    let belief_level = ((1.0 - channel_error_param) / channel_error_param).ln();

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

